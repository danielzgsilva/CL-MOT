from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.decode import mot_decode
from models.losses import FocalLoss, TripletLoss, NTXentLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.utils import _sigmoid, _tranpose_and_gather_feat, extract_feats
from utils.post_process import ctdet_post_process
from utils.utils import create_img_labels

from .base_trainer import BaseTrainer


class MotLoss(torch.nn.Module):
    def __init__(self, opt, loss_states):
        super(MotLoss, self).__init__()
        self.opt = opt
        self.loss_states = loss_states

        # Loss for heatmap
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()

        # Loss for offsets
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
                        RegLoss() if opt.reg_loss == 'sl1' else None

        # Loss for object sizes
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
                        NormRegL1Loss() if opt.norm_wh else \
                        RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg

        # Supervised loss for object IDs
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)

        if opt.unsup and opt.mlp_layer:
            # projection MLP layer for contrastive loss
            self.MLP = nn.Sequential(
                nn.Linear(opt.reid_dim, opt.reid_dim),
                nn.ReLU(),
                nn.Linear(opt.reid_dim, opt.mlp_dim)
            )

        else:
            # FC layer for supervised object ID prediction
            self.classifier = nn.Linear(opt.reid_dim, opt.nID)
            for param in self.classifier.parameters():
                param.requires_grad = False

        # Self supervised loss for object embeddings
        self.SelfSupLoss = NTXentLoss(opt.device, opt.temp) if opt.unsup_loss == 'nt_xent' else \
                            TripletLoss(opt.device, 'batch_all', opt.margin) if opt.unsup_loss == 'triplet_all' else \
                            TripletLoss(opt.device, 'batch_hard', opt.margin) if opt.unsup_loss == 'triplet_hard' else None

        if opt.unsup and self.SelfSupLoss is None:
            raise ValueError('{} is not a supported self-supervised loss. '.format(opt.unsup_loss) + \
                             'Choose nt_xent, triplet_all, or triplet_hard')

        self.emb_scale = math.sqrt(2) * math.log(opt.nID - 1)
        self.s_det = nn.Parameter(-1.85 * torch.ones(1), requires_grad=False)
        self.s_id = nn.Parameter(-1.05 * torch.ones(1), requires_grad=False)

    def forward(self, output_dict, batch):
        opt = self.opt
        loss_results = {loss: 0 for loss in self.loss_states}

        outputs = output_dict['orig']
        flipped_outputs = output_dict['flipped'] if 'flipped' in output_dict else None
        pre_outputs = output_dict['pre'] if 'pre' in output_dict else None
        pre_flipped_outputs = output_dict['pre_flipped'] if 'pre_flipped' in output_dict else None

        # Take loss at each scale
        for s in range(opt.num_stacks):
            output = outputs[s]

            # Supervised loss on predicted heatmap
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])
            loss_results['hm'] += self.crit(output['hm'], batch['hm']) / opt.num_stacks

            # Supervised loss on object sizes
            if opt.wh_weight > 0:
                if opt.dense_wh:
                    mask_weight = batch['dense_wh_mask'].sum() + 1e-4
                    loss_results['wh'] += (self.crit_wh(output['wh'] * batch['dense_wh_mask'],
                                                        batch['dense_wh'] * batch['dense_wh_mask']) /
                                           mask_weight) / opt.num_stacks
                else:
                    loss_results['wh'] += self.crit_reg(
                        output['wh'], batch['reg_mask'],
                        batch['ind'], batch['wh']) / opt.num_stacks

            # Supervised loss on offsets
            if opt.reg_offset and opt.off_weight > 0:
                loss_results['off'] += self.crit_reg(output['reg'], batch['reg_mask'],
                                                     batch['ind'], batch['reg']) / opt.num_stacks

            # Extract object embeddings
            id_head = extract_feats(output['id'], batch['ind'], batch['reg_mask'])
            # Get GT ID labels
            id_labels = batch['ids'][batch['reg_mask'] > 0]

            # Supervised CE loss on object ID predictions
            if opt.id_weight > 0 and not opt.unsup:
                id_head *= self.emb_scale
                id_output = self.classifier(id_head).contiguous()
                loss_results['id'] += self.IDLoss(id_output, id_labels)

            # Self-supervised loss using contrastive learning
            if opt.unsup and flipped_outputs is not None:
                flipped_output = flipped_outputs[s]
                flipped_id_head = extract_feats(flipped_output['id'], batch['flipped_ind'], batch['flipped_reg_mask'])

                # Local object identities 1...N for N objects in the current scene
                flipped_id_labels = batch['flipped_ids'][batch['flipped_reg_mask'] > 0]

                # Track which image in the batch each embedding is from
                img_labels = create_img_labels(batch['num_objs'])
                flipped_img_labels = create_img_labels(batch['flipped_num_objs'])

                if opt.off_center_vecs:
                    off_id_head = extract_feats(output['id'], batch['off_ind'], batch['reg_mask'])
                    id_head = torch.cat([id_head, off_id_head], dim=0)
                    id_labels = torch.cat([id_labels] * 2, dim=0)
                    img_labels *= 2

                    off_flipped_id_head = extract_feats(flipped_output['id'], batch['flipped_off_ind'],
                                                        batch['flipped_reg_mask'])
                    flipped_id_head = torch.cat([flipped_id_head, off_flipped_id_head], dim=0)
                    flipped_id_labels = torch.cat([flipped_id_labels] * 2, dim=0)
                    flipped_img_labels *= 2

                id_head = torch.cat([id_head, flipped_id_head], dim=0)
                id_labels = torch.cat([id_labels, flipped_id_labels])
                img_labels = img_labels + flipped_img_labels

                if opt.pre_img and pre_outputs is not None and pre_flipped_outputs is not None:
                    pre_output = pre_outputs[s]
                    pre_flipped_output = pre_flipped_outputs[s]

                    pre_id_head = extract_feats(pre_output['id'], batch['pre_ind'], batch['pre_reg_mask'])
                    pre_id_labels = batch['pre_ids'][batch['pre_reg_mask'] > 0]
                    pre_img_labels = create_img_labels(batch['pre_num_objs'])

                    pre_flipped_id_head = extract_feats(pre_flipped_output['id'],
                                                        batch['pre_flipped_ind'], batch['pre_flipped_reg_mask'])
                    pre_flipped_id_labels = batch['pre_flipped_ids'][batch['pre_flipped_reg_mask'] > 0]
                    pre_flipped_img_labels = create_img_labels(batch['pre_flipped_num_objs'])

                    if opt.off_center_vecs:
                        off_pre_id_head = extract_feats(pre_output['id'], batch['pre_off_ind'], batch['pre_reg_mask'])
                        pre_id_head = torch.cat([pre_id_head, off_pre_id_head], dim=0)
                        pre_id_labels = torch.cat([pre_id_labels] * 2, dim=0)
                        pre_img_labels *= 2

                        off_pre_flipped_id_head = extract_feats(pre_flipped_output['id'], batch['pre_flipped_off_ind'],
                                                                batch['pre_flipped_reg_mask'])
                        pre_flipped_id_head = torch.cat([pre_flipped_id_head, off_pre_flipped_id_head], dim=0)
                        pre_flipped_id_labels = torch.cat([pre_flipped_id_labels] * 2, dim=0)
                        pre_flipped_img_labels *= 2

                    id_head = torch.cat([id_head, pre_id_head, pre_flipped_id_head], dim=0)
                    id_labels = torch.cat([id_labels, pre_id_labels, pre_flipped_id_labels])
                    img_labels = img_labels + pre_img_labels + pre_flipped_img_labels

                # Feed embeddings through MLP layer
                if opt.mlp_layer:
                    id_head = self.MLP(id_head).contiguous()

                # Compute contrastive loss between reid features
                loss_results[opt.unsup_loss] += self.SelfSupLoss(id_head, id_labels, img_labels)

        # Total supervised loss on detections
        det_loss = opt.hm_weight * loss_results['hm'] + \
                   opt.wh_weight * loss_results['wh'] + \
                   opt.off_weight * loss_results['off']

        # Loss on embeddings
        id_loss = loss_results['id'] if not opt.unsup else loss_results[opt.unsup_loss]

        # Total of supervised and self-supervised losses on object embeddings
        total_loss = torch.exp(-self.s_det) * det_loss + \
                     torch.exp(-self.s_id) * id_loss + \
                     self.s_det + self.s_id

        total_loss *= 0.5
        loss_results['loss'] = total_loss

        return total_loss, loss_results


class MotTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(MotTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        # We always take these losses
        loss_states = ['loss', 'hm', 'wh']

        if opt.reg_offset:
            loss_states.append('off')

        # Use either contrastive or triplet loss on object embeddings if self-supervised training
        if opt.unsup:
            loss_states.append(opt.unsup_loss)

        # Standard cross entropy loss on object IDs when supervised training
        else:
            loss_states.append('id')

        loss = MotLoss(opt, loss_states)
        return loss_states, loss

    def save_result(self, outputs, batch, results):
        output = outputs['orig'][-1]
        reg = output['reg'] if self.opt.reg_offset else None

        dets, inds = mot_decode(output['hm'], output['wh'], reg=reg,
                                cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)

        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])

        dets_out = ctdet_post_process(dets.copy(), batch['meta']['c'].cpu().numpy(),
                                      batch['meta']['s'].cpu().numpy(),
                                      output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])

        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]
