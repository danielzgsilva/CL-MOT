from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from progress.bar import Bar
import torch
import numpy as np

from models.decode import mot_decode
from models.data_parallel import DataParallel
from utils.utils import AverageMeter
from utils.debugger import Debugger


class ModleWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModleWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):
        outputs = dict()

        # Feed image to model
        outputs['orig'] = self.model(batch['img'])

        # When self-supervised learning, we also feed the horizontally flipped version
        if 'flipped_img' in batch:
            outputs['flipped'] = self.model(batch['flipped_img'])

        # Take loss
        loss, loss_stats = self.loss(outputs, batch)

        return outputs, loss, loss_stats


class BaseTrainer(object):
    def __init__(self, opt, model, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        self.loss_stats, self.loss = self._get_losses(opt)
        self.model_with_loss = ModleWithLoss(model, self.loss)

        params = []
        for param in self.loss.parameters():
            if param.requires_grad:
                params.append(param)
        self.optimizer.add_param_group({'params': params})

    def set_device(self, gpus, chunk_sizes, device):
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(
                self.model_with_loss, device_ids=gpus,
                chunk_sizes=chunk_sizes).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, data_loader):
        model_with_loss = self.model_with_loss

        if phase == 'train':
            model_with_loss.train()
        else:
            if len(self.opt.gpus) > 1:
                model_with_loss = self.model_with_loss.module

            model_with_loss.eval()
            torch.cuda.empty_cache()

        opt = self.opt
        results = {}

        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}

        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
        bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
        end = time.time()

        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)

            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True)

            outputs, loss, loss_stats = model_with_loss(batch)
            loss = loss.mean()

            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter_id, num_iters, phase=phase, total=bar.elapsed_td, eta=bar.eta_td)

            for l in avg_loss_stats:
                avg_loss_stats[l].update(loss_stats[l].mean().item(), batch['img'].size(0))
                Bar.suffix = Bar.suffix + '|{} {:.3f} '.format(l, avg_loss_stats[l].avg)

            if not opt.hide_data_time:
                Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                                          '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
            if opt.no_bar:
                print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
            else:
                bar.next()

            if self.opt.debug > 0:
                self.debug(batch, outputs, iter_id, dataset=data_loader.dataset)

            if opt.test:
                self.save_result(outputs, batch, results)

            del outputs, loss, loss_stats, batch

        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        return ret, results

    def debug(self, batch, outputs, iter_id, dataset):
        opt = self.opt
        ds = batch['meta']['dataset']

        # Ground truth detections
        dets_gt = batch['meta']['gt_det']

        # Process predictions on original image
        output = outputs['orig'][-1]
        reg = output['reg'] if self.opt.reg_offset else None
        _dets, inds = mot_decode(output['hm'], output['wh'], reg=reg,
                                 cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        _dets = _dets.detach().cpu().numpy()
        dets = {'bboxes': _dets[:, :, :4], 'scores': _dets[:, :, 4], 'clses': _dets[:, :, 5]}

        # Process predictions on flipped image
        flipped_dets = None
        if 'flipped' in outputs:
            flipped_output = outputs['flipped'][-1]
            f_reg = flipped_output['reg'] if self.opt.reg_offset else None
            _f_dets, f_inds = mot_decode(flipped_output['hm'], flipped_output['wh'], reg=f_reg,
                                         cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
            _f_dets = _f_dets.detach().cpu().numpy()
            flipped_dets = {'bboxes': _f_dets[:, :, :4], 'scores': _f_dets[:, :, 4], 'clses': _f_dets[:, :, 5]}

        # Batch size should be 1 if debug is set
        for i in range(opt.batch_size):
            debugger = Debugger(opt=opt, dataset=dataset)

            img = batch['img'][i].detach().cpu().numpy().transpose(1, 2, 0)
            # img = np.clip(((img * dataset.std + dataset.mean) * 255.), 0, 255).astype(np.uint8)
            img = np.clip(img * 255., 0, 255).astype(np.uint8)[:, :, ::-1]  # RGB to BGR for opencv

            flipped_img = None
            if 'flipped_img' in batch:
                flipped_img = batch['flipped_img'][i].detach().cpu().numpy().transpose(1, 2, 0)
                flipped_img = np.clip(flipped_img * 255., 0, 255).astype(np.uint8)[:, :, ::-1] # RGB to BGR

            pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm')
            debugger.add_blend_img(img, gt, 'gt_hm')

            # Predictions
            debugger.add_img(img, img_id='out_pred')
            for k in range(len(dets['scores'][i])):
                if dets['scores'][i, k] > opt.vis_thresh:
                    debugger.add_coco_bbox(dets['bboxes'][i, k] * opt.down_ratio, dets['clses'][i, k],
                                           dets['scores'][i, k], img_id='out_pred')

            # Ground truth
            debugger.add_img(img, img_id='out_gt')
            for k in range(len(dets_gt['scores'][i])):
                if dets_gt['scores'][i][k] > opt.vis_thresh:
                    debugger.add_coco_bbox(dets_gt['bboxes'][i][k] * opt.down_ratio, dets_gt['clses'][i][k],
                                           dets_gt['scores'][i][k], img_id='out_gt')

            if flipped_img is not None and flipped_dets is not None:
                # Flipped predictions
                debugger.add_img(flipped_img, img_id='flipped_pred')
                for k in range(len(flipped_dets['scores'][i])):
                    if flipped_dets['scores'][i, k] > opt.vis_thresh:
                        debugger.add_coco_bbox(flipped_dets['bboxes'][i, k] * opt.down_ratio,
                                               flipped_dets['clses'][i, k],
                                               flipped_dets['scores'][i, k], img_id='flipped_pred')

                # Flipped ground truth
                debugger.add_img(flipped_img, img_id='flipped_gt')
                for k in range(len(dets_gt['flipped_scores'][i])):
                    if dets_gt['flipped_scores'][i][k] > opt.vis_thresh:
                        debugger.add_coco_bbox(dets_gt['flipped_bboxes'][i][k] * opt.down_ratio,
                                               dets_gt['flipped_clses'][i][k],
                                               dets_gt['flipped_scores'][i][k], img_id='flipped_gt')

            debugger.add_dataset_tag(ds[i])

            if opt.debug == 4:
                debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
            else:
                debugger.show_all_imgs(pause=True)

    def val(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)

    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)

    def save_result(self, output, batch, results):
        raise NotImplementedError

    def _get_losses(self, opt):
        raise NotImplementedError
