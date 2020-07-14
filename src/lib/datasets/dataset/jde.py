import glob
import math
import os
import os.path as osp
import random
import time
from collections import OrderedDict

import cv2
import json
import numpy as np
import torch
import lap
from torch.utils.data import Dataset
from torchvision.transforms import transforms as T
from cython_bbox import bbox_overlaps as bbox_ious

from opts import opts
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import letterbox, random_affine
from utils.utils import xyxy2xywh, generate_anchors, xywh2xyxy, encode_delta
from tracker.matching import iou_distance, linear_assignment


class LoadImages:  # for inference
    def __init__(self, path, img_size=(1088, 608)):
        if os.path.isdir(path):
            image_format = ['.jpg', '.jpeg', '.png', '.tif']
            self.files = sorted(glob.glob('%s/*.*' % path))
            self.files = list(filter(lambda x: os.path.splitext(x)[1].lower() in image_format, self.files))
        elif os.path.isfile(path):
            self.files = [path]

        self.nF = len(self.files)  # number of image files
        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        assert self.nF > 0, 'No images found in ' + path

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nF:
            raise StopIteration
        img_path = self.files[self.count]

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        assert img0 is not None, 'Failed to load ' + img_path

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        # cv2.imwrite(img_path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return img_path, img, img0

    def __getitem__(self, idx):
        idx = idx % self.nF
        img_path = self.files[idx]

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        assert img0 is not None, 'Failed to load ' + img_path

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        return img_path, img, img0

    def __len__(self):
        return self.nF  # number of files


class LoadVideo:  # for inference
    def __init__(self, path, img_size=(1088, 608)):
        self.cap = cv2.VideoCapture(path)
        self.frame_rate = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.vw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vn = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        self.w, self.h = 1920, 1080
        print('Lenth of the video: {:d} frames'.format(self.vn))

    def get_size(self, vw, vh, dw, dh):
        wa, ha = float(dw) / vw, float(dh) / vh
        a = min(wa, ha)
        return int(vw * a), int(vh * a)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == len(self):
            raise StopIteration
        # Read image
        res, img0 = self.cap.read()  # BGR
        assert img0 is not None, 'Failed to load frame {:d}'.format(self.count)
        img0 = cv2.resize(img0, (self.w, self.h))

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        # cv2.imwrite(img_path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return self.count, img, img0

    def __len__(self):
        return self.vn  # number of files


class LoadImagesAndLabels:  # for training
    def __init__(self, path, img_size=(1088, 608), augment=False, transforms=None):
        with open(path, 'r') as file:
            self.img_files = file.readlines()
            self.img_files = [x.replace('\n', '') for x in self.img_files]
            self.img_files = list(filter(lambda x: len(x) > 0, self.img_files))

        self.label_files = [x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
                            for x in self.img_files]

        self.nF = len(self.img_files)  # number of image files
        self.width = img_size[0]
        self.height = img_size[1]
        self.augment = augment
        self.transforms = transforms

    def __getitem__(self, files_index):
        img_path = self.img_files[files_index]
        label_path = self.label_files[files_index]
        return self.get_data(img_path, label_path)

    def load_labels(self, path, unsup=False):
        if os.path.isfile(path):
            # [class, identity, x_center, y_center, width, height]
            labels = np.loadtxt(path, dtype=np.float32).reshape(-1, 6)
            num_objs = len(labels)

            # If self-supervised training, we have no GT object identities,
            # So we use local identities for each object (0...N-1 for N objects in an image)
            if unsup and num_objs > 0:
                labels[:, 1] = np.array([i for i in range(num_objs)])

        else:
            labels = np.array([])

        return labels

    def flip_labels(self, labels):
        num_objs = len(labels)
        flipped_labels = labels.copy()

        if num_objs > 0:
            flipped_labels[:, 2] = 1 - labels[:, 2]

        return flipped_labels

    def load_image(self, path):
        img = cv2.imread(path)  # BGR

        if img is None:
            raise ValueError('File corrupt {}'.format(path))

        return img

    def get_data(self, img_path, label_path, pre_img_path=None, pre_label_path=None, unsup=False):
        height = self.height
        width = self.width

        images = dict()
        labels = dict()

        # Load image and labels
        images['orig'] = self.load_image(img_path)
        labels['orig'] = self.load_labels(label_path, unsup)
        orig_h, orig_w, _ = images['orig'].shape

        # create augmented samples for contrastive learning
        if unsup:
            images['flipped'] = cv2.flip(images['orig'], 1)
            labels['flipped'] = self.flip_labels(labels['orig'])

            # Load t-1 frame
            if pre_img_path is not None and pre_label_path is not None:
                images['pre_orig'] = self.load_image(pre_img_path)
                labels['pre_orig'] = self.load_labels(pre_label_path, unsup)

                images['pre_flipped'] = cv2.flip(images['pre_orig'], 1)
                labels['pre_flipped'] = self.flip_labels(labels['pre_orig'])

                # Match bboxes in previous frame to the current frame's bboxes
                # SHOULD LOOK INTO IAMGES BEING DIFFERENT SIZES FROM THE SAME DATASET
                pre_xyxys = xywh2xyxy(labels['pre_orig'][:, 2:6])
                xyxys = xywh2xyxy(labels['orig'][:, 2:6])

                cost_mat = iou_distance(pre_xyxys, xyxys)
                cost, pre_matches, cur_matches = lap.lapjv(cost_mat, extend_cost=True, cost_limit=0.25)

                # Assign ID labels to previous frame bboxes based on IOU matching with current frame
                for i, match in enumerate(pre_matches):
                    labels['pre_orig'][i][1] = match

                # Match bboxes across the flipped images
                pre_flipped_xyxys = xywh2xyxy(labels['pre_flipped'][:, 2:6])
                flipped_xyxys = xywh2xyxy(labels['flipped'][:, 2:6])

                cost_mat = iou_distance(pre_flipped_xyxys, flipped_xyxys)
                cost, pre_matches, cur_matches = lap.lapjv(cost_mat, extend_cost=True, cost_limit=0.25)

                for i, match in enumerate(pre_matches):
                    labels['pre_flipped'][i][1] = match

        for key, img in images.items():
            h, w, _ = img.shape

            # Saturation and brightness augmentation by 50%
            if unsup:
                continue

            if self.augment:
                fraction = 0.50
                img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                S = img_hsv[:, :, 1].astype(np.float32)
                V = img_hsv[:, :, 2].astype(np.float32)

                a = (random.random() * 2 - 1) * fraction + 1
                S *= a
                if a > 1:
                    np.clip(S, a_min=0, a_max=255, out=S)

                a = (random.random() * 2 - 1) * fraction + 1
                V *= a
                if a > 1:
                    np.clip(V, a_min=0, a_max=255, out=V)

                img_hsv[:, :, 1] = S.astype(np.uint8)
                img_hsv[:, :, 2] = V.astype(np.uint8)
                cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

            img, ratio, padw, padh = letterbox(img, height=height, width=width)

            lbls = labels[key].copy()
            num_objs = len(lbls)

            # Normalize xywh bboxes to pixel xyxy format
            if num_objs > 0:
                lbls[:, 2] = ratio * w * (labels[key][:, 2] - labels[key][:, 4] / 2) + padw
                lbls[:, 3] = ratio * h * (labels[key][:, 3] - labels[key][:, 5] / 2) + padh
                lbls[:, 4] = ratio * w * (labels[key][:, 2] + labels[key][:, 4] / 2) + padw
                lbls[:, 5] = ratio * h * (labels[key][:, 3] + labels[key][:, 5] / 2) + padh

            # Augment image and labels
            if self.augment:
                img, lbls, M = random_affine(img, lbls, degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.50, 1.20))

            if num_objs > 0:
                # convert xyxy to xywh
                lbls[:, 2:6] = xyxy2xywh(lbls[:, 2:6].copy())  # / height
                lbls[:, 2] /= width
                lbls[:, 3] /= height
                lbls[:, 4] /= width
                lbls[:, 5] /= height

            if not unsup and self.augment:
                # random left-right flip during supervised learning
                if random.random() > 0.5:
                    img = np.fliplr(img)
                    if num_objs > 0:
                        lbls[:, 2] = 1 - lbls[:, 2]

            img = np.ascontiguousarray(img[:, :, ::-1])  # BGR to RGB

            if self.transforms is not None:
                img = self.transforms(img)

            images[key] = img
            labels[key] = lbls

        return images, labels, img_path, (orig_h, orig_w)

    def format_gt_det(self, gt_det):
        if len(gt_det['scores']) == 0:
            gt_det = {'bboxes': np.array([[0, 0, 1, 1]], dtype=np.float32),
                      'scores': np.array([1], dtype=np.float32),
                      'clses': np.array([0], dtype=np.float32),
                      'ids': np.array([0], dtype=np.float32),
                      'cts': np.array([[0, 0]], dtype=np.float32)
                      }
        else:
            gt_det = {k: np.array(gt_det[k], dtype=np.float32) for k in gt_det}

        return gt_det

    def __len__(self):
        return self.nF  # number of images


def collate_fn(batch):
    imgs, labels, paths, sizes = zip(*batch)
    batch_size = len(labels)
    imgs = torch.stack(imgs, 0)
    max_box_len = max([l.shape[0] for l in labels])
    labels = [torch.from_numpy(l) for l in labels]
    filled_labels = torch.zeros(batch_size, max_box_len, 6)
    labels_len = torch.zeros(batch_size)

    for i in range(batch_size):
        isize = labels[i].shape[0]
        if len(labels[i]) > 0:
            filled_labels[i, :isize, :] = labels[i]
        labels_len[i] = isize

    return imgs, filled_labels, paths, sizes, labels_len.unsqueeze(1)


class JointDataset(LoadImagesAndLabels):  # for training
    default_resolution = [1088, 608]
    mean = None
    std = None
    num_classes = 1

    def __init__(self, opt, root, paths, img_size=(1088, 608), augment=False, transforms=None):
        self.opt = opt
        self.img_files = OrderedDict()
        self.label_files = OrderedDict()
        self.img_num = OrderedDict()
        self.tid_num = OrderedDict()
        self.tid_start_index = OrderedDict()
        self.num_classes = 1

        # Self supervised training flag
        self.unsup = opt.unsup

        # Get image and annotation file names
        for ds, path in paths.items():
            with open(path, 'r') as file:
                self.img_files[ds] = file.readlines()
                self.img_files[ds] = [osp.join(root, x.strip()) for x in self.img_files[ds]]
                self.img_files[ds] = list(filter(lambda x: len(x) > 0, self.img_files[ds]))

            self.label_files[ds] = [
                x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
                for x in self.img_files[ds]]

        # Counting unique identities in each dataset
        for ds, label_paths in self.label_files.items():
            max_index = -1
            for lp in label_paths:
                lb = np.loadtxt(lp)
                if len(lb) < 1:
                    continue
                if len(lb.shape) < 2:
                    img_max = lb[1]
                else:
                    img_max = np.max(lb[:, 1])
                if img_max > max_index:
                    max_index = img_max
            self.tid_num[ds] = max_index + 1

        # Finding the first identity (unique object) in each dataset
        last_index = 0
        for i, (k, v) in enumerate(self.tid_num.items()):
            self.tid_start_index[k] = last_index
            last_index += v

        # Total identities in all datasets
        self.nID = int(last_index + 1)

        # Count images and find starting file index for each dataset
        self.cds = []
        self.nF = 0
        for i, (ds, img_files) in enumerate(self.img_files.items()):
            img_cnt = len(img_files)
            self.img_num[ds] = img_cnt
            self.nF += img_cnt
            self.cds.append(self.nF - img_cnt)

        self.width = img_size[0]
        self.height = img_size[1]
        self.max_objs = opt.K
        self.augment = augment
        self.transforms = transforms
        self.draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else draw_umich_gaussian

        print('Dataset Summary')
        print('=' * 100)
        print('Images per dataset: {}'.format(self.img_num))
        print('Identities per dataset: {}'.format(self.tid_num))
        print('Total images: {}'.format(self.nF))
        print('Total identities: {}'.format(self.nID))
        print('=' * 100)

    def add_to_dict(self, dict_, key, item):
        if item is not None:
            dict_[key] = item

    def build_targets(self, img, labels, ret, gt_det, full_ret=False, prepend=''):
        if prepend != '':
            prepend = prepend + '_'

        output_h = img.shape[1] // self.opt.down_ratio
        output_w = img.shape[2] // self.opt.down_ratio
        num_classes = self.num_classes
        num_objs = labels.shape[0]

        # object heat map
        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32) if full_ret else None

        # width and height of each object
        wh = np.zeros((self.max_objs, 2), dtype=np.float32) if full_ret else None

        # object center offset due to resizing and decimal error
        reg = np.zeros((self.max_objs, 2), dtype=np.float32) if full_ret else None

        # offsetted object centers
        off_ind = np.zeros((self.max_objs,), dtype=np.int64) if self.opt.off_center_vecs else None

        # object centers
        ind = np.zeros((self.max_objs,), dtype=np.int64)

        # mask representing the gt number of objects in this frame
        reg_mask = np.zeros((self.max_objs,), dtype=np.uint8)

        # object IDs
        ids = np.zeros((self.max_objs,), dtype=np.int64)

        # Ground truth metadata (used by debugger)
        gt_det[prepend + 'bboxes'] = []
        gt_det[prepend + 'scores'] = []
        gt_det[prepend + 'clses'] = []
        gt_det[prepend + 'ids'] = []
        gt_det[prepend + 'cts'] = []

        # Build gt target tensors
        for k in range(num_objs):
            label = labels[k]  # [class, identity, x_center, y_center, width, height]
            bbox = label[2:]
            cls_id = int(label[0])
            obj_id = int(label[1])
            bbox[[0, 2]] = bbox[[0, 2]] * output_w
            bbox[[1, 3]] = bbox[[1, 3]] * output_h
            bbox[0] = np.clip(bbox[0], 0, output_w - 1)
            bbox[1] = np.clip(bbox[1], 0, output_h - 1)
            h = bbox[3]
            w = bbox[2]

            if h > 0 and w > 0:
                ct = np.array([bbox[0], bbox[1]], dtype=np.float32)
                ct_int = ct.astype(np.int32)

                ind[k] = ct_int[1] * output_w + ct_int[0]
                ids[k] = label[1]
                reg_mask[k] = 1

                if full_ret:
                    radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                    radius = max(0, int(radius))
                    radius = self.opt.hm_gauss if self.opt.mse_loss else radius
                    self.draw_gaussian(hm[cls_id], ct_int, radius)

                    wh[k] = 1. * w, 1. * h
                    reg[k] = ct - ct_int

                if off_ind is not None:
                    # Randomly select point from within the object's bbox
                    w_offset = np.random.uniform(-w / 4, w / 4)
                    h_offset = np.random.uniform(-h / 2, h / 2)
                    off_ct = np.array([bbox[0] + w_offset, bbox[1] + h_offset], dtype=np.int32)
                    off_ct[0] = np.clip(off_ct[0], 0, output_w - 1)
                    off_ct[1] = np.clip(off_ct[1], 0, output_h - 1)
                    off_ind[k] = off_ct[1] * output_w + off_ct[0]

                    if self.opt.debug > 0 and full_ret:
                        self.draw_gaussian(hm[cls_id], off_ct, radius)

                gt_det[prepend + 'bboxes'].append(np.array([ct[0] - w / 2, ct[1] - h / 2,
                                                  ct[0] + w / 2, ct[1] + h / 2], dtype=np.float32))
                gt_det[prepend + 'cts'].append(ct)
                gt_det[prepend + 'scores'].append(1)
                gt_det[prepend + 'clses'].append(cls_id)
                gt_det[prepend + 'ids'].append(obj_id)

        self.add_to_dict(ret, prepend + 'img', img)
        self.add_to_dict(ret, prepend + 'hm', hm)
        self.add_to_dict(ret, prepend + 'wh', wh)
        self.add_to_dict(ret, prepend + 'ids', ids)
        self.add_to_dict(ret, prepend + 'ind', ind)
        self.add_to_dict(ret, prepend + 'off_ind', off_ind)
        self.add_to_dict(ret, prepend + 'reg', reg)
        self.add_to_dict(ret, prepend + 'reg_mask', reg_mask)
        self.add_to_dict(ret, prepend + 'num_objs', torch.tensor([num_objs]))

    def __getitem__(self, files_index):
        # Find which dataset this index falls in
        ds = None
        start_index = 0
        for i, c in enumerate(self.cds):
            if files_index >= c:
                ds = list(self.label_files.keys())[i]
                start_index = c

        # Get t-1 frame index
        pre_img_path, pre_label_path = None, None
        if self.opt.pre_img:
            # Ensure t-1 frame is retrieved from correct dataset and sequence
            if files_index == start_index:
                pre_files_index = start_index
                files_index = start_index + 1
            else:
                pre_files_index = files_index - 1

            pre_img_path = self.img_files[ds][pre_files_index - start_index]
            pre_label_path = self.label_files[ds][pre_files_index - start_index]

        # Get image and annotation file names
        img_path = self.img_files[ds][files_index - start_index]
        label_path = self.label_files[ds][files_index - start_index]

        # Load images and labels
        img_dict, label_dict, img_path, _ = self.get_data(img_path, label_path,
                                                          pre_img_path, pre_label_path, self.unsup)

        img, labels = img_dict['orig'], label_dict['orig']

        flipped_img = img_dict['flipped'] if 'flipped' in img_dict else None
        flipped_labels = label_dict['flipped'] if 'flipped' in label_dict else None

        pre_img = img_dict['pre_orig'] if 'pre_orig' in img_dict else None
        pre_labels = label_dict['pre_orig'] if 'pre_orig' in label_dict else None

        pre_flipped_img = img_dict['pre_flipped'] if 'pre_flipped' in img_dict else None
        pre_flipped_labels = label_dict['pre_flipped'] if 'pre_flipped' in label_dict else None

        # Offset object IDs with starting ID index for this dataset
        if not self.unsup:
            for i, _ in enumerate(labels):
                if labels[i, 1] > -1:
                    labels[i, 1] += self.tid_start_index[ds]

        ret = dict()  # Return dictionary
        gt_det = dict()  # Ground truth metadata (used by debugger)

        # Build targets for current frame
        self.build_targets(img, labels, ret, gt_det, full_ret=True)

        # Build targets for flipped current frame
        if flipped_img is not None and flipped_labels is not None:
            self.build_targets(flipped_img, flipped_labels, ret, gt_det, full_ret=False, prepend='flipped')

        # Build targets for t-1 frame
        if pre_img is not None and pre_labels is not None:
            self.build_targets(pre_img, pre_labels, ret, gt_det, full_ret=False, prepend='pre')

        # Build targets for t-1 flipped frame
        if pre_flipped_img is not None and pre_flipped_labels is not None:
            self.build_targets(pre_flipped_img, pre_flipped_labels, ret, gt_det, full_ret=False, prepend='pre_flipped')

        if self.opt.debug > 0:
            gt_det = self.format_gt_det(gt_det)
            meta = {'gt_det': gt_det, 'img_path': img_path, 'dataset': ds}
            self.add_to_dict(meta, 'pre_img_path', pre_img_path)
            ret['meta'] = meta

        return ret


class DetDataset(LoadImagesAndLabels):  # for training
    def __init__(self, root, paths, img_size=(1088, 608), augment=False, transforms=None):

        dataset_names = paths.keys()
        self.img_files = OrderedDict()
        self.label_files = OrderedDict()
        self.tid_num = OrderedDict()
        self.tid_start_index = OrderedDict()
        for ds, path in paths.items():
            with open(path, 'r') as file:
                self.img_files[ds] = file.readlines()
                self.img_files[ds] = [osp.join(root, x.strip()) for x in self.img_files[ds]]
                self.img_files[ds] = list(filter(lambda x: len(x) > 0, self.img_files[ds]))

            self.label_files[ds] = [
                x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
                for x in self.img_files[ds]]

        for ds, label_paths in self.label_files.items():
            max_index = -1
            for lp in label_paths:
                lb = np.loadtxt(lp)
                if len(lb) < 1:
                    continue
                if len(lb.shape) < 2:
                    img_max = lb[1]
                else:
                    img_max = np.max(lb[:, 1])
                if img_max > max_index:
                    max_index = img_max
            self.tid_num[ds] = max_index + 1

        last_index = 0
        for i, (k, v) in enumerate(self.tid_num.items()):
            self.tid_start_index[k] = last_index
            last_index += v

        self.nID = int(last_index + 1)
        self.nds = [len(x) for x in self.img_files.values()]
        self.cds = [sum(self.nds[:i]) for i in range(len(self.nds))]
        self.nF = sum(self.nds)
        self.width = img_size[0]
        self.height = img_size[1]
        self.augment = augment
        self.transforms = transforms

        print('=' * 80)
        print('dataset summary')
        print(self.tid_num)
        print('total # identities:', self.nID)
        print('start index')
        print(self.tid_start_index)
        print('=' * 80)

    def __getitem__(self, files_index):

        for i, c in enumerate(self.cds):
            if files_index >= c:
                ds = list(self.label_files.keys())[i]
                start_index = c

        img_path = self.img_files[ds][files_index - start_index]
        label_path = self.label_files[ds][files_index - start_index]
        if os.path.isfile(label_path):
            labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)

        imgs, labels, img_path, (h, w) = self.get_data(img_path, label_path)
        for i, _ in enumerate(labels):
            if labels[i, 1] > -1:
                labels[i, 1] += self.tid_start_index[ds]

        return imgs['orig'], labels0, img_path, (h, w)
