from .flow_dataset import FlowDataset, UnsupDataset
import tensorflow as tf
import wb_data.path as datapath
import os
import cv2
from glob import glob
BASE_PATH = datapath.HD1kBasePath


class HD1k(FlowDataset):
    def __init__(self, augment, training, shuffle=True, **augment_params):
        super(HD1k, self).__init__(augment, sparse=True, return_mask=True, shuffle=shuffle, **augment_params)

        root = BASE_PATH
        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows) - 1):
                self.flow_path.append(flows[i])
                self.image_path.append([images[i], images[i + 1]])

            seq_ix += 1

    @property
    def default_augment_params(self):
        params = super(HD1k, self).default_augment_params
        params.crop_size = (400, 720)
        params.min_scale = -0.1
        params.max_scale = 1.0
        params.do_flip = True
        return params


class HD1kUnsup(UnsupDataset):
    def __init__(self, augment, training, shuffle=True, **augment_params):
        super(HD1kUnsup, self).__init__(augment, sparse=True, return_mask=True, shuffle=shuffle, **augment_params)
        root = BASE_PATH
        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows) - 1):
                self.flow_path.append([flows[i]])
                self.image_path.append([images[i], images[i + 1]])

            seq_ix += 1

    @property
    def default_augment_params(self):
        params = super(HD1kUnsup, self).default_augment_params
        params.crop_size = (400, 720)
        params.min_scale = -0.1
        params.max_scale = 1.0
        params.do_flip = True
        return params
