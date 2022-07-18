from .flow_dataset import FlowDataset, UnsupDataset
import tensorflow as tf
import wb_data.path as datapath
from util.things_io import readFlow
import os
import cv2
from glob import glob
BASE_PATH = datapath.FlyingThingsBasePath


class FlyingThings(FlowDataset):
    def __init__(self, augment, training, shuffle=True, dstype='frames_cleanpass', **augment_params):
        super(FlyingThings, self).__init__(augment, sparse=False, shuffle=shuffle, **augment_params)

        osp = os.path
        root = BASE_PATH
        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')))
                    flows = sorted(glob(osp.join(fdir, '*.pfm')))
                    for i in range(len(flows) - 1):
                        if direction == 'into_future':
                            self.image_path.append([images[i], images[i + 1]])
                            self.flow_path.append(flows[i])
                        elif direction == 'into_past':
                            self.image_path.append([images[i + 1], images[i]])
                            self.flow_path.append(flows[i + 1])

    @property
    def default_augment_params(self):
        params = super(FlyingThings, self).default_augment_params
        params.crop_size = (368, 768)
        params.min_scale = -0.1
        params.max_scale = 1.0
        params.do_flip = True
        return params


class FlyingThingsUnsup(UnsupDataset):
    def __init__(self, augment, training, shuffle=True, dstype='frames_cleanpass', **augment_params):
        super(FlyingThingsUnsup, self).__init__(augment, sparse=False, shuffle=shuffle, **augment_params)

        osp = os.path
        root = BASE_PATH
        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')))
                    flows = sorted(glob(osp.join(fdir, '*.pfm')))
                    for i in range(len(flows) - 1):
                        if direction == 'into_future':
                            self.image_path.append([images[i], images[i + 1]])
                            self.flow_path.append([flows[i]])
                        elif direction == 'into_past':
                            self.image_path.append([images[i + 1], images[i]])
                            self.flow_path.append([flows[i + 1]])
