from .flow_dataset import FlowDataset, UnsupDataset
import tensorflow as tf
import wb_data.path as datapath
import os
import cv2
from glob import glob
BASE_PATH = datapath.SpringBasePath


class Spring(FlowDataset):
    def __init__(self, augment, shuffle=True,**augment_params):
        super(Spring, self).__init__(augment, sparse=False, shuffle=shuffle, **augment_params)

        image_root = os.path.join(BASE_PATH, "frames")
        image_list = sorted(glob(os.path.join(image_root, '*.png')))

        for i in range(len(image_list) - 1):
            self.image_path.append([image_list[i], image_list[i+1]])


    @property
    def default_augment_params(self):
        params = super(Spring, self).default_augment_params
        params.crop_size = (400, 720)
        params.min_scale = -0.1
        params.max_scale = 1.0
        params.do_flip = True
        return params

class SpringUnsup(UnsupDataset):
    def __init__(self, augment, shuffle=True,**augment_params):
        super(SpringUnsup, self).__init__(augment, sparse=False, shuffle=shuffle, **augment_params)

        image_root = os.path.join(BASE_PATH, "frames")
        image_list = sorted(glob(os.path.join(image_root, '*.png')))

        for i in range(len(image_list) - 1):
            self.image_path.append([image_list[i], image_list[i+1]])


    @property
    def default_augment_params(self):
        params = super(SpringUnsup, self).default_augment_params
        params.crop_size = (400, 720)
        params.min_scale = -0.1
        params.max_scale = 1.0
        params.do_flip = True
        return params


class SpringUnsupInterval(UnsupDataset):
    def __init__(self, augment, shuffle=True,**augment_params):
        super(SpringUnsupInterval, self).__init__(augment, sparse=False, shuffle=shuffle, **augment_params)

        image_root = os.path.join(BASE_PATH, "frames")
        image_list = sorted(glob(os.path.join(image_root, '*.png')))

        for i in range(len(image_list) - 2):
            self.image_path.append([image_list[i], image_list[i+2]])


    @property
    def default_augment_params(self):
        params = super(SpringUnsupInterval, self).default_augment_params
        params.crop_size = (400, 720)
        params.min_scale = -0.1
        params.max_scale = 1.0
        params.do_flip = True
        return params