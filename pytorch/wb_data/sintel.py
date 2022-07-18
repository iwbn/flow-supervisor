from .flow_dataset import FlowDataset, MultiFrameDataset, UnsupDataset
import tensorflow as tf
import wb_data.path as datapath
import os
import cv2
from glob import glob
BASE_PATH = datapath.SintelBasePath


class Sintel(FlowDataset):
    def __init__(self, augment, training, shuffle=True, dstype='final', **augment_params):
        super(Sintel, self).__init__(augment, sparse=False, shuffle=shuffle, **augment_params)

        self.extra_info = []

        split = "training" if training else "test"
        image_root = os.path.join(BASE_PATH, split, dstype)
        flow_root = os.path.join(BASE_PATH, split, 'flow')

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(os.path.join(image_root, scene, '*.png')))
            flow_list = sorted(glob(os.path.join(flow_root, scene, '*.flo')))
            for i in range(len(image_list)-1):
                self.image_path.append([image_list[i], image_list[i+1]])
                self.extra_info.append((scene, i))

                if split != 'test':
                    self.flow_path.append(flow_list[i])


    @property
    def default_augment_params(self):
        params = super(Sintel, self).default_augment_params
        params.crop_size = (400, 720)
        params.min_scale = -0.1
        params.max_scale = 1.0
        params.do_flip = True
        return params


class SintelMultiFrame(MultiFrameDataset):
    def __init__(self, augment, training, shuffle=True, dstype='final',**augment_params):
        super(SintelMultiFrame, self).__init__(augment, sparse=False, shuffle=shuffle, **augment_params)

        self.extra_info = []

        split = "training" if training else "test"
        image_root = os.path.join(BASE_PATH, split, dstype)
        flow_root = os.path.join(BASE_PATH, split, 'flow')

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(os.path.join(image_root, scene, '*.png')))
            flow_list = sorted(glob(os.path.join(flow_root, scene, '*.flo')))
            for i in range(len(image_list)-2):
                self.image_path.append([image_list[i], image_list[i+1], image_list[i+2]])
                self.extra_info.append((scene, i))

                if split != 'test':
                    self.flow_path.append([flow_list[i], flow_list[i+1]])


    @property
    def default_augment_params(self):
        params = super(SintelMultiFrame, self).default_augment_params
        params.crop_size = (400, 720)
        params.min_scale = -0.1
        params.max_scale = 1.0
        params.do_flip = True
        return params


class SintelUnsup(UnsupDataset):
    def __init__(self, augment, training, shuffle=True, dstype='final',**augment_params):
        super(SintelUnsup, self).__init__(augment, sparse=False, shuffle=shuffle, **augment_params)

        self.extra_info = []

        split = "training" if training else "test"
        image_root = os.path.join(BASE_PATH, split, dstype)
        flow_root = os.path.join(BASE_PATH, split, 'flow')

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(os.path.join(image_root, scene, '*.png')))
            flow_list = sorted(glob(os.path.join(flow_root, scene, '*.flo')))
            for i in range(len(image_list)-1):
                self.image_path.append([image_list[i], image_list[i+1]])
                self.extra_info.append((scene, i))

                if split != 'test':
                    self.flow_path.append([flow_list[i]])


    @property
    def default_augment_params(self):
        params = super(UnsupDataset, self).default_augment_params
        params.crop_size = (400, 720)
        params.min_scale = -0.1
        params.max_scale = 1.0
        params.do_flip = True
        return params


class SintelUnsupPart(UnsupDataset):
    def __init__(self, augment, part=1, shuffle=True, dstype='final',**augment_params):
        assert part in [1,2]
        super(SintelUnsupPart, self).__init__(augment, sparse=False, shuffle=shuffle, **augment_params)

        self.extra_info = []

        part1 = ['alley_1', 'ambush_2', 'bamboo_1', 'bandage_1', 'cave_2', 'market_2', 'mountain_1', 'shaman_2', 'sleeping_2', 'temple_2']

        split = "training"
        image_root = os.path.join(BASE_PATH, split, dstype)
        flow_root = os.path.join(BASE_PATH, split, 'flow')

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            if (part == 1 and scene in part1) or (part == 2 and scene not in part1):
                image_list = sorted(glob(os.path.join(image_root, scene, '*.png')))
                flow_list = sorted(glob(os.path.join(flow_root, scene, '*.flo')))
                for i in range(len(image_list)-1):
                    self.image_path.append([image_list[i], image_list[i+1]])
                    self.extra_info.append((scene, i))

                    if split != 'test':
                        self.flow_path.append([flow_list[i]])


    @property
    def default_augment_params(self):
        params = super(UnsupDataset, self).default_augment_params
        params.crop_size = (400, 720)
        params.min_scale = -0.1
        params.max_scale = 1.0
        params.do_flip = True
        return params

class SintelUnsupInterval(UnsupDataset):
    def __init__(self, augment, training, shuffle=True, dstype='final',**augment_params):
        super(SintelUnsupInterval, self).__init__(augment, sparse=False, shuffle=shuffle, **augment_params)

        self.extra_info = []

        split = "training" if training else "test"
        image_root = os.path.join(BASE_PATH, split, dstype)
        flow_root = os.path.join(BASE_PATH, split, 'flow')

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(os.path.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-2):
                self.image_path.append([image_list[i], image_list[i+2]])
                self.extra_info.append((scene, i))


    @property
    def default_augment_params(self):
        params = super(UnsupDataset, self).default_augment_params
        params.crop_size = (400, 720)
        params.min_scale = -0.1
        params.max_scale = 1.0
        params.do_flip = True
        return params




class SintelMovie(FlowDataset):
    def __init__(self, augment, training, shuffle=True, dstype='final',**augment_params):
        super(SintelMovie, self).__init__(augment, sparse=False, shuffle=shuffle, **augment_params)
