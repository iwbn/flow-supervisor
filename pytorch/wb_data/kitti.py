from .flow_dataset import FlowDataset, UnsupDataset, _load_flow, _load_image
import tensorflow as tf
import wb_data.path as datapath
import os
import cv2
from glob import glob
BASE_PATH = datapath.KITTIBasePath


class KITTI(FlowDataset):
    def __init__(self, augment, training, shuffle=True, **augment_params):
        super(KITTI, self).__init__(augment, sparse=True, return_mask=True, shuffle=shuffle, **augment_params)

        split = "training" if training else "testing"
        if split == 'testing':
            self.is_test = True

        osp = os.path
        root = osp.join(BASE_PATH, "data_scene_flow")
        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))
        flows = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))
        self.extra_info = []

        for i, (img1, img2) in enumerate(zip(images1, images2)):
            frame_id = img1.split('/')[-1]
            self.extra_info.append([frame_id])
            self.image_path.append([img1, img2])

            if split == 'training':
                self.flow_path.append(flows[i])

    @property
    def default_augment_params(self):
        params = super(KITTI, self).default_augment_params
        params.crop_size = (400, 720)
        params.min_scale = -0.1
        params.max_scale = 1.0
        params.do_flip = True
        return params

class KITTIUnsup(UnsupDataset):
    def __init__(self, augment, training, shuffle=True, **augment_params):
        super(KITTIUnsup, self).__init__(augment, sparse=True, return_mask=True, shuffle=shuffle, **augment_params)

        split = "training" if training else "testing"
        if split == 'testing':
            self.is_test = True

        osp = os.path
        root = osp.join(BASE_PATH, "data_scene_flow")
        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))
        flows = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))
        self.extra_info = []

        for i, (img1, img2) in enumerate(zip(images1, images2)):
            frame_id = img1.split('/')[-1]
            self.extra_info.append([frame_id])
            self.image_path.append([img1, img2])

            if split == 'training':
                self.flow_path.append([flows[i]])

    @property
    def default_augment_params(self):
        params = super(KITTIUnsup, self).default_augment_params
        params.crop_size = (400, 720)
        params.min_scale = -0.1
        params.max_scale = 1.0
        params.do_flip = True
        return params

class KITTI2012(FlowDataset):
    def __init__(self, augment, training, shuffle=True, **augment_params):
        super(KITTI2012, self).__init__(augment, sparse=True, return_mask=True, shuffle=shuffle, **augment_params)

        split = "training" if training else "testing"
        if split == 'testing':
            self.is_test = True

        osp = os.path
        root = osp.join(BASE_PATH, "data_stereo_flow")
        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'colored_0/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'colored_0/*_11.png')))
        flows = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))
        self.extra_info = []

        for i, (img1, img2) in enumerate(zip(images1, images2)):
            frame_id = img1.split('/')[-1]
            self.extra_info.append([frame_id])
            self.image_path.append([img1, img2])

            if split == 'training':
                self.flow_path.append(flows[i])

    @property
    def default_augment_params(self):
        params = super(KITTI2012, self).default_augment_params
        params.crop_size = (400, 720)
        params.min_scale = -0.1
        params.max_scale = 1.0
        params.do_flip = True
        return params

ORIG_IMAGE_SIZE = (375, 1242)

class KITTI_Multiview(UnsupDataset):
    def __init__(self, augment, training, shuffle=True, **augment_params):
        super(KITTI_Multiview, self).__init__(augment, sparse=True, return_mask=True, shuffle=shuffle, **augment_params)

        split = "training" if training else "testing"
        if split == 'testing':
            self.is_test = True

        osp = os.path
        root = osp.join(BASE_PATH, "data_scene_flow_multiview")
        root = osp.join(root, split)
        images = sorted(glob(osp.join(root, 'image_2/*.png')) + glob(osp.join(root, 'image_3/*.png')))
        self.extra_info = []

        prev_frame_id = images[0].split("/")[-1]
        for i, img in enumerate(images[1::]):
            frame_id = img.split('/')[-1]
            if frame_id.split("_")[0] != prev_frame_id.split("_")[0]:
                prev_frame_id = frame_id
                continue

            self.extra_info.append([prev_frame_id])
            self.image_path.append([images[i-1], images[i]])
            prev_frame_id = frame_id

    @staticmethod
    def load_image(abs_image_path):
        im = tf.numpy_function(_load_image, [abs_image_path], tf.float32)
        s = tf.shape(im)
        im = tf.reshape(im, [s[0], s[1], 3])
        im = tf.image.resize_with_crop_or_pad(im, ORIG_IMAGE_SIZE[0], ORIG_IMAGE_SIZE[1])
        return im

    @staticmethod
    def load_flow(abs_flow_path):
        if tf.equal(abs_flow_path, ""):
            return tf.zeros([1, 1, 2], tf.float32), tf.zeros([1, 1, 1], tf.float32)
        flow = tf.numpy_function(_load_flow, [abs_flow_path], tf.float32)
        s = tf.shape(flow)
        flow = tf.reshape(flow, [s[0], s[1], s[2]])

        mask = tf.cond(tf.equal(s[2], 3), lambda: flow[..., 2:3], lambda: tf.ones_like(flow)[..., 0:1])
        flow = flow[..., 0:2]
        flow = tf.image.resize_with_crop_or_pad(flow, ORIG_IMAGE_SIZE[0], ORIG_IMAGE_SIZE[1])
        mask = tf.image.resize_with_crop_or_pad(mask, ORIG_IMAGE_SIZE[0], ORIG_IMAGE_SIZE[1])
        return flow, mask

    @property
    def default_augment_params(self):
        params = super(KITTI_Multiview, self).default_augment_params
        params.crop_size = (400, 720)
        params.min_scale = -0.1
        params.max_scale = 1.0
        params.do_flip = True
        return params


class KITTI_MultiviewInterval(KITTI_Multiview):
    def __init__(self, augment, training, shuffle=True, **augment_params):
        super(KITTI_Multiview, self).__init__(augment, sparse=True, return_mask=True, shuffle=shuffle, **augment_params)

        split = "training" if training else "testing"
        if split == 'testing':
            self.is_test = True

        osp = os.path
        root = osp.join(BASE_PATH, "data_scene_flow_multiview")
        root = osp.join(root, split)
        images = sorted(glob(osp.join(root, 'image_2/*.png')) + glob(osp.join(root, 'image_3/*.png')))
        self.extra_info = []

        prev_frame_id = images[0].split("/")[-1]
        for i, img in enumerate(images[2::]):
            frame_id = img.split('/')[-1]
            if frame_id.split("_")[0] != prev_frame_id.split("_")[0]:
                prev_frame_id = frame_id
                continue
            elif frame_id.split("_")[0] != images[i-1].split('/')[-1].split("_")[0]:
                prev_frame_id = frame_id
                continue

            self.extra_info.append([prev_frame_id])
            self.image_path.append([images[i-2], images[i]])
            prev_frame_id = frame_id

class KITTI_Raw(FlowDataset):
    pass