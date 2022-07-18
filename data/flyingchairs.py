from .flow_dataset import FlowDataset, UnsupDataset
import tensorflow as tf
import data.path
import os
import cv2

BASE_PATH = data.path.FlyingChairsBasePath
TRAIN_VAL_TXT = data.path.FlyingChairsMetaFilePath

NUM_DATA = 22872


class FlyingChairs(FlowDataset):
    def __init__(self, augment, training, shuffle=True, **augment_params):
        super(FlyingChairs, self).__init__(augment, sparse=False, shuffle=shuffle, **augment_params)

        code = 1 if training else 2
        with open(TRAIN_VAL_TXT, 'r') as f:
            for s, l in enumerate(f):
                if int(l) == code:
                    self.image_path.append([os.path.join(BASE_PATH, "%05d_img%d.ppm" % (s+1,i)) for i in [1, 2]])
                    self.flow_path.append(os.path.join(BASE_PATH, "%05d_flow.flo" % (s+1)))

    @staticmethod
    def load_image(abs_image_path):
        return _decode_ppm(abs_image_path)

    @property
    def default_augment_params(self):
        params = super(FlyingChairs, self).default_augment_params
        params.crop_size = (368, 496)
        params.min_scale = -0.1
        params.max_scale = 1.0
        params.do_flip = True
        return params


class FlyingChairsUnsup(UnsupDataset):
    def __init__(self, augment, training, shuffle=True, **augment_params):
        super(FlyingChairsUnsup, self).__init__(augment, sparse=False, shuffle=shuffle, **augment_params)

        code = 1 if training else 2
        with open(TRAIN_VAL_TXT, 'r') as f:
            for s, l in enumerate(f):
                if int(l) == code:
                    self.image_path.append([os.path.join(BASE_PATH, "%05d_img%d.ppm" % (s+1,i)) for i in [1, 2]])
                    self.flow_path.append([os.path.join(BASE_PATH, "%05d_flow.flo" % (s+1))])


def _decode_ppm(abs_image_path):
    IMG_SIZE = (384, 512)
    img = tf.numpy_function(_read_ppm, [abs_image_path], tf.float32)
    img = tf.reshape(img, [IMG_SIZE[0], IMG_SIZE[1], 3])
    return img


def _read_ppm(filename):
    filename = filename.decode("utf-8")
    img = cv2.imread(filename).astype('float32') / 255.
    img = img[:,:,[2,1,0]]
    return img
