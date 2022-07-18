import tensorflow as tf
from box import Box
import raft_utils.augmentor as raftaug
import os
import cv2
import numpy as np
from util.things_io import readPFM, readFlow

Dataset = tf.data.Dataset


class FlowDataset:
    def __init__(self, augment:bool = True, sparse=False, return_mask=False, shuffle=True, **augment_params):
        self.augment_params = self.default_augment_params
        update_params(self.augment_params, augment_params)
        self.do_augment = augment
        self.sparse = sparse
        self.shuffle = shuffle
        self.return_mask = return_mask

        self.augmentor = raftaug.FlowAugmentor(**self.augment_params)
        self.sparse_augmentor = raftaug.SparseFlowAugmentor(**self.augment_params)

        self.image_path = []
        self.flow_path = []

    @property
    def default_augment_params(self):
        params = Box()
        params.crop_size = (480, 640)
        params.min_scale = -0.2
        params.max_scale=0.5
        params.do_flip=False

        return params

    @staticmethod
    def load_image(abs_image_path):
        im = tf.numpy_function(_load_image, [abs_image_path], tf.float32)
        s = tf.shape(im)
        im = tf.reshape(im, [s[0], s[1], 3])
        return im

    @staticmethod
    def load_flow(abs_flow_path):
        if tf.equal(abs_flow_path, ""):
            return tf.zeros([1,1,2], tf.float32), tf.zeros([1,1,1], tf.float32)
        flow = tf.numpy_function(_load_flow, [abs_flow_path], tf.float32)
        s = tf.shape(flow)
        flow = tf.reshape(flow, [s[0], s[1], s[2]])

        mask = tf.cond(tf.equal(s[2], 3), lambda: flow[...,2:3], lambda: tf.ones_like(flow)[...,0:1])
        flow = flow[...,0:2]

        return flow, mask

    def _prepare(self):
        if len(self.flow_path) == 0:
            seq_len = len(self.image_path[0])
            if seq_len == 2:
                self.flow_path.extend([""] * len(self.image_path))
            else:
                self.flow_path.extend([[""] * (seq_len - 1)] * len(self.image_path))

    @property
    def dataset(self):
        self._prepare()
        x = Dataset.from_tensor_slices(self.image_path)
        y = Dataset.from_tensor_slices(self.flow_path)

        d = Dataset.zip((x, y))
        if self.shuffle:
            d = d.shuffle(len(x))

        _load_flow = self.load_flow

        d = d.map(lambda x, y: (tf.map_fn(self.load_image, x, fn_output_signature=tf.float32), _load_flow(y)),
                  num_parallel_calls=8)

        def _maybe_make_dummy(x, y):
            y, m = y
            y_r = tf.image.resize(y, tf.shape(x[0])[0:2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            m_r = tf.image.resize(m, tf.shape(x[0])[0:2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            m = tf.cond(tf.reduce_all(tf.shape(y) == tf.constant([1, 1, 2])), lambda: m_r, lambda: m)
            y = tf.cond(tf.reduce_all(tf.shape(y) == tf.constant([1, 1, 2])), lambda: y_r, lambda: y)
            return x, (y, m)

        d = d.map(_maybe_make_dummy, num_parallel_calls=8)

        if self.do_augment:
            d = d.map(self.augment, num_parallel_calls=8)
        else:
            ret_mask = self.return_mask
            def _sep(x, y):
                if isinstance(y, (tuple, list)) and len(y) == 2:
                    y, m = y
                else:
                    m = tf.ones_like(y[..., 0:1])
                x1 = x[0]
                x2 = x[1]
                if ret_mask:
                    return (x1, x2), (y, m)
                else:
                    return (x1, x2), y
            d = d.map(_sep, num_parallel_calls=8)

        return d.map(lambda x, y: ((x[0], x[1]), y), num_parallel_calls=8)

    def append(self, flow_dataset):
        self._prepare()
        flow_dataset._prepare()

        if flow_dataset.return_mask:
            self.return_mask = True
        self.image_path.extend(flow_dataset.image_path)
        self.flow_path.extend(flow_dataset.flow_path)

    def augment(self, x, y):
        if isinstance(y, (tuple, list)) and len(y) == 2:
            y, m = y
        else:
            m = tf.ones_like(y[...,0:1])

        is_dense = tf.reduce_all(m > 0.5)
        d_res = self.augmentor(x[0], x[1], y)
        d_m = tf.ones_like(d_res[0][..., 0:1])

        s_res = self.sparse_augmentor(x[0], x[1], y, m)
        #s_m = tf.ones_like(s_res[0][..., 0:1])
        s_m = s_res[3]

        x1 = tf.cond(is_dense, lambda: d_res[0], lambda: s_res[0])
        x2 = tf.cond(is_dense, lambda: d_res[1], lambda: s_res[1])
        y = tf.cond(is_dense, lambda: d_res[2], lambda: s_res[2])
        m = tf.cond(is_dense, lambda: d_m, lambda: s_m)
        s = tf.shape(y)
        m = tf.reshape(m, (s[0], s[1], 1))

        if self.return_mask:
            return (x1, x2), (y, m)
        else:
            return (x1, x2), y


class UnsupDataset(FlowDataset):
    def __init__(self, *args, **kwargs):
        super(UnsupDataset, self).__init__(*args, **kwargs)
        self.augmentor = raftaug.UnsupAugmentor(**self.augment_params)
        self.sparse_augmentor = raftaug.UnsupAugmentor(**self.augment_params)

    def backward(self):
        if len(self.flow_path) != 0:
            self.flow_path = []
        new_image_path = []
        for p in self.image_path:
            new_image_path.append(list(reversed(p)))
        self.image_path = new_image_path
        return self

    def _prepare(self):
        if len(self.flow_path) == 0:
            seq_len = len(self.image_path[0])
            self.flow_path.extend([[""] * (seq_len - 1)] * len(self.image_path))

    @property
    def dataset(self):
        self._prepare()
        x = Dataset.from_tensor_slices(self.image_path)
        y = Dataset.from_tensor_slices(self.flow_path)

        d = Dataset.zip((x, y))
        if self.shuffle:
            d = d.shuffle(len(x))

        _load_flow = self.load_flow

        d = d.map(lambda x, y: (tf.map_fn(self.load_image, x, fn_output_signature=tf.float32), (_load_flow(y[0]))),
                  num_parallel_calls=8)

        def _maybe_make_dummy(x, y):
            y, m = y

            y_r = tf.image.resize(y, tf.shape(x[0])[0:2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            m_r = tf.image.resize(m, tf.shape(x[0])[0:2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            m = tf.cond(tf.reduce_all(tf.shape(y) == tf.constant([1, 1, 2])), lambda: m_r, lambda: m)
            y = tf.cond(tf.reduce_all(tf.shape(y) == tf.constant([1, 1, 2])), lambda: y_r, lambda: y)
            return x, (y, m)

        d = d.map(lambda x, y: (x, (_maybe_make_dummy(x, y)[1])), num_parallel_calls=8)
        d = d.map(lambda x, y: (x, (y[0], y[1])))

        if self.do_augment:
            d = d.map(self.augment, num_parallel_calls=8)
        else:
            pass
            #raise ValueError("enable augment to use this dataset")

        return d


    def augment(self, x, y):
        y, m = y

        res = self.augmentor(x[0], x[1], y, m)
        return res

class MultiFrameDataset(FlowDataset):
    def __init__(self, *args, **kwargs):
        super(MultiFrameDataset, self).__init__(*args, **kwargs)
        self.augmentor = raftaug.MultiFrameAugmentor(**self.augment_params)
        self.sparse_augmentor = raftaug.MultiFrameAugmentor(**self.augment_params)

    @property
    def dataset(self):
        self._prepare()
        x = Dataset.from_tensor_slices(self.image_path)
        y = Dataset.from_tensor_slices(self.flow_path)

        d = Dataset.zip((x, y))
        if self.shuffle:
            d = d.shuffle(len(x))

        _load_flow = self.load_flow

        d = d.map(lambda x, y: (tf.map_fn(self.load_image, x, fn_output_signature=tf.float32), (_load_flow(y[0]), _load_flow(y[1]))),
                  num_parallel_calls=8)

        def _maybe_make_dummy(x, y):
            y, m = y

            y_r = tf.image.resize(y, tf.shape(x[0])[0:2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            m_r = tf.image.resize(m, tf.shape(x[0])[0:2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            m = tf.cond(tf.reduce_all(tf.shape(y) == tf.constant([1, 1, 2])), lambda: m_r, lambda: m)
            y = tf.cond(tf.reduce_all(tf.shape(y) == tf.constant([1, 1, 2])), lambda: y_r, lambda: y)
            return x, (y, m)

        d = d.map(lambda x, y: (x, (_maybe_make_dummy(x, y[0])[1], _maybe_make_dummy(x, y[1])[1])), num_parallel_calls=8)
        d = d.map(lambda x, y: (x, ([y[0][0], y[1][0]], [y[0][1], y[1][1]])))

        if self.do_augment:
            d = d.map(self.augment, num_parallel_calls=8)
        else:
            pass
            #raise ValueError("enable augment to use this dataset")

        return d


    def augment(self, x, y):
        y, m = y

        res = self.augmentor(x[0], x[1], x[2], y[0], y[1], m[0], m[1])
        return res


def update_params(box:dict, new_box:dict):
    for k, v in new_box.items():
        if isinstance(v, dict):
            update_params(box[k], v)
        else:
            box[k] = v


def _load_image(abs_image_path):
    filename = abs_image_path.decode("utf-8")
    ext = os.path.split(filename)[-1].split(".")[-1]
    im = cv2.imread(filename)
    if im is None:
        print(filename)
    im = im[...,[2,1,0]]
    im = np.float32(im) / 255.
    return im


def _load_flow(abs_flow_path):
    filename = abs_flow_path.decode("utf-8")
    ext = os.path.split(filename)[-1].split(".")[-1]
    if ext == "flo":
        return readFlo(filename).astype(np.float32)
    elif ext == "pfm":
        return readPFM(filename)[0][:,:,0:2].astype(np.float32)
    elif ext == "png":
        return _read_png_flow(filename)
    else:
        return readFlow(filename).astype(np.float32)


def readFlo(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))


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


def _read_png_flow(abs_flow_path):
    with tf.device("/CPU:0"):
        flo_string = tf.io.read_file(abs_flow_path)
        gt_uint16 = tf.image.decode_png(flo_string, dtype=tf.uint16)
        gt = tf.cast(gt_uint16, tf.float32)
        flow = (gt[:, :, 0:2] - 2 ** 15) / 64.0
        mask = gt[:, :, 2:3]
        flow = flow
        out = tf.concat([flow, mask], axis=2)
    return out


def make_semi_dataset(unsup_dataset: tf.data.Dataset, sup_dataset:tf.data.Dataset):
    def prepend_sup(x, y):
        res_x = {}
        res_y = {}
        for k, v in x.items():
            res_x["sup_" + k] = v
        for k, v in y.items():
            res_y["sup_" + k] = v
        return res_x, res_y

    def combine_xy(sup, unsup):
        x = {}
        for k, v in sup[0].items():
            x[k] = v
        for k, v in unsup[0].items():
            x[k] = v

        y = {}
        for k, v in sup[1].items():
            y[k] = v
        for k, v in unsup[1].items():
            y[k] = v

        return x, y

    sup_dataset = sup_dataset.repeat(-1)
    unsup_dataset = unsup_dataset.repeat(-1)

    sup_dataset = sup_dataset.map(prepend_sup, num_parallel_calls=8)
    dataset = tf.data.Dataset.zip((unsup_dataset, sup_dataset))
    dataset = dataset.map(combine_xy, num_parallel_calls=8)
    return dataset

