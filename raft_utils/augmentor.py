"""
Original Repository:
https://github.com/princeton-vl/RAFT/blob/master/core/utils/augmentor.py
"""

import numpy as np
import random
import math
from PIL import Image
from uflow.uflow_augmentation import random_rotation

import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import tensorflow as tf
import tensorflow_addons as tfa

class ColorJitter:
    def __init__(self, brightness, contrast, saturation, hue):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, im: tf.Tensor):
        # brightness adjustment
        l, u = np.maximum(0., 1. - self.brightness), 1. + self.brightness
        b_map = tf.random.uniform([], l, u, dtype=im.dtype)
        im *= b_map

        # contrast jitter
        im = tf.image.random_contrast(im, np.maximum(0., 1. - self.contrast), 1. + self.contrast)

        # saturation jitter
        im = tf.image.random_saturation(im, np.maximum(0., 1. - self.saturation), 1. + self.saturation)

        # hue jitter
        im = tf.image.random_hue(im, self.hue)

        return im

class FlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True, eraser_aug_prob=0.5):

        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2
        self.do_rotation = False
        self.max_rotation = 10

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5 / 3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = eraser_aug_prob

    @tf.function
    def color_transform(self, img1:tf.Tensor, img2:tf.Tensor):
        """ Photometric augmentation """

        # asymmetric
        if tf.random.uniform([]) < self.asymmetric_color_aug_prob:
            img1 = self.photo_aug(img1)
            img2 = self.photo_aug(img2)

        # symmetric
        else:
            image_stack = tf.concat([img1, img2], axis=0)
            image_stack = self.photo_aug(image_stack)
            img1, img2 = tf.unstack(tf.split(image_stack, 2, axis=0))

        img1 = tf.clip_by_value(img1, 0., 1.)
        img2 = tf.clip_by_value(img2, 0., 1.)

        return img1, img2
    
    @tf.function
    def eraser_transform(self, img1, img2, bounds=(50, 100)):
        """ Occlusion augmentation """

        ht, wd = tf.unstack(tf.shape(img1)[0:2])
        if tf.random.uniform([]) < self.eraser_aug_prob:
            mean_color = tf.reduce_mean(tf.reshape(img2,(-1, 3)), axis=0)
            for _ in range(tf.random.uniform([],1, 3, dtype=tf.int32)):
                x0 = tf.random.uniform([], 0, wd, dtype=tf.int32)
                y0 = tf.random.uniform([], 0, ht, dtype=tf.int32)
                dx = tf.random.uniform([],
                                       tf.minimum(bounds[0], wd - x0),
                                       tf.minimum(bounds[1], wd - x0 + 1), dtype=tf.int32)
                dy = tf.random.uniform([],
                                       tf.minimum(bounds[0], ht - y0),
                                       tf.minimum(bounds[1], ht - y0 + 1), dtype=tf.int32)

                mask = tf.ones([dy,dx], dtype=mean_color.dtype)
                mask = tf.pad(mask, [[y0, ht-(y0 + dy)], [x0, wd-(x0 + dx)]],
                              constant_values=0.)
                mask = mask[..., tf.newaxis]
                print(img2.shape, mask.shape, mean_color.shape)
                img2 = (1.-mask) * img2 + mask * mean_color[tf.newaxis, tf.newaxis]

        return img1, img2

    @tf.function
    def spatial_transform(self, img1:tf.Tensor, img2:tf.Tensor, flow:tf.Tensor):

        if self.do_rotation:
            imgs, flow, _ = random_rotation(tf.stack([img1, img2]), flow, tf.ones_like(flow)[...,0:1], max_rotation=self.max_rotation)
            img1, img2 = tf.unstack(imgs, axis=0)

        # randomly sample scale
        ht, wd = tf.unstack(tf.shape(img1)[0:2])

        min_scale = tf.maximum(
            (self.crop_size[0] + 8.) / tf.cast(ht, tf.float32),
            (self.crop_size[1] + 8.) / tf.cast(wd, tf.float32))

        scale = tf.pow(2., tf.random.uniform([], self.min_scale, self.max_scale))
        scale_x = scale
        scale_y = scale
        if tf.random.uniform([]) < self.stretch_prob:
            scale_x *= tf.pow(2., tf.random.uniform([], -self.max_stretch, self.max_stretch))
            scale_y *= tf.pow(2., tf.random.uniform([], -self.max_stretch, self.max_stretch))

        scale_x = tf.clip_by_value(scale_x, min_scale, scale_x)
        scale_y = tf.clip_by_value(scale_y, min_scale, scale_y)

        if tf.random.uniform([]) < self.spatial_aug_prob:
            # rescale the images
            t_h = tf.cast(tf.round(tf.cast(ht, tf.float32) * scale_y), tf.int32)
            t_w = tf.cast(tf.round(tf.cast(wd, tf.float32) * scale_x), tf.int32)

            scale_y = tf.cast(t_h, tf.float32) / tf.cast(ht, tf.float32)
            scale_x = tf.cast(t_w, tf.float32) / tf.cast(wd, tf.float32)

            img1 = tf.image.resize(img1, (t_h, t_w), method=tf.image.ResizeMethod.BILINEAR)
            img2 = tf.image.resize(img2, (t_h, t_w), method=tf.image.ResizeMethod.BILINEAR)
            flow = tf.image.resize(flow, (t_h, t_w), method=tf.image.ResizeMethod.BILINEAR)
            flow = flow * tf.reshape([scale_x, scale_y], [1,1,2])

        elif min_scale > 1.0:
            scale_y = scale_x = min_scale

            t_h = tf.cast(tf.round(tf.cast(ht, tf.float32) * scale_y), tf.int32)
            t_w = tf.cast(tf.round(tf.cast(wd, tf.float32) * scale_x), tf.int32)

            img1 = tf.image.resize(img1, (t_h, t_w), method=tf.image.ResizeMethod.BILINEAR)
            img2 = tf.image.resize(img2, (t_h, t_w), method=tf.image.ResizeMethod.BILINEAR)
            flow = tf.image.resize(flow, (t_h, t_w), method=tf.image.ResizeMethod.BILINEAR)
            flow = flow * tf.reshape([scale_x, scale_y], [1, 1, 2])


        if self.do_flip:
            if tf.random.uniform([]) < self.h_flip_prob:  # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * tf.reshape([-1.0, 1.0], [1,1,2])

            if tf.random.uniform([]) < self.v_flip_prob:  # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * tf.reshape([1.0, -1.0], [1,1,2])

        ht, wd = tf.unstack(tf.shape(img1))[0:2]

        y0 = tf.random.uniform([], 0, ht - self.crop_size[0], tf.int32)
        x0 = tf.random.uniform([], 0, wd - self.crop_size[1], tf.int32)

        img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        flow = flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

        img1 = tf.ensure_shape(img1, [self.crop_size[0], self.crop_size[1], 3])
        img2 = tf.ensure_shape(img2, [self.crop_size[0], self.crop_size[1], 3])
        flow = tf.ensure_shape(flow, [self.crop_size[0], self.crop_size[1], 2])

        return img1, img2, flow

    def __call__(self, img1:tf.Tensor, img2:tf.Tensor, flow:tf.Tensor):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow = self.spatial_transform(img1, img2, flow)

        return img1, img2, flow


class SparseFlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=False, eraser_aug_prob=0.5):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1
        self.do_rotation = False
        self.max_rotation = 10

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3 / 3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = eraser_aug_prob

    @tf.function
    def color_transform(self, img1: tf.Tensor, img2: tf.Tensor):
        image_stack = tf.concat([img1, img2], axis=0)
        image_stack = self.photo_aug(image_stack)
        img1, img2 = tf.unstack(tf.split(image_stack, 2, axis=0))

        img1 = tf.clip_by_value(img1, 0., 1.)
        img2 = tf.clip_by_value(img2, 0., 1.)

        return img1, img2

    @tf.function
    def eraser_transform(self, img1, img2, bounds=(50, 100)):
        """ Occlusion augmentation """

        ht, wd = tf.unstack(tf.shape(img1))[0:2]
        if tf.random.uniform([]) < self.eraser_aug_prob:
            mean_color = tf.reduce_mean(tf.reshape(img2, (-1, 3)), axis=0)
            for _ in range(tf.random.uniform([], 1, 3, dtype=tf.int32)):
                x0 = tf.random.uniform([], 0, wd, dtype=tf.int32)
                y0 = tf.random.uniform([], 0, ht, dtype=tf.int32)
                dx = tf.random.uniform([],
                                       tf.minimum(bounds[0], wd - x0),
                                       tf.minimum(bounds[1], wd - x0 + 1), dtype=tf.int32)
                dy = tf.random.uniform([],
                                       tf.minimum(bounds[0], ht - y0),
                                       tf.minimum(bounds[1], ht - y0 + 1), dtype=tf.int32)

                mask = tf.ones([dy, dx], dtype=mean_color.dtype)
                mask = tf.pad(mask, [[y0, ht - (y0 + dy)], [x0, wd - (x0 + dx)]],
                              constant_values=0.)
                mask = mask[..., tf.newaxis]
                print(img2.shape, mask.shape, mean_color.shape)
                img2 = (1. - mask) * img2 + mask * mean_color[tf.newaxis, tf.newaxis]

        return img1, img2

    @tf.function
    def resize_sparse_flow_map(self, flow: tf.Tensor, valid: tf.Tensor, fx=1.0, fy=1.0):
        ht, wd = tf.unstack(tf.shape(flow))[0:2]

        ht1 = tf.cast(tf.round(tf.cast(ht, tf.float32) * fy), tf.int32)
        wd1 = tf.cast(tf.round(tf.cast(wd, tf.float32) * fx), tf.int32)

        flow = tf.image.resize(flow, (ht1, wd1), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        flow = flow * tf.reshape([fx, fy], [1,1,2])
        valid = tf.image.resize(valid, (ht1, wd1), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return flow, valid

    @tf.function
    def spatial_transform(self, img1: tf.Tensor, img2: tf.Tensor, flow: tf.Tensor, valid: tf.Tensor):
        if self.do_rotation:
            imgs, flow, _ = random_rotation(tf.stack([img1, img2]), flow, tf.ones_like(flow)[..., 0:1],
                                            max_rotation=self.max_rotation)
            img1, img2 = tf.unstack(imgs, axis=0)
        # randomly sample scale
        ht, wd = tf.unstack(tf.shape(img1)[0:2])
        min_scale = tf.maximum(
            (self.crop_size[0] + 8.) / tf.cast(ht, tf.float32),
            (self.crop_size[1] + 8.) / tf.cast(wd, tf.float32))

        scale = tf.pow(2., tf.random.uniform([], self.min_scale, self.max_scale))
        scale_x = scale
        scale_y = scale
        if tf.random.uniform([]) < self.stretch_prob:
            scale_x *= tf.pow(2., tf.random.uniform([], -self.max_stretch, self.max_stretch))
            scale_y *= tf.pow(2., tf.random.uniform([], -self.max_stretch, self.max_stretch))

        scale_x = tf.clip_by_value(scale_x, min_scale, scale_x)
        scale_y = tf.clip_by_value(scale_y, min_scale, scale_y)

        if tf.random.uniform([]) < self.spatial_aug_prob:
            # rescale the images
            t_h = tf.cast(tf.round(tf.cast(ht, tf.float32) * scale_y), tf.int32)
            t_w = tf.cast(tf.round(tf.cast(wd, tf.float32) * scale_x), tf.int32)

            scale_y = tf.cast(t_h, tf.float32) / tf.cast(ht, tf.float32)
            scale_x = tf.cast(t_w, tf.float32) / tf.cast(wd, tf.float32)

            img1 = tf.image.resize(img1, (t_h, t_w), method=tf.image.ResizeMethod.BILINEAR)
            img2 = tf.image.resize(img2, (t_h, t_w), method=tf.image.ResizeMethod.BILINEAR)
            flow, valid = self.resize_sparse_flow_map(flow, valid, fx=scale_x, fy=scale_y)

        if self.do_flip:
            if tf.random.uniform([]) < self.h_flip_prob:  # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * tf.reshape([-1.0, 1.0], [1, 1, 2])
                valid = valid[:, ::-1]

            if tf.random.uniform([]) < self.v_flip_prob:  # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * tf.reshape([1.0, -1.0], [1, 1, 2])
                valid = valid[::-1, :]

        ht, wd = tf.unstack(tf.shape(img1))[0:2]
        y0 = tf.random.uniform([], 0, ht - self.crop_size[0], tf.int32)
        x0 = tf.random.uniform([], 0, wd - self.crop_size[1], tf.int32)

        img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        flow = flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        valid = valid[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

        img1 = tf.ensure_shape(img1, [self.crop_size[0], self.crop_size[1], 3])
        img2 = tf.ensure_shape(img2, [self.crop_size[0], self.crop_size[1], 3])
        flow = tf.ensure_shape(flow, [self.crop_size[0], self.crop_size[1], 2])

        return img1, img2, flow, valid

    def __call__(self, img1, img2, flow, valid):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow, valid = self.spatial_transform(img1, img2, flow, valid)

        return img1, img2, flow, valid


class MultiFrameAugmentor(SparseFlowAugmentor):
    def __init__(self, *args, **kwargs):
        super(MultiFrameAugmentor, self).__init__(*args, **kwargs)
        self.min_scale = 1.

    @tf.function
    def color_transform(self, img1: tf.Tensor, img2: tf.Tensor, img3: tf.Tensor):
        """ Photometric augmentation """

        # asymmetric
        if tf.random.uniform([]) < self.asymmetric_color_aug_prob:
            img1 = self.photo_aug(img1)
            img2 = self.photo_aug(img2)
            img3 = self.photo_aug(img3)

        # symmetric
        else:
            image_stack = tf.concat([img1, img2, img3], axis=0)
            image_stack = self.photo_aug(image_stack)
            img1, img2, img3 = tf.unstack(tf.split(image_stack, 3, axis=0))

        img1 = tf.clip_by_value(img1, 0., 1.)
        img2 = tf.clip_by_value(img2, 0., 1.)
        img3 = tf.clip_by_value(img3, 0., 1.)

        return img1, img2, img3

    @tf.function
    def eraser_transform(self, img1, img2, img3, bounds=(50, 100)):
        """ Occlusion augmentation """

        ht, wd = tf.unstack(tf.shape(img1)[0:2])
        results = []
        for img_targ in [img1, img3]:
            if tf.random.uniform([]) < self.eraser_aug_prob:
                mean_color = tf.reduce_mean(tf.reshape(img_targ, (-1, 3)), axis=0)
                for _ in range(tf.random.uniform([], 1, 3, dtype=tf.int32)):
                    x0 = tf.random.uniform([], 0, wd, dtype=tf.int32)
                    y0 = tf.random.uniform([], 0, ht, dtype=tf.int32)
                    dx = tf.random.uniform([],
                                           tf.minimum(bounds[0], wd - x0),
                                           tf.minimum(bounds[1], wd - x0 + 1), dtype=tf.int32)
                    dy = tf.random.uniform([],
                                           tf.minimum(bounds[0], ht - y0),
                                           tf.minimum(bounds[1], ht - y0 + 1), dtype=tf.int32)

                    mask = tf.ones([dy, dx], dtype=mean_color.dtype)
                    mask = tf.pad(mask, [[y0, ht - (y0 + dy)], [x0, wd - (x0 + dx)]],
                                  constant_values=0.)
                    mask = mask[..., tf.newaxis]
                    print(img_targ.shape, mask.shape, mean_color.shape)
                    img_targ = (1. - mask) * img_targ + mask * mean_color[tf.newaxis, tf.newaxis]
            results.append(img_targ)
        img1, img2, img3 = results[0], img2, results[1]

        return img1, img2, img3

    @tf.function
    def spatial_transform(self, img1: tf.Tensor, img2: tf.Tensor, img3: tf.Tensor,
                          flow1: tf.Tensor, flow2: tf.Tensor,
                          valid1: tf.Tensor, valid2: tf.Tensor):
        # randomly sample scale
        ht, wd = tf.unstack(tf.shape(img1)[0:2])
        min_scale = tf.maximum(
            (self.crop_size[0] + 8.) / tf.cast(ht, tf.float32),
            (self.crop_size[1] + 8.) / tf.cast(wd, tf.float32))

        scale = tf.pow(2., tf.random.uniform([], self.min_scale, self.max_scale))
        scale_x = scale
        scale_y = scale
        if tf.random.uniform([]) < self.stretch_prob:
            scale_x *= tf.pow(2., tf.random.uniform([], -self.max_stretch, self.max_stretch))
            scale_y *= tf.pow(2., tf.random.uniform([], -self.max_stretch, self.max_stretch))

        scale_x = tf.clip_by_value(scale_x, min_scale, scale_x)
        scale_y = tf.clip_by_value(scale_y, min_scale, scale_y)

        f_img1, f_img2, f_img3 = img1, img2, img3
        f_flow1, f_flow2, f_valid1, f_valid2 = flow1, flow2, valid1, valid2

        if tf.random.uniform([]) < self.spatial_aug_prob:
            # rescale the images
            t_h = tf.cast(tf.round(tf.cast(ht, tf.float32) * scale_y), tf.int32)
            t_w = tf.cast(tf.round(tf.cast(wd, tf.float32) * scale_x), tf.int32)

            scale_y = tf.cast(t_h, tf.float32) / tf.cast(ht, tf.float32)
            scale_x = tf.cast(t_w, tf.float32) / tf.cast(wd, tf.float32)

            img1 = tf.image.resize(img1, (t_h, t_w), method=tf.image.ResizeMethod.BILINEAR)
            img2 = tf.image.resize(img2, (t_h, t_w), method=tf.image.ResizeMethod.BILINEAR)
            img3 = tf.image.resize(img3, (t_h, t_w), method=tf.image.ResizeMethod.BILINEAR)
            flow1, valid1 = self.resize_sparse_flow_map(flow1, valid1, fx=scale_x, fy=scale_y)
            flow2, valid2 = self.resize_sparse_flow_map(flow2, valid2, fx=scale_x, fy=scale_y)

            y0 = tf.random.uniform([], 0, t_h - ht, tf.int32)
            x0 = tf.random.uniform([], 0, t_w - wd, tf.int32)
            f_img1 = img1[y0:y0 + ht, x0:x0 + wd]
            f_img2 = img2[y0:y0 + ht, x0:x0 + wd]
            f_img3 = img3[y0:y0 + ht, x0:x0 + wd]
            f_flow1 = flow1[y0:y0 + ht, x0:x0 + wd]
            f_valid1 = valid1[y0:y0 + ht, x0:x0 + wd]
            f_flow2 = flow2[y0:y0 + ht, x0:x0 + wd]
            f_valid2 = valid2[y0:y0 + ht, x0:x0 + wd]

        if self.do_flip:
            if tf.random.uniform([]) < self.h_flip_prob:  # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                img3 = img3[:, ::-1]
                flow1 = flow1[:, ::-1] * tf.reshape([-1.0, 1.0], [1, 1, 2])
                valid1 = valid1[:, ::-1]
                flow2 = flow2[:, ::-1] * tf.reshape([-1.0, 1.0], [1, 1, 2])
                valid2 = valid2[:, ::-1]

            if tf.random.uniform([]) < self.v_flip_prob:  # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                img3 = img3[::-1, :]
                flow1 = flow1[::-1, :] * tf.reshape([1.0, -1.0], [1, 1, 2])
                valid1 = valid1[::-1, :]
                flow2 = flow2[::-1, :] * tf.reshape([1.0, -1.0], [1, 1, 2])
                valid2 = valid2[::-1, :]

        ht, wd = tf.unstack(tf.shape(f_img1))[0:2]
        y0 = tf.random.uniform([], 0, ht - self.crop_size[0], tf.int32)
        x0 = tf.random.uniform([], 0, wd - self.crop_size[1], tf.int32)

        img1 = f_img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img2 = f_img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img3 = f_img3[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        flow1 = f_flow1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        valid1 = f_valid1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        flow2 = f_flow2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        valid2 = f_valid2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

        img1 = tf.ensure_shape(img1, [self.crop_size[0], self.crop_size[1], 3])
        img2 = tf.ensure_shape(img2, [self.crop_size[0], self.crop_size[1], 3])
        img3 = tf.ensure_shape(img3, [self.crop_size[0], self.crop_size[1], 3])
        flow1 = tf.ensure_shape(flow1, [self.crop_size[0], self.crop_size[1], 2])
        flow2 = tf.ensure_shape(flow2, [self.crop_size[0], self.crop_size[1], 2])

        return img1, img2, img3, flow1, valid1, flow2, valid2, f_img1, f_img2, f_img3, x0, y0

    def __call__(self, img1, img2, img3, flow1, valid1, flow2, valid2):
        o_img1, o_img2, o_img3 = img1, img2, img3
        img1, img2, img3 = self.color_transform(img1, img2, img3)
        img1, img2, img3 = self.eraser_transform(img1, img2, img3)
        img1, img2, img3, flow1, valid1, flow2, valid2, f_img1, f_img2, f_img3, x0, y0 = self.spatial_transform(
                                                                                img1, img2, img3,
                                                                                flow1, valid1, flow2, valid2)

        ret = ({'augmented_img': (img1, img2, img3),
               'original_img': (f_img1, f_img2, f_img3),
                'crop_x': x0,
                'crop_y': y0
                },

               {'flows':(flow1, flow2),
               'valids': (valid1, valid2)
        })
        return ret


class UnsupAugmentor(SparseFlowAugmentor):
    def __init__(self, *args, **kwargs):
        super(UnsupAugmentor, self).__init__(*args, **kwargs)
        self.min_scale = 1.
        self.full_size = None

    @tf.function
    def color_transform(self, img1: tf.Tensor, img2: tf.Tensor):
        """ Photometric augmentation """

        # asymmetric
        if tf.random.uniform([]) < self.asymmetric_color_aug_prob:
            img1 = self.photo_aug(img1)
            img2 = self.photo_aug(img2)

        # symmetric
        else:
            image_stack = tf.concat([img1, img2], axis=0)
            image_stack = self.photo_aug(image_stack)
            img1, img2 = tf.unstack(tf.split(image_stack, 2, axis=0))

        img1 = tf.clip_by_value(img1, 0., 1.)
        img2 = tf.clip_by_value(img2, 0., 1.)

        return img1, img2

    @tf.function
    def eraser_transform(self, img1, img2, bounds=(50, 100)):
        """ Occlusion augmentation """

        ht, wd = tf.unstack(tf.shape(img1)[0:2])
        results = []
        for img_targ in [img2]:
            if tf.random.uniform([]) < self.eraser_aug_prob:
                mean_color = tf.reduce_mean(tf.reshape(img_targ, (-1, 3)), axis=0)
                for _ in range(tf.random.uniform([], 1, 3, dtype=tf.int32)):
                    x0 = tf.random.uniform([], 0, wd, dtype=tf.int32)
                    y0 = tf.random.uniform([], 0, ht, dtype=tf.int32)
                    dx = tf.random.uniform([],
                                           tf.minimum(bounds[0], wd - x0),
                                           tf.minimum(bounds[1], wd - x0 + 1), dtype=tf.int32)
                    dy = tf.random.uniform([],
                                           tf.minimum(bounds[0], ht - y0),
                                           tf.minimum(bounds[1], ht - y0 + 1), dtype=tf.int32)

                    mask = tf.ones([dy, dx], dtype=mean_color.dtype)
                    mask = tf.pad(mask, [[y0, ht - (y0 + dy)], [x0, wd - (x0 + dx)]],
                                  constant_values=0.)
                    mask = mask[..., tf.newaxis]
                    print(img_targ.shape, mask.shape, mean_color.shape)
                    img_targ = (1. - mask) * img_targ + mask * mean_color[tf.newaxis, tf.newaxis]
            results.append(img_targ)
        img1, img2 = img1, results[0]

        return img1, img2

    @tf.function
    def spatial_transform(self, img1: tf.Tensor, img2: tf.Tensor,
                          flow1: tf.Tensor,
                          valid1: tf.Tensor):
        # randomly sample scale
        ht, wd = tf.unstack(tf.shape(img1)[0:2])
        if self.full_size is None:
            full_size = get_proc_size_floor((ht, wd), multiple=8)
        else:
            inst_full_size = get_proc_size_floor((ht, wd), multiple=8)
            full_size = self.full_size
            fh = tf.minimum(inst_full_size[0], full_size[0])
            fw = tf.minimum(inst_full_size[1], full_size[1])
            full_size = (fh, fw)

        min_scale = tf.maximum(
            (self.crop_size[0] + 8.) / tf.cast(full_size[0], tf.float32),
            (self.crop_size[1] + 8.) / tf.cast(full_size[1], tf.float32))

        scale = tf.pow(2., tf.random.uniform([], self.min_scale, self.max_scale))
        scale_x = scale
        scale_y = scale
        if tf.random.uniform([]) < self.stretch_prob:
            scale_x *= tf.pow(2., tf.random.uniform([], -self.max_stretch, self.max_stretch))
            scale_y *= tf.pow(2., tf.random.uniform([], -self.max_stretch, self.max_stretch))

        scale_x = tf.clip_by_value(scale_x, min_scale, scale_x)
        scale_y = tf.clip_by_value(scale_y, min_scale, scale_y)

        if tf.random.uniform([]) < self.spatial_aug_prob:
            # rescale the images
            t_h = tf.cast(tf.round(tf.cast(ht, tf.float32) * scale_y), tf.int32)
            t_w = tf.cast(tf.round(tf.cast(wd, tf.float32) * scale_x), tf.int32)

            scale_y = tf.cast(t_h, tf.float32) / tf.cast(ht, tf.float32)
            scale_x = tf.cast(t_w, tf.float32) / tf.cast(wd, tf.float32)

            img1 = tf.image.resize(img1, (t_h, t_w), method=tf.image.ResizeMethod.BILINEAR)
            img2 = tf.image.resize(img2, (t_h, t_w), method=tf.image.ResizeMethod.BILINEAR)
            flow1, valid1 = self.resize_sparse_flow_map(flow1, valid1, fx=scale_x, fy=scale_y)

            y0 = tf.random.uniform([], 0, t_h - full_size[0]+1, tf.int32)
            x0 = tf.random.uniform([], 0, t_w - full_size[1]+1, tf.int32)
            f_img1 = img1[y0:y0 + full_size[0], x0:x0 + full_size[1]]
            f_img2 = img2[y0:y0 + full_size[0], x0:x0 + full_size[1]]
            f_flow1 = flow1[y0:y0 + full_size[0], x0:x0 + full_size[1]]
            f_valid1 = valid1[y0:y0 + full_size[0], x0:x0 + full_size[1]]
        else:
            y0 = tf.random.uniform([], 0, ht - full_size[0]+1, tf.int32)
            x0 = tf.random.uniform([], 0, wd - full_size[1]+1, tf.int32)
            f_img1 = img1[y0:y0 + full_size[0], x0:x0 + full_size[1]]
            f_img2 = img2[y0:y0 + full_size[0], x0:x0 + full_size[1]]
            f_flow1 = flow1[y0:y0 + full_size[0], x0:x0 + full_size[1]]
            f_valid1 = valid1[y0:y0 + full_size[0], x0:x0 + full_size[1]]

        if self.do_flip:
            if tf.random.uniform([]) < self.h_flip_prob:  # h-flip
                f_img1 = f_img1[:, ::-1]
                f_img2 = f_img2[:, ::-1]
                f_flow1 = f_flow1[:, ::-1] * tf.reshape([-1.0, 1.0], [1, 1, 2])
                f_valid1 = f_valid1[:, ::-1]

            if tf.random.uniform([]) < self.v_flip_prob:  # v-flip
                f_img1 = f_img1[::-1, :]
                f_img2 = f_img2[::-1, :]
                f_flow1 = f_flow1[::-1, :] * tf.reshape([1.0, -1.0], [1, 1, 2])
                f_valid1 = f_valid1[::-1, :]

        ht, wd = tf.unstack(tf.shape(f_img1))[0:2]
        y0 = tf.random.uniform([], 0, (ht - self.crop_size[0])//8+1, tf.int32) * 8
        x0 = tf.random.uniform([], 0, (wd - self.crop_size[1])//8+1, tf.int32) * 8

        img1 = f_img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img2 = f_img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

        flow1 = f_flow1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        valid1 = f_valid1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

        img1 = tf.ensure_shape(img1, [self.crop_size[0], self.crop_size[1], 3])
        img2 = tf.ensure_shape(img2, [self.crop_size[0], self.crop_size[1], 3])
        flow1 = tf.ensure_shape(flow1, [self.crop_size[0], self.crop_size[1], 2])
        valid1 = tf.ensure_shape(valid1, [self.crop_size[0], self.crop_size[1], 1])
        f_img1 = tf.reshape(f_img1, [full_size[0], full_size[1], 3])
        f_img2 = tf.reshape(f_img2, [full_size[0], full_size[1], 3])
        f_flow1 = tf.reshape(f_flow1, [full_size[0], full_size[1], 2])
        f_valid1 = tf.reshape(f_valid1, [full_size[0], full_size[1], 1])

        return img1, img2, flow1, valid1, f_img1, f_img2, f_flow1, f_valid1, x0, y0

    def __call__(self, img1, img2, flow1, valid1):
        img1, img2, flow1, valid1, f_img1, f_img2, f_flow1, f_valid1, x0, y0 = self.spatial_transform(img1, img2, flow1, valid1)
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)

        ret = ({'augmented_img': (img1, img2),
               'original_img': (f_img1, f_img2),
                'crop_x': x0,
                'crop_y': y0
                },

               {'flows':[flow1],
                'original_flows': [f_flow1],
               'valids': [valid1],
                'original_valids': [f_valid1]
        })
        return ret


def get_proc_size_floor(size, multiple=8):
    im_size = size
    im_size_m = tf.math.floor(tf.cast(im_size, tf.float32) / multiple) * multiple
    im_size_m = tf.cast(im_size_m, tf.int32)
    return im_size_m