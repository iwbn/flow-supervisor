import tensorflow as tf
import tensorflow_addons as tfa
from scipy import interpolate
import numpy as np

def crop_bboxes(images: tf.Tensor, offsets: tf.Tensor, target_size: tf.Tensor):
    a = tf.TensorArray(images.dtype, size=tf.shape(images)[0])

    h, w = tf.unstack(target_size)

    i = tf.constant(0)
    c = lambda i, a: tf.less(i, tf.shape(images)[0])

    def _crop(i, a):
        image_i = images[i]
        y, x = tf.unstack(offsets[i])

        cropped = image_i[y:y + h, x:x + w]
        a = a.write(i, cropped)
        return i + 1, a

    _, a = tf.while_loop(c, _crop, [i, a])

    res = a.stack()
    return res


def pad_bboxes(images: tf.Tensor, offsets: tf.Tensor, target_size: tf.Tensor):
    a = tf.TensorArray(images.dtype, size=tf.shape(images)[0])

    h, w = tf.unstack(target_size)

    i = tf.constant(0)
    c = lambda i, a: tf.less(i, tf.shape(images)[0])

    def _crop(i, a):
        image_i = images[i]
        ih, iw = tf.unstack(tf.shape(image_i)[0:2])
        y, x = tf.unstack(offsets[i])

        pad = [[y, h-y-ih],[x, w-x-iw], [0,0]]
        padded = tf.pad(image_i, pad)
        a = a.write(i, padded)
        return i + 1, a

    _, a = tf.while_loop(c, _crop, [i, a])

    res = a.stack()
    return res


def central_pad(images: tf.Tensor, target_size: tf.Tensor):
    h, w = tf.unstack(target_size)

    ih, iw = tf.unstack(tf.shape(images)[1:3])
    y = (h - ih) // 2
    x = (w - iw) // 2

    pad = [[0,0], [y, h - y - ih], [x, w - x - iw], [0, 0]]
    padded = tf.pad(images, pad)

    return padded


def central_crop(images: tf.Tensor, target_size: tf.Tensor):
    h, w = tf.unstack(target_size)

    ih, iw = tf.unstack(tf.shape(images)[1:3])
    y = (ih - h) // 2
    x = (iw - w) // 2

    cropped = images[:, y: y+h, x: x+w]

    return cropped


def warp_image(image: tf.Tensor, flow: tf.Tensor, occlusion: str="ZERO", background_image: tf.Tensor=None):
    flow = tf.gather(flow, [1, 0], axis=3)
    flow = -flow
    img = tfa.image.dense_image_warp(image, flow)
    # img = dense_image_warp(image, flow, name)
    img = tf.reshape(img, tf.shape(image))
    mask = create_outgoing_mask(flow)
    if occlusion.lower() == "zero":
        img = img * mask
    elif occlusion.lower() == "input":
        img = img * mask + image * (1. - mask)
    elif occlusion.lower() == "background":
        img = img * mask + background_image * (1. - mask)
    else:
        raise ValueError
    return img

def create_outgoing_mask(flow):
    """Computes a mask that is zero at all positions where the flow
    would carry a pixel over the image boundary."""
    with tf.name_scope('create_outgoing_mask'):
        s = tf.unstack(tf.shape(flow))
        height, width, _ = s[-3::]

        grid = tf.cast(tf.meshgrid(tf.range(width), tf.range(height)), tf.float32)
        grid = tf.transpose(grid, [1,2,0])
        for i in range(len(s)-3):
            grid = tf.expand_dims(grid, 0)

        pos_x = grid[...,0] + flow[...,0]
        pos_y = grid[...,1] + flow[...,1]

        inside_x = tf.logical_and(pos_x <= tf.cast(width - 1, tf.float32),
                                  pos_x >=  0.0)
        inside_y = tf.logical_and(pos_y <= tf.cast(height - 1, tf.float32),
                                  pos_y >=  0.0)
        inside = tf.logical_and(inside_x, inside_y)
        return tf.expand_dims(tf.cast(inside, tf.float32), -1)


def forward_interpolate(flow: tf.Tensor):
    dx, dy = flow[...,0], flow[...,1]

    ht, wd = tf.unstack(tf.shape(dx)[0:2])
    x0, y0 = tf.meshgrid(tf.range(wd), tf.range(ht))
    x0 = tf.cast(x0, flow.dtype)
    y0 = tf.cast(y0, flow.dtype)

    x0 = x0
    y0 = y0
    x1 = x0 + dx
    y1 = y0 + dy

    valid = (x1 > 0.) & (x1 < tf.cast(wd, x1.dtype)) & (y1 > 0.) & (y1 < tf.cast(ht, y1.dtype))
    valid = valid[...,tf.newaxis]

    x1 = tf.reshape(x1, [-1])
    y1 = tf.reshape(y1, [-1])
    dx = tf.reshape(dx, [-1])
    dy = tf.reshape(dy, [-1])

    flow_x = interpolate_griddata(
        tf.stack((x1, y1), axis=-1), dx, tf.stack((x0, y0), axis=-1),
        method='nearest', fill_value=0)

    flow_y = interpolate_griddata(
        tf.stack((x1, y1), axis=-1), dy, tf.stack((x0, y0), axis=-1),
        method='nearest', fill_value=0)

    flow_warped = tf.stack([flow_x, flow_y], axis=-1)

    return flow_warped

def interpolate_griddata(points: tf.Tensor, values:tf.Tensor, xi:tf.Tensor, method='linear', fill_value=np.nan, rescale=False):
    def _fn(points, values, xi):
        x = interpolate.griddata(
            points, values, xi, method=method, fill_value=fill_value, rescale=rescale)
        return x

    output = tf.numpy_function(_fn, [points, values, xi], [values.dtype])
    return output