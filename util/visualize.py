import tensorflow as tf
import numpy as np
import math

def visualize_flow(flow, max_mag=None):
    s = tf.shape(flow)
    # Use Hue, Saturation, Value colour model
    hsv_2 = tf.ones([s[0], s[1]], dtype=flow.dtype)

    x = flow[:, :, 0]
    y = flow[:, :, 1]
    rho = tf.sqrt(x ** 2 + y ** 2)
    phi = tf.numpy_function(lambda x, y: np.arctan2(y, x, dtype=y.dtype), [x, y], flow.dtype)
    phi = tf.where(tf.less(phi, 0), phi + 2. * math.pi, phi)
    if max_mag:
        rho = rho / max_mag
        rho = tf.clip_by_value(rho, 0., 1.)
    else:
        max_mag = tf.reduce_max(rho)
        max_mag = tf.cond(tf.equal(max_mag, 0.), lambda: tf.constant(1., dtype=max_mag.dtype), lambda: max_mag)
        rho = rho / max_mag

    hsv_0 = tf.cast(phi / (2. * math.pi), flow.dtype)
    hsv_1 = tf.cast(rho, flow.dtype)
    hsv = tf.stack([hsv_0, hsv_1, hsv_2], axis=2)
    rgb = tf.image.hsv_to_rgb(hsv)
    return rgb




def get_epe(pred_flow, gt_flow):
    sqer = tf.square(pred_flow - gt_flow)
    sqer = tf.reduce_sum(sqer, axis=-1, keepdims=True)
    epes = tf.sqrt(sqer)
    epes = tf.reduce_mean(epes, [1,2,3])
    return epes


def visualize_flows(flows, max_mag=None):
    rgbs = tf.map_fn(lambda x: visualize_flow(x, max_mag), flows, dtype=flows.dtype)
    return rgbs