import tensorflow as tf
import numpy as np
from raft.smurf_models.smurf_utils import compute_occlusions as compute_occlusions_smurf


def compute_occlusions(forward_flow,
                       backward_flow,
                       occlusion_estimation = None,
                       occlusions_are_zeros = True,
                       occ_active = None,
                       boundaries_occluded = True):
    forward_flow = tf.reverse(forward_flow, axis=[3])
    backward_flow = tf.reverse(backward_flow, axis=[3])

    result = compute_occlusions_smurf(forward_flow, backward_flow,
                                     occlusion_estimation,
                                     occlusions_are_zeros,
                                     occ_active,
                                     boundaries_occluded)

    return result


def compute_ae(y_true:tf.Tensor, y_pred:tf.Tensor):
    y_true_mask = 1. - tf.cast(tf.reduce_all(y_true == 0., axis=-1, keepdims=True), y_true.dtype)
    y_pred_mask = 1. - tf.cast(tf.reduce_all(y_pred == 0., axis=-1, keepdims=True), y_pred.dtype)

    inner = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
    norms = tf.norm(y_true, axis=-1, keepdims=True) * tf.norm(y_pred, axis=-1, keepdims=True) + (1. - y_true_mask*y_pred_mask)

    cos = (inner / norms)
    rad = tf.math.acos(tf.clip_by_value(cos, -1., 1.))
    deg = rad / np.pi * 180
    return deg, y_true_mask*y_pred_mask