import tensorflow as tf
import numpy as np
import cv2

def numpy_pad_edge(array, paddings):
    return np.pad(array, paddings, 'edge')

#@tf.custom_gradient
def pad_edge(tensor: tf.Tensor, paddings):
    res = tf.numpy_function(numpy_pad_edge, [tensor, paddings], tensor.dtype)
    return res

def pad_edge_tf(tensor: tf.Tensor, paddings):
    while not tf.reduce_all(tf.convert_to_tensor(paddings) == 0):
        tf.autograph.experimental.set_loop_options(
            shape_invariants=[(tensor, tf.TensorShape([None, None, None, None]))]
        )
        paddings_ = tf.convert_to_tensor(paddings)
        paddings_ = tf.cast(paddings_ > 0, tf.int32)

        tensor = tf.pad(tensor, paddings_, mode="SYMMETRIC")
        paddings = tf.convert_to_tensor(paddings) - paddings_
        print(paddings_, paddings)
