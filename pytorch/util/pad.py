import tensorflow as tf
import numpy as np
import cv2

def numpy_pad_edge(array, paddings):
    return np.pad(array, paddings, 'edge')

#@tf.custom_gradient
def pad_edge(tensor: tf.Tensor, paddings):
    res = tf.numpy_function(numpy_pad_edge, [tensor, paddings], tensor.dtype)
    # def grad(upstream):
    #     pad = paddings
    #     ht, wd = tf.unstack(tf.shape(upstream)[1:3])
    #     c = [pad[1][0], ht - pad[1][1], pad[2][0], wd - pad[2][1]]
    #     upstream = upstream[:, c[0]:c[1], c[2]:c[3]]
    #     return [upstream] + [None] * len(tf.nest.flatten(paddings))
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
    return tensor
#
# for _ in range(100):
#     image1 = tf.random.normal([1,100,100,3])
#     image2 = tf.stop_gradient(image1)
#     with tf.GradientTape(persistent=True) as tape:
#         tape.watch([image1, image2])
#         padded = pad_edge(image1, [[0,0], [50,50], [50,50], [0,0]])
#         padded_tf = pad_edge_tf(image2, [[0,0], [50,50], [50,50], [0,0]])
#         loss = padded + padded_tf
#     grad1 = tape.gradient(loss, image1)
#     grad2 = tape.gradient(loss, image2)
#
#     print(grad1.shape)
#     print(grad2.shape)
#     cv2.imshow("im", padded.numpy()[0])
#     cv2.imshow("im2", padded_tf.numpy()[0])
#     cv2.imshow("grad1", grad1.numpy()[0])
#     cv2.imshow("grad2", grad2.numpy()[0])
#     cv2.waitKey()