import tensorflow as tf


class UpsampleConvexWithMask(tf.keras.Model):
    def __init__(self, scale=8, **kwargs):
        super(UpsampleConvexWithMask, self).__init__(**kwargs)
        self.scale = scale
        self.softmax = tf.keras.layers.Softmax(axis=3, dtype='float32')
        self.mul_float32 = tf.keras.layers.Multiply(dtype='float32')

    def call(self, inputs, training=None, mask=None):
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            if len(inputs) == 3:
                x, mask, ref_tensor = inputs
            else:
                x, mask = inputs
                s = tf.shape(x)
                ref_tensor = tf.zeros([s[0], s[1] * self.scale, s[2] * self.scale, s[3]])
        else:
            raise ValueError
        B, H, W, C = tf.unstack(tf.shape(x))

        mask = tf.reshape(mask, [B, H, W, 9, -1, 1])
        mask = self.softmax(mask)

        x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT')
        shifted = []
        for i in range(3):
            for j in range(3):
                shifted.append(x_padded[:, i:i + H, j:j + W])
        p = tf.stack(shifted, axis=3)
        p = tf.reshape(p, [B, H, W, 9, 1, -1])

        upflow = tf.reduce_sum(self.mul_float32([mask, p]), axis=3)
        upflow = tf.reshape(upflow, [B, H, W, self.scale, self.scale, C])
        upflow = tf.transpose(upflow, [0, 1, 3, 2, 4, 5])
        upflow = tf.reshape(upflow, [B, H*self.scale, W*self.scale, C])

        s = tf.shape(ref_tensor)[1:3]
        o_s = tf.TensorShape([x.shape[0], ref_tensor.shape[1], ref_tensor.shape[2], x.shape[3]])
        return tf.ensure_shape(upflow[:,0:s[0], 0:s[1]], o_s)