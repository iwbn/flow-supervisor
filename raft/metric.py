import tensorflow as tf

class EPE(tf.keras.metrics.Metric):
    def __init__(self, name='epe', **kwargs):
        super(EPE, self).__init__(name=name, **kwargs)
        self.mean = tf.keras.metrics.Mean()

    def update_state(self, y_true, y_pred, sample_weight=None):
        if isinstance(y_true, tf.Tensor) and y_true.shape[-1] == 3:
            mask = y_true[..., 2:3]
            y_true = y_true[..., 0:2]
        elif isinstance(y_true, list) or isinstance(y_true, tuple):
            if len(y_true) == 2:
                y_true, mask = y_true
            else:
                y_true = y_true[0]
                mask = tf.ones_like(y_pred)[...,0:1]
        else:
            y_true = y_true[0]
            mask = tf.ones_like(y_pred)[..., 0:1]

        y_true = tf.cast(y_true, y_pred.dtype)
        mask = tf.cast(mask, y_pred.dtype)

        diff = y_pred - tf.cast(y_true, y_pred.dtype)

        sqer = tf.square(diff)
        sqer = tf.reduce_sum(sqer, axis=-1, keepdims=True)
        epes = tf.sqrt(sqer) * mask
        values = tf.reduce_sum(epes, [1, 2, 3])
        result = tf.cond(tf.reduce_all(mask == 0.),
                         lambda: tf.ones_like(values) * -1.0,
                         lambda: values / tf.reduce_sum(mask, [1, 2, 3]))


        self.mean.update_state(result)

    def result(self):
        res = self.mean.result()
        return res

    def reset_state(self):
        self.mean.reset_state()
        super(EPE, self).reset_state()


class SparseEPE(tf.keras.metrics.Metric):
    def __init__(self, name='epe', **kwargs):
        super(SparseEPE, self).__init__(name=name, **kwargs)
        self.mean = tf.keras.metrics.Mean()

    def update_state(self, y_true, y_pred, sample_weight=None):
        if isinstance(y_true, tf.Tensor) and y_true.shape[-1] == 3:
            mask = y_true[..., 2:3]
            y_true = y_true[..., 0:2]
        elif isinstance(y_true, list) or isinstance(y_true, tuple):
            if len(y_true) == 2:
                y_true, mask = y_true
            else:
                y_true = y_true[0]
                mask = tf.ones_like(y_pred)[...,0:1]
        else:
            y_true = y_true[0]
            mask = tf.ones_like(y_pred)[..., 0:1]

        y_true = tf.cast(y_true, y_pred.dtype)
        mask = tf.cast(mask, y_pred.dtype)

        diff = y_pred - tf.cast(y_true, y_pred.dtype)
        raise NotImplementedError
        sqer = tf.square(diff)
        sqer = tf.reduce_sum(sqer, axis=-1, keepdims=True)
        epes = tf.sqrt(sqer) * mask
        values = tf.reduce_sum(epes, [1, 2, 3])
        result = tf.cond(tf.reduce_all(mask == 0.),
                         lambda: tf.ones_like(values) * -1.0,
                         lambda: values / tf.reduce_sum(mask, [1, 2, 3]))

        self.mean.update_state(result)

    def result(self):
        res = self.mean.result()
        return res

    def reset_state(self):
        self.mean.reset_state()
        super(EPE, self).reset_state()