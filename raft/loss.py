import tensorflow as tf
from tensorflow import keras

class FlowLossL1(keras.losses.Loss):
    def __init__(self, **kwargs):
        super(FlowLossL1, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        if isinstance(y_true, tf.Tensor) and y_true.shape[-1] == 3:
            mask = y_true[..., 2:3]
            y_true = y_true[..., 0:2]
        elif isinstance(y_true, list) or isinstance(y_true, tuple):
            if len(y_true) == 2:
                y_true, mask = y_true
            else:
                y_true = y_true[0]
                mask = tf.ones_like(y_true)[...,0:1]
        else:
            mask = tf.ones_like(y_true)[..., 0:1]
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        mask = tf.cast(mask, y_pred.dtype)

        # large disp
        mag = tf.sqrt(tf.reduce_sum(y_true ** 2, axis=-1, keepdims=True))
        valid = (mag < 400)

        diff = y_pred - y_true
        a = tf.abs(diff) * mask * tf.cast(valid, mask.dtype)
        loss = tf.reduce_mean(a, axis=(3))
        return loss


class FlowLossL2(keras.losses.Loss):
    def __init__(self, **kwargs):
        super(FlowLossL2, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        if isinstance(y_true, tf.Tensor) and y_true.shape[-1] == 3:
            mask = y_true[..., 2:3]
            y_true = y_true[..., 0:2]
        elif isinstance(y_true, list) or isinstance(y_true, tuple):
            if len(y_true) == 2:
                y_true, mask = y_true
            else:
                y_true = y_true[0]
                mask = tf.ones_like(y_true)[...,0:1]
        else:
            mask = tf.ones_like(y_true)[..., 0:1]
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        mask = tf.cast(mask, y_pred.dtype)

        # large disp
        mag = tf.sqrt(tf.reduce_sum(y_true ** 2, axis=-1, keepdims=True))
        valid = (mag < 400)

        diff = y_pred - y_true
        a = tf.square(diff) * mask * tf.cast(valid, mask.dtype)
        loss = tf.reduce_mean(a, axis=(3))
        return loss

class FlowLossRobust(keras.losses.Loss):
    def __init__(self, **kwargs):
        super(FlowLossRobust, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        if isinstance(y_true, tf.Tensor) and y_true.shape[-1] == 3:
            mask = y_true[..., 2:3]
            y_true = y_true[..., 0:2]
        elif isinstance(y_true, list) or isinstance(y_true, tuple):
            if len(y_true) == 2:
                y_true, mask = y_true
            else:
                y_true = y_true[0]
                mask = tf.ones_like(y_true)[...,0:1]
        else:
            mask = tf.ones_like(y_true)[..., 0:1]

        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        mask = tf.cast(mask, y_pred.dtype)

        # large disp
        mag = tf.sqrt(tf.reduce_sum(y_true ** 2, axis=-1, keepdims=True))
        valid = (mag < 400)

        diff = y_pred - y_true

        a = (diff ** 2 + 0.001 ** 2) ** 0.5
        a = a * mask * tf.cast(valid, mask.dtype)
        loss = tf.reduce_mean(a, axis=(3))
        return loss


class GradCosSimLoss(keras.losses.Loss):
    def __init__(self, **kwargs):
        super(GradCosSimLoss, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        if isinstance(y_true, tf.Tensor) and y_true.shape[-1] == 3:
            mask = y_true[..., 2:3]
            y_true = y_true[..., 0:2]
        elif isinstance(y_true, list) or isinstance(y_true, tuple):
            if len(y_true) == 2:
                y_true, mask = y_true
            else:
                y_true = y_true[0]
                mask = tf.ones_like(y_true)[...,0:1]
        else:
            mask = tf.ones_like(y_true)[..., 0:1]

        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        mask = tf.cast(mask, y_pred.dtype)

        y_true = tf.nn.l2_normalize(y_true, axis=-1)
        y_pred = tf.nn.l2_normalize(y_pred, axis=-1)
        cossim = tf.reduce_sum(y_true * y_pred, axis=-1)

        valid = tf.cast(cossim < 0., y_pred.dtype)
        loss = cossim * valid
        return loss

class GradL2Loss(keras.losses.Loss):
    def __init__(self, **kwargs):
        super(GradL2Loss, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        if isinstance(y_true, tf.Tensor) and y_true.shape[-1] == 3:
            mask = y_true[..., 2:3]
            y_true = y_true[..., 0:2]
        elif isinstance(y_true, list) or isinstance(y_true, tuple):
            if len(y_true) == 2:
                y_true, mask = y_true
            else:
                y_true = y_true[0]
                mask = tf.ones_like(y_true)[...,0:1]
        else:
            mask = tf.ones_like(y_true)[..., 0:1]

        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        mask = tf.cast(mask, y_pred.dtype)

        diff = y_pred - tf.stop_gradient(y_pred + y_true)

        a = diff ** 2
        a = a * mask
        loss = tf.reduce_mean(a, axis=(3))
        return loss


class GradCrossEntropyLoss(keras.losses.Loss):
    def __init__(self, temperature=1.0, **kwargs):
        self.temperature = temperature
        self.cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
        super(GradCrossEntropyLoss, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        if isinstance(y_true, tf.Tensor) and y_true.shape[-1] == 3:
            mask = y_true[..., 2:3]
            y_true = y_true[..., 0:2]
        elif isinstance(y_true, list) or isinstance(y_true, tuple):
            if len(y_true) == 2:
                y_true, mask = y_true
            else:
                y_true = y_true[0]
                mask = tf.ones_like(y_true)[...,0:1]
        else:
            mask = tf.ones_like(y_true)[..., 0:1]

        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        mask = tf.cast(mask, y_pred.dtype)

        t = tf.cast(self.temperature, y_true.dtype)
        y_true_u = tf.concat([y_true[..., 0:1], -y_true[..., 0:1]], axis=3) / t
        y_true_v = tf.concat([y_true[..., 1:2], -y_true[..., 1:2]], axis=3) / t

        y_pred_u = tf.concat([y_pred[..., 0:1], -y_pred[..., 0:1]], axis=3)
        y_pred_v = tf.concat([y_pred[..., 1:2], -y_pred[..., 1:2]], axis=3)

        y_true_u_s = tf.nn.softmax(y_true_u)
        y_true_v_s = tf.nn.softmax(y_true_v)

        y_pred_u_s = tf.nn.softmax(y_pred_u)
        y_pred_v_s = tf.nn.softmax(y_pred_v)

        loss = self.cross_entropy_loss(y_true_u_s, y_pred_u_s)
        loss += self.cross_entropy_loss(y_true_v_s, y_pred_v_s)

        return loss