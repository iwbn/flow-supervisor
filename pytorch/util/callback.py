import tensorflow as tf


class CheckpointManagerCallback(tf.keras.callbacks.Callback):
    def __init__(self, ckpt_man: tf.train.CheckpointManager, *args, **kwargs):
        super(CheckpointManagerCallback, self).__init__()
        self.ckpt_man = ckpt_man
        self.batch = 0

    def set_model(self, model):
        self.model = model
        self._train_step = self.model._train_counter

    def on_epoch_end(self, epoch, logs=None):
        self.ckpt_man.save(self.model.optimizer.iterations.numpy())
