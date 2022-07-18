import tensorflow as tf
from contextlib import nullcontext

def freeze_bn(model:tf.keras.Model):
    layers = model.layers
    for layer in layers:
        submodules = layer.submodules
        for m in submodules:
            if isinstance(m, tf.keras.layers.BatchNormalization):
                m.trainable = False

                print("%s is frozen" % (m.name))

class DefaultStrategy:
    def __init__(self):
        self.scope = lambda: nullcontext()

