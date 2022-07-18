import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, ReLU

class FlowHead(keras.Model):
    def __init__(self, hidden_dim=256, **kwargs):
        super(FlowHead, self).__init__(**kwargs)
        self.conv1 = Conv2D(hidden_dim, 3, padding='same')
        self.conv2 = Conv2D(2, 3, padding='same', dtype='float32')
        self.relu = ReLU()

    def call(self, x, training=None, mask=None):
        return self.conv2(self.relu(self.conv1(x)))

class ClsHead(keras.Model):
    def __init__(self, hidden_dim=256, **kwargs):
        super(ClsHead, self).__init__(**kwargs)
        self.conv1 = Conv2D(hidden_dim, 3, padding='same')
        self.conv2 = Conv2D(4, 3, padding='same', dtype=tf.float32)
        self.relu = ReLU()

    def call(self, x, training=None, mask=None):
        return self.conv2(self.relu(self.conv1(x)))

class ConvGRU(keras.Model):
    def __init__(self, hidden_dim=128, **kwargs):
        super(ConvGRU, self).__init__(**kwargs)
        self.convz = Conv2D(hidden_dim, 3, padding='same')
        self.convr = Conv2D(hidden_dim, 3, padding='same')
        self.convq = Conv2D(hidden_dim, 3, padding='same')

    def call(self, inputs, training=None, mask=None):
        h, x = inputs
        hx = tf.concat([h, x], axis=3)

        z = tf.sigmoid(self.convz(hx))
        r = tf.sigmoid(self.convr(hx))
        q = tf.tanh(self.convq(tf.concat([r*h, x], axis=3)))

        h = (1-z) * h + z * q
        return h

class SepConvGRU(keras.Model):
    def __init__(self, hidden_dim=128, **kwargs):
        super(SepConvGRU, self).__init__(**kwargs)
        self.convz1 = Conv2D(hidden_dim, (1,5), padding='same')
        self.convr1 = Conv2D(hidden_dim, (1,5), padding='same')
        self.convq1 = Conv2D(hidden_dim, (1,5), padding='same')

        self.convz2 = Conv2D(hidden_dim, (5,1), padding='same')
        self.convr2 = Conv2D(hidden_dim, (5,1), padding='same')
        self.convq2 = Conv2D(hidden_dim, (5,1), padding='same')

    def call(self, inputs, training=None, mask=None):
        h, x = inputs
        # horizontal
        hx = tf.concat([h, x], axis=3)
        z = tf.sigmoid(self.convz1(hx))
        r = tf.sigmoid(self.convr1(hx))
        q = tf.tanh(self.convq1(tf.concat([r*h, x], axis=3)))
        h = (1-z) * h + z * q

        # vertical
        hx = tf.concat([h, x], axis=3)
        z = tf.sigmoid(self.convz2(hx))
        r = tf.sigmoid(self.convr2(hx))
        q = tf.tanh(self.convq2(tf.concat([r*h, x], axis=3)))
        h = (1-z) * h + z * q

        return h


class ConvDecoder(keras.Model):
    def __init__(self, hidden_dim=128, **kwargs):
        super(ConvDecoder, self).__init__(**kwargs)

        self.activate = keras.layers.LeakyReLU(0.1)
        self.conv_0 = Conv2D(hidden_dim, 3, padding='same', activation=self.activate)
        self.conv_1 = Conv2D(hidden_dim, 3, padding='same', activation=self.activate)
        self.conv_2 = Conv2D(hidden_dim, 3, padding='same', activation=self.activate)
        self.conv_3 = Conv2D(hidden_dim//2, 3, padding='same', activation=self.activate)
        self.conv_4 = Conv2D(hidden_dim//2, 3, padding='same', activation=self.activate)

    def call(self, inputs, training=None, mask=None):
        _, x = inputs

        x = tf.concat((self.conv_0(x), x), axis=3)
        x = tf.concat((self.conv_1(x), x), axis=3)
        x = tf.concat((self.conv_2(x), x), axis=3)
        x = tf.concat((self.conv_3(x), x), axis=3)
        x = tf.concat((self.conv_4(x), x), axis=3)

        return x


class SmallMotionEncoder(keras.Model):
    def __init__(self, **kwargs):
        super(SmallMotionEncoder, self).__init__(**kwargs)

        self.convc1 = Conv2D(96, 1, padding='same')
        self.convf1 = Conv2D(64, 7, padding='same')
        self.convf2 = Conv2D(32, 3, padding='same')
        self.conv = Conv2D(80, 3, padding='same')
        self.relu = ReLU()

    def call(self, inputs, training=None, mask=None):
        flow, corr = inputs
        cor = self.relu(self.convc1(corr))
        flo = self.relu(self.convf1(flow))
        flo = self.relu(self.convf2(flo))
        cor_flo = tf.concat([cor, flo], axis=3)
        out = self.relu(self.conv(cor_flo))
        return tf.concat([out, flow], axis=3)


class BasicMotionEncoder(keras.Model):
    def __init__(self, **kwargs):
        super(BasicMotionEncoder, self).__init__(**kwargs)
        self.convc1 = Conv2D(256, 1, padding='same', activation='relu')
        self.convc2 = Conv2D(192, 3, padding='same', activation='relu')
        self.convf1 = Conv2D(128, 7, padding='same', activation='relu')
        self.convf2 = Conv2D(64, 3, padding='same', activation='relu')
        self.conv = Conv2D(126, 3, padding='same', activation='relu')
        self.concat = tf.keras.layers.Concatenate(axis=3)

    def call(self, inputs, training=None, mask=None):
        flow, corr = inputs
        cor = self.convc1(corr)
        cor = self.convc2(cor)
        flo = self.convf1(flow)
        flo = self.convf2(flo)

        cor_flo = self.concat([cor, flo])
        out = self.conv(cor_flo)
        res = self.concat([out, flow])
        return res


class SmallUpdateBlock(keras.Model):
    def __init__(self, hidden_dim=96, **kwargs):
        super(SmallUpdateBlock, self).__init__(**kwargs)
        self.encoder = SmallMotionEncoder()
        self.gru = ConvGRU(hidden_dim=hidden_dim)
        self.flow_head = FlowHead(hidden_dim)

    def call(self, inputs, training=None, mask=None):
        net, inp, corr, flow = inputs
        motion_features = self.encoder([flow, corr], training, mask)
        inp = tf.concat([inp, motion_features], axis=3)
        net = self.gru([net, inp], training, mask)
        delta_flow = self.flow_head(net)

        return net, None, delta_flow

import sys

class BasicUpdateBlock(keras.Model):
    def __init__(self, hidden_dim=128, **kwargs):
        super(BasicUpdateBlock, self).__init__(**kwargs)
        self.encoder = BasicMotionEncoder()
        self.gru = SepConvGRU(hidden_dim=hidden_dim)
        self.flow_head = FlowHead(hidden_dim=256)

        self.mask = keras.Sequential([
            Conv2D(256, 3, padding='same'),
            ReLU(),
            Conv2D(64*9, 1, padding='same', dtype='float32')])

    def call(self, inputs, training=None, mask=None):
        net, inp, corr, flow = inputs
        motion_features = self.encoder([flow, corr], training, mask)
        inp = tf.concat([inp, motion_features], axis=3)

        net = self.gru([net, inp], training, mask)
        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow



class BasicUpdateBlockCls(keras.Model):
    def __init__(self, hidden_dim=128, **kwargs):
        super(BasicUpdateBlockCls, self).__init__(**kwargs)
        self.encoder = BasicMotionEncoder()
        self.gru = SepConvGRU(hidden_dim=hidden_dim)
        self.cls_head = ClsHead(hidden_dim=256)
        self.flow_head = FlowHead(hidden_dim=256)

        self.mask = keras.Sequential([
            Conv2D(256, 3, padding='same'),
            ReLU(),
            Conv2D(64*9, 1, padding='same', dtype='float32')])

        self.mag = keras.Sequential([
            Conv2D(256, 3, padding='same'),
            ReLU(),
            Conv2D(2, 1, padding='same', dtype='float32')])

        self.last_mul = tf.keras.layers.Multiply(dtype='float32')

    def call(self, inputs, training=None, mask=None):
        net, inp, corr, flow = inputs
        motion_features = self.encoder([flow, corr], training, mask)
        inp = tf.concat([inp, motion_features], axis=3)

        net = self.gru([net, inp], training, mask)
        delta_cls = self.cls_head(net)
        #delta_flow = self.flow_head(net)
        mag = self.mag(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_cls, mag #delta_flow


class BasicUpdateBlockClsIndep(keras.Model):
    def __init__(self, hidden_dim=128, **kwargs):
        super(BasicUpdateBlockClsIndep, self).__init__(**kwargs)
        self.encoder = SmallMotionEncoder()
        self.decoder = ConvDecoder(hidden_dim=hidden_dim)
        self.cls_head = ClsHead(hidden_dim=256)

        self.mask = keras.Sequential([
            Conv2D(256, 3, padding='same'),
            ReLU(),
            Conv2D(64*9, 1, padding='same')])

        self.mag = keras.Sequential([
            Conv2D(256, 3, padding='same'),
            ReLU(),
            Conv2D(2, 1, padding='same', dtype='float32')])

    def call(self, inputs, training=None, mask=None):
        net, inp, corr, flow = inputs
        motion_features = self.encoder([flow, corr], training, mask)
        inp = tf.concat([inp, motion_features], axis=3)

        net = self.decoder([net, inp], training, mask)
        delta_cls = self.cls_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        mag = self.mag(net)
        return net, mask, delta_cls, mag