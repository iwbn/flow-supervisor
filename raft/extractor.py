import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, ReLU, BatchNormalization, Dropout
from tensorflow_addons.layers import GroupNormalization, InstanceNormalization

class ResidualBlock(keras.Model):
    def __init__(self, planes, norm_fn='group', stride=1, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)

        self.conv1 = Conv2D(planes, kernel_size=3, padding='same', strides=stride)
        self.conv2 = Conv2D(planes, kernel_size=3, padding='same')
        self.relu = ReLU()

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = GroupNormalization(groups=num_groups)
            self.norm2 = GroupNormalization(groups=num_groups)
            if not stride == 1:
                self.norm3 = GroupNormalization(groups=num_groups)

        elif norm_fn == 'batch':
            self.norm1 = BatchNormalization()
            self.norm2 = BatchNormalization()
            if not stride == 1:
                self.norm3 = BatchNormalization()

        elif norm_fn == 'instance':
            self.norm1 = InstanceNormalization()
            self.norm2 = InstanceNormalization()
            if not stride == 1:
                self.norm3 = InstanceNormalization()

        elif norm_fn == 'none':
            self.norm1 = keras.Sequential()
            self.norm2 = keras.Sequential()
            if not stride == 1:
                self.norm3 = keras.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = keras.Sequential(
                [Conv2D(planes, kernel_size=1, strides=stride), self.norm3])

    def call(self, x, training=None, mask=None):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class BottleneckBlock(keras.Model):
    def __init__(self, planes, norm_fn='group', stride=1, **kwargs):
        super(BottleneckBlock, self).__init__(**kwargs)

        self.conv1 = Conv2D(planes // 4, kernel_size=1, padding='same')
        self.conv2 = Conv2D(planes // 4, kernel_size=3, padding='same', strides=stride)
        self.conv3 = Conv2D(planes, kernel_size=1, padding='same')
        self.relu = ReLU()

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = GroupNormalization(groups=num_groups)
            self.norm2 = GroupNormalization(groups=num_groups)
            self.norm3 = GroupNormalization(groups=num_groups)
            if not stride == 1:
                self.norm4 = GroupNormalization(num_groups=num_groups)

        elif norm_fn == 'batch':
            self.norm1 = BatchNormalization()
            self.norm2 = BatchNormalization()
            self.norm3 = BatchNormalization()
            if not stride == 1:
                self.norm4 = BatchNormalization()

        elif norm_fn == 'instance':
            self.norm1 = InstanceNormalization()
            self.norm2 = InstanceNormalization()
            self.norm3 = InstanceNormalization()
            if not stride == 1:
                self.norm4 = InstanceNormalization()

        elif norm_fn == 'none':
            self.norm1 = keras.Sequential()
            self.norm2 = keras.Sequential()
            self.norm3 = keras.Sequential()
            if not stride == 1:
                self.norm4 = keras.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = keras.Sequential(
                [Conv2D(planes, kernel_size=1, strides=stride), self.norm4])

    def call(self, x, training=None, mask=None):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class BasicEncoder(keras.Model):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0, **kwargs):
        super(BasicEncoder, self).__init__(**kwargs)
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = GroupNormalization(groups=8)

        elif self.norm_fn == 'batch':
            self.norm1 = BatchNormalization()

        elif self.norm_fn == 'instance':
            self.norm1 = InstanceNormalization()

        elif self.norm_fn == 'none':
            self.norm1 = keras.Sequential()

        self.conv1 = Conv2D(64, kernel_size=7, strides=2, padding='same')
        self.relu1 = ReLU()

        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)

        # output convolution
        self.conv2 = Conv2D(output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = Dropout(dropout)

        layers = self.layers.copy()
        for l in layers:
            if isinstance(l, keras.Model):
                layers.extend(l.layers)
            else:
                if isinstance(l, Conv2D):
                    l.kernel_initializer = keras.initializers.he_normal()

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return keras.Sequential(layers)

    def call(self, x, training=None, mask=None):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = tf.shape(x[0])[0]
            x = tf.concat(x, axis=0)
        else:
            batch_dim = tf.shape(x)[0]

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        if self.dropout is not None:
            x = self.dropout(x, training=training)

        if is_list:
            x = tf.split(x, [batch_dim, batch_dim], axis=0)

        return x


class BasicEncoderFullres(keras.Model):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0, **kwargs):
        super(BasicEncoderFullres, self).__init__(**kwargs)
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = GroupNormalization(groups=8)

        elif self.norm_fn == 'batch':
            self.norm1 = BatchNormalization()

        elif self.norm_fn == 'instance':
            self.norm1 = InstanceNormalization()

        elif self.norm_fn == 'none':
            self.norm1 = keras.Sequential()

        self.conv1 = Conv2D(64, kernel_size=3, strides=1, padding='same')
        self.relu1 = ReLU()

        self.layer1 = self._make_layer(32, stride=2)
        self.layer2 = self._make_layer(32, stride=2)
        self.layer3 = self._make_layer(64, stride=2)

        # output convolution
        self.conv2 = Conv2D(output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = Dropout(dropout)

        layers = self.layers.copy()
        for l in layers:
            if isinstance(l, keras.Model):
                layers.extend(l.layers)
            else:
                if isinstance(l, Conv2D):
                    l.kernel_initializer = keras.initializers.he_normal()

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return keras.Sequential(layers)

    def call(self, x, training=None, mask=None):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = tf.shape(x[0])[0]
            x = tf.concat(x, axis=0)
        else:
            batch_dim = tf.shape(x)[0]

        x = self.conv1(x)
        x = self.norm1(x)
        x = l1 = self.relu1(x)

        x = l2 = self.layer1(x)
        x = l4 = self.layer2(x)
        x = l8 = self.layer3(x)

        multi = [l1]
        for l in [l2, l4, l8]:
            res = tf.image.resize(l, tf.shape(l1)[1:3], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            multi.append(res)


        x = tf.concat(multi, axis=3)

        x = self.conv2(x)

        if self.dropout is not None:
            x = self.dropout(x, training=training)

        if is_list:
            x = tf.split(x, [batch_dim, batch_dim], axis=0)

        return x


class BasicEncoderAtrous(keras.Model):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0, **kwargs):
        super(BasicEncoderAtrous, self).__init__(**kwargs)
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = GroupNormalization(groups=8)

        elif self.norm_fn == 'batch':
            self.norm1 = BatchNormalization()

        elif self.norm_fn == 'instance':
            self.norm1 = InstanceNormalization()

        elif self.norm_fn == 'none':
            self.norm1 = keras.Sequential()

        self.conv1 = Conv2D(64, kernel_size=3, strides=1, padding='same', dilation_rate=1)
        self.conv2 = Conv2D(64, kernel_size=3, strides=1, padding='same', dilation_rate=2)
        self.conv3 = Conv2D(96, kernel_size=3, strides=1, padding='same', dilation_rate=4)
        self.conv4 = Conv2D(96, kernel_size=3, strides=1, padding='same', dilation_rate=8)
        self.conv5 = Conv2D(output_dim, kernel_size=3, strides=1, padding='same', dilation_rate=16)
        self.conv6 = Conv2D(output_dim, kernel_size=3, strides=1, padding='same', dilation_rate=1)

        if dropout > 0.0:
            self.dropout = dropout
        else:
            self.dropout = None

    def call(self, x, training=None, mask=None):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = tf.shape(x[0])[0]
            x = tf.concat(x, axis=0)
        else:
            batch_dim = tf.shape(x)[0]

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        if self.dropout is not None:
            x = self.dropout(x, training=training)

        if is_list:
            x = tf.split(x, [batch_dim, batch_dim], axis=0)

        return x


class SmallEncoder(keras.Model):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0, **kwargs):
        super(SmallEncoder, self).__init__(**kwargs)
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = GroupNormalization(groups=8)

        elif self.norm_fn == 'batch':
            self.norm1 = BatchNormalization()

        elif self.norm_fn == 'instance':
            self.norm1 = InstanceNormalization()

        elif self.norm_fn == 'none':
            self.norm1 = keras.Sequential()

        self.conv1 = Conv2D(32, kernel_size=7, strides=2, padding='same')
        self.relu1 = ReLU()

        self.layer1 = self._make_layer(32, stride=1)
        self.layer2 = self._make_layer(64, stride=2)
        self.layer3 = self._make_layer(96, stride=2)

        self.dropout = None
        if dropout > 0:
            self.dropout = Dropout(rate=dropout)

        self.conv2 = Conv2D(output_dim, kernel_size=1)

        layers = self.layers.copy()
        for l in layers:
            if isinstance(l, keras.Model):
                layers.extend(l.layers)
            else:
                if isinstance(l, Conv2D):
                    l.kernel_initializer = keras.initializers.he_normal()

    def _make_layer(self, dim, stride=1):
        layer1 = BottleneckBlock(dim, self.norm_fn, stride=stride)
        layer2 = BottleneckBlock(dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return keras.Sequential(layers)

    def call(self, x, training=None, mask=None):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = tf.concat(x, axis=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)

        if self.dropout is not None:
            x = self.dropout(x, training=training)

        if is_list:
            x = tf.split(x, [batch_dim, batch_dim], axis=0)

        return x

if __name__ == "__main__":
    layer = SmallEncoder(128)
    layer.build([1, 128, 128, 128])

    print()
