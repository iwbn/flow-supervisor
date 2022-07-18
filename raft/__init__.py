import tensorflow as tf
from tensorflow import keras
from raft.smurf_models.raft_extractor import SmallEncoder, BasicEncoder
from raft.smurf_models.raft_update import SmallUpdateBlock, BasicUpdateBlock
from .upsample import UpsampleConvexWithMask
from .allfield import calc_all_field
from .corr import CorrBlock
from argparse import Namespace, ArgumentParser
from box import Box


class RAFTArgs(object):
  """RAFT arguments."""

  def __init__(self,
               small=False,
               use_norms=True,
               corr_levels=None,
               corr_radius=None,
               convex_upsampling=True,
               dropout=0.0,
               max_rec_iters=12):
    self.small = small
    self.use_norms = use_norms
    self.convex_upsampling = convex_upsampling
    self.dropout = dropout
    self.max_rec_iters = max_rec_iters

    if self.small:
      self.hidden_dim = 96
      self.context_dim = 64
      self.corr_levels = 4 if corr_levels is None else corr_levels
      self.corr_radius = 3 if corr_radius is None else corr_radius
    else:
      self.hidden_dim = 128
      self.context_dim = 128
      self.corr_levels = 4 if corr_levels is None else corr_levels
      self.corr_radius = 4 if corr_radius is None else corr_radius

    if small and convex_upsampling:
      raise ValueError('Convex upsampling is not implemented for the small '
                       'setting of raft.')

class RAFT(keras.Model):
    def __init__(self, params, *args, **kwargs):
        super(RAFT, self).__init__(*args, **kwargs)

        if isinstance(params, Namespace):
            params = Box(vars(params))

        self.params = params
        self.iters = params.iters

        if params.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            self.params.corr_levels = 4
            self.params.corr_radius = 3

        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            self.params.corr_levels = 4
            self.params.corr_radius = 4

        self.params.corr_max_disp = True

        self.dropout = params.dropout


        if params.alternate_corr:
            raise NotImplementedError

            # feature network, context network, and update block
        if params.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=self.dropout)
            self.cnet = SmallEncoder(output_dim=hdim + cdim, norm_fn='none', dropout=self.dropout)
            self.update_block = SmallUpdateBlock(args=RAFTArgs(), hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=self.dropout)
            self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn='batch', dropout=self.dropout)
            self.update_block = BasicUpdateBlock(args=RAFTArgs(), hidden_dim=hdim)

        self.upsample = UpsampleConvexWithMask(scale=8)
        self.corr_fn = CorrBlock(num_levels=self.params.corr_levels, radius=self.params.corr_radius,
                                 is_max_disp=self.params.corr_max_disp)
        self.add_float32 = keras.layers.Add(dtype=tf.float32)

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, H, W, C = tf.unstack(tf.shape(img))
        r = lambda x: tf.math.ceil(tf.cast(x, tf.float32) / float(2))
        h = r(r(r(H)))
        w = r(r(r(W)))
        coords0 = tf.cast(coords_grid(N, h, w), img.dtype)
        coords1 = tf.cast(coords_grid(N, h, w), img.dtype)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def call_dict(self, inputs, training=None, mask=None):
        outputs = self.call(inputs, training=training, mask=mask)
        flow_predictions = outputs
        flow_lows = self.flow_lows
        res = {'flow_predictions': flow_predictions,
               'flow_lows': flow_lows}
        return res

    def call(self, inputs, training=None, mask=None):
        flow_init = None
        if len(inputs) == 2:
            image1, image2 = inputs
        elif len(inputs) == 3:
            image1, image2, flow_init = inputs
        else:
            raise ValueError
        """ Estimate optical flow between pair of frames """

        orig_size = tf.shape(image1)[1:3]
        proc_size = get_proc_size(orig_size, 8)
        #image1 = resize(image1, proc_size)
        #image2 = resize(image2, proc_size)



        image1 = 2 * (image1) - 1.0
        image2 = 2 * (image2) - 1.0

        image1 = image1
        image2 = image2

        hdim = self.hidden_dim
        cdim = self.context_dim

        fmap12 = self.fnet(tf.concat((image1, image2), axis=0), training=training)
        fmap1, fmap2 = tf.split(fmap12, num_or_size_splits=2, axis=0)

        corr_pyramid = calc_all_field(fmap1, fmap2, self.corr_fn.num_levels - 1)

        cnet = self.cnet(image1, training=training)
        net, inp = tf.split(cnet, [hdim, cdim], axis=3)
        net = tf.tanh(net)
        inp = tf.nn.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            flow_init = resize_flow(flow_init, tf.shape(coords1)[1:3], scaling=True)
            coords1 = coords1 + flow_init

        flow_predictions = []
        flow_lows = []
        for itr in range(self.iters):
            coords1 = tf.stop_gradient(coords1)
            flow = coords1 - coords0

            corr = self.corr_fn(corr_pyramid, coords1)

            net, up_mask, delta_flow = self.update_block([net, inp, corr, flow], training=training)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = self.add_float32((coords1, delta_flow))
            flow_lows.append(coords1 - coords0)

            # upsample predictions
            if up_mask is None:
                raise NotImplementedError
            else:
                flow_up = self.upsample([coords1 - coords0, up_mask, image1]) * 8.0
                #flow_up = resize_flow(flow_up, orig_size, scaling=True)
            flow_predictions.append(flow_up)

        self.flow_lows = flow_lows

        return flow_predictions

    @staticmethod
    def get_argparse():
        parser = ArgumentParser(add_help=False)
        parser.add_argument("--iters", type=int, default=12, help="Number of refinement iterations")
        parser.add_argument("--small", action="store_true", help="Whether to use the small model")
        parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate")
        parser.add_argument("--alternate_corr", action="store_true", help="Whether to use the alternative correlation")
        return parser

    def freeze_bn(self):
        layers = self.layers


def coords_grid(batch, ht, wd):
    x, y = tf.meshgrid(tf.range(wd), tf.range(ht))
    g = tf.stack([x, y], axis=-1)[tf.newaxis]
    g = tf.cast(g, tf.float32)
    g = tf.tile(g, [batch, 1,1,1])
    return g


def get_proc_size(size, multiple=8):
    im_size = size
    im_size_m = tf.math.ceil(tf.cast(im_size, tf.float32) / multiple) * multiple
    im_size_m = tf.cast(im_size_m, tf.int32)
    return im_size_m


def resize(im, size):
    cur_size = tf.shape(im)[1:3]
    res = tf.cond(tf.logical_and(cur_size[0] == size[0], cur_size[1] == size[1]),
                  lambda: im,
                  lambda: tf.image.resize(im, size))
    return res

def resize_flow(flow, size, scaling=True):
    flow_size = tf.cast(tf.shape(flow)[-3:-1], tf.float32)

    flow = resize(flow, size)
    if scaling:
        scale = tf.cast(size, tf.float32) / flow_size
        scale = tf.stack([scale[1], scale[0]])
        scale = tf.reshape(scale, [1,1,1,2])
        flow = tf.multiply(flow, scale)
    return flow

