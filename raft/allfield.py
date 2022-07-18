import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


def coordinates_within_radius(radius: int, is_max_disp=False):
    k = np.arange(2 * radius + 1) - radius
    k_u = np.tile(k[..., np.newaxis], [1, 2 * radius + 1])
    k_v = np.tile(k[np.newaxis], [2 * radius + 1, 1])
    k = np.stack([k_u, k_v], -1)
    if is_max_disp:
        return k.reshape([-1,2])

    k_ = k.reshape([-1, 2])
    k_ = np.abs(k_[:, 0]) + np.abs(k_[:, 1])
    coors = k.reshape([-1, 2])[k_ <= radius]

    return coors

@tf.function
def forward_lookup(cost_volume:tf.Tensor, init_flow:tf.Tensor, scale=tf.Tensor, radius=4, is_flow:bool=True, is_max_disp:bool=False):
    b, H_orig, W_orig, H, W = tf.unstack(tf.shape(cost_volume))
    #scale = tf.cast(H / H_orig, cost_volume.dtype)
    c_volume_batched = tf.reshape(cost_volume, [-1, H, W, 1])
    x, y = tf.meshgrid(tf.range(W_orig), tf.range(H_orig))
    g = tf.stack([x, y], axis=-1)[tf.newaxis]
    g = tf.cast(g, cost_volume.dtype)

    sample_coords_np = coordinates_within_radius(radius, is_max_disp)
    sample_coords = tf.convert_to_tensor(sample_coords_np, dtype=cost_volume.dtype)

    flow = init_flow

    if is_flow:
        flow_grid = flow + g
    else:
        flow_grid = init_flow

    flow_grid_r = tf.cast(tf.reshape(flow_grid, [-1, 1, 2]), cost_volume.dtype)
    flow_grid_r = tf.tile(flow_grid_r, [1, sample_coords.shape[0], 1])
    flow_grid_r = tf.cast(flow_grid_r / scale, cost_volume.dtype) + tf.reshape(sample_coords, [1, -1, 2])

    mask_x = tf.logical_and(flow_grid_r[:,:,0:1] <= tf.cast(W, flow_grid_r.dtype)-1.,
                            flow_grid_r[:,:,0:1] >= 0.)
    mask_y = tf.logical_and(flow_grid_r[:,:,1:2] <= tf.cast(H, flow_grid_r.dtype)-1.,
                            flow_grid_r[:,:,1:2] >= 0.)
    mask = tf.cast(tf.logical_and(mask_x, mask_y), c_volume_batched.dtype)

    values = tfa.image.interpolate_bilinear(c_volume_batched, flow_grid_r, 'xy') * mask

    new_volume = tf.reshape(values, [b, H_orig, W_orig, -1])
    new_volume = tf.ensure_shape(new_volume,
                                 [cost_volume.shape[0],
                                  cost_volume.shape[1],
                                  cost_volume.shape[2],
                                  len(sample_coords_np)])

    return new_volume


@tf.function
def calc_all_field(a: tf.Tensor, b: tf.Tensor, num_pool: int=0):
    # a: batch x H x W x c
    # b: batch x H x W x c

    # a_expand: batch x H x W x 1 x 1 x c
    # b_expand: batch x 1 x 1 x H x W x c

    batch, H, W, c = tf.unstack(tf.shape(a))
    dtype = a.dtype

    fmap1 = tf.reshape(a, [batch, -1, c])
    fmap2 = tf.reshape(b, [batch, -1, c])

    corr = tf.matmul(fmap1, tf.transpose(fmap2, [0,2,1]))
    c_volume = tf.reshape(corr, [batch, H, W, H, W])

    c_volume /= tf.sqrt(tf.cast(c, dtype))

    b, H, W, _, _ = tf.unstack(tf.shape(c_volume))

    c_volume_batched = tf.reshape(c_volume, [-1, H, W, 1])
    scale=2
    pyramid = [c_volume]
    for i in range(num_pool):
        pooled = tf.nn.avg_pool2d(c_volume_batched, scale, scale, 'SAME')
        _, H_, W_, _ = tf.unstack(tf.shape(pooled))
        pooled = tf.reshape(pooled, [b, H, W, H_, W_])
        pyramid.append(pooled)
        scale *= 2

    return pyramid

def build_pyramid(c_volume: tf.Tensor, num_pool: int=0):
    pyramid = [c_volume]
    b, H, W, _, _ = tf.unstack(tf.shape(c_volume))
    c_volume_batched = tf.reshape(c_volume, [-1, H, W, 1])
    scale = 2
    for i in range(num_pool):
        pooled = tf.nn.avg_pool2d(c_volume_batched, scale, scale, 'SAME')
        _, H_, W_, _ = tf.unstack(tf.shape(pooled))
        pooled = tf.reshape(pooled, [b, H, W, H_, W_])
        pyramid.append(pooled)
        scale *= 2

    return pyramid


@tf.function
def smurf_corr_block(corr_pyramid_inst, coords, radius):
  r = int(radius)
  b, h1, w1, _ = tf.unstack(tf.shape(coords))
  out_pyramid = []
  for i, corr in enumerate(corr_pyramid_inst):
    b, h1, w1, h2, w2 = tf.unstack(tf.shape(corr))
    corr = tf.reshape(corr, (b*h1*w1, h2, w2, 1))
    start = tf.cast(-r, dtype=tf.float32)
    stop = tf.cast(r, dtype=tf.float32)
    num = tf.cast(2 * r + 1, tf.int32)

    dx = tf.linspace(start, stop, num)
    dy = tf.linspace(start, stop, num)
    delta = tf.stack(tf.meshgrid(dy, dx)[::-1], axis=-1)

    centroid_lvl = tf.reshape(coords, (b * h1 * w1, 1, 1, 2)) / 2**i
    delta_lvl = tf.reshape(delta, (1, 2 * r + 1, 2 * r + 1, 2))
    coords_lvl = centroid_lvl + delta_lvl

    corr = tfa.image.resampler(corr, coords_lvl)

    channel_dim = (2 * r + 1) * (2 * r + 1)
    corr = tf.reshape(corr, (b, h1, w1, channel_dim))
    out_pyramid.append(corr)
  out = tf.concat(out_pyramid, axis=-1)
  return out
