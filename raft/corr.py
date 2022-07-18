import tensorflow as tf
from tensorflow import keras
from .allfield import calc_all_field, forward_lookup, smurf_corr_block

class CorrBlock:
    def __init__(self, num_levels: int=4, radius: int=4, is_max_disp: bool=False):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        self.is_max_disp = is_max_disp

        # all pairs correlation

    def __call__(self, corr_pyramid, coords, is_coord=True):
        r = self.radius
        # res = []
        # for i, c in enumerate(corr_pyramid):
        #     out = forward_lookup(c, coords, 2**c, radius=r, is_flow=not is_coord, is_max_disp=self.is_max_disp)
        #     res.append(out)
        # res = tf.concat(res, axis=-1)
        res = smurf_corr_block(corr_pyramid, coords, radius=r)
        return res



class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        raise NotImplementedError

    def __call__(self, coords):
        raise NotImplementedError