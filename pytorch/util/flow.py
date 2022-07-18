import tensorflow as tf
from raft.smurf_models.smurf_utils import compute_occlusions as compute_occlusions_smurf


def compute_occlusions(forward_flow,
                       backward_flow,
                       occlusion_estimation = None,
                       occlusions_are_zeros = True,
                       occ_active = None,
                       boundaries_occluded = True):
    forward_flow = tf.reverse(forward_flow, axis=[3])
    backward_flow = tf.reverse(backward_flow, axis=[3])

    result = compute_occlusions_smurf(forward_flow, backward_flow,
                                     occlusion_estimation,
                                     occlusions_are_zeros,
                                     occ_active,
                                     boundaries_occluded)

    return result