import tensorflow as tf
from tensorflow import keras
import raft.smurf_models.smurf_utils as smurf_utils
from raft.smurf_models.smurf_utils import unsupervised_loss, compute_occlusions, unsupervised_sequence_loss
import functools
import util.image


class UnsupervisedLoss(keras.losses.Loss):
    def __init__(self, census=1.0, smooth1=2.5, smooth2=0., selfsup=0.3,
                 occlusion='wang', **kwargs):
        super(UnsupervisedLoss, self).__init__(**kwargs, reduction=tf.keras.losses.Reduction.SUM)
        self.gamma = 0.8
        self.occlusion = occlusion

        lw = {
            'supervision': 0.0,
            'census': census,
            'smooth1': smooth1,
            'smooth2': smooth2,
            'selfsup': selfsup,
        }

        self.loss_weights = {}

        for k, v in lw.items():
            if v > 0.0:
                self.loss_weights[k] = v

    def call(self, y_true, y_pred):

        if not isinstance(y_pred, dict):
            raise ValueError("y_pred only allowed in Dict")
        if not isinstance(y_true, dict):
            raise ValueError("y_true only allowed in Dict")
        print(y_pred['flows_fw'])
        flows_fw = [tf.reverse(flow, axis=[3]) for flow in y_pred['flows_fw']]
        flows_bw = [tf.reverse(flow, axis=[3]) for flow in y_pred['flows_bw']]

        try:
            teacher_flows_fw = [tf.reverse(flow, axis=[3]) for flow in y_pred['teacher_flows_fw']]
            teacher_flows_bw = [tf.reverse(flow, axis=[3]) for flow in y_pred['teacher_flows_bw']]
        except KeyError:
            teacher_flows_fw = [None]
            teacher_flows_bw = [None]

        orig_image1 = y_true['orig_image1']
        orig_image2 = y_true['orig_image2']

        orig_images = tf.stack((orig_image1, orig_image2), axis=1)

        crop_height, crop_width = tf.unstack(tf.shape(y_true['image1'])[1:3])
        crop_h, crop_w = y_true['crop_y'], y_true['crop_x']


        image1 = util.image.crop_bboxes(orig_image1, tf.stack([crop_h, crop_w], axis=-1),
                                        tf.stack([crop_height, crop_width]))
        image2 = util.image.crop_bboxes(orig_image2, tf.stack([crop_h, crop_w], axis=-1),
                                        tf.stack([crop_height, crop_width]))
        images = tf.stack((image1, image2), axis=1)

        def _selfsup_transform(images, crop_height, crop_width, crop_h, crop_w, is_flow):
            images_t = util.image.crop_bboxes(images, tf.stack([crop_h, crop_w], axis=-1), tf.stack([crop_height, crop_width]))
            target_shape = tf.unstack(tf.shape(images))
            target_shape[1] = crop_height
            target_shape[2] = crop_width
            images_t = tf.reshape(images_t, target_shape)

            return images_t

        occlusion_estimation_fn = functools.partial(compute_occlusions,
                                                    occlusion_estimation=self.occlusion)

        unsupervised_loss_fn = functools.partial(
            unsupervised_loss,
            weights=self.loss_weights,
            occlusion_estimation_fn=occlusion_estimation_fn,
            only_forward=False,
            selfsup_transform_fn=_selfsup_transform,
            smoothness_edge_weighting='exponential',
            smoothness_edge_constant=150.0,
            selfsup_mask='gaussian',)

        unsupervised_sequence_loss_fn = functools.partial(
            unsupervised_sequence_loss,
            unsupervised_loss_fn=unsupervised_loss_fn,
            loss_decay=self.gamma,
            supervision_weight=0.0,
            mode='unsup_per_update',
            crop_h=crop_h,
            crop_w=crop_w,
            pad_h=tf.zeros_like(crop_h),
            pad_w=tf.zeros_like(crop_w),
        )

        flow_sequence = []
        for i, (flow_fw, flow_bw) in enumerate(zip(flows_fw, flows_bw)):
            # Compute the losses.
            flows = {
                (0, 1, 'augmented-student'): [flow_fw],
                (1, 0, 'augmented-student'): [flow_bw],
                (0, 1, 'transformed-student'): [flow_fw],
                (1, 0, 'transformed-student'): [flow_bw],
                (0, 1, 'original-teacher'): [teacher_flows_fw[-1]],
                (1, 0, 'original-teacher'): [teacher_flows_bw[-1]]
            }
            flow_sequence.append(flows)

        loss_dict = unsupervised_sequence_loss_fn(
            images=images,
            flows_sequence=flow_sequence,
            full_size_images=orig_images, )

        loss = tf.constant(0.)
        for k, v in loss_dict.items():
            loss += v

        return loss


def build_selfsup_transformations(num_flow_levels=3,
                                  crop_height=0,
                                  crop_width=0,
                                  resize=True):
    """Apply augmentations to a list of student images."""

    def transform(images, is_flow, crop_height, crop_width, resize):

        height = images.shape[-3]
        width = images.shape[-2]

        op5 = tf.compat.v1.assert_greater(
            height,
            2 * crop_height,
            message='Image height is too small for cropping.')
        op6 = tf.compat.v1.assert_greater(
            width, 2 * crop_width, message='Image width is too small for cropping.')
        with tf.control_dependencies([op5, op6]):
            images = images[:, crop_height:height - crop_height,
                     crop_width:width - crop_width, :]
        if resize:
            images = smurf_utils.resize(images, height, width, is_flow=is_flow)
            images.set_shape((images.shape[0], height, width, images.shape[3]))
        else:
            images.set_shape((images.shape[0], height - 2 * crop_height,
                              width - 2 * crop_width, images.shape[3]))
        return images

    max_divisor = 2 ** (num_flow_levels - 1)
    assert crop_height % max_divisor == 0
    assert crop_width % max_divisor == 0
    # Compute random shifts for different images in a sequence.
    return functools.partial(
        transform,
        crop_height=crop_height,
        crop_width=crop_width,
        resize=resize)
