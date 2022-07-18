from box import Box
import tensorflow as tf
from raft import RAFT
from tensorflow.python.keras.engine import data_adapter
from . import get_proc_size, resize, resize_flow
from .loss import FlowLossL1, FlowLossL2, FlowLossRobust, GradCosSimLoss, GradCrossEntropyLoss, GradL2Loss
from . import UpsampleConvexWithMask
from . import RAFTArgs, BasicEncoder, BasicUpdateBlock
from .unsup import Unsupervised
from .allfield import calc_all_field, build_pyramid
from .unsup_loss import UnsupervisedLoss
from util.image import crop_bboxes, pad_bboxes, central_crop, central_pad, warp_image
from util.flow import compute_occlusions

class Semisupervised(Unsupervised):
    def __init__(self, *args, **kwargs):
        super(Semisupervised, self).__init__(*args, **kwargs)
        hdim = 128
        cdim = 128
        self.use_grad = True

        self.teacher_fnet = self.fnet
        self.teacher_cnet = self.cnet
        self.teacher_update_block = BasicUpdateBlock(args=RAFTArgs(), hidden_dim=hdim)
        self.teacher_upsample = UpsampleConvexWithMask(scale=8)

    def compile(self, *args, **kwargs):
        super(Semisupervised, self).compile(*args, **kwargs)
        if self.params.lfr_loss_type == "l1":
            self.lfr_loss_fn = FlowLossL1(reduction=tf.keras.losses.Reduction.NONE)
        if self.params.lfr_loss_type == "l2":
            self.lfr_loss_fn = FlowLossL2(reduction=tf.keras.losses.Reduction.NONE)
        if self.params.lfr_loss_type == "robust":
            self.lfr_loss_fn = FlowLossRobust(reduction=tf.keras.losses.Reduction.NONE)

        self.teacher_unsup_loss = UnsupervisedLoss(census=self.params.census_weight,
                                           smooth1=self.params.smooth1_weight,
                                           smooth2=self.params.smooth2_weight,
                                           selfsup=0.0,
                                           occlusion=self.params.smurf_occlusion
                                           )

    """
    defines teacher network
    depending on options, parameter sharing is applied differently.
    """
    def _teacher_net(self, inputs, training=None, mask=None, augmented=False):
        image1, corr_pyramid, coords0, coords1, inp, net = inputs

        image1 = 2 * (image1) - 1.0

        cnet = self.cnet(image1, training=training)
        net_temp, inp = tf.split(cnet, [128, 128], axis=3)
        inp = tf.stop_gradient(tf.nn.relu(inp))

        flow_predictions = []
        delta_flows = []
        for itr in range(self.params.teacher_iters):
            coords1 = tf.stop_gradient(coords1)
            flow = coords1 - coords0

            corr = self.corr_fn(corr_pyramid, coords1)

            net, up_mask, delta_flow = self.teacher_update_block([net, inp, corr, flow], training=training)
            delta_flows.append(delta_flow)
            # F(t+1) = F(t) + \Delta(t)
            coords1 = self.add_float32((coords1, delta_flow))

            self.delta_flows = delta_flows
            # upsample predictions
            if up_mask is None:
                raise NotImplementedError
            else:
                flow_up = self.teacher_upsample([coords1 - coords0, up_mask, image1]) * 8.0
            flow_predictions.append(flow_up)

        return flow_predictions

    """
    given correlation pyramid, it computes the optical flows
    """
    def _flow_net(self, inputs, training=None, mask=None):
        image1, corr_pyramid, coords0, coords1  = inputs

        image1 = 2 * (image1) - 1.0

        hdim = self.hidden_dim
        cdim = self.context_dim

        cnet = self.cnet(image1, training=training)
        net, inp = tf.split(cnet, [hdim, cdim], axis=3)
        net = tf.tanh(net)
        inp = tf.nn.relu(inp)

        flow_predictions = []
        flow_lowres_predictions = []
        delta_flows = []
        for itr in range(self.iters):
            coords1 = tf.stop_gradient(coords1)
            flow = coords1 - coords0

            corr = self.corr_fn(corr_pyramid, coords1)

            net, up_mask, delta_flow = self.update_block([net, inp, corr, flow], training=training)
            delta_flows.append(delta_flow)
            # F(t+1) = F(t) + \Delta(t)
            coords1 = self.add_float32((coords1, delta_flow))

            # upsample predictions
            if up_mask is None:
                raise NotImplementedError
            else:
                flow_up = self.upsample([coords1 - coords0, up_mask, image1]) * 8.0
            flow_lowres_predictions.append(coords1 - coords0)
            flow_predictions.append(flow_up)
        self.delta_flows = delta_flows
        return flow_predictions, flow_lowres_predictions, inp, net

    """
    compute features for teacher model
    """
    def _teacher_feature_net(self, inputs, training=None, mask=None):
        flow_init = None
        if len(inputs) == 2:
            image1, image2 = inputs
        elif len(inputs) == 3:
            image1, image2, flow_init = inputs
        else:
            raise ValueError
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1) - 1.0
        image2 = 2 * (image2) - 1.0

        image1 = image1
        image2 = image2


        images = [image1, image2]

        fmap12 = self.teacher_fnet(tf.concat(images, axis=0), training=training)
        fmap1, fmap2 = tf.split(fmap12, num_or_size_splits=2, axis=0)
        fmaps = [fmap1, fmap2]

        corr_pyramids = [calc_all_field(fmap1, fmap2, self.corr_fn.num_levels - 1)]

        return fmaps, corr_pyramids

    """
    main call
    """
    def call(self, inputs, training=None, mask=None):
        flow_init = None
        orig_image1 = None
        offsets = None
        if len(inputs) == 2:
            image1, image2 = inputs
        elif len(inputs) == 3:
            image1, image2, flow_init = inputs
        elif len(inputs) == 4:
            image1, image2, orig_image1, orig_image2 = inputs
            inputs = (image1, image2)
        elif len(inputs) == 5:
            image1, image2, orig_image1, orig_image2, flow_init = inputs
            inputs = (image1, image2, flow_init)
        elif len(inputs) == 6:
            image1, image2, orig_image1, orig_image2, offset_x, offset_y = inputs
            offsets = (offset_x, offset_y)
            inputs = (image1, image2)
        else:
            raise ValueError
        """ Estimate optical flow between pair of frames """

        orig_size = tf.shape(image1)[1:3]
        # for training it is requrired that input size is a multiple of 8
        proc_size = get_proc_size(orig_size, 8)

        image1 = resize(image1, proc_size)
        image2 = resize(image2, proc_size)

        # resizing function
        def _r(flow_list):
            new_list = [resize_flow(f, orig_size, scaling=True) for f in flow_list]
            return new_list

        # extract features for student net
        fmaps, corr_pyramids = self._feature_net(inputs, training=training, mask=mask)

        # initialize forward flows
        coords0, coords1 = self.initialize_flow(image1)
        coords1_init = coords1

        if flow_init is not None:
            flow_init = resize_flow(flow_init, tf.shape(coords1)[1:3], scaling=True)
            coords1 = coords1 + flow_init

        # forward flow predictions
        flow_predictions, flow_lowres, inp, net = self._flow_net([image1, corr_pyramids[0], coords0, coords1], training=training, mask=mask)
        self.student_delta_flows = self.delta_flows[:]
        self.flow_lows = flow_lowres

        # teacher inputs start
        teacher_inputs = (image1, image2, flow_init)
        if orig_image1 is not None:
            teacher_inputs = (orig_image1, orig_image2, flow_init)


        if orig_image1 is not None:
            _, teacher_corr_pyramids = self._feature_net(teacher_inputs, training=training, mask=mask)
            teacher_corr_pyramid = [tf.stop_gradient(v) for v in teacher_corr_pyramids[0]]
        else:
            teacher_corr_pyramid = [tf.stop_gradient(v) for v in corr_pyramids[0]]

        # if full image
        if offsets is not None:
            x0, y0 = offsets
            oh, ow = tf.unstack(tf.shape(orig_image1)[1:3])

            # pad each inputs
            teacher_inp = pad_bboxes(inp, tf.stack((y0//8, x0//8), axis=1), tf.stack((oh//8, ow//8)))
            teacher_net = pad_bboxes(net, tf.stack((y0//8, x0//8), axis=1), tf.stack((oh//8, ow//8)))
            teacher_flow_lowres = pad_bboxes(flow_lowres[-1], tf.stack((y0//8, x0//8), axis=1), tf.stack((oh//8, ow//8)))

            teacher_coords0, _ = self.initialize_flow(orig_image1)
        else:
            teacher_inp = inp
            teacher_net = net
            teacher_coords0 = coords0
            teacher_flow_lowres = flow_lowres[-1]

        teacher_predictions = self._teacher_net([teacher_inputs[0], teacher_corr_pyramid, teacher_coords0,
                                                 tf.stop_gradient(teacher_flow_lowres) + teacher_coords0,
                                                 tf.stop_gradient(teacher_inp), tf.stop_gradient(teacher_net)])

        self.teacher_delta_flows = self.delta_flows[:]
        self.teacher_pred_fullsize = teacher_predictions[:]

        # if full image
        if offsets is not None:
            res = []
            for pred in teacher_predictions:
                res.append(crop_bboxes(pred, tf.stack((y0, x0), axis=1), tf.stack((proc_size[0], proc_size[1]))))
            teacher_predictions = res

            res = []
            for pred in self.teacher_delta_flows:
                res.append(crop_bboxes(pred, tf.stack((y0//8, x0//8), axis=1), tf.stack((proc_size[0]//8, proc_size[1]//8))))
            self.teacher_delta_flows = res

        if self.use_bw:
            c_volume_bw = tf.transpose(corr_pyramids[0][0], [0, 3, 4, 1, 2])
            coor_pyramid_bw = build_pyramid(c_volume_bw, num_pool=self.corr_fn.num_levels - 1)

            flow_predictions_bw, flow_lowres_bw, inp_bw, net_bw = self._flow_net([image2, coor_pyramid_bw, coords0, coords1_init],
                                                 training=training, mask=mask)
            self.student_delta_flows_bw = self.delta_flows[:]

            teacher_c_volume_bw = tf.transpose(teacher_corr_pyramid[0], [0, 3, 4, 1, 2])
            teacher_corr_pyramid_bw = build_pyramid(teacher_c_volume_bw, num_pool=self.corr_fn.num_levels - 1)

            if offsets is not None:
                x0, y0 = offsets
                oh, ow = tf.unstack(tf.shape(orig_image1)[1:3])
                teacher_inp_bw = pad_bboxes(inp_bw, tf.stack((y0 // 8, x0 // 8), axis=1), tf.stack((oh // 8, ow // 8)))
                teacher_net_bw = pad_bboxes(net_bw, tf.stack((y0 // 8, x0 // 8), axis=1), tf.stack((oh // 8, ow // 8)))
                teacher_flow_lowres_bw = pad_bboxes(flow_lowres_bw[-1], tf.stack((y0 // 8, x0 // 8), axis=1),
                                          tf.stack((oh // 8, ow // 8)))
                teacher_coords0, _ = self.initialize_flow(orig_image1)
            else:
                teacher_inp_bw = inp_bw
                teacher_net_bw = net_bw
                teacher_coords0 = coords0
                teacher_flow_lowres_bw = flow_lowres_bw[-1]


            teacher_predictions_bw = self._teacher_net([teacher_inputs[1], teacher_corr_pyramid_bw, teacher_coords0,
                                                     tf.stop_gradient(teacher_flow_lowres_bw) + teacher_coords0,
                                                     tf.stop_gradient(teacher_inp_bw), tf.stop_gradient(teacher_net_bw)])

            self.teacher_delta_flows_bw = self.delta_flows[:]
            self.teacher_pred_fullsize_bw = teacher_predictions_bw[:]

            if offsets is not None:
                res = []
                for pred in teacher_predictions_bw:
                    res.append(crop_bboxes(pred, tf.stack((y0, x0), axis=1), tf.stack((proc_size[0], proc_size[1]))))
                teacher_predictions_bw = res

                res = []
                for pred in self.teacher_delta_flows_bw:
                    res.append(crop_bboxes(pred, tf.stack((y0 // 8, x0 // 8), axis=1),
                                           tf.stack((proc_size[0] // 8, proc_size[1] // 8))))
                self.teacher_delta_flows_bw = res

            if self.use_grad:
                return _r(flow_predictions), _r(flow_predictions_bw), _r(teacher_predictions), _r(teacher_predictions_bw)
            else:
                return _r(flow_predictions), _r(flow_predictions_bw)
        else:
            if self.use_grad:
                return _r(flow_predictions), _r(teacher_predictions)
            else:
                return _r(flow_predictions)

    @staticmethod
    def parse_inputs(data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        if isinstance(x, dict):
            return x, y, sample_weight

        if len(x) == 2:
            image1, image2 = x
            init_flow = tf.zeros_like(image1)[..., 0:2]
            for _ in range(3):
                init_flow = tf.nn.avg_pool2d(init_flow, 2, 2, padding="SAME")
        elif len(x) == 3:
            image1, image2, init_flow = x
        else:
            raise ValueError

        if (isinstance(y, tuple) or isinstance(y, list)) and len(y) == 2:
            flow, valid = y
        elif (isinstance(y, tuple) or isinstance(y, list)) and len(y) == 1:
            flow = y[0]
            valid = tf.ones_like(flow)[..., 0:1]
        elif isinstance(y, tf.Tensor):
            flow = y
            valid = tf.ones_like(flow)[..., 0:1]
        else:
            raise ValueError

        res_x = {
            'augmented_img': (image1, image2),
            'init_flow': init_flow,
        }

        res_y = {
            'valids': (valid),
            'flows': (flow)
        }

        return res_x, res_y, sample_weight

    def train_step(self, data):
        x, y, sample_weight = self.parse_inputs(data)

        # define training log (store scalars)
        log = {}

        # only support dict dataset
        assert(isinstance(x, dict) and isinstance(y, dict))

        """
        SUPERVISED STAGE
        """
        sup_image1, sup_image2 = x['sup_augmented_img']
        sup_orig_image1, sup_orig_image2 = x['sup_original_img']
        sup_gt_flow = y['sup_flows'][:, 0]
        sup_gt_valid = y['sup_valids'][:, 0]
        y_sup = tf.concat((sup_gt_flow, sup_gt_valid), axis=3)
        crop_x = x['sup_crop_x'][:, 0]
        crop_y = x['sup_crop_y'][:, 0]
        h, w = tf.unstack(tf.shape(sup_image1)[1:3])

        gamma = self.params.loss_decay_rate
        if self.params.sup_weight > 0.0:
            with tf.GradientTape(persistent=True) as tape_sup:
                sup_flow_predictions, sup_flow_predictions_bw, teacher_predictions, teacher_predictions_bw = self.call(
                    (sup_image1, sup_image2, sup_orig_image1, sup_orig_image2, crop_x, crop_y), training=True)
                sup_losses = []
                for i, y_pred in enumerate(sup_flow_predictions):
                    if i == len(sup_flow_predictions) - 1:
                        regularization_loss = self.losses
                    else:
                        regularization_loss = None

                    i_weight = gamma ** (len(sup_flow_predictions) - i - 1)

                    loss = self.compiled_loss(
                        y_sup, y_pred, sample_weight, regularization_losses=regularization_loss) * i_weight

                    sup_losses.append(loss * self.params.sup_label_loss_weight)
                log['sup_label_loss'] = tf.add_n(sup_losses)

                if self.params.lfl_weight > 0.0:
                    lfl_losses = []
                    teacher_preds = teacher_predictions
                    teacher_y = y_sup

                    gamma = self.params.lfl_loss_decay_rate
                    for i, y_pred in enumerate(teacher_preds):
                        i_weight = gamma ** (len(teacher_preds) - i - 1)

                        loss = self.compiled_loss(
                            teacher_y, y_pred, sample_weight, regularization_losses=None) * i_weight * self.params.lfl_weight

                        lfl_losses.append(loss)
                    lfl_loss = tf.add_n(lfl_losses)
                    log["lfl_loss"] = lfl_loss
                    sup_losses.append(lfl_loss)

                sup_loss = tf.add_n(sup_losses)
                log['sup_loss'] = sup_loss

            # store supervised gradient which is to be summed with unsupervised grad below
            sup_grad = tape_sup.gradient(sup_loss, self.trainable_weights, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        else:
            sup_grad = None

        """
        UNSUPERVISED STAGE
        """
        image1, image2 = x['augmented_img']
        orig_image1, orig_image2 = x['original_img']
        crop_x = x['crop_x'][:, 0]
        crop_y = x['crop_y'][:, 0]
        h, w = tf.unstack(tf.shape(image1)[1:3])

        unsup_loss = tf.constant(0.0)
        if self.params.unsup_weight > 0.0:
            with tf.GradientTape(persistent=True) as tape_unsup:
                if self.use_bw:
                    flow_predictions, flow_predictions_bw, teacher_predictions, teacher_predictions_bw = self.call(
                        (image1, image2, orig_image1, orig_image2, crop_x, crop_y), training=True)

                    if self.params.teacher_smurf_weight > 0.0:
                        y_true = {'image1': image1,
                                  'image2': image2,
                                  'orig_image1': orig_image1,
                                  'orig_image2': orig_image2,
                                  'crop_x': crop_x,
                                  'crop_y': crop_y}

                        y_pred = {'flows_fw': teacher_predictions,
                                  'flows_bw': teacher_predictions_bw,
                                  }

                        teacher_smurf_loss_unweighted = self.teacher_unsup_loss(y_true, y_pred)

                        teacher_smurf_loss = teacher_smurf_loss_unweighted * self.params.teacher_smurf_weight
                        unsup_loss += teacher_smurf_loss
                        log['teacher_smurf_loss'] = teacher_smurf_loss_unweighted

                    # compute LFR loss
                    if self.params.lfr_weight > 0.0:
                        lfr_losses = []
                        target_flow = teacher_predictions[-1]
                        target_flow_bw = teacher_predictions_bw[-1]

                        gamma = self.params.loss_decay_rate
                        for i, (y_pred, y_pred_bw) in enumerate(zip(flow_predictions, flow_predictions_bw)):
                            i_weight = gamma ** (len(flow_predictions) - i - 1)

                            mask_fw = tf.ones_like(target_flow)[..., 0:1]
                            mask_bw = tf.ones_like(target_flow)[..., 0:1]

                            y_joint = tf.concat((target_flow, mask_fw), axis=3)
                            y_joint = tf.stop_gradient(y_joint)

                            y_joint_bw = tf.concat((target_flow_bw, mask_bw), axis=3)
                            y_joint_bw = tf.stop_gradient(y_joint_bw)

                            loss = self.lfr_loss_fn(y_joint, y_pred) * i_weight
                            loss += self.lfr_loss_fn(y_joint_bw, y_pred_bw) * i_weight
                            lfr_losses.append(loss)
                        lfr_loss = tf.add_n(lfr_losses) * self.params.lfr_weight

                        unsup_loss += lfr_loss
                        log['lfr_loss'] = lfr_loss
                else:
                    # this code only works with backward flow..
                    raise ValueError("Please use use_bw=True")
                log['unsup_loss'] = unsup_loss
            unsup_grad = tape_unsup.gradient(unsup_loss, self.trainable_weights, unconnected_gradients=tf.UnconnectedGradients.NONE)


            grads = []
            for i, w in enumerate(self.trainable_weights):
                if unsup_grad[i] is not None:
                    if self.params.sup_weight == 0.0:
                        g = self.params.unsup_weight * unsup_grad[i]
                    else:
                        g = self.params.sup_weight * sup_grad[i] + self.params.unsup_weight * unsup_grad[i]
                else:
                    if self.params.sup_weight > 0.0:
                        g = self.params.sup_weight * sup_grad[i]
                    else:
                        g = None
                grads.append(g)
            vars = self.trainable_weights
        else:
            grads = sup_grad
            vars = self.trainable_weights

        self.optimizer.apply_gradients(zip(grads, vars))

        if self.params.sup_weight > 0.0:
            self.compiled_metrics.update_state((y['sup_flows'][0], y['sup_valids'][0]), sup_flow_predictions[-1], sample_weight)

        # Collect metrics to return
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return_metrics.update(log)
        return return_metrics

    def test_step(self, data):
        image1, image2, init_flow, flow, valid, sample_weight = self.parse_inputs(data)
        y = tf.concat((flow, valid), axis=3)

        flow_predictions, _ = self.call((image1, image2), training=False)

        gamma = self.params.loss_decay_rate

        for i, y_pred in enumerate(flow_predictions):
            i_weight = gamma ** (len(flow_predictions) - i - 1)

            self.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.losses) * i_weight * len(flow_predictions)

        self.compiled_metrics.update_state(y, flow_predictions[-1], sample_weight)

        # Collect metrics to return
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics

    @staticmethod
    def get_argparse():
        argparse = Unsupervised.get_argparse()
        argparse.add_argument('--sup_weight', type=float, default=1.0, help="supervised grad weight")
        argparse.add_argument('--lfr_weight', type=float, default=1.0, help="[student model] weight for the residual learning loss")
        argparse.add_argument('--lfl_weight', type=float, default=1.0, help="[teacher model] weight for learning from label")
        argparse.add_argument('--sup_label_loss_weight', type=float, default=1.0, help="[student model] weight for learning from label")
        argparse.add_argument('--teacher_smurf_weight', type=float, default=0.0, help="[teacher model] weight to the smurf loss")

        argparse.add_argument('--lfr_loss_type', type=str, default="l2", help="lfr loss type (l1 | l2 | robust)")

        argparse.add_argument('--teacher_iters', type=int, default=12, help="teacher model (separated or not) iteration")
        argparse.add_argument("--lfl_loss_decay_rate", type=float, default=0.8)
        return argparse

    def initialize_teacher_net(self):
        self.build([[None, ] + [512, 512] + [3]] * 2)
        self.teacher_update_block.set_weights(self.update_block.get_weights())
        self.teacher_upsample.set_weights(self.upsample.get_weights())
