from box import Box
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
from . import get_proc_size, resize, resize_flow
from .baseline import Baseline
from .allfield import calc_all_field, build_pyramid
from .unsup_loss import UnsupervisedLoss
from raft import RAFT


class Unsupervised(Baseline):
    def __init__(self, *args, **kwargs):
        super(Unsupervised, self).__init__(*args, **kwargs)
        self.use_bw = True

    def compile(self, *args, **kwargs):
        super(Unsupervised, self).compile(*args, **kwargs)
        self.unsup_loss = UnsupervisedLoss(census=self.params.census_weight,
                                           smooth1=self.params.smooth1_weight,
                                           smooth2=self.params.smooth2_weight,
                                           selfsup=self.params.selfsup_weight,
                                           occlusion=self.params.smurf_occlusion
                                           )

    def _feature_net(self, inputs, training=None, mask=None):
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
        image1 = resize(image1, proc_size)
        image2 = resize(image2, proc_size)

        image1 = 2 * (image1) - 1.0
        image2 = 2 * (image2) - 1.0

        image1 = image1
        image2 = image2


        images = [image1, image2]

        fmap12 = self.fnet(tf.concat(images, axis=0), training=training)
        fmap1, fmap2 = tf.split(fmap12, num_or_size_splits=2, axis=0)
        fmaps = [fmap1, fmap2]

        corr_pyramids = [calc_all_field(fmap1, fmap2, self.corr_fn.num_levels - 1)]

        return fmaps, corr_pyramids

    def _flow_net(self, inputs, training=None, mask=None):
        image1, corr_pyramid, coords0, coords1  = inputs

        image1 = 2 * (image1) - 1.0

        orig_size = tf.shape(image1)[1:3]
        proc_size = get_proc_size(orig_size, 8)
        image1 = resize(image1, proc_size)

        hdim = self.hidden_dim
        cdim = self.context_dim

        cnet = self.cnet(image1, training=training)
        net, inp = tf.split(cnet, [hdim, cdim], axis=3)
        net = tf.tanh(net)
        inp = tf.nn.relu(inp)

        flow_predictions = []
        for itr in range(self.iters):
            coords1 = tf.stop_gradient(coords1)
            flow = coords1 - coords0

            corr = self.corr_fn(corr_pyramid, coords1)

            net, up_mask, delta_flow = self.update_block([net, inp, corr, flow], training=training)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = self.add_float32((coords1, delta_flow))

            # upsample predictions
            if up_mask is None:
                raise NotImplementedError
            else:
                flow_up = self.upsample([coords1 - coords0, up_mask, image1]) * 8.0
                flow_up = resize_flow(flow_up, orig_size, scaling=True)
            flow_predictions.append(flow_up)

        return flow_predictions

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

        if flow_init is not None:
            flow_init = resize_flow(flow_init, proc_size, scaling=True)

        fmaps, corr_pyramids = self._feature_net(inputs, training=training, mask=mask)

        coords0, coords1 = self.initialize_flow(image1)
        coords1_init = coords1

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = self._flow_net([image1, corr_pyramids[0], coords0, coords1], training=training, mask=mask)

        if self.use_bw:
            c_volume_bw = tf.transpose(corr_pyramids[0][0], [0, 3, 4, 1, 2])
            coor_pyramid_bw = build_pyramid(c_volume_bw, num_pool=self.corr_fn.num_levels - 1)

            flow_predictions_bw = self._flow_net([image2, coor_pyramid_bw, coords0, coords1_init],
                                                 training=training, mask=mask)
            return flow_predictions, flow_predictions_bw
        else:
            return flow_predictions

    def predict_step(self, data):
        res = super(Unsupervised, self).predict_step(data)
        if self.use_bw:
            prediction_fw, _ = res
        else:
            prediction_fw = res
        return prediction_fw

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

        assert(isinstance(x, dict) and isinstance(y, dict))
        image1, image2 = x['augmented_img']
        orig_image1, orig_image2 = x['original_img']
        crop_x = x['crop_x'][:,0]
        crop_y = x['crop_y'][:,0]

        teacher_flow_predictions, teacher_flow_predictions_bw = self.call((orig_image1, orig_image2), training=True)

        with tf.GradientTape() as tape:
            if self.use_bw:
                flow_predictions, flow_predictions_bw = self.call((image1, image2), training=True)
                y_pred = {'flows_fw': flow_predictions,
                          'flows_bw': flow_predictions_bw,
                          'teacher_flows_fw': teacher_flow_predictions,
                          'teacher_flows_bw': teacher_flow_predictions_bw,}
                y_true = {'image1': image1,
                          'image2': image2,
                          'orig_image1': orig_image1,
                          'orig_image2': orig_image2,
                          'crop_x': crop_x,
                          'crop_y': crop_y}

                loss = self.unsup_loss(y_true, y_pred)
            else:
                raise ValueError("Please use use_bw=True")

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)

        self.compiled_metrics.update_state((y['flows'][0], y['valids'][0]), flow_predictions[-1], sample_weight)

        # Collect metrics to return
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return_metrics['loss'] = loss
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

    def initialize_weight_from_baseline(self, raft:RAFT):
        raft.build([[None, ] + [512, 512] + [3]] * 2)
        self.build([[None, ] + [512, 512] + [3]] * 2)
        self.fnet.set_weights(raft.fnet.get_weights())
        self.cnet.set_weights(raft.cnet.get_weights())
        self.update_block.set_weights(raft.update_block.get_weights())
        self.upsample.set_weights(raft.upsample.get_weights())

    @staticmethod
    def get_argparse():
        argparse = Baseline.get_argparse()
        """
        'census': 1.0,
        'smooth1': 2.5,
        'smooth2': 0.0,
        'selfsup': 0.3,
        """
        argparse.add_argument('--unsup_weight', type=float, default=1.0, help="weight to the unsupervised loss")
        argparse.add_argument('--smooth1_weight', type=float, default=2.5, help="weight to the smooth loss")
        argparse.add_argument('--smooth2_weight', type=float, default=0.0, help="weight to the smooth2 loss")
        argparse.add_argument('--census_weight', type=float, default=1.0, help="weight to the census loss")
        argparse.add_argument('--selfsup_weight', type=float, default=0.3, help="weight to the selfsup loss")
        argparse.add_argument('--smurf_occlusion', type=str, default="wang", help="smurf loss type (brox|wang)")
        return argparse