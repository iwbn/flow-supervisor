from box import Box
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
from . import RAFT


class Baseline(RAFT):
    def train_step(self, data):
        image1, image2, init_flow, flow, valid, sample_weight = self.parse_inputs(data)
        y = tf.concat((flow, valid), axis=3)

        gamma = self.params.loss_decay_rate

        with tf.GradientTape() as tape:
            flow_predictions = self.call((image1, image2), training=True)

            losses = []
            for i, y_pred in enumerate(flow_predictions):
                if i == len(flow_predictions) - 1:
                    regularization_loss = self.losses
                else:
                    regularization_loss = None

                i_weight = gamma ** (len(flow_predictions) - i - 1)

                loss = self.compiled_loss(
                    y, y_pred, sample_weight, regularization_losses=regularization_loss) * i_weight

                losses.append(loss)

            loss = tf.add_n(losses)

        # grads = tape.gradient(loss, self.trainable_weights)
        # new_grads = []
        # for g in grads:
        #     new_grads.append(tf.clip_by_norm(g, 1.0))
        #
        # self.optimizer.apply_gradients(zip(new_grads, self.trainable_weights))
        self.optimizer.minimize(loss, self.trainable_weights, tape=tape)

        self.compiled_metrics.update_state(y, flow_predictions[-1], sample_weight)

        # Collect metrics to return
        return_metrics = {}

        return_metrics['bn_sample'] = tf.reduce_mean(self.cnet.norm1.moving_mean)


        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics

    def test_step(self, data):
        image1, image2, init_flow, flow, valid, sample_weight = self.parse_inputs(data)
        y = tf.concat((flow, valid), axis=3)

        flow_predictions = self.call((image1, image2), training=False)

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

    def predict_step(self, data):
        image1, image2, init_flow, _, _, _ = self.parse_inputs(data)
        result = self.call((image1, image2), training=False)[-1]
        return result

    @staticmethod
    def parse_inputs(data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

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

        return image1, image2, init_flow, flow, valid, sample_weight

    @staticmethod
    def get_argparse():
        argparse = RAFT.get_argparse()
        argparse.add_argument("--loss_decay_rate", type=float, default=0.8)
        return argparse