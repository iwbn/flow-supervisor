import numpy as np
import tensorflow as tf
from data.flyingchairs import FlyingChairs
from data.sintel import Sintel
from data.kitti import KITTI
from raft.metric import EPE
from util.pad import pad_edge
from util.visualize import visualize_flow
from util.image import forward_interpolate

class ValidateCallback(tf.keras.callbacks.Callback):
    def __init__(self, skip_on_start=False):
        super(ValidateCallback, self).__init__()
        self.skip_on_start = skip_on_start


class ValidateOnChairs(ValidateCallback):
    def __init__(self, tb_callback:tf.keras.callbacks.TensorBoard, *args, **kwargs):
        super(ValidateOnChairs, self).__init__(**kwargs)
        self.batch = 0
        self.tb_callback = tb_callback
        self.dataset_name = "chairs"

    def set_model(self, model):
        self.model = model
        self._train_step = self.model._train_counter

    def on_epoch_begin(self, epoch, logs=None):
        if not self.skip_on_start and epoch == 0:
            obj = FlyingChairs(augment=False, training=False, shuffle=True)
            dataset = obj.dataset
            dataset = dataset.batch(1).prefetch(10)
            print("validating...", end=" ")
            res = validate_on_dataset(self.model, dataset, sparse=False)
            print("done!")
            print("Epoch_%d, %s:" % (epoch, self.dataset_name))
            with self.tb_callback._val_writer.as_default():
                for name, value in res.items():
                    print("%s = %.4f" % (name, value))
                    tf.summary.scalar("epoch_%s_%s" % (self.dataset_name, name) , value, step=epoch)

    def on_epoch_end(self, epoch, logs=None):
        obj = FlyingChairs(augment=False, training=False, shuffle=True)
        dataset = obj.dataset
        dataset = dataset.batch(1).prefetch(10)
        print("validating...", end=" ")
        res = validate_on_dataset(self.model, dataset, sparse=False)
        print("done!")
        print("Epoch_%d, %s:" % (epoch, self.dataset_name))
        with self.tb_callback._val_writer.as_default():
            for name, value in res.items():
                print("%s = %.4f" % (name, value))
                tf.summary.scalar("epoch_%s_%s" % (self.dataset_name, name) , value, step=epoch+1)


class ValidateOnKITTI(ValidateCallback):
    def __init__(self, tb_callback:tf.keras.callbacks.TensorBoard, *args, **kwargs):
        super(ValidateOnKITTI, self).__init__(**kwargs)
        self.batch = 0
        self.tb_callback = tb_callback
        self.dataset_name = "kitti"

    def set_model(self, model):
        self.model = model
        self._train_step = self.model._train_counter

    def on_epoch_begin(self, epoch, logs=None):
        if not self.skip_on_start and epoch == 0:
            print("validating...", end=" ")
            obj = KITTI(augment=False, training=True, shuffle=False)
            dataset = obj.dataset
            dataset = dataset.batch(1).prefetch(10)
            res = validate_on_dataset(self.model, dataset, sparse=True)
            print("done!")
            print("Epoch_%d, %s:" % (epoch, self.dataset_name))
            with self.tb_callback._val_writer.as_default():
                for name, value in res.items():
                    print("%s = %.4f" % (name, value))
                    tf.summary.scalar("epoch_%s_%s" % (self.dataset_name, name) , value, step=epoch)

    def on_epoch_end(self, epoch, logs=None):
        print("validating...", end=" ")
        obj = KITTI(augment=False, training=True, shuffle=False)
        dataset = obj.dataset
        dataset = dataset.batch(1).prefetch(10)
        res = validate_on_dataset(self.model, dataset, sparse=True)
        print("done!")
        print("Epoch_%d, %s:" % (epoch, self.dataset_name))
        with self.tb_callback._val_writer.as_default():
            for name, value in res.items():
                print("%s = %.4f" % (name, value))
                tf.summary.scalar("epoch_%s_%s" % (self.dataset_name, name), value, step=epoch+1)

class ValidateOnSintel(ValidateCallback):
    def __init__(self, tb_callback:tf.keras.callbacks.TensorBoard, *args, **kwargs):
        super(ValidateOnSintel, self).__init__(**kwargs)
        self.batch = 0
        self.tb_callback = tb_callback
        self.dataset_name = "sintel"

    def set_model(self, model):
        self.model = model
        self._train_step = self.model._train_counter

    def on_epoch_begin(self, epoch, logs=None):
        if not self.skip_on_start and epoch == 0:
            print("validating...", end=" ")
            for split in ['final']:
                obj = Sintel(augment=False, training=True, shuffle=False, dstype=split)
                dataset = obj.dataset
                dataset = dataset.batch(1).prefetch(10)
                res = validate_on_dataset(self.model, dataset, sparse=False)
                print("done!")
                print("Epoch_%d, %s_%s:" % (epoch, self.dataset_name, split))
                with self.tb_callback._val_writer.as_default():
                    for name, value in res.items():
                        print("%s = %.4f" % (name, value))
                        tf.summary.scalar("epoch_%s_%s_%s" % (self.dataset_name, split, name) , value, step=epoch)

    def on_epoch_end(self, epoch, logs=None):
        print("validating...", end=" ")
        for split in ['final']:
            obj = Sintel(augment=False, training=True, shuffle=False, dstype=split)
            dataset = obj.dataset
            dataset = dataset.batch(1).prefetch(10)
            res = validate_on_dataset(self.model, dataset, sparse=False)
            print("done!")
            print("Epoch_%d, %s_%s:" % (epoch, self.dataset_name, split))
            with self.tb_callback._val_writer.as_default():
                for name, value in res.items():
                    print("%s = %.4f" % (name, value))
                    tf.summary.scalar("epoch_%s_%s_%s" % (self.dataset_name, split, name), value, step=epoch+1)

def validate_on_dataset(model:tf.keras.Model, dataset:tf.data.Dataset, warm_start=False, sparse=False):
    _get_scene_id = lambda s: "_".join(s.split("_")[0:-1])
    if sparse:
        lists = {}
        model_call = tf.function(model.call)
        prev_s = "none_none_0"
        prev_flow = None
        for d in dataset:
            if warm_start:
                (x, y), s = d
                s = s[0].numpy().decode("utf-8")
                if prev_flow is not None:
                    if _get_scene_id(s) == _get_scene_id(prev_s):
                        x = (x[0], x[1], forward_interpolate(prev_flow[0])[tf.newaxis])
                prev_s = s
            else:
                x, y = d

            x_padded, pad = pad_inputs(*x, mode='kitti')
            flow_predictions = model_call(x_padded, training=False)
            if not isinstance(flow_predictions[0], tf.Tensor):
                flow_preds = flow_predictions[0]
                test_flows = {'student': flow_preds}
                teacher_idx = -1
                if len(flow_predictions) == 4:
                    teacher_idx = 2
                elif len(flow_predictions) == 2:
                    teacher_idx = 1
                if teacher_idx >= 0:
                    t_flow_predictions = flow_predictions[teacher_idx]
                    test_flows['teacher'] = t_flow_predictions
            else:
                test_flows = {'student': flow_predictions}


            for k, flow_preds in test_flows.items():
                epe_list = lists.setdefault(k + "_epe", [])
                epe_1px_list = lists.setdefault(k + "_epe_1px", [])
                epe_3px_list = lists.setdefault(k + "_epe_3px", [])
                epe_5px_list = lists.setdefault(k + "_epe_5px", [])
                out_list = lists.setdefault(k + "_fl", [])

                flow_preds = unpad_inputs(*flow_preds, pad=pad)

                if k == 'student':
                    prev_flow = flow_preds[-1]

                pred = flow_preds[-1]

                # _s("im1", x_padded[0][0])
                # _s("im2", x_padded[1][0])
                # _s("pred", visualize_flow(pred[0]))
                # _s("gt", visualize_flow(y[0][0]))
                # cv2.imshow("mask", y[1][0].numpy())
                # cv2.waitKey(1)

                y_pred = pred
                y_true, m = y

                mag = tf.sqrt(tf.reduce_sum(y_true ** 2, axis=-1, keepdims=True))
                diff = y_pred - tf.cast(y_true, y_pred.dtype)

                sqer = tf.square(diff)
                sqer = tf.reduce_sum(sqer, axis=-1, keepdims=True)
                epes = tf.sqrt(sqer)
                valid = m > 0.5

                valid_epes = epes[valid]

                epe = tf.reduce_mean(valid_epes)[tf.newaxis]
                valid_mags = mag[valid]
                out = tf.logical_and(valid_epes > 3.0, (valid_epes / valid_mags) > 0.05)
                out = tf.cast(out, tf.float32)

                acc_1px = tf.reduce_mean(tf.cast(valid_epes < 1., tf.float32))[tf.newaxis]
                acc_3px = tf.reduce_mean(tf.cast(valid_epes < 3., tf.float32))[tf.newaxis]
                acc_5px = tf.reduce_mean(tf.cast(valid_epes < 5., tf.float32))[tf.newaxis]

                epe_list.append(epe)
                epe_1px_list.append(acc_1px)
                epe_3px_list.append(acc_3px)
                epe_5px_list.append(acc_5px)
                out_list.append(tf.reduce_mean(out))

        returnres = {}
        for k, v in lists.items():
            returnres[k] = tf.reduce_mean(tf.concat(v, axis=0))
        res = returnres
    else:
        if warm_start:
            model_call = tf.function(model.call_dict)
        else:
            model_call = tf.function(model.call)
        lists = {}
        prev_s = "none_none_0"
        prev_flow = None
        for d in dataset:
            if warm_start:
                (x, y), s = d
                s = s[0].numpy().decode("utf-8")
                if prev_flow is not None:
                    if _get_scene_id(s) == _get_scene_id(prev_s):
                        x = (x[0], x[1], forward_interpolate(prev_flow[0])[tf.newaxis])
                prev_s = s

            else:
                x, y = d
            x_padded, pad = pad_inputs(*x, mode='sintel')
            flow_predictions = model_call(x_padded, training=False)
            if isinstance(flow_predictions, dict):
                flow_lows = flow_predictions['flow_lows']
                flow_predictions = flow_predictions['flow_predictions']
                prev_flow = flow_lows[-1]

            if not isinstance(flow_predictions[0], tf.Tensor):
                flow_preds = flow_predictions[0]
                test_flows = {'student': flow_preds}
                teacher_idx = -1
                if len(flow_predictions) == 4:
                    teacher_idx = 2
                elif len(flow_predictions) == 2:
                    teacher_idx = 1
                if teacher_idx >= 0:
                    t_flow_predictions = flow_predictions[teacher_idx]
                    test_flows['teacher'] = t_flow_predictions
            else:
                test_flows = {'student': flow_predictions}

            for k, flow_preds in test_flows.items():
                epe_list = lists.setdefault(k + "_epe", [])
                epe_1px_list = lists.setdefault(k + "_epe_1px", [])
                epe_3px_list = lists.setdefault(k + "_epe_3px", [])
                epe_5px_list = lists.setdefault(k + "_epe_5px", [])

                flow_preds = unpad_inputs(*flow_preds, pad=pad)
                pred = flow_preds[-1]


                y_pred = pred
                y_true = y
                diff = y_pred - tf.cast(y_true, y_pred.dtype)

                sqer = tf.square(diff)
                sqer = tf.reduce_sum(sqer, axis=-1, keepdims=True)
                epes = tf.sqrt(sqer)
                epe = tf.reduce_mean(epes, [1, 2, 3])
                acc_1px = tf.reduce_mean(tf.cast(epes < 1., tf.float32), [1, 2, 3])
                acc_3px = tf.reduce_mean(tf.cast(epes < 3., tf.float32), [1, 2, 3])
                acc_5px = tf.reduce_mean(tf.cast(epes < 5., tf.float32), [1, 2, 3])

                epe_list.append(epe)
                epe_1px_list.append(acc_1px)
                epe_3px_list.append(acc_3px)
                epe_5px_list.append(acc_5px)
        returnres = {}
        for k, v in lists.items():
            returnres[k] = tf.reduce_mean(tf.concat(v, axis=0))
        res = returnres
    return res

def _s(name, value):
    import cv2
    value = value.numpy()
    value = value[...,[2,1,0]]
    cv2.imshow(name, value)


def pad_inputs(*inputs, mode=None):
    ht, wd = tf.unstack(tf.shape(inputs[0])[1:3])
    pad_ht = (((ht // 8) + 1) * 8 - ht) % 8
    pad_wd = (((wd // 8) + 1) * 8 - wd) % 8

    if mode == 'sintel':
        _pad = [[0,0], [pad_ht // 2, pad_ht - pad_ht // 2], [pad_wd // 2, pad_wd - pad_wd // 2], [0,0]]
    else:
        _pad = [[0,0], [0, pad_ht], [pad_wd // 2, pad_wd - pad_wd // 2], [0,0]]

    out = []
    for inp in inputs:
        inp = pad_edge(inp, _pad)
        out.append(inp)
    return out, _pad


def unpad_inputs(*inputs, pad=None):
    ht, wd = tf.unstack(tf.shape(inputs[0])[1:3])
    c = [pad[1][0], ht - pad[1][1], pad[2][0], wd - pad[2][1]]
    out = []
    for inp in inputs:
        inp = inp[:,c[0]:c[1], c[2]:c[3]]
        out.append(inp)
    return out