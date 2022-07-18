import tensorflow as tf
import tensorflow_addons as tfa

tf.random.set_seed(1234)
from cargbox import CargBox
from box import Box
from raft.baseline import Baseline
from raft.unsup import Unsupervised
from raft.semi import Semisupervised
from argparse import ArgumentParser
from tensorflow.keras import mixed_precision
import os
from raft.loss import FlowLossL1, FlowLossRobust
from raft.metric import EPE
from util.learning_rate import OneCycleLearningRate
from util.callback import CheckpointManagerCallback
from util.validate import ValidateOnChairs, ValidateOnSintel, ValidateOnKITTI
from math import ceil
from data.sintel import Sintel
from data.flyingchairs import FlyingChairs
from data.kitti import KITTI
from util.train import freeze_bn, DefaultStrategy
import numpy as np

if __name__ == "__main__":
    main_parser = ArgumentParser()
    def main_parser_def(main_parser):
        main_parser.add_argument("ckpt_path", type=str, help="log and ckpts are saved")
        main_parser.add_argument("--dataset", type=str, nargs='+', help="datasets to evaluate on")
        main_parser.add_argument("--model_type", type=str, default="raft-baseline", help="type of model to use")

        main_parser.add_argument("--mixed_precision","-m", action="store_true", help="use mixed precision")
        main_parser.add_argument("--warm_start", "-w", action="store_true", help="use warm start (only for sintel)")
        main_parser.add_argument('--gpus', '-g', type=int, nargs='+', help='gpus to use')
        main_parser.add_argument('--eval_iters', type=int, help='gpus to use')
        main_parser.add_argument('--run_eagerly', '-e', action="store_true", help='run eagerly')

    main_parser_def(main_parser)
    args, _ = main_parser.parse_known_args()

    if args.model_type == "raft-baseline":
        model_fn = Baseline
    elif args.model_type == "raft-unsup":
        model_fn = Unsupervised
    elif args.model_type == "raft-semi":
        model_fn = Semisupervised
    else:
        raise ValueError("args.model_type does not support %s" % args.model_type)

    opt_parser = model_fn.get_argparse()
    main_parser = ArgumentParser(parents=[opt_parser])
    main_parser_def(main_parser)
    args = main_parser.parse_args()

    if args.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join([str(i) for i in args.gpus])
        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            for d in physical_devices:
                tf.config.experimental.set_memory_growth(d, True)
        except:
            # Invalid device or cannot modify virtual devices once initialized.
            pass
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""

    if args.gpus and len(args.gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = DefaultStrategy()

    ckpt_dir = args.ckpt_path
    if not os.path.isdir(ckpt_dir):
        ckpt_dir = os.path.split(ckpt_dir)[0]

    cargbox = CargBox(save_path=ckpt_dir, argparse=opt_parser, main_parser=main_parser)
    cargbox.parse_args()
    cargbox.restore_from_yaml(show_diff=False)
    print(cargbox.args)

    if args.mixed_precision:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)

    if args.run_eagerly:
       tf.config.run_functions_eagerly(True)

    with strategy.scope():
        raft = model_fn(cargbox.args)


    """
    DATASET
    """
    datasets = {}
    for dataset_name in args.dataset:
        if dataset_name == "sintel":
            ds = Sintel(augment=False, training=True, shuffle=False, dstype='clean')
            dataset = ds.dataset

            if args.warm_start:
                extra_info = np.array([a[0] + "_" + str(a[1]) for a in ds.extra_info])
                extra_dataset = tf.data.Dataset.from_tensor_slices(extra_info)
                dataset = tf.data.Dataset.zip((dataset, extra_dataset))

            datasets['Sintel Clean'] = dataset.batch(1).prefetch(10)

            ds = Sintel(augment=False, training=True, shuffle=False, dstype='final')
            dataset = ds.dataset

            if args.warm_start:
                extra_info = np.array([a[0] + "_" + str(a[1]) for a in ds.extra_info])
                extra_dataset = tf.data.Dataset.from_tensor_slices(extra_info)
                dataset = tf.data.Dataset.zip((dataset, extra_dataset))

            datasets['Sintel Final'] = dataset.batch(1).prefetch(10)

        elif dataset_name == "chairs":
            dataset = FlyingChairs(augment=False, training=False, shuffle=False).dataset.batch(1).prefetch(10)
            datasets['FlyingChairs'] = dataset
        elif dataset_name == "kitti":
            dataset = KITTI(augment=False, training=True, shuffle=False).dataset.batch(1).prefetch(10)
            datasets['KITTI 2015'] = dataset
        else:
            raise ValueError

    if len(datasets) == 0:
        exit(0)

    ckpt = tf.train.Checkpoint(model=raft)
    ckpt_man = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=None)

    freeze_bn(raft)

    loss = FlowLossRobust()

    initial_epoch = 0
    # restore from chackpoint
    if ckpt_man.latest_checkpoint is not None or not os.path.isdir(args.ckpt_path):
        optimizer = tfa.optimizers.AdamW(
            weight_decay=1.,
            learning_rate=1.,
            epsilon=1e-8,
            clipnorm=1.0
        )
        raft.compile(loss=loss, optimizer=optimizer, run_eagerly=args.run_eagerly)
        if not os.path.isdir(args.ckpt_path):
            status = ckpt.restore(args.ckpt_path, )
        else:
            status = ckpt.restore(ckpt_man.latest_checkpoint,)

        step = raft.optimizer.iterations.numpy()
        print("step %d restored" % step)
    else:
        raise ValueError

    from util.validate import validate_on_dataset
    try:
        raft.use_bw = False
        print("Backward flow disabled for evaluation")
    except AttributeError:
        pass


    for k, v in datasets.items():
        print("Validation on %s" % k)
        raft.iters = 24
        sparse = False
        if args.eval_iters:
            raft.iters = args.eval_iters
            print("set iters %d" % args.eval_iters)
        elif k in ["Sintel Final", "Sintel Clean"]:
            raft.iters = 32
            print("set iters %d" % raft.iters)
        if k in ["KITTI 2015"]:
            sparse = True
        res = validate_on_dataset(raft, v, warm_start=args.warm_start, sparse=sparse)

        print("Step_%d, %s:" % (step, k))
        for name, value in res.items():
            print("%s = %.4f" % (name, value))



