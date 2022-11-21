import os
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from argparse import ArgumentParser
from cargbox import CargBox

from raft.baseline import Baseline
from raft.semi import Semisupervised

from util.train import freeze_bn, DefaultStrategy
from util.validate import pad_inputs, unpad_inputs

from raft.loss import FlowLossL1, FlowLossRobust

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from raft_utils import frame_utils
from util.visualize import visualize_flow


import cv2

from glob import glob

if __name__ == "__main__":
    main_parser = ArgumentParser()
    def main_parser_def(main_parser):
        main_parser.add_argument("ckpt_path", type=str, help="ckpt containing weights")
        main_parser.add_argument("--source_dirs", type=str, nargs='+', help="source directories with image files (jpgs and pngs)")
        main_parser.add_argument("--target_dirs", type=str, nargs='+', help="target directories")
        main_parser.add_argument("--model_type", type=str, default="raft-baseline", help="type of model to use")

        main_parser.add_argument('--gpus', '-g', type=int, nargs='+', help='gpus to use')
        main_parser.add_argument('--eval_iters', default=12, type=int, help='gpus to use')
        main_parser.add_argument('--run_eagerly', '-e', action="store_true", help='run eagerly')


    main_parser_def(main_parser)
    args, _ = main_parser.parse_known_args()

    if args.model_type == "raft-baseline":
        model_fn = Baseline
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

    if args.run_eagerly:
       tf.config.run_functions_eagerly(True)

    with strategy.scope():
        raft = model_fn(cargbox.args)

    ckpt = tf.train.Checkpoint(model=raft)
    ckpt_man = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=None)

    initial_epoch = 0
    # restore from chackpoint
    if ckpt_man.latest_checkpoint is not None or not os.path.isdir(args.ckpt_path):
        optimizer = tfa.optimizers.AdamW(
            weight_decay=1.,
            learning_rate=1.,
            epsilon=1e-8,
            clipnorm=1.0
        )
        raft.compile(run_eagerly=args.run_eagerly)
        if not os.path.isdir(args.ckpt_path):
            status = ckpt.restore(args.ckpt_path, )
        else:
            status = ckpt.restore(ckpt_man.latest_checkpoint, )

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

    for source_dir, target_dir in zip(args.source_dirs, args.target_dirs):
        print(dir)
        files = glob(os.path.join(source_dir, "*.jpg"))
        files += glob(os.path.join(source_dir, "*.png"))

        os.makedirs(target_dir, exist_ok=True)
        flo_path = os.path.join(target_dir, 'flo')
        vis_path = os.path.join(target_dir, 'vis')
        os.makedirs(flo_path, exist_ok=True)
        os.makedirs(vis_path, exist_ok=True)

        files = sorted(files)
        print(files)

        for i in range(len(files) - 1):
            im1_path = files[i]
            im2_path = files[i+1]

            im1 = tf.image.decode_image(tf.io.read_file(im1_path), dtype=tf.float32)
            im2 = tf.image.decode_image(tf.io.read_file(im2_path), dtype=tf.float32)

            inputs = [im1[None], im2[None]]

            x_padded, pad = pad_inputs(*inputs, mode='sintel')
            flow_predictions = raft(x_padded, training=False)
            if not isinstance(flow_predictions[0], tf.Tensor):
                flow_preds = flow_predictions[0]
            flow_preds = unpad_inputs(*flow_preds, pad=pad)

            flow_pred = flow_preds[-1][0]

            target_flo = os.path.join(flo_path, os.path.split(im1_path)[-1] + "_flow.flo")
            target_vis = os.path.join(vis_path, os.path.split(im1_path)[-1] + "_flow.png")

            vis_flow = visualize_flow(flow_pred).numpy()
            frame_utils.writeFlow(target_flo, flow_pred)
            cv2.imwrite(target_vis, np.uint8(vis_flow[...,[2,1,0]]*255.))

            print(target_flo)

