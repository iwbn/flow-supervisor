import tensorflow as tf
import tensorflow_addons as tfa

tf.random.set_seed(1234)
from cargbox import CargBox
from box import Box
from raft.baseline import Baseline
from raft.unsup import Unsupervised
from raft.semi import Semisupervised
from argparse import ArgumentParser
import os
from raft.loss import FlowLossL1, FlowLossRobust
from raft.metric import EPE
from util.learning_rate import OneCycleLearningRate, ExponentialLearningRateSmurf
from util.callback import CheckpointManagerCallback
from util.validate import ValidateOnChairs, ValidateOnSintel, ValidateOnKITTI
from util.train import freeze_bn, DefaultStrategy
from math import ceil
from ckpt_cleaner import get_clean_ckpt_path

if __name__ == "__main__":
    main_parser = ArgumentParser()
    def main_parser_def(main_parser):
        main_parser.add_argument("ckpt_path", type=str, help="log and ckpts are saved")
        main_parser.add_argument("--pretrained_ckpt", type=str, help="weight initialization from")
        main_parser.add_argument("--arg_path", type=str, help="net_arguments_are_parsed")
        main_parser.add_argument("--max_step", type=int, default=100000, help="maximum step to train")
        main_parser.add_argument("--val_step", type=int, default=5000, help="validation every n step")
        main_parser.add_argument("--model_type", type=str, default="raft-baseline", help="type of model to use")

        main_parser.add_argument("--learning_rate", "-l", type=float, default=1e-4, help="learning_rate")
        main_parser.add_argument("--lr_schedule", type=str, default="one_cycle", help="learning_rate schedule (one_cycle|exponential|none)")
        main_parser.add_argument("--lr_decay_steps", type=int, default=50000, help="learning rate decay steps")
        main_parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")

        main_parser.add_argument('--gpus', '-g', type=int, nargs='+', help='gpus to use')
        main_parser.add_argument('--run_eagerly', '-e', action="store_true", help='run eagerly')
        main_parser.add_argument('--skip_validation_at_start', action="store_true", help='skip validation at start')
        main_parser.add_argument("--batch_size", "-b", type=int, default=4, help="maximum step to train")
        main_parser.add_argument("--image_size", type=int, nargs='+', help="Input image size")
        main_parser.add_argument("--sup_image_size", type=int, nargs='+', help="Input image size")
        main_parser.add_argument("--unsup_image_size", type=int, nargs='+', help="Input image size")
        main_parser.add_argument("--stage", type=str, default="chairs", help="Training stage")
        main_parser.add_argument("--main_loss", type=str, default="default", help="default|l1|robust")

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

    if args.arg_path:
        cargbox = CargBox(save_path=args.arg_path, argparse=opt_parser, main_parser=main_parser)
        cargbox.maybe_restore(update=True)
        cargbox._config['save_path'] = args.ckpt_path
        cargbox.save_to_yaml(save_main_parser=True)
    else:
        cargbox = CargBox(save_path=args.ckpt_path, argparse=opt_parser, main_parser=main_parser)
        cargbox.save_to_yaml(save_main_parser=True)


    with strategy.scope():
        raft = model_fn(cargbox.args)

    from data.flow_datasets import fetch_dataloader, make_semi_dataset

    """
    DATASET
    """
    if args.stage.startswith("semi-"):
        unsup_stage = args.stage.split("-")[1]
        sup_stage = args.stage.split("-")[2]

        data_args = Box({"stage": unsup_stage, "image_size": tuple(args.unsup_image_size)})
        unsup_dataset = fetch_dataloader(data_args)

        data_args = Box({"stage": sup_stage, "image_size": tuple(args.sup_image_size)})
        sup_dataset = fetch_dataloader(data_args)

        trainset = make_semi_dataset(unsup_dataset=unsup_dataset, sup_dataset=sup_dataset)

        batch_size = args.batch_size
        if isinstance(strategy, tf.distribute.Strategy):
            if batch_size % strategy.num_replicas_in_sync != 0:
                raise ValueError("Batch_size must be divisible by the number of GPUs")

        trainset = trainset.batch(batch_size).prefetch(10)

    else:
        data_args = Box({"stage": args.stage, "image_size": tuple(args.image_size)})
        dataset = fetch_dataloader(data_args)
        trainset = dataset

        batch_size = args.batch_size
        if isinstance(strategy, tf.distribute.Strategy) :
            if batch_size % strategy.num_replicas_in_sync != 0:
                raise ValueError("Batch_size must be divisible by the number of GPUs")

        trainset = trainset.batch(batch_size).repeat(-1).prefetch(10)

    """
    Configs
    """
    steps_per_epoch = args.val_step
    max_epochs = int(ceil(args.max_step / steps_per_epoch))
    max_steps = steps_per_epoch * max_epochs

    if args.lr_schedule == "one_cycle":
        learning_rate = OneCycleLearningRate(args.learning_rate,
                                             max_steps + 100,
                                             pct_start=0.05,
                                             anneal_strategy='linear')

        weight_decay_rate = OneCycleLearningRate(args.weight_decay * args.learning_rate,
                                       max_steps + 100,
                                       pct_start=0.05,
                                       anneal_strategy='linear')
    elif args.lr_schedule == "exponential":
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate,
                                                                       args.lr_decay_steps,
                                                                       decay_rate=0.5,
                                                                       staircase=True)

        weight_decay_rate = tf.keras.optimizers.schedules.ExponentialDecay(args.weight_decay * args.learning_rate,
                                                                           args.lr_decay_steps,
                                                                           decay_rate=0.5,
                                                                           staircase=True)
    elif args.lr_schedule == "smurf":
        learning_rate = ExponentialLearningRateSmurf(max_lr=args.learning_rate,
                                                     min_lr=args.learning_rate/1000.,
                                                     total_steps=max_steps,
                                                     const_portion=0.8)

        weight_decay_rate = ExponentialLearningRateSmurf(max_lr=args.learning_rate,
                                                     min_lr=args.learning_rate/1000.,
                                                     total_steps=max_steps,
                                                     const_portion=0.8)
    else:
        learning_rate = args.learning_rate
        weight_decay_rate = args.weight_decay * args.learning_rate

    optimizer = tfa.optimizers.AdamW(
        weight_decay=weight_decay_rate,
        learning_rate=learning_rate,
        epsilon=1e-8,
        clipnorm=1.0
    )

    ckpt = tf.train.Checkpoint(model=raft)
    ckpt_man = tf.train.CheckpointManager(ckpt, args.ckpt_path, max_to_keep=None)

    if args.stage != "chairs" and args.stage != "chairs_unsup" and args.stage != "autoflow":
        freeze_bn(raft)

    if args.main_loss == "l1" or (args.stage.find("semi") == -1 and args.stage.find("unsup") != -1):
        loss = FlowLossL1()
    else:
        loss = FlowLossRobust()

    initial_epoch = 0
    # restore from chackpoint
    if ckpt_man.latest_checkpoint is not None:
        with strategy.scope():
            raft.compile(loss=loss, optimizer=optimizer, metrics=[EPE()], run_eagerly=args.run_eagerly)
            ckpt_man.restore_or_initialize()
        initial_epoch = raft.optimizer.iterations.numpy() // steps_per_epoch

    # initialize from pretrained_model
    elif args.pretrained_ckpt:
        ckpt_tmp = tf.train.Checkpoint(model=raft)
        status = ckpt_tmp.restore(get_clean_ckpt_path(args.pretrained_ckpt))
        with strategy.scope():
            raft.compile(loss=loss, optimizer=optimizer, metrics=[EPE()], run_eagerly=args.run_eagerly)
            try:
                raft.initialize_teacher_net()
            except AttributeError:
                pass
    # initialize from scratch
    else:
        with strategy.scope():
            raft.compile(loss=loss, optimizer=optimizer, metrics=[EPE()], run_eagerly=args.run_eagerly)

    # define callbacks
    callback_box = Box()
    callback_box.tensorboard = tf.keras.callbacks.TensorBoard(log_dir=args.ckpt_path, update_freq=10)
    callback_box.checkpoint = CheckpointManagerCallback(ckpt_man)
    callback_box.epe2 = ValidateOnSintel(callback_box.tensorboard, skip_on_start=args.skip_validation_at_start)
    callback_box.epe3 = ValidateOnKITTI(callback_box.tensorboard, skip_on_start=args.skip_validation_at_start)
    callbacks = [v for k,v in callback_box.items()]

    # train
    raft.fit(x=trainset, epochs=max_epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks,
             initial_epoch=initial_epoch)
