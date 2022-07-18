import tensorflow as tf
from argparse import ArgumentParser
from cargbox import CargBox
from raft import RAFT
import subprocess


def get_clean_ckpt_path(pretrained_ckpt):
    try:
        save_path = pretrained_ckpt + "-weights"
        tf.train.load_checkpoint(save_path)
        return save_path
    except tf.errors.NotFoundError:
        return clean_ckpt(pretrained_ckpt)

def clean_ckpt(pretrained_ckpt):
    out = subprocess.run(["python", "ckpt_cleaner.py", "--pretrained_ckpt", pretrained_ckpt], capture_output=True)
    print(out.stdout.decode())
    out_lines = out.stdout.decode().strip().split("\n")
    return out_lines[-1]

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    main_parser = ArgumentParser()
    def main_parser_def(main_parser):
        main_parser.add_argument("--pretrained_ckpt", type=str, help="weight initialization from")
    main_parser_def(main_parser)
    args, _ = main_parser.parse_known_args()

    ckpt_path = "/".join(args.pretrained_ckpt.split("/")[0:-1])
    print(ckpt_path)

    model_fn = RAFT

    opt_parser = model_fn.get_argparse()
    main_parser = ArgumentParser(parents=[opt_parser])

    main_parser_def(main_parser)
    args = main_parser.parse_args()
    print(args)

    cargbox = CargBox(save_path=ckpt_path, argparse=opt_parser, main_parser=main_parser)
    cargbox.maybe_restore(update=True)

    model = RAFT(cargbox.args)
    to_save = RAFT(cargbox.args)

    if args.pretrained_ckpt:
        ckpt_tmp = tf.train.Checkpoint(model=model)
        print("restoring model from %s..." % args.pretrained_ckpt)
        status = ckpt_tmp.restore(args.pretrained_ckpt)
        print("building model...")
        model.build([[None, ] + [512, 512] + [3]] * 2)
        to_save.build([[None, ] + [512, 512] + [3]] * 2)
        to_save.set_weights(model.get_weights())
        save_path = args.pretrained_ckpt + '-weights'
        print("saving...")
        tf.train.Checkpoint(model=to_save).write(save_path)
        print(save_path)