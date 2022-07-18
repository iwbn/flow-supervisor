#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python -u train.py --name semi-raft-sintel-spring \
--stage semi-sintel_unsup_train-sintel_unsup_labeled_train --validation sintel --restore_ckpt checkpoints/raft-sintel.pth \
--gpus 0 --num_steps 50000 --batch_size 1 --lr 0.000005 --unsup_lambda 1.0  \
--image_size 368 768 --unsup_image_size 368 768 --wdecay 0.0

CUDA_VISIBLE_DEVICES=0 python -u train_gma.py --name semi-gma-sintel-spring --gamma 0.85 \
--stage semi-sintel_unsup_train-sintel_unsup_labeled_train --validation sintel --restore_ckpt GMA/checkpoints/gma-sintel.pth \
--gpus 0 --num_steps 50000 --batch_size 1 --lr 0.000001 --unsup_lambda 0.25 \
--image_size 368 768 --unsup_image_size 368 768 --wdecay 0.0


CUDA_VISIBLE_DEVICES=0 python -u train.py --name semi-kitti-kitti \
--stage semi-kitti_unsup-kitti2015_unsup --validation kitti --restore_ckpt checkpoints/raft-kitti.pth \
--gpus 0 --num_steps 50000 --batch_size 1 --lr 0.000005 --unsup_lambda 1.0  \
--image_size 288 960 --unsup_image_size 288 960 --wdecay 0.0


CUDA_VISIBLE_DEVICES=0 python -u train_gma.py --name semi-gma-kitti-kitti --gamma 0.85 \
--stage semi-kitti_unsup-kitti2015_unsup --validation kitti --restore_ckpt GMA/checkpoints/gma-kitti.pth \
--gpus 0 --num_steps 50000 --batch_size 1 --lr 0.000001 --unsup_lambda 0.05  \
--image_size 288 960 --unsup_image_size 288 960 --wdecay 0.0
