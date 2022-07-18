from __future__ import print_function, division

import argparse
import os
import cv2
import time
import numpy as np
import tensorflow as tf
tf.random.set_seed(1234)
try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#from torch.utils.data import DataLoader
from core.raft import RAFT
from core.gma_l2l import GMAL2L
import evaluate
import core.datasets as datasets

from box import Box

from torch.utils.tensorboard import SummaryWriter

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000


def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, gamma2=1.0, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    nm_predictions = len(flow_preds)
    n_predictions = len(flow_preds[0:nm_predictions//2])
    m_predictions = len(flow_preds[nm_predictions // 2::])
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    mask = (valid >= 0.5) & (mag < max_flow)
    valid = (valid > 0.5)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        diff = flow_preds[i] - flow_gt
        i_loss = (diff ** 2 + 0.001 ** 2) ** 0.5
        flow_loss += i_weight * (mask[:, None] * i_loss).mean()

    for i in range(m_predictions):
        i_weight = gamma2**(n_predictions - i - 1)
        #i_loss = (flow_preds[n_predictions+i] - flow_gt).abs()
        diff = flow_preds[n_predictions+i] - flow_gt
        i_loss = (diff ** 2 + 0.001 ** 2) ** 0.5
        flow_loss += i_weight * (mask[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[n_predictions-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def sequence_loss_unsup(flow_preds, flow_gt, valid, gamma=0.8, unsup_weight=1.0, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    nm_predictions = len(flow_preds)
    n_predictions = len(flow_preds[0:nm_predictions//2])
    m_predictions = len(flow_preds[nm_predictions // 2::])
    flow_loss = 0.0
    valid = (valid > 0.5)
    # exlude invalid pixels and extremely large diplacements

    flow_pseudo = flow_preds[-1]  #torch.mean(torch.stack(flow_preds), dim=0)
    flow_pseudo = flow_pseudo.detach()

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)

        diff = flow_preds[i] - flow_pseudo
        i_loss = (diff ** 2 + 0.001 ** 2) ** 0.5
        flow_loss += unsup_weight * i_weight * (i_loss).mean()

    epe = torch.sum((flow_preds[n_predictions-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.StepLR(optimizer, args.num_steps // 5, gamma=0.5)

    return optimizer, scheduler
    

class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):
    l2l_model = nn.DataParallel(GMAL2L(args), device_ids=args.gpus)
    model = l2l_model
    print("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)
        model.module.grad_update_block.load_state_dict(model.module.update_block.state_dict())

    model.cuda()
    model.train()

    #if args.stage != 'chairs':
    #    model.module.freeze_bn()

    from wb_data.flow_datasets import fetch_dataloader, make_semi_dataset

    """
    DATASET
    """
    if args.stage.startswith("semi-"):
        unsup_stage = args.stage.split("-")[1]
        sup_stage = args.stage.split("-")[2]

        data_args = Box({"stage": unsup_stage, "image_size": tuple(args.unsup_image_size)})
        unsup_dataset = fetch_dataloader(data_args)

        data_args = Box({"stage": sup_stage, "image_size": tuple(args.image_size)})
        sup_dataset = fetch_dataloader(data_args)

        trainset = make_semi_dataset(unsup_dataset=unsup_dataset, sup_dataset=sup_dataset)

        batch_size = args.batch_size

        trainset = trainset.batch(batch_size).prefetch(10)
    else:
        raise NotImplementedError

    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler)

    add_noise = True

    def _t(img:tf.Tensor):
        np_arr = img.numpy()
        res = torch.from_numpy(np.transpose(np_arr, [0,3,1,2]))
        return res.cuda()


    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(trainset):
            d = data_blob
            x, y = d
            optimizer.zero_grad()
            # image1, image2, flow, valid = [x.cuda() for x in data_blob]
            image1, image2 = _t(x['sup_augmented_img'][0]) * 255.0, _t(x['sup_augmented_img'][1]) * 255.0
            oi1, oi2 = _t(x['sup_original_img'][0]) * 255.0, _t(x['sup_original_img'][1]) * 255.0
            flow, valid = _t(y['sup_flows'][:, 0]), _t(y['sup_valids'][:, 0])
            ox = torch.from_numpy(x['sup_crop_x'][:].numpy()).cuda()
            oy = torch.from_numpy(x['sup_crop_y'][:].numpy()).cuda()

            image1_u, image2_u = _t(x['augmented_img'][0]) * 255.0, _t(x['augmented_img'][1]) * 255.0
            oi1_u, oi2_u = _t(x['original_img'][0]) * 255.0, _t(x['original_img'][1]) * 255.0
            flow_u, valid_u = _t(y['flows'][:, 0]), _t(y['valids'][:, 0])
            ox_u = torch.from_numpy(x['crop_x'][:].numpy()).cuda()
            oy_u = torch.from_numpy(x['crop_y'][:].numpy()).cuda()

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)
                image1_u = (image1_u + stdv * torch.randn(*image1_u.shape).cuda()).clamp(0.0, 255.0)
                image2_u = (image2_u + stdv * torch.randn(*image2_u.shape).cuda()).clamp(0.0, 255.0)

            flow_predictions = l2l_model(image1, image2, oi1, oi2, ox, oy, iters=args.iters*2)
            loss, metrics = sequence_loss(flow_predictions, flow, valid, args.gamma)

            scaler.scale(loss).backward()

            flow_predictions_unsup = l2l_model(image1_u, image2_u, oi1_u, oi2_u, ox_u, oy_u, iters=args.iters * 2)
            unsup_loss, unsup_metrics = sequence_loss_unsup(flow_predictions_unsup, flow_u, valid_u, unsup_weight=args.unsup_lambda)
            scaler.scale(unsup_loss).backward()

            scaler.unscale_(optimizer)                
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, args.name)
                torch.save(model.state_dict(), PATH)

                results = {}
                for val_dataset in args.validation:
                    if val_dataset == 'chairs':
                        results.update(evaluate.validate_chairs(model.module))
                    elif val_dataset == 'sintel':
                        results.update(evaluate.validate_sintel(model.module))
                    elif val_dataset == 'kitti':
                        results.update(evaluate.validate_kitti(model.module))

                logger.write_dict(results)
                
                model.train()
                if args.stage != 'chairs':
                    model.module.freeze_bn()
            
            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--unsup_lambda', type=float, default=1.0)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--unsup_image_size', type=int, nargs='+', default=[384, 512])

    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')

    parser.add_argument('--upsample-learn', action='store_true', default=False,
                        help='If True, use learned upsampling, otherwise, use bilinear upsampling.')

    parser.add_argument('--model_name', default='', help='specify model name')
    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')

    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(args)