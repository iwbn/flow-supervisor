import tensorflow as tf
from .flyingchairs import FlyingChairs, FlyingChairsUnsup
from .flyingthings import FlyingThings, FlyingThingsUnsup
from .autoflow import Autoflow, AutoflowUnsup
from .sintel import Sintel, SintelUnsup, SintelUnsupInterval
from .flow_dataset import make_semi_dataset
from .kitti import KITTI, KITTI_Multiview, KITTI_MultiviewInterval, KITTIUnsup
from .vkitti import VKITTI, VKITTIUnsup, VKITTI2, VKITTI2Unsup
from .hd1k import HD1k, HD1kUnsup
from .spring import Spring, SpringUnsup, SpringUnsupInterval
from copy import deepcopy


def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding trainign set """

    if args.stage == 'sintel_unsup_train':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.6, 'do_flip': True}

        obj_final = SintelUnsup(augment=True, training=True, dstype='final', shuffle=True, **aug_params)
        obj_clean = SintelUnsup(augment=True, training=True, dstype='clean', shuffle=True, **aug_params)
        obj_final2 = SintelUnsupInterval(augment=True, training=True, dstype='final', shuffle=True, **aug_params)
        obj_clean2 = SintelUnsupInterval(augment=True, training=True, dstype='clean', shuffle=True, **aug_params)
        obj_final_bw = SintelUnsup(augment=True, training=True, dstype='final', shuffle=True, **aug_params).backward()
        obj_clean_bw = SintelUnsup(augment=True, training=True, dstype='clean', shuffle=True, **aug_params).backward()
        obj_final2_bw = SintelUnsupInterval(augment=True, training=True, dstype='final', shuffle=True, **aug_params).backward()
        obj_clean2_bw = SintelUnsupInterval(augment=True, training=True, dstype='clean', shuffle=True, **aug_params).backward()
        spring = SpringUnsup(augment=True, shuffle=True, **aug_params)
        spring_bw = SpringUnsup(augment=True, shuffle=True, **aug_params).backward()
        spring2 = SpringUnsupInterval(augment=True, shuffle=True, **aug_params)
        spring2_bw = SpringUnsupInterval(augment=True, shuffle=True, **aug_params).backward()
        obj_final.append(obj_clean)
        obj_final.append(obj_final2)
        obj_final.append(obj_clean2)
        obj_final.append(obj_final_bw)
        obj_final.append(obj_clean_bw)
        obj_final.append(obj_final2_bw)
        obj_final.append(obj_clean2_bw)
        obj_final.append(spring)
        obj_final.append(spring_bw)
        obj_final.append(spring2)
        obj_final.append(spring2_bw)

        train_dataset = obj_final.dataset

    elif args.stage == 'sintel_unsup_labeled_train':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.6, 'do_flip': True}

        obj_final = SintelUnsup(augment=True, training=True, dstype='final', shuffle=True, **aug_params)
        obj_clean = SintelUnsup(augment=True, training=True, dstype='clean', shuffle=True, **aug_params)
        obj_final.append(obj_clean)
        train_dataset = obj_final.dataset

    elif args.stage == 'kitti2015_unsup':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        obj = KITTIUnsup(augment=True, training=True, shuffle=True, **aug_params)
        train_dataset = obj.dataset

    elif args.stage == 'kitti_unsup':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        obj = KITTI_Multiview(augment=True, training=True, shuffle=True, **aug_params)
        #obj2 = KITTI_MultiviewInterval(augment=True, training=True, shuffle=True, **aug_params)
        #obj.append(obj2)
        train_dataset = obj.dataset

    elif args.stage == 'hd1k':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        obj = HD1k(augment=True, training=True, shuffle=True, **aug_params)
        train_dataset = obj.dataset
    else:
        raise NotImplementedError

    print('Training with %d image pairs' % train_dataset.cardinality())
    return train_dataset
