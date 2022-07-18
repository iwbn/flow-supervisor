import tensorflow as tf
from data.flyingchairs import FlyingChairs, FlyingChairsUnsup
from data.flyingthings import FlyingThings, FlyingThingsUnsup
from data.sintel import Sintel, SintelUnsup, SintelUnsupInterval
from data.flow_dataset import make_semi_dataset
from data.kitti import KITTI, KITTI_Multiview, KITTI_MultiviewInterval, KITTIUnsup
from copy import deepcopy


def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding trainign set """

    if args.stage == 'chairs':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        obj = FlyingChairs(augment=True, training=True, shuffle=True, **aug_params)
        train_dataset = obj.dataset

    elif args.stage == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.0, 'max_scale': 0.8, 'do_flip': True}
        obj_final = FlyingThings(augment=True, training=True, dstype='frames_finalpass', shuffle=True, **aug_params)
        obj_clean = FlyingThings(augment=True, training=True, dstype='frames_cleanpass', shuffle=True, **aug_params)
        obj_final.append(obj_clean)
        train_dataset = obj_final.dataset

    elif args.stage == 'things_unsup':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        obj_final = FlyingThingsUnsup(augment=True, training=True, dstype='frames_finalpass', shuffle=True, **aug_params)
        obj_clean = FlyingThingsUnsup(augment=True, training=True, dstype='frames_cleanpass', shuffle=True, **aug_params)
        obj_final.append(obj_clean)
        train_dataset = obj_final.dataset

    elif args.stage == 'sintel_unsup_test':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.6, 'do_flip': True}

        obj_final = SintelUnsup(augment=True, training=False, dstype='final', shuffle=True, **aug_params)
        obj_clean = SintelUnsup(augment=True, training=False, dstype='clean', shuffle=True, **aug_params)
        obj_final2 = SintelUnsupInterval(augment=True, training=False, dstype='final', shuffle=True, **aug_params)
        obj_clean2 = SintelUnsupInterval(augment=True, training=False, dstype='clean', shuffle=True, **aug_params)
        obj_final_bw = SintelUnsup(augment=True, training=False, dstype='final', shuffle=True, **aug_params).backward()
        obj_clean_bw = SintelUnsup(augment=True, training=False, dstype='clean', shuffle=True, **aug_params).backward()
        obj_final2_bw = SintelUnsupInterval(augment=True, training=False, dstype='final', shuffle=True, **aug_params).backward()
        obj_clean2_bw = SintelUnsupInterval(augment=True, training=False, dstype='clean', shuffle=True, **aug_params).backward()
        obj_final.append(obj_clean)
        obj_final.append(obj_final2)
        obj_final.append(obj_clean2)
        obj_final.append(obj_final_bw)
        obj_final.append(obj_clean_bw)
        obj_final.append(obj_final2_bw)
        obj_final.append(obj_clean2_bw)
        train_dataset = obj_final.dataset

    elif args.stage == 'kitti_unsup_test':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        obj = KITTI_Multiview(augment=True, training=False, shuffle=True, **aug_params)
        obj2 = KITTI_MultiviewInterval(augment=True, training=False, shuffle=True, **aug_params)
        obj_bw = KITTI_Multiview(augment=True, training=False, shuffle=True, **aug_params).backward()
        obj2_bw = KITTI_MultiviewInterval(augment=True, training=False, shuffle=True, **aug_params).backward()
        obj.append(obj2)
        obj.append(obj_bw)
        obj.append(obj2_bw)

        train_dataset = obj.dataset

    else:
        raise NotImplementedError

    print('Training with %d image pairs' % train_dataset.cardinality())
    return train_dataset
