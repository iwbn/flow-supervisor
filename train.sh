# base model training

python train.py --stage chairs --iters 12 --image_size 368 496 --max_step 100000 --val_step 5000 --learning_rate 0.0004 \
--weight_decay 0.0001 --gpus 0 1 --batch_size 10 ckpts/raft_baseline/chairs/
python train.py --stage things --pretrained_ckpt ckpts/raft_baseline/chairs/ckpt-100000 --image_size 400 720 --iters 12 \
--max_step 100000 --val_step 5000 --learning_rate 0.000125 --weight_decay 0.0001 --gpus 0 1 --batch_size 6 ckpts/raft_baseline/things/


# semi-sintel
python train.py --stage semi-sintel_unsup_test-things_unsup --pretrained_ckpt ckpts/raft_baseline/things/ckpt-100000 \
--unsup_weight 1.0 --unsup_image_size 368 768 --sup_image_size 400 720 --model_type raft-semi --gpus 0 --iters 12 \
--max_step 100000 --val_step 5000 --learning_rate 0.00001 --lr_schedule exponential --lr_decay_steps 25000 --weight_decay 0.0 \
--batch_size 1 --lfr_weight 1.0 --lfl_weight 1.0 \
--lfr_loss_type robust \
--lfl_loss_decay_rate 1.0 \
ckpts/semi/sintel/

# semi-kitti
python train.py --stage semi-kitti_unsup_test-things_unsup --pretrained_ckpt ckpts/raft_baseline/things/ckpt-100000 \
--unsup_weight 1.0 --unsup_image_size 288 640 --sup_image_size 360 640 --model_type raft-semi --gpus 0 --iters 12 \
--max_step 100000 --val_step 5000 --learning_rate 0.00001 --lr_schedule exponential --lr_decay_steps 25000 --weight_decay 0.0 \
--batch_size 1 --lfr_weight 1.0 --lfl_weight 1.0 \
--teacher_smurf_weight 1.0 --census_weight 1.0 --smooth2_weight 2.0 --smooth1_weight 0.0 --smurf_occlusion brox \
--lfr_loss_type robust \
--lfl_loss_decay_rate 0.8 \
ckpts/semi/kitti2/

python train.py --stage semi-kitti_unsup_test-things_unsup --pretrained_ckpt ckpts/raft_baseline/things/ckpt-100000 \
--unsup_weight 1.0 --unsup_image_size 288 640 --sup_image_size 360 640 --model_type raft-semi --gpus 0 --iters 12 \
--max_step 100000 --val_step 5000 --learning_rate 0.00001 --lr_schedule exponential --lr_decay_steps 25000 --weight_decay 0.0 \
--batch_size 1 --lfr_weight 1.0 --lfl_weight 1.0 \
--teacher_smurf_weight 0.0 --census_weight 1.0 --smooth2_weight 2.0 --smooth1_weight 0.0 --smurf_occlusion wang \
--lfr_loss_type robust \
--lfl_loss_decay_rate 0.8 \
ckpts/semi/kitti/
