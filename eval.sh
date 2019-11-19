#!/bin/bash

GPU=0
ROOT_DIR=data

# compute different hyperparameter combinations on validation set
VAL_SEQUENCES=('MOT17/train/MOT17-09-DPM'
               'MOT17/train/MOT17-09-FRCNN'
                'MOT17/train/MOT17-09-SDP'
                'MOT17/train/MOT17-10-DPM'
                'MOT17/train/MOT17-10-FRCNN'
                'MOT17/train/MOT17-10-SDP')

########################### Baseline Tracker ###########################

# python eval.py --codec=h264 --vector_type=p --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=20 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=8 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=6 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=2 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
#
# python eval.py --codec=h264 --vector_type=p+b --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=20 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p+b --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p+b --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p+b --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=8 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p+b --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=6 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p+b --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p+b --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=2 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
#
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=20 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=8 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=6 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=2 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}

########################### Deep Tracker (Dense) ###########################

# python eval.py --codec=mpeg4 --tracker_type=deep --mvs_mode=dense --tracker_iou_thres=0.1 --detector_interval=20 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-11-08_04-15-42/model_lowest_loss.pth
# python eval.py --codec=mpeg4 --tracker_type=deep --mvs_mode=dense --tracker_iou_thres=0.1 --detector_interval=15 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-11-08_04-15-42/model_lowest_loss.pth
# python eval.py --codec=mpeg4 --tracker_type=deep --mvs_mode=dense --tracker_iou_thres=0.1 --detector_interval=10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-11-08_04-15-42/model_lowest_loss.pth
# python eval.py --codec=mpeg4 --tracker_type=deep --mvs_mode=dense --tracker_iou_thres=0.1 --detector_interval=8 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-11-08_04-15-42/model_lowest_loss.pth
# python eval.py --codec=mpeg4 --tracker_type=deep --mvs_mode=dense --tracker_iou_thres=0.1 --detector_interval=6 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-11-08_04-15-42/model_lowest_loss.pth
# python eval.py --codec=mpeg4 --tracker_type=deep --mvs_mode=dense --tracker_iou_thres=0.1 --detector_interval=4 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-11-08_04-15-42/model_lowest_loss.pth
# python eval.py --codec=mpeg4 --tracker_type=deep --mvs_mode=dense --tracker_iou_thres=0.1 --detector_interval=2 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-11-08_04-15-42/model_lowest_loss.pth
#
# python eval.py --codec=h264 --tracker_type=deep --mvs_mode=dense --tracker_iou_thres=0.1 --detector_interval=20 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-11-12_05-17-34/model_highest_iou.pth
# python eval.py --codec=h264 --tracker_type=deep --mvs_mode=dense --tracker_iou_thres=0.1 --detector_interval=15 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-11-12_05-17-34/model_highest_iou.pth
# python eval.py --codec=h264 --tracker_type=deep --mvs_mode=dense --tracker_iou_thres=0.1 --detector_interval=10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-11-12_05-17-34/model_highest_iou.pth
# python eval.py --codec=h264 --tracker_type=deep --mvs_mode=dense --tracker_iou_thres=0.1 --detector_interval=8 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-11-12_05-17-34/model_highest_iou.pth
# python eval.py --codec=h264 --tracker_type=deep --mvs_mode=dense --tracker_iou_thres=0.1 --detector_interval=6 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-11-12_05-17-34/model_highest_iou.pth
# python eval.py --codec=h264 --tracker_type=deep --mvs_mode=dense --tracker_iou_thres=0.1 --detector_interval=4 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-11-12_05-17-34/model_highest_iou.pth
# python eval.py --codec=h264 --tracker_type=deep --mvs_mode=dense --tracker_iou_thres=0.1 --detector_interval=2 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-11-12_05-17-34/model_highest_iou.pth

########################### Deep Tracker (Upsampled) ###########################

# python eval.py --codec=mpeg4 --tracker_type=deep --mvs_mode=upsampled --tracker_iou_thres=0.1 --detector_interval=20 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-10-30_02-47-42/model_highest_iou.pth
# python eval.py --codec=mpeg4 --tracker_type=deep --mvs_mode=upsampled --tracker_iou_thres=0.1 --detector_interval=15 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-10-30_02-47-42/model_highest_iou.pth
# python eval.py --codec=mpeg4 --tracker_type=deep --mvs_mode=upsampled --tracker_iou_thres=0.1 --detector_interval=10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-10-30_02-47-42/model_highest_iou.pth
# python eval.py --codec=mpeg4 --tracker_type=deep --mvs_mode=upsampled --tracker_iou_thres=0.1 --detector_interval=8 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-10-30_02-47-42/model_highest_iou.pth
# python eval.py --codec=mpeg4 --tracker_type=deep --mvs_mode=upsampled --tracker_iou_thres=0.1 --detector_interval=6 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-10-30_02-47-42/model_highest_iou.pth
# python eval.py --codec=mpeg4 --tracker_type=deep --mvs_mode=upsampled --tracker_iou_thres=0.1 --detector_interval=4 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-10-30_02-47-42/model_highest_iou.pth
# python eval.py --codec=mpeg4 --tracker_type=deep --mvs_mode=upsampled --tracker_iou_thres=0.1 --detector_interval=2 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-10-30_02-47-42/model_highest_iou.pth
#
# python eval.py --codec=h264 --vector_type=p --tracker_type=deep --mvs_mode=upsampled --tracker_iou_thres=0.1 --detector_interval=20 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-10-29_09-35-25/model_lowest_loss.pth
# python eval.py --codec=h264 --vector_type=p --tracker_type=deep --mvs_mode=upsampled --tracker_iou_thres=0.1 --detector_interval=15 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-10-29_09-35-25/model_lowest_loss.pth
# python eval.py --codec=h264 --vector_type=p --tracker_type=deep --mvs_mode=upsampled --tracker_iou_thres=0.1 --detector_interval=10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-10-29_09-35-25/model_lowest_loss.pth
# python eval.py --codec=h264 --vector_type=p --tracker_type=deep --mvs_mode=upsampled --tracker_iou_thres=0.1 --detector_interval=8 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-10-29_09-35-25/model_lowest_loss.pth
# python eval.py --codec=h264 --vector_type=p --tracker_type=deep --mvs_mode=upsampled --tracker_iou_thres=0.1 --detector_interval=6 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-10-29_09-35-25/model_lowest_loss.pth
# python eval.py --codec=h264 --vector_type=p --tracker_type=deep --mvs_mode=upsampled --tracker_iou_thres=0.1 --detector_interval=4 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-10-29_09-35-25/model_lowest_loss.pth
# python eval.py --codec=h264 --vector_type=p --tracker_type=deep --mvs_mode=upsampled --tracker_iou_thres=0.1 --detector_interval=2 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-10-29_09-35-25/model_lowest_loss.pth

########################### Experiment: Effect of input scale on speed and MOTA ###########################
#
# python eval.py --codec=mpeg4 --scale=0.25 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=20 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=mpeg4 --scale=0.25 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=mpeg4 --scale=0.25 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=mpeg4 --scale=0.25 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=8 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=mpeg4 --scale=0.25 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=6 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=mpeg4 --scale=0.25 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=mpeg4 --scale=0.25 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=2 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
#
# python eval.py --codec=mpeg4 --scale=0.5 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=20 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=mpeg4 --scale=0.5 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=mpeg4 --scale=0.5 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=mpeg4 --scale=0.5 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=8 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=mpeg4 --scale=0.5 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=6 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=mpeg4 --scale=0.5 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=mpeg4 --scale=0.5 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=2 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
#
# python eval.py --codec=mpeg4 --scale=0.75 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=20 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=mpeg4 --scale=0.75 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=mpeg4 --scale=0.75 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=mpeg4 --scale=0.75 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=8 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=mpeg4 --scale=0.75 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=6 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=mpeg4 --scale=0.75 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=mpeg4 --scale=0.75 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=2 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
#
# python eval.py --codec=h264 --vector_type=p+b --scale=0.25 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=20 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p+b --scale=0.25 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p+b --scale=0.25 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p+b --scale=0.25 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=8 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p+b --scale=0.25 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=6 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p+b --scale=0.25 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p+b --scale=0.25 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=2 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
#
# python eval.py --codec=h264 --vector_type=p+b --scale=0.5 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=20 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p+b --scale=0.5 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p+b --scale=0.5 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p+b --scale=0.5 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=8 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p+b --scale=0.5 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=6 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p+b --scale=0.5 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p+b --scale=0.5 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=2 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
#
# python eval.py --codec=h264 --vector_type=p+b --scale=0.75 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=20 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p+b --scale=0.75 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p+b --scale=0.75 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p+b --scale=0.75 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=8 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p+b --scale=0.75 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=6 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p+b --scale=0.75 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p+b --scale=0.75 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=2 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
#
# python eval.py --codec=h264 --vector_type=p --scale=0.25 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=20 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p --scale=0.25 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p --scale=0.25 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p --scale=0.25 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=8 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p --scale=0.25 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=6 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p --scale=0.25 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p --scale=0.25 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=2 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
#
# python eval.py --codec=h264 --vector_type=p --scale=0.5 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=20 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p --scale=0.5 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p --scale=0.5 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p --scale=0.5 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=8 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p --scale=0.5 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=6 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p --scale=0.5 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p --scale=0.5 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=2 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
#
# python eval.py --codec=h264 --vector_type=p --scale=0.75 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=20 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p --scale=0.75 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p --scale=0.75 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p --scale=0.75 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=8 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p --scale=0.75 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=6 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p --scale=0.75 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
# python eval.py --codec=h264 --vector_type=p --scale=0.75 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=2 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}

# Note: Results for scale 1.0 have been computed above already as default scale is 1.0

# python eval.py --codec=h264 --vector_type=p --scale=0.25 --tracker_type=deep --mvs_mode=dense --tracker_iou_thres=0.1 --detector_interval=20 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-11-12_05-17-34/model_highest_iou.pth
# python eval.py --codec=h264 --vector_type=p --scale=0.25 --tracker_type=deep --mvs_mode=dense --tracker_iou_thres=0.1 --detector_interval=15 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-11-12_05-17-34/model_highest_iou.pth
# python eval.py --codec=h264 --vector_type=p --scale=0.25 --tracker_type=deep --mvs_mode=dense --tracker_iou_thres=0.1 --detector_interval=10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-11-12_05-17-34/model_highest_iou.pth
# python eval.py --codec=h264 --vector_type=p --scale=0.25 --tracker_type=deep --mvs_mode=dense --tracker_iou_thres=0.1 --detector_interval=8 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-11-12_05-17-34/model_highest_iou.pth
# python eval.py --codec=h264 --vector_type=p --scale=0.25 --tracker_type=deep --mvs_mode=dense --tracker_iou_thres=0.1 --detector_interval=6 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-11-12_05-17-34/model_highest_iou.pth
# python eval.py --codec=h264 --vector_type=p --scale=0.25 --tracker_type=deep --mvs_mode=dense --tracker_iou_thres=0.1 --detector_interval=4 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-11-12_05-17-34/model_highest_iou.pth
# python eval.py --codec=h264 --vector_type=p --scale=0.25 --tracker_type=deep --mvs_mode=dense --tracker_iou_thres=0.1 --detector_interval=2 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-11-12_05-17-34/model_highest_iou.pth
#
# python eval.py --codec=h264 --vector_type=p --scale=0.5 --tracker_type=deep --mvs_mode=dense --tracker_iou_thres=0.1 --detector_interval=20 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-11-12_05-17-34/model_highest_iou.pth
# python eval.py --codec=h264 --vector_type=p --scale=0.5 --tracker_type=deep --mvs_mode=dense --tracker_iou_thres=0.1 --detector_interval=15 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-11-12_05-17-34/model_highest_iou.pth
# python eval.py --codec=h264 --vector_type=p --scale=0.5 --tracker_type=deep --mvs_mode=dense --tracker_iou_thres=0.1 --detector_interval=10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-11-12_05-17-34/model_highest_iou.pth
# python eval.py --codec=h264 --vector_type=p --scale=0.5 --tracker_type=deep --mvs_mode=dense --tracker_iou_thres=0.1 --detector_interval=8 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-11-12_05-17-34/model_highest_iou.pth
# python eval.py --codec=h264 --vector_type=p --scale=0.5 --tracker_type=deep --mvs_mode=dense --tracker_iou_thres=0.1 --detector_interval=6 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-11-12_05-17-34/model_highest_iou.pth
# python eval.py --codec=h264 --vector_type=p --scale=0.5 --tracker_type=deep --mvs_mode=dense --tracker_iou_thres=0.1 --detector_interval=4 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-11-12_05-17-34/model_highest_iou.pth
# python eval.py --codec=h264 --vector_type=p --scale=0.5 --tracker_type=deep --mvs_mode=dense --tracker_iou_thres=0.1 --detector_interval=2 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-11-12_05-17-34/model_highest_iou.pth
#
# python eval.py --codec=h264 --vector_type=p --scale=0.75 --tracker_type=deep --mvs_mode=dense --tracker_iou_thres=0.1 --detector_interval=20 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-11-12_05-17-34/model_highest_iou.pth
# python eval.py --codec=h264 --vector_type=p --scale=0.75 --tracker_type=deep --mvs_mode=dense --tracker_iou_thres=0.1 --detector_interval=15 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-11-12_05-17-34/model_highest_iou.pth
# python eval.py --codec=h264 --vector_type=p --scale=0.75 --tracker_type=deep --mvs_mode=dense --tracker_iou_thres=0.1 --detector_interval=10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-11-12_05-17-34/model_highest_iou.pth
# python eval.py --codec=h264 --vector_type=p --scale=0.75 --tracker_type=deep --mvs_mode=dense --tracker_iou_thres=0.1 --detector_interval=8 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-11-12_05-17-34/model_highest_iou.pth
# python eval.py --codec=h264 --vector_type=p --scale=0.75 --tracker_type=deep --mvs_mode=dense --tracker_iou_thres=0.1 --detector_interval=6 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-11-12_05-17-34/model_highest_iou.pth
# python eval.py --codec=h264 --vector_type=p --scale=0.75 --tracker_type=deep --mvs_mode=dense --tracker_iou_thres=0.1 --detector_interval=4 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-11-12_05-17-34/model_highest_iou.pth
# python eval.py --codec=h264 --vector_type=p --scale=0.75 --tracker_type=deep --mvs_mode=dense --tracker_iou_thres=0.1 --detector_interval=2 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-11-12_05-17-34/model_highest_iou.pth

########################### Create Jobfile (DO NOt EDIT) ###########################
#python create_jobfile_for_compute_metrics.py
