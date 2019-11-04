#!/bin/bash

GPU=1
ROOT_DIR=data

# compute different hyperparameter combinations on validation set
VAL_SEQUENCES=('MOT17/train/MOT17-09-DPM'
               'MOT17/train/MOT17-09-FRCNN'
                'MOT17/train/MOT17-09-SDP'
                'MOT17/train/MOT17-10-DPM'
                'MOT17/train/MOT17-10-FRCNN'
                'MOT17/train/MOT17-10-SDP')

########################### Baseline Tracker ###########################

python eval.py --codec=h264 --vector_type=p --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=20 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
python eval.py --codec=h264 --vector_type=p --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
python eval.py --codec=h264 --vector_type=p --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
python eval.py --codec=h264 --vector_type=p --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=8 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
python eval.py --codec=h264 --vector_type=p --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=6 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
python eval.py --codec=h264 --vector_type=p --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
python eval.py --codec=h264 --vector_type=p --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=2 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}

python eval.py --codec=h264 --vector_type=pb --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=20 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
python eval.py --codec=h264 --vector_type=pb --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
python eval.py --codec=h264 --vector_type=pb --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
python eval.py --codec=h264 --vector_type=pb --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=8 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
python eval.py --codec=h264 --vector_type=pb --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=6 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
python eval.py --codec=h264 --vector_type=pb --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
python eval.py --codec=h264 --vector_type=pb --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=2 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}

python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=20 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=8 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=6 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}
python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=2 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]}


########################### Deep Tracker ###########################

# MPEG4, Full dataset, P vectors, old train code
python eval.py --codec=mpeg4 --tracker_type=deep --tracker_iou_thres=0.1 --detector_interval=4 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-10-23_09-25-34/model_final.pth
python eval.py --codec=mpeg4 --tracker_type=deep --tracker_iou_thres=0.1 --detector_interval=4 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-10-23_09-25-34/model_highest_iou.pth
python eval.py --codec=mpeg4 --tracker_type=deep --tracker_iou_thres=0.1 --detector_interval=4 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-10-23_09-25-34/model_lowest_loss.pth

# MPEG4, Full dataset, P vectors, new train code
python eval.py --codec=mpeg4 --tracker_type=deep --tracker_iou_thres=0.1 --detector_interval=4 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-10-30_02-47-42/model_final.pth
python eval.py --codec=mpeg4 --tracker_type=deep --tracker_iou_thres=0.1 --detector_interval=4 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-10-30_02-47-42/model_highest_iou.pth
python eval.py --codec=mpeg4 --tracker_type=deep --tracker_iou_thres=0.1 --detector_interval=4 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-10-30_02-47-42/model_lowest_loss.pth

# H264, Full dataset, P vectors, new train code
python eval.py --codec=h264 --vector_type=p --tracker_type=deep --tracker_iou_thres=0.1 --detector_interval=4 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-10-29_09-35-25/model_final.pth
python eval.py --codec=h264 --vector_type=p --tracker_type=deep --tracker_iou_thres=0.1 --detector_interval=4 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-10-29_09-35-25/model_highest_iou.pth
python eval.py --codec=h264 --vector_type=p --tracker_type=deep --tracker_iou_thres=0.1 --detector_interval=4 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --gpu=$GPU --deep_tracker_weights_file=models/tracker/2019-10-29_09-35-25/model_lowest_loss.pth


# baseline tracker
# python eval.py MOT17 train mpeg4 baseline 0.1 20 --root_dir=$ROOT_DIR
# python eval.py MOT17 train mpeg4 baseline 0.1 15 --root_dir=$ROOT_DIR
# python eval.py MOT17 train mpeg4 baseline 0.1 10 --root_dir=$ROOT_DIR
# python eval.py MOT17 train mpeg4 baseline 0.1 8 --root_dir=$ROOT_DIR
# python eval.py MOT17 train mpeg4 baseline 0.1 6 --root_dir=$ROOT_DIR
# python eval.py MOT17 train mpeg4 baseline 0.1 5 --root_dir=$ROOT_DIR
# python eval.py MOT17 train mpeg4 baseline 0.1 4 --root_dir=$ROOT_DIR
# python eval.py MOT17 train mpeg4 baseline 0.1 3 --root_dir=$ROOT_DIR
# python eval.py MOT17 train mpeg4 baseline 0.1 2 --root_dir=$ROOT_DIR
# python eval.py MOT17 train mpeg4 baseline 0.1 1 --root_dir=$ROOT_DIR
#
# python eval.py MOT17 train mpeg4 baseline 0.2 20 --root_dir=$ROOT_DIR
# python eval.py MOT17 train mpeg4 baseline 0.2 15 --root_dir=$ROOT_DIR
# python eval.py MOT17 train mpeg4 baseline 0.2 10 --root_dir=$ROOT_DIR
# python eval.py MOT17 train mpeg4 baseline 0.2 8 --root_dir=$ROOT_DIR
# python eval.py MOT17 train mpeg4 baseline 0.2 6 --root_dir=$ROOT_DIR
# python eval.py MOT17 train mpeg4 baseline 0.2 5 --root_dir=$ROOT_DIR
# python eval.py MOT17 train mpeg4 baseline 0.2 4 --root_dir=$ROOT_DIR
# python eval.py MOT17 train mpeg4 baseline 0.2 3 --root_dir=$ROOT_DIR
# python eval.py MOT17 train mpeg4 baseline 0.2 2 --root_dir=$ROOT_DIR
# python eval.py MOT17 train mpeg4 baseline 0.2 1 --root_dir=$ROOT_DIR
#
# python eval.py MOT17 train mpeg4 baseline 0.3 20 --root_dir=$ROOT_DIR
# python eval.py MOT17 train mpeg4 baseline 0.3 15 --root_dir=$ROOT_DIR
# python eval.py MOT17 train mpeg4 baseline 0.3 10 --root_dir=$ROOT_DIR
# python eval.py MOT17 train mpeg4 baseline 0.3 8 --root_dir=$ROOT_DIR
# python eval.py MOT17 train mpeg4 baseline 0.3 6 --root_dir=$ROOT_DIR
# python eval.py MOT17 train mpeg4 baseline 0.3 5 --root_dir=$ROOT_DIR
# python eval.py MOT17 train mpeg4 baseline 0.3 4 --root_dir=$ROOT_DIR
# python eval.py MOT17 train mpeg4 baseline 0.3 3 --root_dir=$ROOT_DIR
# python eval.py MOT17 train mpeg4 baseline 0.3 2 --root_dir=$ROOT_DIR
# python eval.py MOT17 train mpeg4 baseline 0.3 1 --root_dir=$ROOT_DIR




# deep tracker upsampled
# python eval.py MOT17 train mpeg4 deep 0.1 20 --deep_tracker_weights_file=models/tracker/2019-10-23_09-25-34/model_final.pth --root_dir=$ROOT_DIR --gpu=$GPU
# python eval.py MOT17 train mpeg4 deep 0.1 15 --deep_tracker_weights_file=models/tracker/2019-10-23_09-25-34/model_final.pth --root_dir=$ROOT_DIR --gpu=$GPU
# python eval.py MOT17 train mpeg4 deep 0.1 10 --deep_tracker_weights_file=models/tracker/2019-10-23_09-25-34/model_final.pth --root_dir=$ROOT_DIR --gpu=$GPU
# python eval.py MOT17 train mpeg4 deep 0.1 8 --deep_tracker_weights_file=models/tracker/2019-10-23_09-25-34/model_final.pth --root_dir=$ROOT_DIR --gpu=$GPU
# python eval.py MOT17 train mpeg4 deep 0.1 6 --deep_tracker_weights_file=models/tracker/2019-10-23_09-25-34/model_final.pth --root_dir=$ROOT_DIR --gpu=$GPU
# python eval.py MOT17 train mpeg4 deep 0.1 5 --deep_tracker_weights_file=models/tracker/2019-10-23_09-25-34/model_final.pth --root_dir=$ROOT_DIR --gpu=$GPU
# python eval.py MOT17 train mpeg4 deep 0.1 4 --deep_tracker_weights_file=models/tracker/2019-10-23_09-25-34/model_final.pth --root_dir=$ROOT_DIR --gpu=$GPU
# python eval.py MOT17 train mpeg4 deep 0.1 3 --deep_tracker_weights_file=models/tracker/2019-10-23_09-25-34/model_final.pth --root_dir=$ROOT_DIR --gpu=$GPU
# python eval.py MOT17 train mpeg4 deep 0.1 2 --deep_tracker_weights_file=models/tracker/2019-10-23_09-25-34/model_final.pth --root_dir=$ROOT_DIR --gpu=$GPU
# python eval.py MOT17 train mpeg4 deep 0.1 1 --deep_tracker_weights_file=models/tracker/2019-10-23_09-25-34/model_final.pth --root_dir=$ROOT_DIR --gpu=$GPU
