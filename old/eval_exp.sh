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

# test confidence threshold param at two different detector intervals
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=-1 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.1 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.2 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.3 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.4 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.6 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.7 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.8 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.9 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.95 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.99 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
#
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --det_conf_thres=-1 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --det_conf_thres=0.1 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --det_conf_thres=0.2 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --det_conf_thres=0.3 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --det_conf_thres=0.4 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --det_conf_thres=0.5 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --det_conf_thres=0.6 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --det_conf_thres=0.7 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --det_conf_thres=0.8 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --det_conf_thres=0.9 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --det_conf_thres=0.95 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --det_conf_thres=0.99 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1

# test different state change thresholds
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 0 0 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 47.7
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 0 1 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 48.7
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 0 2 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 47.8
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 0 3 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 46.3
#
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 1 0 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 46.6
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 1 1 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 47.7
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 1 2 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 47.1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 1 3 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 45.9
#
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 2 0 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 43.2
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 2 1 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 45.6
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 2 2 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 45.7
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 2 3 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 44.8
#
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 3 0 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 40.1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 3 1 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 42.7
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 3 2 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 43.4
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 3 3 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 42.5

# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --det_conf_thres=0.5 --state_thres 0 0 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 41.7
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --det_conf_thres=0.5 --state_thres 0 1 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 38.8
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --det_conf_thres=0.5 --state_thres 0 2 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 32.1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --det_conf_thres=0.5 --state_thres 0 3 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 25.6
#
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --det_conf_thres=0.5 --state_thres 1 0 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 37.5
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --det_conf_thres=0.5 --state_thres 1 1 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 34.9
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --det_conf_thres=0.5 --state_thres 1 2 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 29.9
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --det_conf_thres=0.5 --state_thres 1 3 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 24.9

# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.1 --state_thres 0 0 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 48.1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.1 --state_thres 0 1 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 47.4
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.1 --state_thres 0 2 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 45.0
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.1 --state_thres 1 0 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 47.4
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.1 --state_thres 1 1 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 47.0
#
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.2 --state_thres 0 0 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 48.2
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.2 --state_thres 0 1 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 48.0
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.2 --state_thres 0 2 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 46.1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.2 --state_thres 1 0 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 47.4
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.2 --state_thres 1 1 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 47.4
#
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.3 --state_thres 0 0 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 48.1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.3 --state_thres 0 1 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 48.1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.3 --state_thres 0 2 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 46.4
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.3 --state_thres 1 0 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 47.2
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.3 --state_thres 1 1 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 47.5
#
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.4 --state_thres 0 0 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 47.8
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.4 --state_thres 0 1 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 48.3
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.4 --state_thres 0 2 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 47.0
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.4 --state_thres 1 0 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 46.8
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.4 --state_thres 1 1 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 47.5

# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.6 --state_thres 0 0 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 47.5
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.6 --state_thres 0 1 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 48.9
#
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.7 --state_thres 0 0 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 47.3
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.7 --state_thres 0 1 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 49.0
#
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.8 --state_thres 0 0 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 46.7
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.8 --state_thres 0 1 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 48.8
#
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.9 --state_thres 0 0 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 45.9
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.9 --state_thres 0 1 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 48.5

# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.65 --state_thres 0 1 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 48.9
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.75 --state_thres 0 1 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 48.9

# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.3 --detector_interval=4 --det_conf_thres=0.6 --state_thres 0 1 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 48.6
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.3 --detector_interval=4 --det_conf_thres=0.7 --state_thres 0 1 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 48.8
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.3 --detector_interval=4 --det_conf_thres=0.8 --state_thres 0 1 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 48.6

# python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 0 0 10 --root_dir=$ROOT_DIR --repeats=1  # 25.9
# python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 0 1 10 --root_dir=$ROOT_DIR --repeats=1  # 28.0
# python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 0 2 10 --root_dir=$ROOT_DIR --repeats=1  # 28.7
# python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 1 0 10 --root_dir=$ROOT_DIR --repeats=1
# python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 1 1 10 --root_dir=$ROOT_DIR --repeats=1
# python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 1 2 10 --root_dir=$ROOT_DIR --repeats=1

# python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 2 0 10 --root_dir=$ROOT_DIR --repeats=1
# python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 2 1 10 --root_dir=$ROOT_DIR --repeats=1
# python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 2 2 10 --root_dir=$ROOT_DIR --repeats=1
# python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 2 3 10 --root_dir=$ROOT_DIR --repeats=1
#
# python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 3 0 10 --root_dir=$ROOT_DIR --repeats=1
# python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 3 1 10 --root_dir=$ROOT_DIR --repeats=1
# python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 3 2 10 --root_dir=$ROOT_DIR --repeats=1
# python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 3 3 10 --root_dir=$ROOT_DIR --repeats=1

# change IoU threshold and see if it is better -> generally 0.3 seems to be slightly better (but very little only)
# python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.3 --detector_interval=4 --det_conf_thres=0.5 --state_thres 0 0 10 --root_dir=$ROOT_DIR --repeats=1  # 25.9
# python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.3 --detector_interval=4 --det_conf_thres=0.5 --state_thres 0 1 10 --root_dir=$ROOT_DIR --repeats=1  # 28.1
# python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.3 --detector_interval=4 --det_conf_thres=0.5 --state_thres 0 2 10 --root_dir=$ROOT_DIR --repeats=1  # 28.7

# repeat for best state thresholds and vary detector confidence threshold
# python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=-1 --state_thres 0 2 10 --root_dir=$ROOT_DIR --repeats=1   # -2.2
# python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.1 --state_thres 0 2 10 --root_dir=$ROOT_DIR --repeats=1  # 28.9
# python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.2 --state_thres 0 2 10 --root_dir=$ROOT_DIR --repeats=1  # 29.6
# python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.3 --state_thres 0 2 10 --root_dir=$ROOT_DIR --repeats=1  # 29.8
# python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.4 --state_thres 0 2 10 --root_dir=$ROOT_DIR --repeats=1  # 29.6
# python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.6 --state_thres 0 2 10 --root_dir=$ROOT_DIR --repeats=1  # 28.7
# python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.7 --state_thres 0 2 10 --root_dir=$ROOT_DIR --repeats=1  # 28.0
# python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.8 --state_thres 0 2 10 --root_dir=$ROOT_DIR --repeats=1  # 27.0
# python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.9 --state_thres 0 2 10 --root_dir=$ROOT_DIR --repeats=1  # 25.8
# python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.95 --state_thres 0 2 10 --root_dir=$ROOT_DIR --repeats=1 # 24.5

# TODO: For MOT16 evaluate which detector interval is best (e.g. [1,2,3,4,5,6,8])
# python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.3 --detector_interval=20 --det_conf_thres=0.3 --state_thres 0 0 10 --root_dir=$ROOT_DIR --repeats=1  # 24.8
# python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.3 --detector_interval=15 --det_conf_thres=0.3 --state_thres 0 0 10 --root_dir=$ROOT_DIR --repeats=1  # 26.0
# python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.3 --detector_interval=10 --det_conf_thres=0.3 --state_thres 0 0 10 --root_dir=$ROOT_DIR --repeats=1  # 27.4
# python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.3 --detector_interval=10 --det_conf_thres=0.3 --state_thres 0 1 10 --root_dir=$ROOT_DIR --repeats=1  # 27.8
# python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.3 --detector_interval=8 --det_conf_thres=0.3 --state_thres 0 1 10 --root_dir=$ROOT_DIR --repeats=1   # 28.8
# python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.3 --detector_interval=8 --det_conf_thres=0.3 --state_thres 0 2 10 --root_dir=$ROOT_DIR --repeats=1   # 27.8
# python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.3 --detector_interval=6 --det_conf_thres=0.3 --state_thres 0 2 10 --root_dir=$ROOT_DIR --repeats=1   # 28.6
# python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.3 --detector_interval=4 --det_conf_thres=0.3 --state_thres 0 2 10 --root_dir=$ROOT_DIR --repeats=1   # 29.7
# python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.3 --detector_interval=3 --det_conf_thres=0.3 --state_thres 0 2 10 --root_dir=$ROOT_DIR --repeats=1   # 29.8
# python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.3 --detector_interval=2 --det_conf_thres=0.3 --state_thres 0 2 10 --root_dir=$ROOT_DIR --repeats=1   # 30.1
# python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.3 --detector_interval=1 --det_conf_thres=0.3 --state_thres 0 2 10 --root_dir=$ROOT_DIR --repeats=1   # 29.5

# final parameter choice (for MOT16 and MOT17):
# --tracker_iou_thres=0.3
# --det_conf_thres=0.3
# --detector_interval=[20, 15, 10, 8, 6, 4, 3, 2, 1]
# --state_thres=[(0 0 10),(0 0 10),(0 1 10),(0 1 10),(0 1 10),(0 2 10),(0 2 10),(0 2 10),(0 2 10)]

# for MOT 17 try out both (0 2 10) and (0 1 10) and use the better one
# python eval.py --benchmark="MOT17" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.3 --detector_interval=20 --det_conf_thres=0.3 --state_thres 0 0 10 --root_dir=$ROOT_DIR --repeats=1  # 38.7
# python eval.py --benchmark="MOT17" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.3 --detector_interval=15 --det_conf_thres=0.3 --state_thres 0 0 10 --root_dir=$ROOT_DIR --repeats=1  # 40.9
# python eval.py --benchmark="MOT17" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.3 --detector_interval=10 --det_conf_thres=0.3 --state_thres 0 1 10 --root_dir=$ROOT_DIR --repeats=1  # 41.3
# python eval.py --benchmark="MOT17" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.3 --detector_interval=8 --det_conf_thres=0.3 --state_thres 0 1 10 --root_dir=$ROOT_DIR --repeats=1   # 43.0
# python eval.py --benchmark="MOT17" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.3 --detector_interval=6 --det_conf_thres=0.3 --state_thres 0 1 10 --root_dir=$ROOT_DIR --repeats=1   # 44.2
# python eval.py --benchmark="MOT17" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.3 --detector_interval=4 --det_conf_thres=0.3 --state_thres 0 2 10 --root_dir=$ROOT_DIR --repeats=1   # 44.3
# python eval.py --benchmark="MOT17" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.3 --detector_interval=3 --det_conf_thres=0.3 --state_thres 0 2 10 --root_dir=$ROOT_DIR --repeats=1   # 44.8
# python eval.py --benchmark="MOT17" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.3 --detector_interval=2 --det_conf_thres=0.3 --state_thres 0 2 10 --root_dir=$ROOT_DIR --repeats=1   # 45.6
# python eval.py --benchmark="MOT17" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.3 --detector_interval=1 --det_conf_thres=0.3 --state_thres 0 2 10 --root_dir=$ROOT_DIR --repeats=1   # 45.4
#
# python eval.py --benchmark="MOT17" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.3 --detector_interval=4 --det_conf_thres=0.3 --state_thres 0 1 10 --root_dir=$ROOT_DIR --repeats=1   # 45.3
# python eval.py --benchmark="MOT17" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.3 --detector_interval=3 --det_conf_thres=0.3 --state_thres 0 1 10 --root_dir=$ROOT_DIR --repeats=1   # 45.4
# python eval.py --benchmark="MOT17" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.3 --detector_interval=2 --det_conf_thres=0.3 --state_thres 0 1 10 --root_dir=$ROOT_DIR --repeats=1   # 45.7
# python eval.py --benchmark="MOT17" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.3 --detector_interval=1 --det_conf_thres=0.3 --state_thres 0 1 10 --root_dir=$ROOT_DIR --repeats=1   # 45.4

# python eval.py --benchmark="MOT17" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.3 --detector_interval=2 --det_conf_thres=0.2 --state_thres 0 1 10 --root_dir=$ROOT_DIR --repeats=1   # 45.7
# python eval.py --benchmark="MOT17" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.3 --detector_interval=2 --det_conf_thres=0.1 --state_thres 0 1 10 --root_dir=$ROOT_DIR --repeats=1   # 45.5
#
# python eval.py --benchmark="MOT17" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.3 --detector_interval=2 --det_conf_thres=0.3 --state_thres 0 3 10 --root_dir=$ROOT_DIR --repeats=1   # 45.1
# python eval.py --benchmark="MOT17" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.3 --detector_interval=2 --det_conf_thres=0.3 --state_thres 1 0 10 --root_dir=$ROOT_DIR --repeats=1   # 44.7
# python eval.py --benchmark="MOT17" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.3 --detector_interval=2 --det_conf_thres=0.3 --state_thres 1 1 10 --root_dir=$ROOT_DIR --repeats=1   # 45.4
# python eval.py --benchmark="MOT17" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.3 --detector_interval=2 --det_conf_thres=0.3 --state_thres 1 2 10 --root_dir=$ROOT_DIR --repeats=1   # 45.5
# python eval.py --benchmark="MOT17" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.3 --detector_interval=2 --det_conf_thres=0.3 --state_thres 1 3 10 --root_dir=$ROOT_DIR --repeats=1   # 45.2
# python eval.py --benchmark="MOT17" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.3 --detector_interval=2 --det_conf_thres=0.3 --state_thres 2 0 10 --root_dir=$ROOT_DIR --repeats=1   # 43.0
# python eval.py --benchmark="MOT17" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.3 --detector_interval=2 --det_conf_thres=0.3 --state_thres 2 1 10 --root_dir=$ROOT_DIR --repeats=1   # 44.5
# python eval.py --benchmark="MOT17" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.3 --detector_interval=2 --det_conf_thres=0.3 --state_thres 2 2 10 --root_dir=$ROOT_DIR --repeats=1   # 44.8
# python eval.py --benchmark="MOT17" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.3 --detector_interval=2 --det_conf_thres=0.3 --state_thres 2 3 10 --root_dir=$ROOT_DIR --repeats=1   # 44.7

# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.7 --state_thres 0 0 10 --detector_interval=20 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 39.2 <-
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.7 --state_thres 0 1 10 --detector_interval=20 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 34.6
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.7 --state_thres 0 2 10 --detector_interval=20 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 27.0
#
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.7 --state_thres 0 0 10 --detector_interval=15 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 41.5 <-
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.7 --state_thres 0 1 10 --detector_interval=15 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 39.1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.7 --state_thres 0 2 10 --detector_interval=15 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 33.6
#
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.7 --state_thres 0 0 10 --detector_interval=10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 45.0 <-
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.7 --state_thres 0 1 10 --detector_interval=10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 44.6
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.7 --state_thres 0 2 10 --detector_interval=10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 41.8
#
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.7 --state_thres 0 0 10 --detector_interval=8 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 46.0
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.7 --state_thres 0 1 10 --detector_interval=8 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 46.5 <-
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.7 --state_thres 0 2 10 --detector_interval=8 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 44.6
#
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.7 --state_thres 0 0 10 --detector_interval=6 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 46.9
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.7 --state_thres 0 1 10 --detector_interval=6 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 48.0 <-
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.7 --state_thres 0 2 10 --detector_interval=6 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 47.0
#
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.7 --state_thres 0 0 10 --detector_interval=4 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 47.3
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.7 --state_thres 0 1 10 --detector_interval=4 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 49.0 <-
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.7 --state_thres 0 2 10 --detector_interval=4 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 48.7
#
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.7 --state_thres 0 0 10 --detector_interval=2 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 47.4
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.7 --state_thres 0 1 10 --detector_interval=2 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 49.1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.7 --state_thres 0 2 10 --detector_interval=2 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1  # 49.8 <-


########################### Hyperparameter Tuning for MOT17 ###########################

# procedure: first find good values for det_conf_thres and state_thres, then try all detector_intervals with those values
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=-1 --state_thres 0 0 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1     # 40.5
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.1 --state_thres 0 0 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 45.3
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.2 --state_thres 0 0 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 45.3
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.3 --state_thres 0 0 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 45.0
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.4 --state_thres 0 0 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 44.7
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.5 --state_thres 0 0 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 44.4
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.6 --state_thres 0 0 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 44.0
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.7 --state_thres 0 0 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 43.5
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.8 --state_thres 0 0 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 42.8
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.9 --state_thres 0 0 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 42.1
#
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=-1 --state_thres 0 1 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1     # 37.6
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.1 --state_thres 0 1 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 45.5 <-
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.2 --state_thres 0 1 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 45.6 <-
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.3 --state_thres 0 1 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 45.6 <-
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.4 --state_thres 0 1 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 45.4 <-
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.5 --state_thres 0 1 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 45.4 <-
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.6 --state_thres 0 1 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 45.1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.7 --state_thres 0 1 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 44.7
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.8 --state_thres 0 1 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 44.3
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.9 --state_thres 0 1 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 43.7
#
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=-1 --state_thres 0 2 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1     # 34.6
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.1 --state_thres 0 2 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 44.7
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.2 --state_thres 0 2 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 45.1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.3 --state_thres 0 2 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 45.2
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.4 --state_thres 0 2 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 45.2
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.5 --state_thres 0 2 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 45.3
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.6 --state_thres 0 2 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 45.3
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.7 --state_thres 0 2 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 45.0
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.8 --state_thres 0 2 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 44.7
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.9 --state_thres 0 2 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 44.3
#
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=-1 --state_thres 1 0 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1     # 40.9
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.1 --state_thres 1 0 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 44.8
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.2 --state_thres 1 0 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 44.6
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.3 --state_thres 1 0 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 44.3
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.4 --state_thres 1 0 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 44.0
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.5 --state_thres 1 0 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 43.6
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.6 --state_thres 1 0 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 43.0
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.7 --state_thres 1 0 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 42.5
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.8 --state_thres 1 0 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 41.8
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.9 --state_thres 1 0 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 41.0
#
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=-1 --state_thres 1 1 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1     # 38.9
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.1 --state_thres 1 1 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 45.1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.2 --state_thres 1 1 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 45.2
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.3 --state_thres 1 1 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 45.0
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.4 --state_thres 1 1 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 44.8
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.5 --state_thres 1 1 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 44.6
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.6 --state_thres 1 1 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 44.2
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.7 --state_thres 1 1 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 43.7
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.8 --state_thres 1 1 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 43.2
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.9 --state_thres 1 1 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 42.5
#
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=-1 --state_thres 1 2 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1     # 36.8
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.1 --state_thres 1 2 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 44.7
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.2 --state_thres 1 2 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 44.9
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.3 --state_thres 1 2 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 44.8
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.4 --state_thres 1 2 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 44.7
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.5 --state_thres 1 2 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 44.7
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.6 --state_thres 1 2 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 44.4
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.7 --state_thres 1 2 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 44.0
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.8 --state_thres 1 2 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 43.7
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.9 --state_thres 1 2 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 43.1
#
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=-1 --state_thres 2 2 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1     # 39.2
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.1 --state_thres 2 2 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 44.1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.2 --state_thres 2 2 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 44.0
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.3 --state_thres 2 2 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 43.8
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.4 --state_thres 2 2 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 43.5
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.5 --state_thres 2 2 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 43.2
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.6 --state_thres 2 2 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 42.8
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.7 --state_thres 2 2 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 42.4
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.8 --state_thres 2 2 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 41.9
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.9 --state_thres 2 2 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 41.1

# todo: find best parameter combination abvoe and try tracker_iou_thres = 0.3 and try all detector intervals

# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.3 --det_conf_thres=0.2 --state_thres 0 1 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 45.4
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.3 --det_conf_thres=0.3 --state_thres 0 1 10 --detector_interval=3 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 45.4
#
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.3 --state_thres 0 1 10 --detector_interval=20 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1   # 34.0
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.3 --state_thres 0 1 10 --detector_interval=15 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1   # 37.7
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.3 --state_thres 0 1 10 --detector_interval=10 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1   # 41.8
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.3 --state_thres 0 1 10 --detector_interval=8 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 43.4
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.3 --state_thres 0 1 10 --detector_interval=6 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 44.5
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.3 --state_thres 0 1 10 --detector_interval=4 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 45.4
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.3 --state_thres 0 1 10 --detector_interval=2 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 45.8
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.3 --state_thres 0 1 10 --detector_interval=1 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 45.5
#
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.3 --state_thres 0 0 10 --detector_interval=20 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1   # 38.7
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.3 --state_thres 0 0 10 --detector_interval=15 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1   # 40.9
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.3 --state_thres 0 0 10 --detector_interval=10 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1   # 43.1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.3 --state_thres 0 2 10 --detector_interval=2 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 45.8
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --det_conf_thres=0.3 --state_thres 0 2 10 --detector_interval=1 --root_dir=$ROOT_DIR --benchmark="MOT17" --mode="train" --repeats=1    # 45.6

########################### Create Jobfile (DO NOT EDIT) ###########################
python create_jobfile_for_compute_metrics.py

########################### Compute MOT metrics (DO NOT EDIT) ###########################
parallel --jobs 10 < compute_metrics_jobfile.txt
