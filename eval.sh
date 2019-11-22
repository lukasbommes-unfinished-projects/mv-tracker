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

STATE_THRESHOLDS=(2 3 10)

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
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 0 0 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 0 1 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 0 2 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 0 3 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
#
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 1 0 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 1 1 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 1 2 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 1 3 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
#
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 2 0 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 2 1 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 2 2 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 2 3 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
#
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 3 0 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 3 1 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 3 2 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 3 3 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1

# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --det_conf_thres=0.5 --state_thres 0 0 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --det_conf_thres=0.5 --state_thres 0 1 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --det_conf_thres=0.5 --state_thres 0 2 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --det_conf_thres=0.5 --state_thres 0 3 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
#
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --det_conf_thres=0.5 --state_thres 1 0 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --det_conf_thres=0.5 --state_thres 1 1 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --det_conf_thres=0.5 --state_thres 1 2 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
# python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --det_conf_thres=0.5 --state_thres 1 3 10 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1

python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 0 0 10 --root_dir=$ROOT_DIR --repeats=1
python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 0 1 10 --root_dir=$ROOT_DIR --repeats=1
python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 0 2 10 --root_dir=$ROOT_DIR --repeats=1
python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 1 0 10 --root_dir=$ROOT_DIR --repeats=1
python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 1 1 10 --root_dir=$ROOT_DIR --repeats=1
python eval.py --benchmark="MOT16" --mode="train" --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.5 --state_thres 1 2 10 --root_dir=$ROOT_DIR --repeats=1

# TODO: For MOT16 evaluate which detector interval is best (e.g. [1,2,3,4,5,6,8])

########################### Create Jobfile (DO NOT EDIT) ###########################
python create_jobfile_for_compute_metrics.py
