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
python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.7 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.8 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.9 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.95 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --det_conf_thres=0.99 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --det_conf_thres=0.7 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --det_conf_thres=0.8 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --det_conf_thres=0.9 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --det_conf_thres=0.95 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1
python eval.py --codec=mpeg4 --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=15 --det_conf_thres=0.99 --root_dir=$ROOT_DIR --sequences ${VAL_SEQUENCES[*]} --repeats=1

########################### Create Jobfile (DO NOT EDIT) ###########################
python create_jobfile_for_compute_metrics.py
