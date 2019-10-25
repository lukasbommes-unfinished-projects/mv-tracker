import os
import time

import torch
import numpy as np
import cv2
import pickle

from video_cap import VideoCap
from mvt.utils import draw_motion_vectors, draw_boxes

from detector import DetectorTF

from mvt.tracker import MotionVectorTracker as MotionVectorTrackerBaseline
from lib.tracker import MotionVectorTracker as MotionVectorTrackerDeep
from lib.dataset.stats import StatsMpeg4UpsampledStatic, \
    StatsMpeg4UpsampledFull, StatsMpeg4DenseStatic


if __name__ == "__main__":

    video_file = "data/MOT17/test/MOT17-08-FRCNN/MOT17-08-FRCNN-mpeg4.mp4"  # test set, static cam
    #video_file = "data/MOT17/test/MOT17-12-FRCNN/MOT17-12-FRCNN-mpeg4.mp4"  # test set, moving cam
    #video_file = "data/MOT17/train/MOT17-09-FRCNN/MOT17-09-FRCNN-mpeg4.mp4"  # val set, static cam
    #video_file = "data/MOT17/train/MOT17-10-FRCNN/MOT17-10-FRCNN-mpeg4.mp4"  # val set, moving cam

    detector_path = "models/detector/faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.pb"  # detector frozen inferenze graph (*.pb)
    detector_box_size_thres = None #(0.25*1920, 0.6*1080) # discard detection boxes larger than this threshold
    detector_interval = 20
    tracker_iou_thres = 0.05

    tracker_baseline = MotionVectorTrackerBaseline(iou_threshold=tracker_iou_thres)
    tracker_deep = MotionVectorTrackerDeep(
        iou_threshold=tracker_iou_thres,
        weights_file="models/tracker/2019-10-23_09-25-34/model_final.pth", #2019-10-16_09-24-32/model_lowest_loss.pth,
        mvs_mode="upsampled",
        codec="mpeg4",
        stats=StatsMpeg4UpsampledFull,
        device=torch.device("cuda:0"))
    # tracker_deep = MotionVectorTrackerDeep(
    #     iou_threshold=tracker_iou_thres,
    #     weights_file="models/tracker/2019-10-24_09-26-27/model_final.pth",
    #     mvs_mode="dense",
    #     codec="mpeg4",
    #     stats=StatsMpeg4DenseStatic,
    #     device=torch.device("cuda:0"))

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("frame", 640, 360)

    detector = DetectorTF(path=detector_path,
                        box_size_threshold=detector_box_size_thres,
                        scaling_factor=1.0,
                        gpu=0)

    cap = VideoCap()

    ret = cap.open(video_file)
    if not ret:
        raise RuntimeError("Could not open the video file")

    frame_idx = 0
    step_wise = False

    # box colors
    color_detection = (0, 0, 150)
    color_tracker_baseline = (0, 0, 255)
    color_previous_baseline = (150, 150, 255)
    color_tracker_deep = (0, 255, 255)
    color_previous_deep = (150, 255, 255)

    prev_boxes_baseline = None
    prev_boxes_deep = None

    while True:
        ret, frame, motion_vectors, frame_type, _ = cap.read()
        if not ret:
            print("Could not read the next frame")
            break

        # draw entire field of motion vectors
        frame = draw_motion_vectors(frame, motion_vectors)

        # draw info
        frame = cv2.putText(frame, "Frame Type: {}".format(frame_type), (1000, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        # draw color legend
        frame = cv2.putText(frame, "Detection", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_detection, 2, cv2.LINE_AA)
        frame = cv2.putText(frame, "Baseline Previous Prediction", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_previous_baseline, 2, cv2.LINE_AA)
        frame = cv2.putText(frame, "Baseline Tracker Prediction", (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_tracker_baseline, 2, cv2.LINE_AA)
        frame = cv2.putText(frame, "Deep Previous Prediction", (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_previous_deep, 2, cv2.LINE_AA)
        frame = cv2.putText(frame, "Deep Tracker Prediction", (15, 165), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_tracker_deep, 2, cv2.LINE_AA)

        # update with detections
        if frame_idx % detector_interval == 0:
            detections = detector.detect(frame)
            det_boxes = detections['detection_boxes']
            tracker_baseline.update(motion_vectors, frame_type, det_boxes)
            tracker_deep.update(motion_vectors, frame_type, det_boxes, frame.shape)
            if prev_boxes_baseline is not None:
               frame = draw_boxes(frame, prev_boxes_baseline, color=color_previous_baseline)
            prev_boxes_baseline = np.copy(det_boxes)
            if prev_boxes_deep is not None:
               frame = draw_boxes(frame, prev_boxes_deep, color=color_previous_deep)
            prev_boxes_deep = np.copy(det_boxes)

        # prediction by tracker
        else:
            tracker_baseline.predict(motion_vectors, frame_type)
            track_boxes_baseline = tracker_baseline.get_boxes()

            tracker_deep.predict(motion_vectors, frame_type, frame.shape)
            track_boxes_deep = tracker_deep.get_boxes()

            if prev_boxes_baseline is not None:
               frame = draw_boxes(frame, prev_boxes_baseline, color=color_previous_baseline)
            prev_boxes_baseline = np.copy(track_boxes_baseline)

            if prev_boxes_deep is not None:
               frame = draw_boxes(frame, prev_boxes_deep, color=color_previous_deep)
            prev_boxes_deep = np.copy(track_boxes_deep)

            frame = draw_boxes(frame, track_boxes_baseline, color=color_tracker_baseline)
            frame = draw_boxes(frame, track_boxes_deep, color=color_tracker_deep)

        frame = draw_boxes(frame, det_boxes, color=color_detection)

        frame_idx += 1
        cv2.imshow("frame", frame)

        # handle key presses
        # 'q' - Quit the running program
        # 's' - enter stepwise mode
        # 'a' - exit stepwise mode
        key = cv2.waitKey(1)
        if not step_wise and key == ord('s'):
            step_wise = True
        if key == ord('q'):
            break
        if step_wise:
            while True:
                key = cv2.waitKey(1)
                if key == ord('s'):
                    break
                elif key == ord('a'):
                    step_wise = False
                    break

    cap.release()
    cv2.destroyAllWindows()
