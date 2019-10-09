import os
import time

import torch
import numpy as np
import cv2
import pickle

from video_cap import VideoCap
from mvt.utils import draw_motion_vectors, draw_boxes

from detector import DetectorTF
from config import Config

if Config.TRACKER_MODEL == "baseline":
    from mvt.tracker import MotionVectorTracker
elif Config.TRACKER_MODEL == "deep":
    from lib.tracker import MotionVectorTracker


if __name__ == "__main__":

    video_file = "data/MOT17/test/MOT17-08-FRCNN/MOT17-08-FRCNN-mpeg4.mp4"

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("frame", 640, 360)

    detector = DetectorTF(path=Config.DETECTOR_PATH,
                        box_size_threshold=Config.DETECTOR_BOX_SIZE_THRES,
                        scaling_factor=Config.SCALING_FACTOR,
                        gpu=0)
    if Config.TRACKER_MODEL == "baseline":
        tracker = MotionVectorTracker(iou_threshold=Config.TRACKER_IOU_THRES)
    elif Config.TRACKER_MODEL == "deep":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        tracker = MotionVectorTracker(iou_threshold=Config.TRACKER_IOU_THRES,
            device=device, weights_file=Config.TRACKER_WEIGHTS_FILE)

    cap = VideoCap()

    ret = cap.open(video_file)
    if not ret:
        raise RuntimeError("Could not open the video file")

    frame_idx = 0
    step_wise = False

    # box colors
    color_detection = (0, 0, 150)
    color_tracker = (0, 0, 255)
    color_previous = (200, 200, 200)

    prev_boxes = None

    while True:
        ret, frame, motion_vectors, frame_type, _ = cap.read()
        if not ret:
            print("Could not read the next frame")
            break

        print("-------------------")
        print("Frame Index: ", frame_idx)
        print("Frame type: ", frame_type)

        # draw entire field of motion vectors
        frame = draw_motion_vectors(frame, motion_vectors)

        # draw info
        frame = cv2.putText(frame, "Frame Type: {}".format(frame_type), (1000, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        # draw color legend
        frame = cv2.putText(frame, "Detection", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_detection, 2, cv2.LINE_AA)
        frame = cv2.putText(frame, "Previous Prediction", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_previous, 2, cv2.LINE_AA)
        frame = cv2.putText(frame, "Tracker Prediction", (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_tracker, 2, cv2.LINE_AA)

        # update with detections
        if frame_idx % Config.DETECTOR_INTERVAL == 0:
            detections = detector.detect(frame)
            det_boxes = detections['detection_boxes']
            if Config.TRACKER_MODEL == "baseline":
                tracker.update(motion_vectors, frame_type, det_boxes)
            elif Config.TRACKER_MODEL == "deep":
                tracker.update(motion_vectors, frame.shape, frame_type, det_boxes)
            if prev_boxes is not None:
                frame = draw_boxes(frame, prev_boxes, color=color_previous)
            prev_boxes = np.copy(det_boxes)

        # prediction by tracker
        else:
            if Config.TRACKER_MODEL == "baseline":
                tracker.predict(motion_vectors, frame_type)
            elif Config.TRACKER_MODEL == "deep":
                tracker.predict(motion_vectors, frame.shape, frame_type)
            track_boxes = tracker.get_boxes()
            frame = draw_boxes(frame, track_boxes, color=color_tracker)
            if prev_boxes is not None:
                frame = draw_boxes(frame, prev_boxes, color=color_previous)
            prev_boxes = np.copy(track_boxes)

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
