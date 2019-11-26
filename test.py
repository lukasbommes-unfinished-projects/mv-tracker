import os
import glob

import torch
import numpy as np
import cv2

from video_cap import VideoCap
from mvt.utils import draw_motion_vectors, draw_boxes, draw_box_ids, draw_scores

from detector import DetectorTF
from lib.dataset.loaders import load_detections

from mvt.tracker import MotionVectorTracker as MotionVectorTrackerBaseline
from lib.tracker import MotionVectorTracker as MotionVectorTrackerDeep
from lib.dataset.stats import StatsMpeg4DenseStaticSinglescale, \
    StatsMpeg4DenseFullSinglescale, StatsMpeg4UpsampledStaticSinglescale, \
    StatsMpeg4UpsampledFullSinglescale, StatsH264UpsampledStaticSinglescale,\
    StatsH264UpsampledFullSinglescale, StatsH264DenseStaticSinglescale, \
    StatsH264DenseFullSinglescale


if __name__ == "__main__":

    scaling_factor = 1.0
    codec = "mpeg4"

    # reminder: when evaluating h264 models, use h264 videos
    #video_file = "data/MOT17/train/MOT17-02-FRCNN/MOT17-02-FRCNN-{}-{}.mp4".format(codec, scaling_factor)  # train set, static cam
    #video_file = "data/MOT17/train/MOT17-11-FRCNN/MOT17-11-FRCNN-{}-{}.mp4".format(codec, scaling_factor)  # train set, moving cam
    video_file = "data/MOT17/test/MOT17-08-FRCNN/MOT17-08-FRCNN-{}-{}.mp4".format(codec, scaling_factor)  # test set, static cam
    #video_file = "data/MOT17/test/MOT17-12-FRCNN/MOT17-12-FRCNN-{}-{}.mp4".format(codec, scaling_factor)  # test set, moving cam
    #video_file = "data/MOT17/train/MOT17-09-FRCNN/MOT17-09-FRCNN-{}-{}.mp4".format(codec, scaling_factor)  # val set, static cam
    #video_file = "data/MOT17/train/MOT17-10-FRCNN/MOT17-10-FRCNN-{}-{}.mp4".format(codec, scaling_factor)  # val set, moving cam

    use_offline_detections = True
    detections_file = "data/MOT17/test/MOT17-08-DPM/det/det.txt"

    detector_path = "models/detector/faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.pb"  # detector frozen inferenze graph (*.pb)
    detector_box_size_thres = None #(0.25*1920, 0.6*1080) # discard detection boxes larger than this threshold
    detector_interval = 6
    tracker_iou_thres = 0.1
    det_conf_threshold = 0.5
    state_thresholds = (0, 1, 10)

    tracker_baseline = MotionVectorTrackerBaseline(
        iou_threshold=tracker_iou_thres,
        det_conf_threshold=det_conf_threshold,
        state_thresholds=state_thresholds,
        use_only_p_vectors=False,
        use_numeric_ids=True,
        measure_timing=True)
    tracker_deep = MotionVectorTrackerDeep(
        iou_threshold=tracker_iou_thres,
        det_conf_threshold=det_conf_threshold,
        state_thresholds=state_thresholds,
        weights_file="models/tracker/2019-10-30_02-47-42/model_highest_iou.pth",
        mvs_mode="upsampled",
        vector_type="p",
        codec=codec,
        stats=StatsMpeg4UpsampledFullSinglescale,
        device=torch.device("cuda:0"),
        use_numeric_ids=True,
        measure_timing=True)
    # tracker_deep = MotionVectorTrackerDeep(
    #     iou_threshold=tracker_iou_thres,
    #     det_conf_threshold=det_conf_threshold,
    #     state_thresholds=state_thresholds,
    #     weights_file="models/tracker/2019-11-19_16-16-13/model_highest_iou.pth",
    #     mvs_mode="dense",
    #     vector_type="p+b",
    #     codec=codec,
    #     stats=StatsH264DenseFullSinglescale,
    #     device=torch.device("cuda:0"),
    #     use_numeric_ids=True,
    #     measure_timing=True)

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("frame", 640, 360)

    if use_offline_detections:
        base_dir = str.split(video_file, "/")[:-1]
        num_frames = len(glob.glob(os.path.join(*base_dir, 'img1', '*.jpg')))
        det_boxes_all, det_scores_all = load_detections(detections_file, num_frames)
    else:
        detector = DetectorTF(path=detector_path,
                            box_size_threshold=detector_box_size_thres,
                            scaling_factor=1.0,
                            gpu=0)

    cap = VideoCap()

    ret = cap.open(video_file)
    if not ret:
        raise RuntimeError("Could not open the video file")

    frame_idx = 0
    step_wise = True

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
            if use_offline_detections:
                det_boxes = det_boxes_all[frame_idx] * scaling_factor
                det_scores = det_scores_all[frame_idx] * scaling_factor
            else:
                detections = detector.detect(frame)
                det_boxes = detections['detection_boxes']
                det_scores = detections['detection_scores']
            tracker_baseline.update(motion_vectors, frame_type, det_boxes, det_scores)
            tracker_deep.update(motion_vectors, frame_type, det_boxes, det_scores, frame.shape)
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
            box_ids_baseline = tracker_baseline.get_box_ids()

            tracker_deep.predict(motion_vectors, frame_type, frame.shape)
            track_boxes_deep = tracker_deep.get_boxes()
            box_ids_deep = tracker_deep.get_box_ids()

            if prev_boxes_baseline is not None:
               frame = draw_boxes(frame, prev_boxes_baseline, color=color_previous_baseline)
            prev_boxes_baseline = np.copy(track_boxes_baseline)

            if prev_boxes_deep is not None:
               frame = draw_boxes(frame, prev_boxes_deep, color=color_previous_deep)
            prev_boxes_deep = np.copy(track_boxes_deep)

            frame = draw_boxes(frame, track_boxes_baseline, color=color_tracker_baseline)
            frame = draw_boxes(frame, track_boxes_deep, color=color_tracker_deep)
            frame = draw_box_ids(frame, track_boxes_baseline, box_ids_baseline, color=color_tracker_baseline)
            frame = draw_box_ids(frame, track_boxes_deep, box_ids_deep, color=color_tracker_deep)

        frame = draw_boxes(frame, det_boxes, color=color_detection)
        frame = draw_scores(frame, det_boxes, det_scores, color=color_detection)

        # print FPS
        print("### FPS ###")
        print("Baseline: Predict {}, Update {}".format(
            1/tracker_baseline.last_predict_dt, 1/tracker_baseline.last_update_dt))
        print("Deep: Predict {}, Update {}, Inference {}".format(
            1/tracker_deep.last_predict_dt, 1/tracker_deep.last_update_dt,
            1/tracker_deep.last_inference_dt))

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
