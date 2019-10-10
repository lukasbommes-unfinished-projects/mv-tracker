import os
import glob
import csv

import numpy as np
from tqdm import tqdm

from video_cap import VideoCap
from mvt.tracker import MotionVectorTracker

from lib.dataset.loaders import load_detections

import motmetrics as mm


if __name__ == "__main__":

    root_dir = "data/MOT17"
    eval_detectors = ["FRCNN", "SDP", "DPM"]  # which detections to use, can contain "FRCNN", "SDP", "DPM"
    eval_datasets = ["train"]  # which datasets to use, can contain "train" and "test"
    detector_interval = 5
    tracker_iou_thres = 0.1

    print("Evaluating datasets: {}".format(eval_datasets))
    print("Evaluating with detections: {}".format(eval_detectors))

    train_dirs = sorted(glob.glob(os.path.join(root_dir, "train/*")))
    test_dirs = sorted(glob.glob(os.path.join(root_dir, "test/*")))
    data_dirs = []
    if "test" in eval_datasets:
        data_dirs += test_dirs
    if "train" in eval_datasets:
        data_dirs += train_dirs

    for data_dir in data_dirs:
        video_file = os.path.join(data_dir, 'seq.mp4')

        num_frames = len(glob.glob(os.path.join(data_dir, 'img1/*.jpg')))

        detections = load_detections(os.path.join(data_dir, 'det/det.txt'), num_frames)
        sequence_name = data_dir.split('/')[-1]

        detector_name = sequence_name.split('-')[-1]
        if detector_name not in eval_detectors:
            continue

        print("Computing MOT metrics for sequence {}".format(sequence_name))

        tracker = MotionVectorTracker(iou_threshold=tracker_iou_thres)
        cap = VideoCap()

        ret = cap.open(video_file)
        if not ret:
            raise RuntimeError("Could not open the video file")

        frame_idx = 0

        with open(os.path.join('eval_output', '{}.txt'.format(sequence_name)), mode="w") as csvfile:

            csv_writer = csv.writer(csvfile, delimiter=',')

            pbar = tqdm(total=len(detections))
            while True:
                ret, frame, motion_vectors, frame_type, _ = cap.read()
                if not ret:
                    break

                # update with detections
                if frame_idx % detector_interval == 0:
                    tracker.update(motion_vectors, frame_type, detections[frame_idx])

                # prediction by tracker
                else:
                    tracker.predict(motion_vectors, frame_type)

                track_boxes = tracker.get_boxes()
                track_ids = tracker.get_box_ids()

                for track_box, track_id in zip(track_boxes, track_ids):
                    csv_writer.writerow([frame_idx+1, track_id, track_box[0], track_box[1],
                        track_box[2], track_box[3], -1, -1, -1, -1])

                frame_idx += 1
                pbar.update(1)

        cap.release()
        pbar.close()
