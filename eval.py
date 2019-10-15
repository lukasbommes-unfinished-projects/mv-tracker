import os
import glob
import csv
import numpy as np
from tqdm import tqdm
import motmetrics as mm

from video_cap import VideoCap

from lib.dataset.loaders import load_detections

from mvt.tracker import MotionVectorTracker as MotionVectorTrackerBaseline
from lib.tracker import MotionVectorTracker as MotionVectorTrackerDeep


if __name__ == "__main__":

    root_dir = "data"
    benchmark = "MOT17"  # either "MOT17" or "MOT16"
    codec = "mpeg4"
    eval_detectors = ["FRCNN", "SDP", "DPM"]  # which detections to use, can contain "FRCNN", "SDP", "DPM"
    eval_datasets = ["test", "train"]  # which datasets to use, can contain "train" and "test"
    tracker_type = "baseline"  # which tracker(s) to evaluate, can be "baseline", "deep"
    deep_tracker_weights_file = "models/tracker/12_10_2019_03.pth"
    detector_interval = 5
    tracker_iou_thres = 0.1

    print("Evaluating datasets: {}".format(eval_datasets))
    print("Evaluating with detections: {}".format(eval_detectors))

    train_dirs = sorted(glob.glob(os.path.join(root_dir, benchmark, "train/*")))
    test_dirs = sorted(glob.glob(os.path.join(root_dir, benchmark, "test/*")))
    data_dirs = []
    if "test" in eval_datasets:
        data_dirs += test_dirs
    if "train" in eval_datasets:
        data_dirs += train_dirs

    print(data_dirs)

    for data_dir in data_dirs:
        num_frames = len(glob.glob(os.path.join(data_dir, 'img1/*.jpg')))
        detections = load_detections(os.path.join(data_dir, 'det/det.txt'), num_frames)
        sequence_name = data_dir.split('/')[-1]
        sequence_path = '/'.join(data_dir.split('/')[:-1])
        detector_name = sequence_name.split('-')[-1]

        print("Loading annotation data from", data_dir)

        if benchmark == "MOT17":
            if detector_name not in eval_detectors:
                continue

            # get the video file from FRCNN sub directory
            sequence_name_without_detector = '-'.join(sequence_name.split('-')[:-1])
            sequence_name_frcnn = "{}-FRCNN".format(sequence_name_without_detector)
            video_file = os.path.join(sequence_path, sequence_name_frcnn, "{}-{}.mp4".format(sequence_name_frcnn, codec))

        else:
            video_file = os.path.join(data_dir, "{}-{}.mp4".format(sequence_name, codec))

        print("Loading video file from", video_file)

        # init tracker
        if tracker_type == "baseline":
            tracker = MotionVectorTrackerBaseline(iou_threshold=tracker_iou_thres)

        elif tracker_type == "deep":
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            tracker = MotionVectorTrackerDeep(iou_threshold=tracker_iou_thres,
                device=device, weights_file=deep_tracker_weights_file)

        print("Computing {} metrics for sequence {}".format(benchmark, sequence_name))

        cap = VideoCap()
        ret = cap.open(video_file)
        if not ret:
            raise RuntimeError("Could not open the video file")

        frame_idx = 0

        with open(os.path.join('eval_output', benchmark, '{}.txt'.format(sequence_name)), mode="w") as csvfile:

            csv_writer = csv.writer(csvfile, delimiter=',')

            pbar = tqdm(total=len(detections))
            while True:
                ret, frame, motion_vectors, frame_type, _ = cap.read()
                if not ret:
                    break

                # update with detections
                if frame_idx % detector_interval == 0:
                    if tracker_type == "baseline":
                        tracker.update(motion_vectors, frame_type, detections[frame_idx])
                    elif tracker_type == "deep":
                        tracker.update(motion_vectors, frame_type, detections[frame_idx], frame.shape)

                # prediction by tracker
                else:
                    if tracker_type == "baseline":
                        tracker.predict(motion_vectors, frame_type)
                    elif tracker_type == "deep":
                        tracker.predict(motion_vectors, frame_type, frame.shape)

                track_boxes = tracker.get_boxes()
                track_ids = tracker.get_box_ids()

                for track_box, track_id in zip(track_boxes, track_ids):
                    csv_writer.writerow([frame_idx+1, track_id, track_box[0], track_box[1],
                        track_box[2], track_box[3], -1, -1, -1, -1])

                frame_idx += 1
                pbar.update(1)

        cap.release()
        pbar.close()
