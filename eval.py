import os
import glob
import csv
import time
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
    mode = "train"  # which datasets to use, "train" or "test"
    tracker_type = "deep"  # which tracker(s) to evaluate, "baseline" or "deep"
    deep_tracker_weights_file = "models/tracker/14_10_2019_01.pth"
    detector_interval = 10
    tracker_iou_thres = 0.1

    data_dirs = sorted(glob.glob(os.path.join(root_dir, benchmark, "{}/*".format(mode))))

    output_directory = os.path.join('eval_output', benchmark, mode, codec, tracker_type, "iou-thres-{}".format(tracker_iou_thres), "det-interval-{}".format(detector_interval))
    os.makedirs(output_directory)

    print("Created output directory {}".format(output_directory))

    dts = {}
    for data_dir in data_dirs:
        num_frames = len(glob.glob(os.path.join(data_dir, 'img1/*.jpg')))
        detections = load_detections(os.path.join(data_dir, 'det/det.txt'), num_frames)
        sequence_name = data_dir.split('/')[-1]
        sequence_path = '/'.join(data_dir.split('/')[:-1])
        detector_name = sequence_name.split('-')[-1]
        dts[sequence_name] = {
            "update": [],
            "predict": [],
            "total": []
        }
        dts["accumulated"] = {
            "update": [],
            "predict": [],
            "total": []
        }

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
            tracker = MotionVectorTrackerDeep(iou_threshold=tracker_iou_thres,
                weights_file=deep_tracker_weights_file)

        print("Computing {} metrics for sequence {}".format(benchmark, sequence_name))

        cap = VideoCap()
        ret = cap.open(video_file)
        if not ret:
            raise RuntimeError("Could not open the video file")

        frame_idx = 0

        with open(os.path.join(output_directory, '{}.txt'.format(sequence_name)), mode="w") as csvfile:

            csv_writer = csv.writer(csvfile, delimiter=',')

            pbar = tqdm(total=len(detections))
            while True:
                ret, frame, motion_vectors, frame_type, _ = cap.read()
                if not ret:
                    break

                t_start_total = time.perf_counter()

                # update with detections
                if frame_idx % detector_interval == 0:
                    t_start_update = time.perf_counter()
                    if tracker_type == "baseline":
                        tracker.update(motion_vectors, frame_type, detections[frame_idx])
                    elif tracker_type == "deep":
                        tracker.update(motion_vectors, frame_type, detections[frame_idx], frame.shape)
                    dts[sequence_name]["update"].append(time.perf_counter() - t_start_update)


                # prediction by tracker
                else:
                    t_start_predict = time.perf_counter()
                    if tracker_type == "baseline":
                        tracker.predict(motion_vectors, frame_type)
                    elif tracker_type == "deep":
                        tracker.predict(motion_vectors, frame_type, frame.shape)
                    dts[sequence_name]["predict"].append(time.perf_counter() - t_start_predict)

                dts[sequence_name]["total"].append(time.perf_counter() - t_start_total)

                track_boxes = tracker.get_boxes()
                track_ids = tracker.get_box_ids()

                for track_box, track_id in zip(track_boxes, track_ids):
                    csv_writer.writerow([frame_idx+1, track_id, track_box[0], track_box[1],
                        track_box[2], track_box[3], -1, -1, -1, -1])

                frame_idx += 1
                pbar.update(1)

        dts["accumulated"]["update"].extend(dts[sequence_name]["update"])
        dts["accumulated"]["predict"].extend(dts[sequence_name]["predict"])
        dts["accumulated"]["total"].extend(dts[sequence_name]["total"])

        cap.release()
        pbar.close()

    # write frame rate output file
    with open(os.path.join(output_directory, 'time_perf.log'), mode="w") as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(["sequence", "predict mean dt", "predict std dt",
            "update mean dt", "update std dt", "total mean dt", "total std dt"])
        for sequence_name, subdict in dts.items():
            if sequence_name != "accumulated":
                csv_writer.writerow([sequence_name, np.mean(subdict["predict"]),
                    np.std(subdict["predict"]), np.mean(subdict["update"]),
                    np.std(subdict["update"]), np.mean(subdict["total"]),
                    np.std(subdict["total"])])

        # compute average over entire dataset
        csv_writer.writerow(["Dataset averages:",'','','','','',''])
        csv_writer.writerow(["predict mean dt", "predict std dt", "update mean dt",
            "update std dt", "total mean dt", "total std dt", ""])
        csv_writer.writerow([np.mean(dts["accumulated"]["predict"]),
            np.std(dts["accumulated"]["predict"]),
            np.mean(dts["accumulated"]["update"]),
            np.std(dts["accumulated"]["update"]),
            np.mean(dts["accumulated"]["total"]),
            np.std(dts["accumulated"]["total"]), ""])
        csv_writer.writerow(["predict mean fps",'update mean fps','total mean fps','','','',''])
        csv_writer.writerow([1/np.mean(dts["accumulated"]["predict"]),
            1/np.mean(dts["accumulated"]["update"]),
            1/np.mean(dts["accumulated"]["total"]),'','','',''])
