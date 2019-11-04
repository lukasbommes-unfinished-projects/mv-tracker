import os
import glob
import csv
import time
import argparse
import numpy as np
import torch
from tqdm import tqdm
import motmetrics as mm

from video_cap import VideoCap

from lib.dataset.loaders import load_detections

from mvt.tracker import MotionVectorTracker as MotionVectorTrackerBaseline
from lib.tracker import MotionVectorTracker as MotionVectorTrackerDeep
from lib.dataset.stats import StatsMpeg4UpsampledFull


def parse_args():
    parser = argparse.ArgumentParser(description="""
Compute predicted bounding boxes for any tracker and hyper parameter settings.

This script runs any of the specified trackers with the specified hyper
parameters and writes the predictions into output csv files which can later be
used as input to the `compute_metrics.py` script. Additionally, the frame rate
of the trackers is measured and stored into the `time_perf.log` file.

The output directory is called `eval_output` and has the following structure

eval_output/<benchmark>/<mode>/<codec>/<vector_type>/<tracker_type>/<x>/
    <tracker_iou_thres>/<detector_interval>

In case of the baseline tracker `x` in the above path is ommited. For the deep
trackers, this is the weights file used in the evaluation.

Example usages:
python eval.py --benchmark=MOT17 --mode=train --codec=h264 --vector_type=p --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4
python eval.py --benchmark=MOT17 --mode=train --codec=mpeg4 --tracker_type=deep --tracker_iou_thres=0.1 --detector_interval=4 --deep_tracker_weights_file=models/tracker/14_10_2019_01.pth --root_dir=data --gpu 0

When providing a set of sequences via the "sequences" argument the values for
benchmark and mode are ignored and boxes are computed for the specified
sequences only.

Example:
python eval.py --codec=h264 --vector_type=p --tracker_type=baseline --tracker_iou_thres=0.1 --detector_interval=4 --sequences MOT17/train/MOT17-02-DPM MOT17/train/MOT17-02-FRCNN
""", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--benchmark', type=str, help='Either "MOT16" or "MOT17". Determines which detections are loaded.', default='MOT17')
    parser.add_argument('--mode', type=str, help='Either "train" or "test". Whether to compute metrics on train or test split of MOT data.', default='train')
    parser.add_argument('--sequences', nargs='+', type=str, help='Compute boxes on the specified sequences only. If provided settings for benchmark and mode are ignored.')
    parser.add_argument('--scale', type=float, help='Scaling factor for the input sequence relative to the original sequence, e.g. if original sequence is 1920 x 1080, but was reencoded to 960 x 540 then scale is 0.5.', default=1.0)
    parser.add_argument('--codec', type=str, help='Either "mpeg4" or "h264" determines the encoding type of the video.', default='mpeg4')
    parser.add_argument('--vector_type', type=str, help='Either "p" to use only p vectors or "pb" to use both p and b vectors ("pb" is only valid if codec is h264).', default='p')
    parser.add_argument('--tracker_type', type=str, help='Specifies the tracker model used, e.g. "baseline" or "deep"', default='baseline')
    parser.add_argument('--tracker_iou_thres', type=float, help='The minimum IoU needed to match a tracked boy with a detected box during data assocation step.', default=0.1)
    parser.add_argument('--detector_interval', type=int, help='The interval in which the detector is run, e.g. 10 means the detector is run on every 10th frame.', default=5)
    parser.add_argument('--deep_tracker_weights_file', type=str, help='File path to the weights file of the deep tracker')
    parser.add_argument('--root_dir', type=str, help='Directory containing the MOT data', default='data')
    parser.add_argument('--gpu', type=int, help='Index of the GPU on which to run inference of deep tracker. Pass -1 to run on CPU.', default=0)
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    eval_detectors = ["FRCNN", "SDP", "DPM"]  # which detections to use, can contain "FRCNN", "SDP", "DPM"

    # select either custom sequences or MOT16/MOT17 sequences
    if args.sequences:
        data_dirs = [os.path.join(args.root_dir, sequence) for sequence in args.sequences]
        output_directory_root = os.path.join('eval_output', "custom")
    else:
        data_dirs = sorted(glob.glob(os.path.join(args.root_dir, args.benchmark, "{}/*".format(args.mode))))
        output_directory_root = os.path.join('eval_output', args.benchmark, args.mode)

        # generate output directory
    if args.tracker_type == "baseline":
        if args.codec == "h264":
            output_directory = os.path.join(
                output_directory_root,
                "scale-{}".format(args.scale),
                args.codec,
                args.tracker_type,
                args.vector_type,
                "iou-thres-{}".format(args.tracker_iou_thres),
                "det-interval-{}".format(args.detector_interval))
        elif args.codec == "mpeg4":
            output_directory = os.path.join(
                output_directory_root,
                "scale-{}".format(args.scale),
                args.codec,
                args.tracker_type,
                "iou-thres-{}".format(args.tracker_iou_thres),
                "det-interval-{}".format(args.detector_interval))
    elif args.tracker_type == "deep":
        weights_file_name_date = str.split(args.deep_tracker_weights_file, "/")[-2]
        weights_file_name_model = str.split(args.deep_tracker_weights_file, "/")[-1][:-4]
        weights_file_name = "_".join([weights_file_name_date, weights_file_name_model])
        if args.codec == "h264":
            output_directory = os.path.join(
                output_directory_root,
                "scale-{}".format(args.scale),
                args.codec,
                args.tracker_type,
                weights_file_name,
                args.vector_type,
                "iou-thres-{}".format(args.tracker_iou_thres),
                "det-interval-{}".format(args.detector_interval))
        elif args.codec == "mpeg4":
            output_directory = os.path.join(
                output_directory_root,
                "scale-{}".format(args.scale),
                args.codec,
                args.tracker_type, 
                weights_file_name,
                "iou-thres-{}".format(args.tracker_iou_thres),
                "det-interval-{}".format(args.detector_interval))

    # if output directory exists exit this process
    try:
        os.makedirs(output_directory)
    except FileExistsError:
        print("Output directory {} exists. Skipping.".format(output_directory))
        exit()

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

        if args.benchmark == "MOT17":
            if detector_name not in eval_detectors:
                continue

            # get the video file from FRCNN sub directory
            sequence_name_without_detector = '-'.join(sequence_name.split('-')[:-1])
            sequence_name_frcnn = "{}-FRCNN".format(sequence_name_without_detector)
            video_file = os.path.join(sequence_path, sequence_name_frcnn, "{}-{}-{}.mp4".format(sequence_name_frcnn, args.codec, args.scale))

        else:
            video_file = os.path.join(data_dir, "{}-{}-{}.mp4".format(sequence_name, args.codec, args.scale))

        print("Loading video file from", video_file)

        if args.gpu == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(args.gpu))

        # init tracker
        if args.tracker_type == "baseline":
            use_only_p_vectors = (args.vector_type == "p")
            tracker = MotionVectorTrackerBaseline(iou_threshold=args.tracker_iou_thres,
                use_only_p_vectors=use_only_p_vectors)
        elif args.tracker_type == "deep":
            tracker = MotionVectorTrackerDeep(
                iou_threshold=args.tracker_iou_thres,
                weights_file=args.deep_tracker_weights_file,
                mvs_mode="upsampled",
                codec="mpeg4",
                stats=StatsMpeg4UpsampledFull,
                device=device)
            # TODO: Add other tracker models here

        print("Computing {} metrics for sequence {}".format(args.benchmark, sequence_name))

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

                t_start_total = time.process_time()

                # update with detections
                if frame_idx % args.detector_interval == 0:
                    t_start_update = time.process_time()
                    if args.tracker_type == "baseline":
                        tracker.update(motion_vectors, frame_type, detections[frame_idx]*args.scale)
                    elif args.tracker_type == "deep":
                        tracker.update(motion_vectors, frame_type, detections[frame_idx]*args.scale, frame.shape)
                    dts[sequence_name]["update"].append(time.process_time() - t_start_update)


                # prediction by tracker
                else:
                    t_start_predict = time.process_time()
                    if args.tracker_type == "baseline":
                        tracker.predict(motion_vectors, frame_type)
                    elif args.tracker_type == "deep":
                        tracker.predict(motion_vectors, frame_type, frame.shape)
                    dts[sequence_name]["predict"].append(time.process_time() - t_start_predict)

                dts[sequence_name]["total"].append(time.process_time() - t_start_total)

                track_ids = tracker.get_box_ids()
                track_boxes = tracker.get_boxes()
                # revert scaling so that comparison with unscaled ground truth in compute_metrics.py makes sense
                track_boxes = track_boxes / args.scale


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
