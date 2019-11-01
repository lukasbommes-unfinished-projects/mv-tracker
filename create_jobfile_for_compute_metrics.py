"""Create jobfile for metric computation

This script looks up which output files are available in the `eval_dir`
directory specified below and creates a jobfile "compute_metrics_jobfile.txt"
for usage with GNU parallel. To run all jobs in parallel execute the command
parallel --jobs 12 < compute_metrics_jobfile.txt

The folder structure in `eval_dir` should not be generated manually, but
instead should be generated automatically by running the `eval.py`
script. It should look like follows
    <eval_dir>/<hyperparam_1>/<hyperparam_2>/.../<sequence_1>.txt
    <eval_dir>/<hyperparam_1>/<hyperparam_2>/.../<sequence_2>.txt

Each sub directory in the `eval_dir` corresponds to a specific set of hyper
parameters specified in `eval.py`. If the sub directory already contains a file
named "mot_metrics.log" computation for this set of files is skipped. To
recompute metrics for a set of files, manually delete the "mot_metrics.log" in
the according sub directory.

The second important directory is the `data_dir` directory which contains the
MOT16 or MOT17 ground truth annotations in the following format
    <data_dir>/<SEQUENCE_1>/gt/gt.txt
    <data_dir>/<SEQUENCE_2>/gt/gt.txt
"""

import os

eval_dir = "eval_output"  # the directory where evaluation output is written to
data_dir = "data"  # the directory containing "MOT16" and "MOT17" data

# lookup table for sequence paths
sequence_locations = {
    "MOT17-02-DPM": "MOT17/train",
    "MOT17-02-FRCNN": "MOT17/train",
    "MOT17-02-SDP": "MOT17/train",
    "MOT17-04-DPM": "MOT17/train",
    "MOT17-04-FRCNN": "MOT17/train",
    "MOT17-04-SDP": "MOT17/train",
    "MOT17-05-DPM": "MOT17/train",
    "MOT17-05-FRCNN": "MOT17/train",
    "MOT17-05-SDP": "MOT17/train",
    "MOT17-09-DPM": "MOT17/train",
    "MOT17-09-FRCNN": "MOT17/train",
    "MOT17-09-SDP": "MOT17/train",
    "MOT17-10-DPM": "MOT17/train",
    "MOT17-10-FRCNN": "MOT17/train",
    "MOT17-10-SDP": "MOT17/train",
    "MOT17-11-DPM": "MOT17/train",
    "MOT17-11-FRCNN": "MOT17/train",
    "MOT17-11-SDP": "MOT17/train",
    "MOT17-13-DPM": "MOT17/train",
    "MOT17-13-FRCNN": "MOT17/train",
    "MOT17-13-SDP": "MOT17/train",
    "MOT17-01-DPM": "MOT17/test",
    "MOT17-01-FRCNN": "MOT17/test",
    "MOT17-01-SDP": "MOT17/test",
    "MOT17-03-DPM": "MOT17/test",
    "MOT17-03-FRCNN": "MOT17/test",
    "MOT17-03-SDP": "MOT17/test",
    "MOT17-06-DPM": "MOT17/test",
    "MOT17-06-FRCNN": "MOT17/test",
    "MOT17-06-SDP": "MOT17/test",
    "MOT17-07-DPM": "MOT17/test",
    "MOT17-07-FRCNN": "MOT17/test",
    "MOT17-07-SDP": "MOT17/test",
    "MOT17-08-DPM": "MOT17/test",
    "MOT17-08-FRCNN": "MOT17/test",
    "MOT17-08-SDP": "MOT17/test",
    "MOT17-12-DPM": "MOT17/test",
    "MOT17-12-FRCNN": "MOT17/test",
    "MOT17-12-SDP": "MOT17/test",
    "MOT17-14-DPM": "MOT17/test",
    "MOT17-14-FRCNN": "MOT17/test",
    "MOT17-14-SDP": "MOT17/test",
    "ADL-Rundle-6": "MOT15/train",
    "ADL-Rundle-8": "MOT15/train",
    "ETH-Bahnhof": "MOT15/train",
    "ETH-Pedcross2": "MOT15/train",
    "ETH-Sunnyday": "MOT15/train",
    "KITTI-13": "MOT15/train",
    "KITTI-17": "MOT15/train",
    "PETS09-S2L1": "MOT15/train",
    "TUD-Campus": "MOT15/train",
    "TUD-Stadtmitte": "MOT15/train",
    "Venice-2": "MOT15/train",
    "ADL-Rundle-1": "MOT15/test",
    "ADL-Rundle-3": "MOT15/test",
    "AVG-TownCentre": "MOT15/test",
    "ETH-Crossing": "MOT15/test",
    "ETH-Jelmoli": "MOT15/test",
    "ETH-Linthescher": "MOT15/test",
    "KITTI-16": "MOT15/test",
    "KITTI-19": "MOT15/test",
    "PETS09-S2L2": "MOT15/test",
    "TUD-Crossing": "MOT15/test",
    "Venice-1": "MOT15/test"
}


if __name__ == "__main__":

    # get all paths that have files in their leaves
    path_info = [(path, files) for path, dirs, files in list(os.walk(eval_dir)) if len(files) > 0]

    # remove all the paths which already contain a mot_metrics.log file
    path_info = [(path, files) for path, files in path_info if not 'mot_metrics.log' in files]

    with open("compute_metrics_jobfile.txt", "w") as jobfile:
        for eval_path, files in path_info:

            # remove "time_perf.log" from files
            files.remove("time_perf.log")

            # strip of file extension from files
            for i in range(len(files)):
                files[i] = files[i][:-4]

            # test if sequences in the eval_path have a common ground_truth directory. This is required by the compute_metrics script
            current_sequence_locations = []
            for file in files:
                current_sequence_locations.append(sequence_locations[file])
            if len(set(current_sequence_locations)) != 1:
                raise RuntimeError("You selected a combination of sequences for which no common ground truth location (e.g. MOT17/train or MOT15/test) is available.")

            ground_truth_path = os.path.join(data_dir, sequence_locations[files[0]])

            jobfile.write("python compute_metrics.py {} {}\n".format(ground_truth_path, eval_path))
