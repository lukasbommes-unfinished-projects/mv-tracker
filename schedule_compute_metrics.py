"""Compute py-motmetrics for different configurations in parallel.

This script looks up which output files are available in the `eval_dir`
directory specified below and computes for each set of files the py-motmetrics
asynchronously. The folder structure in `eval_dir` should not be generated
manually, but instead should be generated automatically by running the `eval.py`
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
import time
from datetime import datetime
import subprocess


eval_dir = "eval_output"  # the directory where evaluation output is written to
data_dir = "data"  # the directory containing "MOT16" and "MOT17" data

# get all paths that have files in their leaves
path_info = [(path, files) for path, dirs, files in list(os.walk("eval_output")) if len(files) > 0]

# remove all the paths which already contain a mot_metrics.log file
path_info = [(path, files) for path, files in path_info if not 'mot_metrics.log' in files]

FNULL = open(os.devnull, 'w')

processes = []
for eval_path, _ in path_info:
    print("Scheduling computation of MOT metrics for {}".format(eval_path))
    benchmark = str.split(eval_path, "/")[1]  # determine whether MOT17 or MOT16 is used
    mode = str.split(eval_path, "/")[2]
    ground_truth_path = os.path.join(data_dir, benchmark, mode)
    start_time = datetime.now()
    process = subprocess.Popen(["python", "compute_metrics.py", ground_truth_path, eval_path], stdout=FNULL, stderr=subprocess.STDOUT)
    processes.append({
        "process": process,
        "start_time": start_time,
        "eval_path": eval_path
    })

try:
    while True:
        print("###")
        for p in processes:
            retcode = p["process"].poll()
            dt = datetime.now() - p["start_time"]
            if retcode is not None:
                print("Process-{} ({}): Finished with return code {} after {}".format(p["process"].pid, p["eval_path"], retcode, dt))
            else:
                print("Process-{} ({}): Running for {}".format(p["process"].pid, p["eval_path"], dt))

        time.sleep(60)  # wait one minute

except KeyboardInterrupt:
    for p in processes:
        p["process"].terminate()
        dt = datetime.now() - p["start_time"]
        print("Process-{} ({}): Terminated after {}".format(p["process"].pid, p["eval_path"], dt))
