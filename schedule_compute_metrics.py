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
import curses

eval_dir = "eval_output"  # the directory where evaluation output is written to
data_dir = "data"  # the directory containing "MOT16" and "MOT17" data

def draw(stdscr):
    stdscr.clear()
    stdscr.refresh()
    stdscr.nodelay(True)  # enable non-blocking getch()

    scroll_offset = 0

    # Start colors in curses
    curses.start_color()
    curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_WHITE)

    # Declaration of strings
    statusbarstr = "'q' exit | 't' terminate all jobs | 'UP' scroll up | 'DOWN' scroll down"
    heading = "PID    Status      Uptime         Command"

    while True:

        key = stdscr.getch()
        stdscr.clear()
        height, width = stdscr.getmaxyx()

        # scrolling
        if  key == curses.KEY_DOWN and scroll_offset > -1 * len(processes):
            scroll_offset -= 1

        elif key == curses.KEY_UP and scroll_offset < 0:
            scroll_offset += 1

        elif key == ord('q'):
            exit()

        elif key == ord('t'):
            for p in processes:
                if p["status"] is "running":
                    p["process"].terminate()
                    p["status"] = "terminated"
                    p["stop_time"] = datetime.now()

        else:
            for i, p in enumerate(processes):
                retcode = p["process"].poll()
                try:
                    if retcode is not None:
                        if p["status"] == "terminated":
                            p["dt"] = p["stop_time"] - p["start_time"]
                            stdscr.addnstr(i + 1 + scroll_offset, 0, str(p["process"].pid), 6)
                            stdscr.addnstr(i + 1 + scroll_offset, 7, "TERMINATED", 10)
                            stdscr.addnstr(i + 1 + scroll_offset, 19, str(p["dt"]), 15)
                            stdscr.addnstr(i + 1 + scroll_offset, 34, str(p["command"]), width-34)
                        else:
                            if p["just_finished"]:
                                p["dt"] = datetime.now() - p["start_time"]
                                p["status"] = "finished"
                                p["just_finished"] = False
                            stdscr.addnstr(i + 1 + scroll_offset, 0, str(p["process"].pid), 6)
                            if retcode < 0:
                                stdscr.addnstr(i + 1 + scroll_offset, 7, "FAILED ({})".format(retcode), 10)
                            else:
                                stdscr.addnstr(i + 1 + scroll_offset, 7, "FINISHED", 10)
                            stdscr.addnstr(i + 1 + scroll_offset, 19, str(p["dt"]), 15)
                            stdscr.addnstr(i + 1 + scroll_offset, 34, str(p["command"]), width-34)
                    else:
                        p["dt"] = datetime.now() - p["start_time"]
                        stdscr.addnstr(i + 1 + scroll_offset, 0, str(p["process"].pid), 6)
                        stdscr.addnstr(i + 1 + scroll_offset, 7, "RUNNING", 10)
                        stdscr.addnstr(i + 1 + scroll_offset, 19, str(p["dt"]), 15)
                        stdscr.addnstr(i + 1 + scroll_offset, 34, str(p["command"]), width-34)
                except curses.error:
                    pass

            # Render heading
            try:
                stdscr.attron(curses.color_pair(3))
                stdscr.attron(curses.A_BOLD)
                stdscr.addnstr(0, 0, heading, width)
                stdscr.addnstr(0, len(heading), " " * (width - len(heading)), width)
                stdscr.attroff(curses.color_pair(3))
                stdscr.attroff(curses.A_BOLD)

                # Render status bar
                stdscr.attron(curses.color_pair(3))
                stdscr.addnstr(height-1, 0, statusbarstr, width)
                stdscr.addnstr(height-1, len(statusbarstr), " " * (width - len(statusbarstr) - 1), width)
                stdscr.attroff(curses.color_pair(3))
            except curses.error:
                pass

        # Refresh the screen
        stdscr.refresh()

        time.sleep(0.1)


if __name__ == "__main__":

    # get all paths that have files in their leaves
    path_info = [(path, files) for path, dirs, files in list(os.walk("eval_output")) if len(files) > 0]

    # remove all the paths which already contain a mot_metrics.log file
    path_info = [(path, files) for path, files in path_info if not 'mot_metrics.log' in files]

    processes = []
    for eval_path, _ in path_info:
        benchmark = str.split(eval_path, "/")[1]  # determine whether MOT17 or MOT16 is used
        mode = str.split(eval_path, "/")[2]
        ground_truth_path = os.path.join(data_dir, benchmark, mode)
        start_time = datetime.now()
        command = ["python", "compute_metrics.py", ground_truth_path, eval_path]
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        processes.append({
            "process": process,
            "start_time": start_time,
            "command": command,
            "end_time": None,
            "dt": None,
            "status": "running",
            "just_finished": True
        })

    curses.wrapper(draw)
