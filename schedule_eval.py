"""Run `eval.py` with different hyperparameters asynchronously."""

# put the commands that shall be scheduled here
commands = [
    ["python", "eval.py", "MOT17", "train", "mpeg4", "baseline", "0.1", "20", "--root_dir=data"],
    ["python", "eval.py", "MOT17", "train", "mpeg4", "baseline", "0.1", "15", "--root_dir=data"],
    ["python", "eval.py", "MOT17", "train", "mpeg4", "baseline", "0.1", "10", "--root_dir=data"],
    ["python", "eval.py", "MOT17", "train", "mpeg4", "baseline", "0.1", "8", "--root_dir=data"],
    ["python", "eval.py", "MOT17", "train", "mpeg4", "baseline", "0.1", "6", "--root_dir=data"],
    ["python", "eval.py", "MOT17", "train", "mpeg4", "baseline", "0.1", "4", "--root_dir=data"],
    ["python", "eval.py", "MOT17", "train", "mpeg4", "baseline", "0.1", "2", "--root_dir=data"],
    #
    ["python", "eval.py", "MOT17", "train", "mpeg4", "baseline", "0.2", "20", "--root_dir=data"],
    ["python", "eval.py", "MOT17", "train", "mpeg4", "baseline", "0.2", "15", "--root_dir=data"],
    ["python", "eval.py", "MOT17", "train", "mpeg4", "baseline", "0.2", "10", "--root_dir=data"],
    ["python", "eval.py", "MOT17", "train", "mpeg4", "baseline", "0.2", "8", "--root_dir=data"],
    ["python", "eval.py", "MOT17", "train", "mpeg4", "baseline", "0.2", "6", "--root_dir=data"],
    ["python", "eval.py", "MOT17", "train", "mpeg4", "baseline", "0.2", "4", "--root_dir=data"],
    ["python", "eval.py", "MOT17", "train", "mpeg4", "baseline", "0.2", "2", "--root_dir=data"],
    #
    ["python", "eval.py", "MOT17", "train", "mpeg4", "baseline", "0.3", "20", "--root_dir=data"],
    ["python", "eval.py", "MOT17", "train", "mpeg4", "baseline", "0.3", "15", "--root_dir=data"],
    ["python", "eval.py", "MOT17", "train", "mpeg4", "baseline", "0.3", "10", "--root_dir=data"],
    ["python", "eval.py", "MOT17", "train", "mpeg4", "baseline", "0.3", "8", "--root_dir=data"],
    ["python", "eval.py", "MOT17", "train", "mpeg4", "baseline", "0.3", "6", "--root_dir=data"],
    ["python", "eval.py", "MOT17", "train", "mpeg4", "baseline", "0.3", "4", "--root_dir=data"],
    ["python", "eval.py", "MOT17", "train", "mpeg4", "baseline", "0.3", "2", "--root_dir=data"],
    #
    ["python", "eval.py", "MOT17", "train", "mpeg4", "deep", "0.1", "20", "--deep_tracker_weights_file=models/tracker/14_10_2019_01.pth", "--root_dir=data"],
    ["python", "eval.py", "MOT17", "train", "mpeg4", "deep", "0.1", "15", "--deep_tracker_weights_file=models/tracker/14_10_2019_01.pth", "--root_dir=data"],
    ["python", "eval.py", "MOT17", "train", "mpeg4", "deep", "0.1", "10", "--deep_tracker_weights_file=models/tracker/14_10_2019_01.pth", "--root_dir=data"],
    ["python", "eval.py", "MOT17", "train", "mpeg4", "deep", "0.1", "8", "--deep_tracker_weights_file=models/tracker/14_10_2019_01.pth", "--root_dir=data"],
    ["python", "eval.py", "MOT17", "train", "mpeg4", "deep", "0.1", "6", "--deep_tracker_weights_file=models/tracker/14_10_2019_01.pth", "--root_dir=data"],
    ["python", "eval.py", "MOT17", "train", "mpeg4", "deep", "0.1", "4", "--deep_tracker_weights_file=models/tracker/14_10_2019_01.pth", "--root_dir=data"],
    ["python", "eval.py", "MOT17", "train", "mpeg4", "deep", "0.1", "2", "--deep_tracker_weights_file=models/tracker/14_10_2019_01.pth", "--root_dir=data"],
]



######################## DO NO EDIT BELOW THIS LINE  ########################

import os
import time
from datetime import datetime
import subprocess
import curses


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

    processes = []
    for command in commands:
        start_time = datetime.now()
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        processes.append({
            "process": process,
            "start_time": start_time,
            "end_time": None,
            "dt": None,
            "command": command,
            "status": "running",
            "just_finished": True
        })

    curses.wrapper(draw)
