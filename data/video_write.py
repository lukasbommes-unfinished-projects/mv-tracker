# This script converts each sequence's imaged of the MOT dataset into H.264 encoded video sequences
import os
import subprocess


CODEC = "mpeg4"  # "h264" or "mpeg4"
DATASET = "MOT17"

if __name__ == "__main__":

    scales = [1.0, 0.75, 0.5]  # encode video at 100 %, 75 % and 50 % of original size

    frame_rates = {
        "MOT17": {
                "train": [30,30,14,30,30,30,25],
                "test": [30,30,14,30,30,30,25]
            },
        "MOT16": {
            "train": [30,30,14,30,30,30,25],
            "test": [30,30,14,30,30,30,25]
            },
        "MOT15": {
            "train": [25,25,7,14,14,14,30,30,10,10,30],
            "test": [25,7,14,14,14,5/2,30,30,10,10,30]
            }
    }

    dir_names = {
    "MOT17": {
            "train": [
                'MOT17-02-FRCNN',
                'MOT17-04-FRCNN',
                'MOT17-05-FRCNN',
                'MOT17-09-FRCNN',
                'MOT17-10-FRCNN',
                'MOT17-11-FRCNN',
                'MOT17-13-FRCNN',
            ],
            "test": [
                'MOT17-01-FRCNN',
                'MOT17-03-FRCNN',
                'MOT17-06-FRCNN',
                'MOT17-07-FRCNN',
                'MOT17-08-FRCNN',
                'MOT17-12-FRCNN',
                'MOT17-14-FRCNN',
            ]
        },
    "MOT16": {
            "train": [
                'MOT16-02',
                'MOT16-04',
                'MOT16-05',
                'MOT16-09',
                'MOT16-10',
                'MOT16-11',
                'MOT16-13',
            ],
            "test": [
                'MOT16-01',
                'MOT16-03',
                'MOT16-06',
                'MOT16-07',
                'MOT16-08',
                'MOT16-12',
                'MOT16-14',
            ]
        },
    "MOT15": {
            "train": [
                'TUD-Stadtmitte',
                'TUD-Campus',
                'PETS09-S2L1',
                'ETH-Bahnhof',
                'ETH-Sunnyday',
                'ETH-Pedcross2',
                'ADL-Rundle-6',
                'ADL-Rundle-8',
                'KITTI-13',
                'KITTI-17',
                'Venice-2'
            ],
            "test": [
                'TUD-Crossing',
                'PETS09-S2L2',
                'ETH-Jelmoli',
                'ETH-Linthescher',
                'ETH-Crossing',
                'AVG-TownCentre',
                'ADL-Rundle-1',
                'ADL-Rundle-3',
                'KITTI-16',
                'KITTI-19',
                'Venice-1'
            ]
        }
    }


    cwd = os.getcwd()
    for mode in ["train", "test"]:
        for dir_name, frame_rate in zip(dir_names[DATASET][mode], frame_rates[DATASET][mode]):
            os.chdir(os.path.join(cwd, DATASET, mode, dir_name, 'img1'))
            for scale in scales:
                if CODEC == "h264":
                    subprocess.call(['ffmpeg', '-y', '-r', str(frame_rate),
                        '-i', '%06d.jpg', '-c:v', 'libx264', '-vf',
                        'scale=iw*{}:ih*{}, pad=ceil(iw/2)*2:ceil(ih/2)*2'.format(scale, scale),
                        '-f', 'rawvideo', '../{}-{}-{}.mp4'.format(dir_name, CODEC, scale)])
                elif CODEC == "mpeg4":
                    subprocess.call(['ffmpeg', '-y', '-r', str(frame_rate),
                        '-i', '%06d.jpg', '-c:v', 'mpeg4', '-vf',
                        'scale=iw*{}:ih*{}, pad=ceil(iw/2)*2:ceil(ih/2)*2'.format(scale, scale),
                        '-f', 'rawvideo', '../{}-{}-{}.mp4'.format(dir_name, CODEC, scale)])
