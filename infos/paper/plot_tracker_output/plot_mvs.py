import os
import pickle
import glob
import cv2
import numpy as np

import sys
sys.path.append("../../..")
from lib.visu import draw_motion_vectors, draw_macroblocks, draw_boxes
from lib.dataset.loaders import load_detections

data_path = "../../../data/MOT17/train/MOT17-09-FRCNN"
result_file = "../../../eval_output/custom/scale-1.0/mpeg4/baseline/iou-thres-0.1/conf-thres-0.7/state-thres-0-0-10/det-interval-10/MOT17-09-FRCNN.txt"
frame_idx = 99 #119 # MOT17-09-FRCNN: 239, MOT17-10-FRCNN: 199
codec = "mpeg4"
save_files = True

# values 7, 5, 0 from matplotlib color map 'tab20c'
color_boxes = (0.23529411764705882*255, 0.5529411764705883*255, 0.9921568627450981*255)  # 7
color_boxes_prev = (0.6352941176470588*255, 0.8156862745098039*255, 0.9921568627450981*255)  # 5
color_boxes_det = (0.7411764705882353*255, 0.5098039215686274*255, 0.19215686274509805*255)  # 0
#color_boxes_prev = (255, 255, 255)

detections_file = os.path.join(data_path, "det", "det.txt")
num_frames = len(glob.glob(os.path.join(data_path, "img1", "*.jpg")))


def load_results(res_file, num_frames):
    res_boxes = []
    res_ids = []
    raw_data = np.genfromtxt(res_file, delimiter=',')
    for frame_idx in range(1, num_frames + 1):  # dataset indices are 1-based
        idx = np.where(raw_data[:, 0] == frame_idx)
        if idx[0].size:  # if there are boxes in this frame
            res_box = np.stack(raw_data[idx], axis=0)[:, 2:6]
            res_id = np.stack(raw_data[idx], axis=0)[:, 1]
            consider_in_eval = np.stack(raw_data[idx], axis=0)[:, 6]
            if len(res_id) > 0:
                res_boxes.append(res_box)
                res_ids.append(res_id)
            else:
                res_boxes.append(None)
                res_ids.append(None)
        else:
            res_boxes.append(None)
            res_ids.append(None)
    return res_ids, res_boxes

# load frame
img_file = os.path.join(data_path, "img1", "{:06d}.jpg".format(frame_idx))
frame = cv2.imread(img_file, cv2.IMREAD_COLOR)
frame_shape = (frame.shape[1], frame.shape[0])

# load motion vectors
mvs_file = os.path.join(data_path, "mvs-{}-1.0".format(codec), "{:06d}.pkl".format(frame_idx))
sample = pickle.load(open(mvs_file, "rb"))
motion_vectors = sample["motion_vectors"]

# load tracker results
res_ids, res_boxes = load_results(result_file, num_frames)
det_boxes, _ = load_detections(detections_file, num_frames)

# draw motion vectors on frame
frame = draw_motion_vectors(frame, motion_vectors, format='numpy')
#frame = draw_macroblocks(frame, motion_vectors, alpha=0.4)

# draw boxes
frame = draw_boxes(frame, det_boxes[frame_idx], color=color_boxes_det)
frame = draw_boxes(frame, res_boxes[frame_idx-1], color=color_boxes_prev)
frame = draw_boxes(frame, res_boxes[frame_idx], res_ids[frame_idx].astype(np.int), color=color_boxes)

if save_files:
    cv2.imwrite("frame_{}_{:06d}.png".format(codec, frame_idx), frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])

cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("frame", 640, 360)

while True:
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
