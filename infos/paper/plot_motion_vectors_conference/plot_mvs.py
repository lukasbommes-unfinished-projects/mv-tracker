import os
import pickle
import cv2
import numpy as np

import sys
sys.path.append("../../..")
from lib.visu import draw_motion_vectors, draw_macroblocks
from lib.dataset.motion_vectors import motion_vectors_to_image, \
    motion_vectors_to_hsv_image, motion_vectors_to_hsv_grid, \
    motion_vectors_to_hsv_grid_interpolated, get_vectors_by_source, \
    normalize_vectors

data_path = "../../../data/MOT17/train/MOT17-02-FRCNN"
frame_idx = 381
codec = "mpeg4"
save_files = True

# load frame
img_file = os.path.join(data_path, "img1", "{:06d}.jpg".format(frame_idx))
frame = cv2.imread(img_file, cv2.IMREAD_COLOR)
frame_shape = (frame.shape[1], frame.shape[0])

# load motion vectors
mvs_file = os.path.join(data_path, "mvs-{}-1.0".format(codec), "{:06d}.pkl".format(frame_idx))
sample = pickle.load(open(mvs_file, "rb"))
motion_vectors = sample["motion_vectors"]

# debug to print the motion vector with destination (952, 424)
# idx_x = set(list(np.where(motion_vectors[:, 5] == 952)[0]))
# idx_y = set(list(np.where(motion_vectors[:, 6] == 424)[0]))
# idx = idx_x.intersection(idx_y)
# for i in idx:
#     print(i, motion_vectors[i, :])

# draw motion vectors on frame
frame = draw_motion_vectors(frame, motion_vectors, format='numpy')
frame = draw_macroblocks(frame, motion_vectors, alpha=0.4)

# preprocess motion vectors
motion_vectors = get_vectors_by_source(motion_vectors, "past")  # get only p vectors
motion_vectors = normalize_vectors(motion_vectors)

# convert motion vectors to image (upsampled)
mvs_image_upsampled = motion_vectors_to_hsv_image(motion_vectors, frame_shape)

# convert motion vectors to image (dense)
if codec == "mpeg4":
    mvs_image_dense = motion_vectors_to_hsv_grid(motion_vectors, frame_shape)
elif codec == "h264":
    mvs_image_dense = motion_vectors_to_hsv_grid_interpolated(motion_vectors, frame_shape)

if save_files:
    cv2.imwrite("frame_{}_{:06d}.png".format(codec, frame_idx), frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    cv2.imwrite("mvs_image_upsampled_{}_{:06d}.png".format(codec, frame_idx), mvs_image_upsampled, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    cv2.imwrite("mvs_image_dense_{}_{:06d}.png".format(codec, frame_idx), mvs_image_dense, [cv2.IMWRITE_PNG_COMPRESSION, 0])

cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("frame", 640, 360)
cv2.namedWindow("mvs_image_upsampled", cv2.WINDOW_NORMAL)
cv2.resizeWindow("mvs_image_upsampled", 640, 360)
cv2.namedWindow("mvs_image_dense", cv2.WINDOW_NORMAL)
cv2.resizeWindow("mvs_image_dense", 640, 360)

while True:
    cv2.imshow("frame", frame)
    cv2.imshow("mvs_image_upsampled", mvs_image_upsampled)
    cv2.imshow("mvs_image_dense", mvs_image_dense)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
