import os
import pickle
import glob
import torch
import torchvision
import cv2
import numpy as np
import math

from lib.dataset.loaders import load_groundtruth
from lib.dataset.motion_vectors import get_vectors_by_source, get_nonzero_vectors, \
    normalize_vectors, motion_vectors_to_image, motion_vectors_to_grid, \
    motion_vectors_to_grid_interpolated, motion_vectors_to_hsv_image
from lib.dataset.velocities import velocities_from_boxes
from lib.visu import draw_boxes, draw_velocities, draw_motion_vectors

# for testing
from lib.dataset.stats import StatsMpeg4DenseStaticMultiscale as Stats
from lib.transforms.transforms import StandardizeMotionVectors, \
    StandardizeVelocities, RandomFlip, RandomMotionChange

#
# def motion_vectors_to_hsv_image(motion_vectors, frame_shape=(1920, 1080)):
#
#     # output array with shape 1080, 1920, 3 and channel order H, S, V
#     # assign the direction to the hue H and the magnitude to the saturation S
#     # before returning the array convert image back to BGR space
#
#     # compute necessary frame shape
#     need_width = math.ceil(frame_shape[0] / 16) * 16
#     need_height = math.ceil(frame_shape[1] / 16) * 16
#
#     image = np.zeros((need_height, need_width, 3), dtype=np.uint8)
#
#     if np.shape(motion_vectors)[0] != 0:
#
#         # get minimum and maximum values
#         mvs_dst_x = motion_vectors[:, 5]
#         mvs_dst_y = motion_vectors[:, 6]
#         mb_w = motion_vectors[:, 1]
#         mb_h = motion_vectors[:, 2]
#         mvs_tl_x = (mvs_dst_x - 0.5 * mb_w).astype(np.int64)
#         mvs_tl_y = (mvs_dst_y - 0.5 * mb_h).astype(np.int64)
#
#         # scale motion value with the motion_scale
#         mvs_motion_x = (motion_vectors[:, 7] / motion_vectors[:, 9]).reshape(-1, 1)
#         mvs_motion_y = (motion_vectors[:, 8] / motion_vectors[:, 9]).reshape(-1, 1)
#
#         # transform x and y motion to angle in range 0...180 and magnitude in range 0...255
#         mvs_motion_magnitude, mvs_motion_angle = cv2.cartToPolar(mvs_motion_x, mvs_motion_y)
#         mvs_motion_angle = mvs_motion_angle * 180 / (2 * np.pi)  # hue channel is [0, 180]
#         mvs_motion_magnitude = cv2.normalize(mvs_motion_magnitude, None, 0, 255, cv2.NORM_MINMAX)
#
#         for i, motion_vector in enumerate(motion_vectors):
#             # repeat value in theshape of the underlying macroblock, e.g. 16 x 16 or 16 x 8
#             mvs_motion_angle_repeated = np.repeat(np.repeat(mvs_motion_angle[i, :].reshape(1, 1), mb_h[i], axis=0), mb_w[i], axis=1)
#             mvs_motion_magnitude_repeated = np.repeat(np.repeat(mvs_motion_magnitude[i, :].reshape(1, 1), mb_h[i], axis=0), mb_w[i], axis=1)
#
#             # insert repeated block into image, angle is hue (channel 0), magitude is saturation (channel 1)
#             image[mvs_tl_y[i]:mvs_tl_y[i]+mb_h[i], mvs_tl_x[i]:mvs_tl_x[i]+mb_w[i], 0] = mvs_motion_angle_repeated
#             image[mvs_tl_y[i]:mvs_tl_y[i]+mb_h[i], mvs_tl_x[i]:mvs_tl_x[i]+mb_w[i], 1] = mvs_motion_magnitude_repeated
#
#     # crop the image back to frame_shape
#     image = image[0:frame_shape[1], 0:frame_shape[0], :]
#     image[:, :, 2] = 255
#     image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
#
#     return image

# def motion_vectors_to_hsv_image(motion_vectors, frame_shape=(1920, 1080)):
#
#     # output array with shape 1080, 1920, 3 and channel order H, S, V
#     # assign the direction to the hue H and the magnitude to the saturation S
#     # before returning the array convert image back to BGR space
#
#     # compute necessary frame shape
#     need_width = math.ceil(frame_shape[0] / 16) * 16
#     need_height = math.ceil(frame_shape[1] / 16) * 16
#
#     image = np.zeros((need_height, need_width, 3), dtype=np.uint8)
#     print(image.dtype)
#
#     if np.shape(motion_vectors)[0] != 0:
#
#         # get minimum and maximum values
#         mvs_dst_x = motion_vectors[:, 5]
#         mvs_dst_y = motion_vectors[:, 6]
#         mb_w = motion_vectors[:, 1]
#         mb_h = motion_vectors[:, 2]
#         mvs_tl_x = (mvs_dst_x - 0.5 * mb_w).astype(np.int64)
#         mvs_tl_y = (mvs_dst_y - 0.5 * mb_h).astype(np.int64)
#
#         # scale motion value with the motion_scale
#         mvs_motion_x = (motion_vectors[:, 7] / motion_vectors[:, 9]).reshape(-1, 1)
#         mvs_motion_y = (motion_vectors[:, 8] / motion_vectors[:, 9]).reshape(-1, 1)
#
#         # transform x and y motion to angle, first in range [0, 1)
#         #mvs_motion_angle = np.arctan2(mvs_motion_y, mvs_motion_x)
#         #mvs_motion_angle = ((mvs_motion_angle + 2 * np.pi) % (2 * np.pi)) / (2 * np.pi)
#
#         # transform x and y motion to magnitude (>= 0)
#         #mvs_motion_magnitude = np.sqrt(np.square(mvs_motion_x) + np.square(mvs_motion_y))
#         #mvs_motion_magnitude = (mvs_motion_magnitude - np.min(mvs_motion_magnitude)) / (np.max(mvs_motion_magnitude) - np.min(mvs_motion_magnitude))
#
#         mvs_motion_magnitude, mvs_motion_angle = cv2.cartToPolar(mvs_motion_x, mvs_motion_y)
#
#         mvs_motion_angle = mvs_motion_angle * 180 / (2 * np.pi)  # hue channel is [0, 180]
#         mvs_motion_magnitude = cv2.normalize(mvs_motion_magnitude, None, 0, 255, cv2.NORM_MINMAX)
#
#         print(np.min(mvs_motion_angle))
#         print(np.max(mvs_motion_angle))
#         print(mvs_motion_angle.shape)
#
#         print(np.min(mvs_motion_magnitude))
#         print(np.max(mvs_motion_magnitude))
#         print(np.mean(mvs_motion_magnitude))
#         print(np.std(mvs_motion_magnitude))
#         print(mvs_motion_magnitude.shape)
#
#         for i, motion_vector in enumerate(motion_vectors):
#             # repeat value in theshape of the underlying macroblock, e.g. 16 x 16 or 16 x 8
#             mvs_motion_angle_repeated = np.repeat(np.repeat(mvs_motion_angle[i, :].reshape(1, 1), mb_h[i], axis=0), mb_w[i], axis=1)
#             mvs_motion_magnitude_repeated = np.repeat(np.repeat(mvs_motion_magnitude[i, :].reshape(1, 1), mb_h[i], axis=0), mb_w[i], axis=1)
#
#             # insert repeated block into image, angle is hue (channel 0), magitude is saturation (channel 1)
#             image[mvs_tl_y[i]:mvs_tl_y[i]+mb_h[i], mvs_tl_x[i]:mvs_tl_x[i]+mb_w[i], 0] = mvs_motion_angle_repeated
#             image[mvs_tl_y[i]:mvs_tl_y[i]+mb_h[i], mvs_tl_x[i]:mvs_tl_x[i]+mb_w[i], 2] = mvs_motion_magnitude_repeated
#
#     # crop the image back to frame_shape
#     image = image[0:frame_shape[1], 0:frame_shape[0], :]
#     image[:, :, 1] = 255
#     image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
#
#     # print(np.min(image[:, :, 1]))
#     # print(np.max(image[:, :, 1]))
#     print(image.dtype)
#
#     return image


frame = cv2.imread("000255.jpg", cv2.IMREAD_COLOR)
data = pickle.load(open("000255.pkl", "rb"))
motion_vectors = data["motion_vectors"]
frame_type = data["frame_type"]

motion_vectors = get_vectors_by_source(motion_vectors, "past")  # get only p vectors
motion_vectors = normalize_vectors(motion_vectors)
motion_vectors_for_visu = np.copy(get_nonzero_vectors(motion_vectors))
motion_vectors = get_nonzero_vectors(motion_vectors)
motion_vectors_rgb = motion_vectors_to_image(motion_vectors, (768, 576))
motion_vectors_hsv = motion_vectors_to_hsv_image(motion_vectors, (768, 576))

print(motion_vectors_for_visu.shape)
frame = draw_motion_vectors(frame, motion_vectors_for_visu, format='numpy')

while True:
    cv2.imshow("frame", frame)
    cv2.imshow("motion_vectors_rgb", motion_vectors_rgb)
    cv2.imshow("motion_vectors_hsv", motion_vectors_hsv)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
