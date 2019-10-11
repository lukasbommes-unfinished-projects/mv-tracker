import os
import random
import torch
import cv2
import numpy as np

from lib.dataset.dataset import MotionVectorDataset
from lib.dataset.stats import Stats
from lib.transforms.transforms import standardize_motion_vectors, \
    standardize_velocities, scale_image

class RandomFlip:
    def __init__(self, direction="horizontal", p=0.5):
        self.direction = direction
        self.p = p

    def __call__(self, frame, motion_vectors, boxes_prev):
        if bool(torch.rand(1) <= self.p):
            pass


class RandomScale:
    def __init__(self, scales=[300, 400, 500, 600], max_size=1000):
        self.scales = scales
        self.max_size = max_size

    def __call__(self, frame):
        scale = random.choice(self.scales)
        print(scale)

        # determine the scaling factor
        frame = frame.numpy()
        frame_size_min = np.min(frame.shape[-3:-1])
        frame_size_max = np.max(frame.shape[-3:-1])
        scaling_factor = float(scale) / float(frame_size_min)
        if np.round(scaling_factor * frame_size_max) > self.max_size:
            scaling_factor = float(self.max_size) / float(frame_size_max)

        # scale the frame
        frame_resized = cv2.resize(frame, None, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_LINEAR)
        frame_resized = torch.from_numpy(frame_resized)

        return frame_resized, scaling_factor


root_dir = "data_precomputed"
batch_size=2
dataset = MotionVectorDataset(root_dir='data', batch_size=batch_size,
    codec="mpeg4", mot17_detector="FRCNN", visu=True, debug=False, mode="train")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
    shuffle=False, num_workers=0)

flip = RandomFlip(direction="horizontal", p=1.0)
random_scale = RandomScale()

step_wise = True

for batch_idx in range(batch_size):
    cv2.namedWindow("frame-{}".format(batch_idx), cv2.WINDOW_NORMAL)
    cv2.resizeWindow("frame-{}".format(batch_idx), 640, 360)
    cv2.namedWindow("motion_vectors-{}".format(batch_idx), cv2.WINDOW_NORMAL)
    cv2.resizeWindow("motion_vectors-{}".format(batch_idx), 640, 360)

for step, (frames, motion_vectors, boxes_prev, velocities, num_boxes_mask,
    det_boxes_prev) in enumerate(dataloader):

    print(motion_vectors.shape)

    # all transforms are applied right after the dataset and expect the data:
    # 1) motion vectors shape [batch_size, H, W, C], e.g. [2, 1080, 1920, 3], channel order BGR

    # apply random flip
    # apply random scale
    # apply random translation and crop

    # resize spatial dimensions of motion vectors
    motion_vectors, motion_vector_scale = scale_image(motion_vectors,
        short_side_min_len=600, long_side_max_len=1000)

    # standardize velocities
    velocities = standardize_velocities(velocities,
        mean=Stats.velocities["mean"],
        std=Stats.velocities["std"])

    # standardize motion vectors
    motion_vectors = standardize_motion_vectors(motion_vectors,
        mean=Stats.motion_vectors["mean"],
        std=Stats.motion_vectors["std"])

    # swap channel order of motion vectors from BGR to RGB
    motion_vectors = motion_vectors[..., [2, 1, 0]]

    # swap motion vector axes so that shape is (B, C, H, W) instead of (B, H, W, C)
    motion_vectors = motion_vectors.permute(0, 3, 1, 2)

    motion_vector_scale = torch.tensor(motion_vector_scale)
    motion_vector_scale = motion_vector_scale.repeat(batch_size).view(-1, 1)

    for batch_idx in range(motion_vectors.shape[0]):

        motion_vectors_ = motion_vectors[batch_idx]
        frame = frames[batch_idx]
        boxes_prev = boxes_prev[batch_idx]

        #frame, motion_vectors_, boxes_prev = flip(frame, motion_vectors_, boxes_prev)
        print("before random scale:", frame.shape)
        frame, scale = random_scale(frame)
        print("after random scale:", frame.shape)

        motion_vectors_ = motion_vectors_[[2, 1, 0], ...]
        motion_vectors_ = motion_vectors_.permute(1, 2, 0)
        motion_vectors_ = motion_vectors_.numpy()
        frame = frame.numpy()

        print("step: {}, MVS shape: {}".format(step, motion_vectors_.shape))
        cv2.imshow("frame-{}".format(batch_idx), frame)
        cv2.imshow("motion_vectors-{}".format(batch_idx), motion_vectors_)

    key = cv2.waitKey(1)
    if not step_wise and key == ord('s'):
        step_wise = True
    if key == ord('q'):
        break
    if step_wise:
        while True:
            key = cv2.waitKey(1)
            if key == ord('s'):
                break
            elif key == ord('a'):
                step_wise = False
                break
