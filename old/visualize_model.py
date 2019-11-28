import cv2
import numpy as np
import torch

from lib.visu import draw_boxes
from lib.dataset.velocities import box_from_velocities
from lib.dataset.stats import StatsMpeg4UpsampledFull as Stats
from lib.transforms.transforms import StandardizeVelocities


class Visualizer:
    def __init__(self):
        self.stats = Stats()
        self.standardize_velocities = StandardizeVelocities(
            mean=self.stats.velocities["mean"],
            std=self.stats.velocities["std"],
            inverse=True)
        cv2.namedWindow("motion_vectors", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("motion_vectors", 640, 360)

    def save_inputs(self, motion_vectors, boxes_prev, boxes, velocities):
        motion_vectors = motion_vectors.detach().cpu()
        boxes_prev = boxes_prev.detach().cpu()
        boxes = boxes.detach().cpu()
        self.boxes_gt = boxes.clone()
        velocities = velocities.detach().cpu()

        # undo the standardization of velocities prior to computing and plotting
        # boxes from it
        sample = self.standardize_velocities({"velocities": velocities})
        velocities = sample["velocities"]

        self.batch_size = motion_vectors.shape[0]

        # prepare motion vectors
        batch_idx = 0
        motion_vectors = motion_vectors[batch_idx, ...]
        motion_vectors = motion_vectors[[2, 1, 0], ...]
        motion_vectors = motion_vectors.permute(1, 2, 0)
        motion_vectors = motion_vectors.numpy()
        motion_vectors = (motion_vectors - np.min(motion_vectors)) / (np.max(motion_vectors) - np.min(motion_vectors))

        # show boxes_prev
        boxes_prev = boxes_prev[batch_idx, ...]
        boxes_prev = boxes_prev[..., 1:5]

        # compute boxes based on velocities and boxes_prev
        velocities = velocities[batch_idx, ...]
        boxes = box_from_velocities(boxes_prev, velocities)

        # store frame to write ouputs into it
        self.motion_vectors = motion_vectors
        self.boxes = boxes
        self.boxes_prev = boxes_prev
        self.batch_idx = batch_idx


    def save_outputs(self, velocities_pred, num_boxes_mask):
        velocities_pred = velocities_pred.detach().cpu()
        # add padding back to allow for reshaping into [batch_size, pad_num_boxes, 4]
        batch_size = num_boxes_mask.shape[0]
        pad_num_boxes = num_boxes_mask.shape[1]
        velocities_pred_padded = torch.zeros(batch_size, pad_num_boxes, 4).float()
        for batch_idx in range(batch_size):
            num_boxes = torch.sum(num_boxes_mask[batch_idx, ...])
            velocities_pred_padded[batch_idx, :num_boxes, :] = velocities_pred[batch_idx, ...].view(-1, 4)
        velocities_pred = velocities_pred_padded
        velocities_pred = velocities_pred[self.batch_idx, ...]
        # undo the standardization
        sample = self.standardize_velocities({"velocities": velocities_pred})
        velocities_pred = sample["velocities"]
        # compute boxes from predicted velocities
        self.boxes_pred = box_from_velocities(self.boxes_prev, velocities_pred)


    def show(self):
        # show previous boxes
        self.motion_vectors = draw_boxes(self.motion_vectors, self.boxes_prev.numpy(), None, color=(200, 200, 200))
        # show gt boxes
        self.motion_vectors = draw_boxes(self.motion_vectors, self.boxes.numpy(), None, color=(255, 0, 0))
        # show predicted boxes
        self.motion_vectors = draw_boxes(self.motion_vectors, self.boxes_pred.numpy(), None, color=(255, 255, 0))
        cv2.imshow("motion_vectors", self.motion_vectors)
        key = cv2.waitKey(1)
