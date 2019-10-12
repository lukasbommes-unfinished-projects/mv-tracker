import cv2
import numpy as np

from lib.visu import draw_boxes
from lib.dataset.velocities import box_from_velocities
from lib.dataset.stats import Stats
from lib.transforms.transforms import standardize_velocities


class Visualizer:
    def __init__(self):
        self.stats = Stats()

    def save_inputs(self, motion_vectors, boxes_prev,  motion_vector_scale, velocities):
        motion_vectors = motion_vectors.detach().cpu()
        boxes_prev = boxes_prev.detach().cpu()
        motion_vector_scale = motion_vector_scale.detach().cpu()
        velocities = velocities.detach().cpu()

        # undo the standardization of velocities prior to computing and plotting
        # boxes from it
        velocities = standardize_velocities(velocities,
            mean=self.stats.velocities["mean"],
            std=self.stats.velocities["std"], inverse=True)

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
        self.motion_vector_scale = motion_vector_scale
        self.batch_idx = batch_idx


    def save_outputs(self, velocities_pred):
        velocities_pred = velocities_pred.detach().cpu()
        velocities_pred = velocities_pred.view(self.batch_size, -1, 4)
        velocities_pred = velocities_pred[self.batch_idx, ...]
        # undo the standardization
        velocities_pred = standardize_velocities(velocities_pred,
            mean=self.stats.velocities["mean"],
            std=self.stats.velocities["std"], inverse=True)
        # compute boxes from predicted velocities
        self.boxes_pred = box_from_velocities(self.boxes_prev, velocities_pred)


    def show(self):

        # show previous boxes
        boxes_prev_scaled = self.boxes_prev * self.motion_vector_scale[0, 0]
        self.motion_vectors = draw_boxes(self.motion_vectors, boxes_prev_scaled.numpy(), None, color=(200, 200, 200))

        # show gt boxes
        boxes_scaled = self.boxes * self.motion_vector_scale[0, 0]
        self.motion_vectors = draw_boxes(self.motion_vectors, boxes_scaled.numpy(), None, color=(255, 0, 0))

        # show predicted boxes
        boxes_pred_scaled = self.boxes_pred * self.motion_vector_scale[0, 0]
        self.motion_vectors = draw_boxes(self.motion_vectors, boxes_pred_scaled.numpy(), None, color=(255, 255, 0))

        cv2.imshow("motion_vectors", self.motion_vectors)
        key = cv2.waitKey(1)
