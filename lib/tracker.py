import uuid
from collections import OrderedDict

import torch
import torchvision
import numpy as np
import cv2
import pickle

import sys
sys.path.append("..")
from mvt import trackerlib
from mvt.utils import draw_motion_vectors, draw_boxes

from lib.models.pnet_upsampled import PropagationNetwork as PropagationNetworkUpsampled
from lib.models.pnet_dense import PropagationNetwork as PropagationNetworkDense
from lib.dataset.motion_vectors import get_vectors_by_source, get_nonzero_vectors, \
    normalize_vectors, motion_vectors_to_image, motion_vectors_to_grid, \
    motion_vectors_to_grid_interpolated
from lib.dataset.velocities import box_from_velocities
from lib.transforms.transforms import StandardizeMotionVectors, StandardizeVelocities


class MotionVectorTracker:
    def __init__(self, iou_threshold, weights_file, mvs_mode, codec, stats, device=None):
        self.iou_threshold = iou_threshold
        self.mvs_mode = mvs_mode
        self.codec = codec
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.boxes = np.empty(shape=(0, 4))
        self.box_ids = []
        self.last_motion_vectors = torch.zeros(size=(1, 600, 1000, 3))

        self.standardize_motion_vectors = StandardizeMotionVectors(
            mean=stats.motion_vectors["mean"],
            std=stats.motion_vectors["std"])
        self.standardize_velocities = StandardizeVelocities(
            mean=stats.velocities["mean"],
            std=stats.velocities["std"],
            inverse=True)

        # load model and weigths
        if self.mvs_mode == "upsampled":
            self.model = PropagationNetworkUpsampled()
        elif self.mvs_mode == "dense":
            self.model = PropagationNetworkDense()
        self.model = self.model.to(self.device)
        state_dict = torch.load(weights_file)
        # if model was trained with nn.DataParallel we need to alter the state dict
        if "module" in list(state_dict.keys())[0]:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove 'module.'
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(state_dict)
        self.model.eval()


    def update(self, motion_vectors, frame_type, detection_boxes, frame_shape):

        # bring boxes into next state
        self.predict(motion_vectors, frame_type, frame_shape)

        # match predicted (tracked) boxes with detected boxes
        matches, unmatched_trackers, unmatched_detectors = trackerlib.match_bounding_boxes(self.boxes, detection_boxes, self.iou_threshold)

        #print("####")
        #print("unmatched_trackers", unmatched_trackers, [str(self.box_ids[t])[:6] for t in unmatched_trackers])
        #print("unmatched_detectors", unmatched_detectors)

        # handle matches
        for d, t in matches:
            self.boxes[t] = detection_boxes[d]
            #print("Matched tracker {} with detector {}".format(str(self.box_ids[t])[:6], d))

        # handle unmatched detections by spawning new trackers
        for d in unmatched_detectors:
            uid = uuid.uuid4()
            self.box_ids.append(uid)
            self.boxes = np.vstack((self.boxes, detection_boxes[d]))
            #print("Created new tracker {} for detector {}".format(str(uid)[:6], d))

        # handle unmatched tracker predictions by removing trackers
        for t in unmatched_trackers:
            #print("Removed tracker {}".format(str(self.box_ids[t])[:6]))
            self.boxes = np.delete(self.boxes, t, axis=0)
            self.box_ids.pop(t)


    def predict(self, motion_vectors, frame_type, frame_shape):

        # if there are no boxes skip prediction step
        if np.shape(self.boxes)[0] == 0:
            return

        # I frame has no motion vectors
        if frame_type != "I":

            # preprocess motion vectors
            motion_vectors = get_vectors_by_source(motion_vectors, "past")  # get only p vectors
            motion_vectors = normalize_vectors(motion_vectors)

            if self.mvs_mode == "upsampled":
                motion_vectors = get_nonzero_vectors(motion_vectors)
                motion_vectors = motion_vectors_to_image(motion_vectors, (frame_shape[1], frame_shape[0]))
            elif self.mvs_mode == "dense":
                if self.codec == "mpeg4":
                    motion_vectors = motion_vectors_to_grid(motion_vectors, (frame_shape[1], frame_shape[0]))
                elif self.codec == "h264":
                    motion_vectors = motion_vectors_to_grid_interpolated(motion_vectors, (frame_shape[1], frame_shape[0]))

            motion_vectors = torch.from_numpy(motion_vectors).float()
            motion_vectors = motion_vectors.unsqueeze(0)  # add batch dimension

            sample = self.standardize_motion_vectors({"motion_vectors": motion_vectors})
            motion_vectors = sample["motion_vectors"]

            # swap channel order of motion vectors from BGR to RGB
            motion_vectors = motion_vectors[..., [2, 1, 0]]

            # swap motion vector axes so that shape is (B, C, H, W) instead of (B, H, W, C)
            motion_vectors = motion_vectors.permute(0, 3, 1, 2)

            self.last_motion_vectors = motion_vectors

        # pre process boxes
        boxes_prev = np.copy(self.boxes)
        boxes_prev = torch.from_numpy(boxes_prev)
        num_boxes = (boxes_prev.shape)[0]
        # model expects boxes in format [frame_idx, xmin, ymin, w, h]
        boxes_prev_tmp = torch.zeros(num_boxes, 5).float()
        boxes_prev_tmp[:, 1:] = boxes_prev
        boxes_prev = boxes_prev_tmp
        boxes_prev = boxes_prev.unsqueeze(0)  # add batch dimension

        # feed into model, retrieve output
        with torch.set_grad_enabled(False):
            if self.mvs_mode == "upsampled":
                velocities_pred = self.model(
                    self.last_motion_vectors.to(self.device),
                    boxes_prev.to(self.device),
                    None)
            elif self.mvs_mode == "dense":
                boxes_prev_ = boxes_prev.clone()
                boxes_prev_[:, 1:] = boxes_prev_[:, 1:] / 16.0
                velocities_pred = self.model(
                    self.last_motion_vectors.to(self.device),
                    boxes_prev_.to(self.device))

            # make sure output is on CPU
            velocities_pred = velocities_pred.cpu()
            velocities_pred = velocities_pred.view(1, -1, 4)
            velocities_pred = velocities_pred[0, ...]

            # undo the standardization of predicted velocities
            sample = self.standardize_velocities({"velocities": velocities_pred})
            velocities_pred = sample["velocities"]

        # compute boxes from predicted velocities
        boxes_prev = boxes_prev[0, ...]
        boxes_prev = boxes_prev[..., 1:5]
        self.boxes = box_from_velocities(boxes_prev, velocities_pred).numpy()


    def get_boxes(self):
        return self.boxes


    def get_box_ids(self):
        return self.box_ids
