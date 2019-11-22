import uuid

import numpy as np
import cv2
import pickle

from mvt import trackerlib
from mvt.utils import draw_motion_vectors, draw_boxes


class MotionVectorTracker:
    def __init__(self, iou_threshold, det_conf_threshold, use_only_p_vectors=False,
        use_kalman=False, use_numeric_ids=False):
        self.iou_threshold = iou_threshold
        self.det_conf_threshold = det_conf_threshold
        self.use_only_p_vectors = use_only_p_vectors
        self.use_kalman = use_kalman
        self.boxes = np.empty(shape=(0, 4))
        self.box_ids = []
        self.last_motion_vectors = np.empty(shape=(0, 10))
        self.next_id = 1
        if self.use_kalman:
            self.filters = []
        self.use_numeric_ids = use_numeric_ids


    def _filter_low_confidence_detections(self, detection_boxes, detection_scores):
        idx = np.nonzero(detection_scores >= self.det_conf_threshold)
        detection_boxes[idx]
        return detection_boxes[idx], detection_scores[idx]


    def update(self, motion_vectors, frame_type, detection_boxes, detection_scores):

        # remove detections with confidence lower than det_conf_threshold
        if self.det_conf_threshold is not None:
            detection_boxes, detection_scores = self._filter_low_confidence_detections(detection_boxes, detection_scores)

        # bring boxes into next state
        self.predict(motion_vectors, frame_type)

        # match predicted (tracked) boxes with detected boxes
        matches, unmatched_trackers, unmatched_detectors = trackerlib.match_bounding_boxes(self.boxes, detection_boxes, self.iou_threshold)

        #print("####")
        #print("unmatched_trackers", unmatched_trackers, [str(self.box_ids[t])[:6] for t in unmatched_trackers])
        #print("unmatched_detectors", unmatched_detectors)

        # handle matches
        for d, t in matches:
            if self.use_kalman:
                self.filters[t].predict()
                self.filters[t].update(detection_boxes[d])
                self.boxes[t] = self.filters[t].get_box_from_state()
            else:
                self.boxes[t] = detection_boxes[d]
            #print("Matched tracker {} with detector {}".format(str(self.box_ids[t])[:6], d))

        # handle unmatched detections by spawning new trackers
        for d in unmatched_detectors:
            if self.use_numeric_ids:
                self.box_ids.append(self.next_id)
                self.next_id += 1
            else:
                uid = uuid.uuid4()
                self.box_ids.append(uid)
            self.boxes = np.vstack((self.boxes, detection_boxes[d]))
            if self.use_kalman:
                filter = trackerlib.Kalman()
                filter.set_initial_state(detection_boxes[d])
                self.filters.append(filter)
            #print("Created new tracker {} for detector {}".format(str(uid)[:6], d))

        # handle unmatched tracker predictions by removing trackers
        for t in unmatched_trackers:
            #print("Removed tracker {}".format(str(self.box_ids[t])[:6]))
            self.boxes = np.delete(self.boxes, t, axis=0)
            self.box_ids.pop(t)
            if self.use_kalman:
                self.filters.pop(t)


    def predict(self, motion_vectors, frame_type):
        # I frame has no motion vectors
        if frame_type != "I":

            if self.use_only_p_vectors:
                motion_vectors = trackerlib.get_vectors_by_source(motion_vectors, "past")
            # get non-zero motion vectors and normalize them to point to the past frame (source = -1)
            motion_vectors = trackerlib.get_nonzero_vectors(motion_vectors)
            motion_vectors = trackerlib.normalize_vectors(motion_vectors)

            self.last_motion_vectors = motion_vectors

        # shift the box edges based on the contained motion vectors
        motion_vector_subsets = trackerlib.get_vectors_in_boxes(self.last_motion_vectors, self.boxes)
        shifts = trackerlib.get_box_shifts(motion_vector_subsets, metric="median")
        self.boxes = trackerlib.adjust_boxes(self.boxes, shifts)

        if self.use_kalman:
            for t in range(len(self.filters)):
                self.filters[t].predict()
                self.filters[t].update(self.boxes[t])
                self.boxes[t] = self.filters[t].get_box_from_state()


    def get_boxes(self):
        return self.boxes

    def get_box_ids(self):
        return self.box_ids
