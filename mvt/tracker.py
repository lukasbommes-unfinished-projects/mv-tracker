import uuid

import numpy as np
import cv2
import pickle

from mvt import trackerlib
from mvt.utils import draw_motion_vectors, draw_boxes


class MotionVectorTracker:
    def __init__(self, iou_threshold, vector_type="p"):
        self.iou_threshold = iou_threshold
        self.vector_type = vector_type
        self.boxes = np.empty(shape=(0, 4))
        self.box_ids = []
        self.last_motion_vectors = np.empty(shape=(0, 10))


    def update(self, motion_vectors, frame_type, detection_boxes):

        # bring boxes into next state
        self.predict(motion_vectors, frame_type)

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


    def predict(self, motion_vectors, frame_type):

        # I frame has no motion vectors
        if frame_type != "I":

            # get non-zero motion vectors and normalize them to point to the past frame (source = -1)
            motion_vectors = trackerlib.get_nonzero_vectors(motion_vectors)
            print(motion_vectors[:20, :])
            motion_vectors = trackerlib.normalize_vectors(motion_vectors)
            print(motion_vectors[:20, :])

            self.last_motion_vectors = motion_vectors

        # shift the box edges based on the contained motion vectors
        motion_vector_subsets = trackerlib.get_vectors_in_boxes(self.last_motion_vectors, self.boxes)
        shifts = trackerlib.get_box_shifts(motion_vector_subsets, metric="median")
        self.boxes = trackerlib.adjust_boxes(self.boxes, shifts)


    def get_boxes(self):
        return self.boxes

    def get_box_ids(self):
        return self.box_ids
