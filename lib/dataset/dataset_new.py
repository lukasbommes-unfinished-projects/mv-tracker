import os
import pickle
import torch
import cv2
import numpy as np

from video_cap import VideoCap

from lib.dataset.loaders import load_groundtruth
from lib.dataset.motion_vectors import get_vectors_by_source, get_nonzero_vectors, \
    normalize_vectors, motion_vectors_to_image
from lib.dataset.velocities import velocities_from_boxes
from lib.dataset.stats import Stats
from lib.visu import draw_boxes, draw_velocities, draw_motion_vectors
from lib.transforms.transforms import StandardizeMotionVectors


class MotionVectorDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, mode, codec="mpeg4", visu=False, debug=False):

        self.DEBUG = debug  # whether to print debug information

        self.sequences = {
            "train": [
                "MOT17/train/MOT17-02-FRCNN",  # static cam
                "MOT17/train/MOT17-04-FRCNN",  # static cam
                "MOT17/train/MOT17-05-FRCNN",  # moving cam
                "MOT17/train/MOT17-11-FRCNN",  # moving cam
                "MOT17/train/MOT17-13-FRCNN",  # moving cam
                "MOT15/train/ETH-Bahnhof",  # moving cam
                "MOT15/train/ETH-Sunnyday",  # moving cam
                "MOT15/train/KITTI-13",  # moving cam
                "MOT15/train/KITTI-17",  # static cam
                "MOT15/train/PETS09-S2L1",  # static cam
                "MOT15/train/TUD-Campus",  # static cam
                "MOT15/train/TUD-Stadtmitte"  # static cam
            ],
            "val": [
                "MOT17/train/MOT17-09-FRCNN",  # static cam
                "MOT17/train/MOT17-10-FRCNN"  # moving cam
            ]
        }

        self.lens = {
            "train": [600, 1050, 837, 900, 750, 1000,
               354, 340, 145, 795, 71, 179],
            "val": [525, 654]
        }

        # self.lens = {
        #     "train": [600, 1050, 145, 795, 71, 179],
        #     "val": [525]
        # }

        self.scales = [1.0, 0.75, 0.5]

        self.root_dir = root_dir
        self.mode = mode
        self.codec = codec
        self.visu = visu

        self.index = []   # stores (sequence_idx, scale_idx, frame_idx) for available samples

        self.load_groundtruth_()
        self.build_index_()


    def load_groundtruth_(self):
        self.gt_ids = []
        self.gt_boxes = []
        for sequence, num_frames in zip(self.sequences[self.mode], self.lens[self.mode]):
            gt_file = os.path.join(self.root_dir, sequence, "gt/gt.txt")
            gt_ids, gt_boxes, _ = load_groundtruth(gt_file, num_frames, only_eval=True)
            self.gt_ids.append(gt_ids)
            self.gt_boxes.append(gt_boxes)


    def build_index_(self):
        """Generate index of all usable frames of all sequences.

        Usable frames are those which have a ground truth annotation and for
        which the previous frame also has a ground truth annotation. Only those
        ground truth annotation for which the eval flag is set to 1 are considered
        (see `only_eval` parameter in load_groundtruth).

        The index has the format [(0, 0, 2), (0, 0, 3), ..., (2, 2, 2),
        (2, 2, 3), (2, 2, 4)] where the first item of the tuple is the sequence
        index (0-based), the second item is the scale index (0-based) and the
        third item is the frame index (0-based) within this sequence.
        """
        for sequence_idx, sequence in enumerate(self.sequences[self.mode]):
            for scale_idx in range(len(self.scales)):
                last_none = True
                for frame_idx, gt_id in enumerate(self.gt_ids[sequence_idx]):
                    if gt_id is None:
                        last_none = True
                        continue
                    if last_none:
                        last_none = False
                        continue
                    self.index.append((sequence_idx, scale_idx, frame_idx))


    def __len__(self):
        total_len = len(self.index)
        if self.DEBUG:
            print("Overall length of {} dataset: {}".format(self.mode, total_len))
        return total_len


    def load_frame_(self, sequence_idx, scale_idx, frame_idx):
        frame_file = os.path.join(self.root_dir,
            self.sequences[self.mode][sequence_idx], "img1",
            "{:06d}.jpg".format(frame_idx + 1))
        frame = cv2.imread(frame_file, cv2.IMREAD_COLOR)
        scaling_factor = self.scales[scale_idx]
        frame = cv2.resize(frame, None, None, fx=scaling_factor,
            fy=scaling_factor, interpolation=cv2.INTER_LINEAR)
        return frame


    def load_motion_vectors_(self, sequence_idx, scale_idx, frame_idx):
        mvs_file = os.path.join(self.root_dir,
            self.sequences[self.mode][sequence_idx], "mvs-{}-{}".format(
            self.codec, self.scales[scale_idx]), "{:06d}.pkl".format(
            frame_idx + 1))
        data_item = pickle.load(open(mvs_file, "rb"))
        motion_vectors = data_item["motion_vectors"]
        frame_type = data_item["frame_type"]
        return motion_vectors, frame_type


    def __getitem__(self, idx):

        sequence_idx, scale_idx, frame_idx = self.index[idx]
        #sequence_idx, scale_idx, frame_idx = (0, 0, 0)
        frame = self.load_frame_(sequence_idx, scale_idx, frame_idx)
        motion_vectors, frame_type = self.load_motion_vectors_(sequence_idx, scale_idx, frame_idx)

        if self.DEBUG:
            print(("Loaded frame {}, frame_type {}, mvs shape: {}, "
                "frame shape: {}, scale: {}").format(frame_idx + 1, frame_type,
                motion_vectors.shape, frame.shape, self.scales[scale_idx]))

        # convert motion vectors to image (for I frame black image is returned)
        motion_vectors = get_vectors_by_source(motion_vectors, "past")  # get only p vectors
        motion_vectors = normalize_vectors(motion_vectors)
        motion_vectors = get_nonzero_vectors(motion_vectors)
        motion_vectors_copy = np.copy(motion_vectors)
        motion_vectors = motion_vectors_to_image(motion_vectors, (frame.shape[1], frame.shape[0]))
        motion_vectors = torch.from_numpy(motion_vectors).float()

        if self.visu:
            frame = draw_motion_vectors(frame, motion_vectors_copy, format='numpy')
            sequence_name = str.split(self.sequences[self.mode][sequence_idx], "/")[-1]
            cv2.putText(frame, 'Sequence: {}'.format(sequence_name), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, 'Frame Idx: {}'.format(frame_idx + 1), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, 'Frame Type: {}'.format(frame_type), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        # get ground truth boxes and update previous boxes and ids
        gt_boxes = self.gt_boxes[sequence_idx][frame_idx]
        gt_ids = self.gt_ids[sequence_idx][frame_idx]
        gt_boxes_prev = self.gt_boxes[sequence_idx][frame_idx - 1]
        gt_ids_prev = self.gt_ids[sequence_idx][frame_idx - 1]

        # scale boxes
        gt_boxes = gt_boxes * self.scales[scale_idx]
        gt_boxes_prev = gt_boxes_prev * self.scales[scale_idx]

        # match ids with previous ids and compute box velocities
        _, idx_1, idx_0 = np.intersect1d(gt_ids, gt_ids_prev, assume_unique=True, return_indices=True)
        boxes = torch.from_numpy(gt_boxes[idx_1]).float()
        boxes_prev = torch.from_numpy(gt_boxes_prev[idx_0]).float()
        velocities = velocities_from_boxes(boxes_prev, boxes)
        if self.visu:
            frame = draw_boxes(frame, boxes, gt_ids, color=(255, 255, 255))
            frame = draw_boxes(frame, boxes_prev, gt_ids_prev, color=(200, 200, 200))

        # insert frame index into boxes_prev
        num_boxes = boxes_prev.shape[0]
        boxes_prev_tmp = torch.zeros(num_boxes, 5).float()
        boxes_prev_tmp[:, 1:5] = boxes_prev
        boxes_prev = boxes_prev_tmp

        # insert frame index into boxes
        boxes_tmp = torch.zeros(num_boxes, 5).float()
        boxes_tmp[:, 1:5] = boxes
        boxes = boxes_tmp

        if self.visu:
            frame = draw_velocities(frame, boxes[:, 1:], velocities, scale=1000)

        sample = {
            "motion_vectors": motion_vectors,
            "boxes_prev": boxes_prev,
            "boxes": boxes,
            "velocities": velocities
        }

        # print("sequence_idx", sequence_idx, "scale_idx", scale_idx, "frame_idx", frame_idx)
        # print("boxes_prev", boxes_prev, gt_ids_prev)
        # print("boxes", boxes, gt_ids)
        # print("velocities", velocities)

        if self.visu:
            sample["frame"] = frame

        return sample


# run as python -m lib.dataset.dataset from root dir
if __name__ == "__main__":

    batch_size = 1
    codec = "mpeg4"
    datasets = {x: MotionVectorDataset(root_dir='data', codec=codec, visu=True,
        debug=True, mode=x) for x in ["train", "val"]}
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size,
        shuffle=False, num_workers=0) for x in ["train", "val"]}
    stats = Stats()

    transform = StandardizeMotionVectors(mean=stats.motion_vectors["mean"],
        std=stats.motion_vectors["std"])

    step_wise = True

    for batch_idx in range(batch_size):
        cv2.namedWindow("frame-{}".format(batch_idx), cv2.WINDOW_NORMAL)
        cv2.resizeWindow("frame-{}".format(batch_idx), 640, 360)
        cv2.namedWindow("motion_vectors-{}".format(batch_idx), cv2.WINDOW_NORMAL)
        cv2.resizeWindow("motion_vectors-{}".format(batch_idx), 640, 360)

    for step, sample in enumerate(dataloaders["train"]):

        # apply transforms
        sample = transform(sample)

        for batch_idx in range(batch_size):

            frame = sample["frame"][batch_idx].numpy()
            motion_vectors = sample["motion_vectors"][batch_idx].numpy()
            motion_vectors = (motion_vectors - np.min(motion_vectors)) / (np.max(motion_vectors) - np.min(motion_vectors))

            print("step: {}, MVS shape: {}".format(step, motion_vectors.shape))

            cv2.imshow("frame-{}".format(batch_idx), frame)
            cv2.imshow("motion_vectors-{}".format(batch_idx), motion_vectors)

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
