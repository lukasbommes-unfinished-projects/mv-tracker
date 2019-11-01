import os
import glob
import torch
import cv2
import numpy as np

from video_cap import VideoCap

from lib.dataset.loaders import load_groundtruth
from lib.dataset.motion_vectors import get_vectors_by_source, get_nonzero_vectors, \
    normalize_vectors, motion_vectors_to_image
from lib.dataset.velocities import velocities_from_boxes
from lib.visu import draw_boxes, draw_velocities, draw_motion_vectors

# for testing
from lib.dataset.stats import StatsMpeg4UpsampledFull as Stats
from lib.transforms.transforms import StandardizeMotionVectors


class MotionVectorDataset(torch.utils.data.Dataset):
    """Dataset for object tracking in compressed video domain.

    Args:
        root_dir (`str`): The relative path of the dataset root. The directory
            should containe the following subdirectory structure whereby the
            train and test folders contain the corresponding MOT dataset:
            <root_dir>
                |
                |---MOT15/
                |     |---train/
                |     |---test/
                |
                |---MOT16/
                |     |---train/
                |     |---test/
                |
                |---MOT17/
                      |---train/
                      |---test/

        mode (`str`): Either "train" or "val". Specifies which split of the data
            set to iterate over. Refer to the `self.sequences` attribute below
            to see which videos are contained in each split.

        codec (`str`): Either "mpeg4" or "h264". Determines whether motion
            vectors are loaded from the mpeg4 or h264 encoded video. Ensure to
            provide the raw data by setting the codec in the `video_write.py`
            script accordingly and running the script.

        transforms (`torchvision.transform` of `None`): Transformations which
            are applied to the generated samples. Can be either a single
            transformation or a several transformations composed with
            `torchvision.transforms.Compose`. For available transformations see
            the module `lib.transforms.transforms`. If None no transformation
            is applied.

        static_only (`bool`): If True use only those videos in MOT15 and MOT17
            which have a static (not moving) camera.

        visu (`bool`): If True show frames, motion vectors and boxes of each
            sample while iterating over the dataset. Activating this will slow
            down the data loading.

        debug (`bool`): If True print debug information.
    """
    def __init__(self, root_dir, mode, batch_size, codec="mpeg4",
        static_only=False, pad_num_boxes=54, visu=False, debug=False):

        self.DEBUG = debug  # whether to print debug information

        if static_only:
            self.sequences = {
                "train": [
                    "MOT17/train/MOT17-02-FRCNN",  # static cam
                    "MOT17/train/MOT17-04-FRCNN",  # static cam
                    "MOT15/train/KITTI-17",  # static cam
                    "MOT15/train/PETS09-S2L1",  # static cam
                    "MOT15/train/TUD-Campus",  # static cam
                    "MOT15/train/TUD-Stadtmitte"  # static cam
                ],
                "val": [
                    "MOT17/train/MOT17-09-FRCNN",  # static cam
                ]
            }

        else:
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

        self.root_dir = root_dir
        self.mode = mode
        self.codec = codec
        self.pad_num_boxes = pad_num_boxes
        self.visu = visu
        self.batch_size = batch_size

        self.get_sequence_lengths_()

        self.gt_boxes_prev = None  # store for next iteration
        self.gt_ids_prev = None  # store for next iteration
        self.gt_ids_all = None
        self.gt_boxes_all = None

        self.caps = []
        self.is_open = []
        for _ in self.sequences[self.mode]:
            cap = VideoCap()
            self.caps.append(cap)
            self.is_open.append(False)

        self.current_seq_id = 0
        self.current_frame_idx = 0
        self.frame_return_count = 0  # number of frames returned
        self.last_gt_was_none = True  # flag that is True when the previous frame had no gt annotations,
                                      # set initially to True because frame prior to first frame has no annotation

        self.compute_truncated_length_()

    def get_sequence_lengths_(self):
        """Determine number of frames in each video sequence."""
        self.lens = []
        for sequence in self.sequences[self.mode]:
            frame_files = glob.glob(os.path.join(self.root_dir, sequence, "img1/*.jpg"))
            self.lens.append(len(frame_files))


    def compute_truncated_length_(self):
        """Compute the number of usable frames for each sequence.

        Usable frames are those which have a ground truth annotation and for
        which the previous frame also had a ground truth annotation. Only those
        ground truth annotation for which the eval flag is set to 1 are considered
        (see `only_eval` parameter in load_groundtruth).

        The number of usable frames is stored in the attribute lens_truncated.

        The function also considers the batch_size. The trunacted lenghts are
        corrected so that the number of usable frames can be split into an integer
        number of batches, that is: number_of_frames % batch_size == 0
        """
        self.lens_truncated = []
        for seq_id, sequence in enumerate(self.sequences[self.mode]):
            # open gt file
            gt_file = os.path.join(self.root_dir, sequence, "gt/gt.txt")
            gt_ids, gt_boxes, _ = load_groundtruth(gt_file, num_frames=self.lens[seq_id], only_eval=True)
            # count number of usable frames in the sequence
            count = 0
            last_gt_was_none = True
            for index in range(len(gt_boxes)):
                if gt_boxes[index] is None:
                    last_gt_was_none = True
                    continue
                if last_gt_was_none:
                    last_gt_was_none = False
                    continue
                count += 1
            # consider batch size truncation
            count = count - (count % self.batch_size)
            if self.DEBUG:
                print("Sequence {} has {} usable frames".format(sequence, count))
            self.lens_truncated.append(count)


    def __len__(self):
        total_len = sum(self.lens_truncated)
        if self.DEBUG:
            print("Overall length of {} dataset: {}".format(self.mode, total_len))
        return total_len


    def __getitem__(self, idx):

        while True:

            if self.DEBUG:
                print("Getting item idx {}, Current sequence idx {}, Current frame idx {}, Current frame return count {}".format(idx, self.current_seq_id, self.current_frame_idx, self.frame_return_count))

            # when the end of the sequence is reached switch to the next one
            if self.frame_return_count == self.lens_truncated[self.current_seq_id]:
                if self.DEBUG:
                    print("Sequence {} is being closed...".format(self.sequences[self.mode][self.current_seq_id]))
                self.caps[self.current_seq_id].release()
                self.is_open[self.current_seq_id] = False
                self.current_frame_idx = 0
                self.current_seq_id += 1
                # make sure the sequence index wraps around at the number of sequences
                if self.current_seq_id == len(self.sequences[self.mode]):
                    self.current_seq_id = 0
                if self.DEBUG:
                    print("Updated sequence id to {} and frame index to {}".format(self.current_seq_id, self.current_frame_idx))
                continue

            # this block is executed only once when a new sequence starts
            if not self.is_open[self.current_seq_id]:
                if self.DEBUG:
                    print("Sequence {} is being opened...".format(self.sequences[self.mode][self.current_seq_id]))

                # open groundtruth files
                #self.detections = load_detections(detections_file, num_frames=self.lens[self.mode][self.current_seq_id])
                gt_file = os.path.join(self.root_dir, self.sequences[self.mode][self.current_seq_id], "gt/gt.txt")
                self.gt_ids_all, self.gt_boxes_all, _ = load_groundtruth(gt_file, num_frames=self.lens[self.current_seq_id], only_eval=True)
                if self.DEBUG:
                    print("Groundtruth files loaded")

                # open the video sequence and drop frame
                sequence_name = str.split(self.sequences[self.mode][self.current_seq_id], "/")[-1]
                video_file = os.path.join(self.root_dir, self.sequences[self.mode][self.current_seq_id], "{}-{}.mp4".format(sequence_name, self.codec))
                if self.DEBUG:
                    print("Opening video file {} of sequence {}".format(video_file, sequence_name))
                ret = self.caps[self.current_seq_id].open(video_file)
                if not ret:
                    raise RuntimeError("Could not open the video file")
                if self.DEBUG:
                    print("Opened the video file")

                self.frame_return_count = 0  # reset return counter for the new sequence

                # if the first frame does not contain any valid annotation, loop
                # through the following frames until the first valid annotation
                self.last_gt_was_none = True
                while True:

                    ret, _, _, _, _ = self.caps[self.current_seq_id].read()
                    if not ret:  # should never happen
                        raise RuntimeError("Could not read first frame from video")

                    # read and store first set of ground truth boxes and ids
                    if self.gt_boxes_all[self.current_frame_idx] is None:
                        if self.DEBUG:
                            print("gt is None", self.last_gt_was_none)
                        self.last_gt_was_none = True
                        self.current_frame_idx += 1
                        continue

                    if self.last_gt_was_none:
                        self.last_gt_was_none = False
                        self.gt_boxes_prev = self.gt_boxes_all[self.current_frame_idx]
                        self.gt_ids_prev = self.gt_ids_all[self.current_frame_idx]
                        if self.DEBUG:
                            print("Storing boxes prev for frame_idx ", self.current_frame_idx)
                        break

                self.is_open[self.current_seq_id] = True
                self.current_frame_idx += 1
                if self.DEBUG:
                    print("Incremented frame index to {}".format(self.current_frame_idx))
                continue

            # otherwise just load, process and return the next sample
            ret, frame, motion_vectors, frame_type, _ = self.caps[self.current_seq_id].read()
            if not ret:  # should never happen
                raise RuntimeError("Could not read next frame from video")
            if self.DEBUG:
                print("got frame, frame_type {}, mvs shape: {}, frame shape: {}".format(frame_type, motion_vectors.shape, frame.shape))

            if self.DEBUG:
                print(self.gt_ids_all[self.current_frame_idx])

            # if there is no ground truth annotation for this frame skip it
            if self.gt_boxes_all[self.current_frame_idx] is None:
                if self.DEBUG:
                    print("gt is None", self.last_gt_was_none)
                self.last_gt_was_none = True
                self.current_frame_idx += 1
                continue

            # when the next frame with gt annotation is read, store prev_boxes and prev_ids but still skip this frame
            if self.last_gt_was_none:
                self.last_gt_was_none = False
                self.gt_boxes_prev = self.gt_boxes_all[self.current_frame_idx]
                self.gt_ids_prev = self.gt_ids_all[self.current_frame_idx]
                if self.DEBUG:
                    print("Storing boxes prev for frame_idx ", self.current_frame_idx)
                self.current_frame_idx += 1
                continue

            # convert motion vectors to image (for I frame black image is returned)
            motion_vectors = get_vectors_by_source(motion_vectors, "past")  # get only p vectors
            motion_vectors = normalize_vectors(motion_vectors)
            motion_vectors = get_nonzero_vectors(motion_vectors)
            motion_vectors_copy = np.copy(motion_vectors)
            motion_vectors = motion_vectors_to_image(motion_vectors, (frame.shape[1], frame.shape[0]))
            motion_vectors = torch.from_numpy(motion_vectors).float()

            if self.visu:
                frame = draw_motion_vectors(frame, motion_vectors_copy, format='numpy')
                sequence_name = str.split(self.sequences[self.mode][self.current_seq_id], "/")[-1]
                cv2.putText(frame, 'Sequence: {}'.format(sequence_name), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, 'Frame Idx: {}'.format(self.current_frame_idx), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, 'Frame Type: {}'.format(frame_type), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

            # get ground truth boxes and update previous boxes and ids
            gt_boxes = self.gt_boxes_all[self.current_frame_idx]
            gt_ids = self.gt_ids_all[self.current_frame_idx]
            gt_boxes_prev_ = np.copy(self.gt_boxes_prev)
            gt_ids_prev_ = np.copy(self.gt_ids_prev)
            self.gt_boxes_prev = gt_boxes
            self.gt_ids_prev = gt_ids

            # match ids with previous ids and compute box velocities
            _, idx_1, idx_0 = np.intersect1d(gt_ids, gt_ids_prev_, assume_unique=True, return_indices=True)
            boxes = torch.from_numpy(gt_boxes[idx_1]).float()
            boxes_prev = torch.from_numpy(gt_boxes_prev_[idx_0]).float()
            velocities = velocities_from_boxes(boxes_prev, boxes)
            if self.visu:
                frame = draw_boxes(frame, boxes, gt_ids, color=(255, 255, 255))
                frame = draw_boxes(frame, boxes_prev, gt_ids_prev_, color=(200, 200, 200))

            # pad boxes_prev to the same global length as specified by pad_num_boxes
            num_boxes = (boxes.shape)[0]
            boxes_prev_padded = torch.zeros(self.pad_num_boxes, 4).float()
            boxes_prev_padded[:num_boxes, :] = boxes_prev
            boxes_prev = boxes_prev_padded

            # pad boxes
            boxes_padded = torch.zeros(self.pad_num_boxes, 4).float()
            boxes_padded[:num_boxes, :] = boxes
            boxes = boxes_padded

            # pad velocites
            velocities_padded = torch.zeros(self.pad_num_boxes, 4).float()
            velocities_padded[:num_boxes, :] = velocities
            velocities = velocities_padded

            # insert frame index into boxes_prev
            boxes_prev_tmp = torch.zeros(self.pad_num_boxes, 5).float()
            boxes_prev_tmp[:, 1:5] = boxes_prev
            boxes_prev_tmp[:, 0] = torch.full((self.pad_num_boxes,), self.current_frame_idx).float()
            boxes_prev = boxes_prev_tmp

            # insert frame index into boxes
            boxes_tmp = torch.zeros(self.pad_num_boxes, 5).float()
            boxes_tmp[:, 1:5] = boxes
            boxes_tmp[:, 0] = torch.full((self.pad_num_boxes,), self.current_frame_idx).float()
            boxes = boxes_tmp

            if self.visu:
                frame = draw_velocities(frame, boxes[:, 1:], velocities, scale=1000)

            # create a mask to revert the padding at a later stage
            num_boxes_mask = torch.zeros(self.pad_num_boxes,).bool()
            num_boxes_mask[0:num_boxes] = torch.ones(num_boxes,).bool()

            self.current_frame_idx += 1
            self.frame_return_count += 1

            sample = {
                "motion_vectors": motion_vectors,
                "boxes_prev": boxes_prev,
                "boxes": boxes,
                "velocities": velocities,
                "num_boxes_mask": num_boxes_mask,
            }

            if self.visu:
                sample["frame"] = frame

            return sample


# run as python -m lib.dataset.dataset from root dir
if __name__ == "__main__":
    batch_size = 2
    codec = "mpeg4"
    datasets = {x: MotionVectorDataset(root_dir='data', batch_size=batch_size,
        codec=codec, visu=True, debug=True,
        mode=x) for x in ["train", "val"]}
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size,
        shuffle=False, num_workers=0) for x in ["train", "val"]}
    stats = Stats()

    transform = StandardizeMotionVectors(mean=stats.motion_vectors["mean"],
        std=stats.motion_vectors["std"])

    step_wise = False

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
