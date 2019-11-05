import os
import pickle
import glob
import torch
import torchvision
import cv2
import numpy as np

from lib.dataset.loaders import load_groundtruth
from lib.dataset.motion_vectors import get_vectors_by_source, get_nonzero_vectors, \
    normalize_vectors, motion_vectors_to_image, motion_vectors_to_grid, \
    motion_vectors_to_grid_interpolated
from lib.dataset.velocities import velocities_from_boxes, velocities_from_boxes_2d
from lib.visu import draw_boxes, draw_velocities, draw_motion_vectors

# for testing
from lib.dataset.stats import StatsMpeg4DenseStaticSinglescale as Stats
from lib.transforms.transforms import StandardizeMotionVectors, \
    StandardizeVelocities, RandomFlip, RandomMotionChange


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

        scales (`list` of `float`): Compose the dataset of videos with different
            scales as specified in this list. E.g. [1.0, 0.75, 0.5] means, the
            video snippets are available at 100 %, 75 % and 50 % of their
            original size. Ensure that the scaled videos were created previously
            with the `video_write.py` and `extract_mvs.py` scripts inside the
            data folder.

        mvs_mode (`str`): Either "upsampled" or "dense". In "upsampled" mode
            motion vectors are encoded as image with same dimensions as the
            original video frame. In "dense" mode motion vectors are not
            upsampled and a compact representation in form of an image is
            generated. In this representation each pixel corresponds to a
            macroblock in the original frame. In both cases, green channel is
            used for y motion, red channel for x motion and blue channel is
            always zero.

        static_only (`bool`): If True use only those videos in MOT15 and MOT17
            which have a static (not moving) camera.

        exclude_keyframes (`bool`): If True, keyframes (frame type "I") are
            excluded from the dataset.

        visu (`bool`): If True show frames, motion vectors and boxes of each
            sample while iterating over the dataset. Activating this will slow
            down the data loading.

        debug (`bool`): If True print debug information.

    """
    def __init__(self, root_dir="data", mode="train", codec="mpeg4",
        transforms=None, scales=[1.0, 0.75, 0.5], mvs_mode="upsampled",
        static_only=False, exclude_keyframes=True, visu=False, debug=False):

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
        self.transforms = transforms
        self.mode = mode
        self.codec = codec
        self.scales = scales
        self.mvs_mode = mvs_mode
        self.exclude_keyframes = exclude_keyframes
        self.visu = visu

        self.index = []   # stores (sequence_idx, scale_idx, frame_idx) for available samples

        self.get_sequence_lengths_()
        self.load_groundtruth_()
        if self.DEBUG:
            print("Loaded ground truth files.")
        self.build_index_()
        if self.DEBUG:
            print("Built dataset index.")


    def get_sequence_lengths_(self):
        """Determine number of frames in each video sequence."""
        self.lens = []
        for sequence in self.sequences[self.mode]:
            frame_files = glob.glob(os.path.join(self.root_dir, sequence, "img1/*.jpg"))
            self.lens.append(len(frame_files))


    def load_groundtruth_(self):
        """Load ground truth boxes and IDs from annotation files."""
        self.gt_ids = []
        self.gt_boxes = []
        for sequence, num_frames in zip(self.sequences[self.mode], self.lens):
            gt_file = os.path.join(self.root_dir, sequence, "gt/gt.txt")
            gt_ids, gt_boxes, _ = load_groundtruth(gt_file, num_frames, only_eval=True)
            self.gt_ids.append(gt_ids)
            self.gt_boxes.append(gt_boxes)


    def load_frame_(self, sequence_idx, scale_idx, frame_idx):
        """Load, scale and return a single video frame."""
        frame_file = os.path.join(self.root_dir,
            self.sequences[self.mode][sequence_idx], "img1",
            "{:06d}.jpg".format(frame_idx + 1))
        frame = cv2.imread(frame_file, cv2.IMREAD_COLOR)
        scaling_factor = self.scales[scale_idx]
        frame = cv2.resize(frame, None, None, fx=scaling_factor,
            fy=scaling_factor, interpolation=cv2.INTER_LINEAR)
        return frame


    def load_frame_data_(self, sequence_idx, scale_idx, frame_idx):
        """Load and return motion vectors and frame type of a specific frame."""
        mvs_file = os.path.join(self.root_dir,
            self.sequences[self.mode][sequence_idx], "mvs-{}-{}".format(
            self.codec, self.scales[scale_idx]), "{:06d}.pkl".format(
            frame_idx + 1))
        data_item = pickle.load(open(mvs_file, "rb"))
        motion_vectors = data_item["motion_vectors"]
        frame_type = data_item["frame_type"]
        return motion_vectors, frame_type


    def build_index_(self):
        """Generate index of all usable frames of all sequences.

        Usable frames are those which have a ground truth annotation and for
        which the previous frame also has a ground truth annotation. Only those
        ground truth annotation for which the eval flag is set to 1 are considered
        (see `only_eval` parameter in load_groundtruth). If `excluse_keyframes`
        is True keyframes (frame type "I") are also excluded from the index.

        The index has the format [(0, 0, 2), (0, 0, 3), ..., (2, 2, 2),
        (2, 2, 3), (2, 2, 4)] where the first item of the tuple is the sequence
        index (0-based), the second item is the scale index (0-based) and the
        third item is the frame index (0-based) within this sequence.
        """
        for sequence_idx, sequence in enumerate(self.sequences[self.mode]):
            for scale_idx in range(len(self.scales)):
                last_none = True
                for frame_idx in range(len(self.gt_ids[sequence_idx])):
                    gt_ids = self.gt_ids[sequence_idx][frame_idx]
                    gt_ids_prev = self.gt_ids[sequence_idx][frame_idx - 1]
                    # exclude frames without gt annotation from index
                    if gt_ids is None:
                        last_none = True
                        continue
                    if last_none:
                        last_none = False
                        continue
                    # check if ids can be matched, otherwise exclude frame
                    _, idx_1, idx_0 = np.intersect1d(gt_ids, gt_ids_prev,
                        assume_unique=True, return_indices=True)
                    if len(idx_1) == 0 and len(idx_0) == 0:
                        continue
                    # exclude key frames from index
                    if self.exclude_keyframes:
                        _, frame_type = self.load_frame_data_(sequence_idx,
                            scale_idx, frame_idx)
                        if frame_type == "I":
                            continue
                    self.index.append((sequence_idx, scale_idx, frame_idx))


    def __len__(self):
        """Return the total length of the dataset."""
        total_len = len(self.index)
        if self.DEBUG:
            print("Overall length of {} dataset: {}".format(self.mode, total_len))
        return total_len


    def __getitem__(self, idx):
        """Retrieve item with index `idx` from the dataset."""
        sequence_idx, scale_idx, frame_idx = self.index[idx]
        frame = self.load_frame_(sequence_idx, scale_idx, frame_idx)
        motion_vectors, frame_type = self.load_frame_data_(sequence_idx, scale_idx, frame_idx)

        if self.DEBUG:
            print(("Loaded frame {}, frame_type {}, mvs shape: {}, "
                "frame shape: {}, scale: {}").format(frame_idx + 1, frame_type,
                motion_vectors.shape, frame.shape, self.scales[scale_idx]))

        # convert motion vectors to image (for I frame black image is returned)
        motion_vectors = get_vectors_by_source(motion_vectors, "past")  # get only p vectors
        motion_vectors = normalize_vectors(motion_vectors)
        motion_vectors_for_visu = np.copy(get_nonzero_vectors(motion_vectors))

        if self.mvs_mode == "upsampled":
            motion_vectors = get_nonzero_vectors(motion_vectors)
            motion_vectors = motion_vectors_to_image(motion_vectors, (frame.shape[1], frame.shape[0]))
        elif self.mvs_mode == "dense":
            if self.codec == "mpeg4":
                motion_vectors = motion_vectors_to_grid(motion_vectors, (frame.shape[1], frame.shape[0]))
            elif self.codec == "h264":
                motion_vectors = motion_vectors_to_grid_interpolated(motion_vectors, (frame.shape[1], frame.shape[0]))

        motion_vectors = torch.from_numpy(motion_vectors).float()

        if self.visu:
            frame = draw_motion_vectors(frame, motion_vectors_for_visu, format='numpy')
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
        if self.mvs_mode == "upsampled":
            velocities = velocities_from_boxes(boxes_prev, boxes)
        elif self.mvs_mode == "dense":
            velocities = velocities_from_boxes_2d(boxes_prev, boxes)
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

        # scale down boxes by factor 16 if using dense mvs_mode
        if self.mvs_mode == "dense":
            boxes = boxes / 16.0
            boxes_prev = boxes_prev / 16.0

        sample = {
            "motion_vectors": motion_vectors,
            "boxes_prev": boxes_prev,
            "boxes": boxes,
            "velocities": velocities
        }

        if self.transforms:
            sample = self.transforms(sample)

        # swap channel order of motion vectors from BGR to RGB
        sample["motion_vectors"] = sample["motion_vectors"][..., [2, 1, 0]]

        # swap motion vector axes so that shape is (C, H, W) instead of (H, W, C)
        sample["motion_vectors"] = sample["motion_vectors"].permute(2, 0, 1)

        if self.visu:
            sample["frame"] = frame

        return sample


# run as python -m lib.dataset.dataset from root dir
if __name__ == "__main__":

    batch_size = 1
    codec = "mpeg4"
    mvs_mode = "dense"
    static_only = True
    exclude_keyframes = True
    scales = [1.0]

    transforms = {
        "train": torchvision.transforms.Compose([
            RandomFlip(directions=["x", "y"]),
            StandardizeVelocities(mean=Stats.velocities["mean"], std=Stats.velocities["std"]),
            StandardizeMotionVectors(mean=Stats.motion_vectors["mean"], std=Stats.motion_vectors["std"]),
            RandomMotionChange(scale=1.0),
        ]),
        "val": torchvision.transforms.Compose([
            StandardizeVelocities(mean=Stats.velocities["mean"], std=Stats.velocities["std"]),
            StandardizeMotionVectors(mean=Stats.motion_vectors["mean"], std=Stats.motion_vectors["std"]),
        ])
    }

    datasets = {x: MotionVectorDataset(root_dir='data', transforms=transforms[x],
        codec=codec, scales=scales, mvs_mode=mvs_mode, static_only=static_only,
        exclude_keyframes=exclude_keyframes, visu=True, debug=True,
        mode=x) for x in ["train", "val"]}

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size,
        shuffle=False, num_workers=0) for x in ["train", "val"]}

    step_wise = True

    for batch_idx in range(batch_size):
        cv2.namedWindow("frame-{}".format(batch_idx), cv2.WINDOW_NORMAL)
        cv2.resizeWindow("frame-{}".format(batch_idx), 640, 360)
        cv2.namedWindow("motion_vectors-{}".format(batch_idx), cv2.WINDOW_NORMAL)
        cv2.resizeWindow("motion_vectors-{}".format(batch_idx), 640, 360)

    for step, sample in enumerate(dataloaders["train"]):

        #pickle.dump(sample, open("overfit_dense_model/data/{:06d}".format(step), "wb"))

        for batch_idx in range(batch_size):

            frame = sample["frame"][batch_idx].numpy()
            motion_vectors = sample["motion_vectors"][batch_idx]
            boxes_prev = sample["boxes_prev"][batch_idx]
            boxes = sample["boxes"][batch_idx]
            velocities = sample["velocities"][batch_idx]

            motion_vectors = motion_vectors.permute(1, 2, 0)
            motion_vectors = motion_vectors[..., [2, 1, 0]]
            motion_vectors = motion_vectors.numpy()
            motion_vectors = (motion_vectors - np.min(motion_vectors)) / (np.max(motion_vectors) - np.min(motion_vectors))

            # draw boxes on motion vector image
            #motion_vectors = draw_boxes(motion_vectors, boxes_prev[:, 1:], None, color=(200, 200, 200))
            #motion_vectors = draw_boxes(motion_vectors, boxes[:, 1:], None, color=(255, 255, 255))
            #motion_vectors = draw_velocities(motion_vectors, boxes[:, 1:], velocities, scale=1000)

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
