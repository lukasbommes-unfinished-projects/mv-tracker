import os
import torch
import torchvision
import cv2
import numpy as np

from video_cap import VideoCap

from lib.dataset.loaders import load_detections
from lib.dataset.motion_vectors import get_vectors_by_source, get_nonzero_vectors, \
    normalize_vectors, motion_vectors_to_image
from lib.dataset.stats import StatsMpeg4UpsampledFull as Stats
from lib.visu import draw_boxes, draw_motion_vectors
from lib.transforms.transforms import StandardizeMotionVectors, ScaleImage

class MotionVectorDatasetTest(torch.utils.data.Dataset):
    def __init__(self, root_dir, codec="mpeg4", visu=False, debug=False):

        self.DEBUG = debug  # whether to print debug information

        self.sequences = [
            "MOT17/test/MOT17-01-FRCNN",  # static cam
            "MOT17/test/MOT17-03-FRCNN",  # static cam
            "MOT17/test/MOT17-06-FRCNN",  # moving cam
            "MOT17/test/MOT17-07-FRCNN",  # moving cam
            "MOT17/test/MOT17-08-FRCNN",  # static cam
            "MOT17/test/MOT17-12-FRCNN",  # moving cam
            "MOT17/test/MOT17-14-FRCNN"  # moving cam
        ]

        #self.lens = [450, 1500, 1194, 500, 625, 900, 750]
        self.lens = [450, 1500, 625]

        self.root_dir = root_dir
        self.codec = codec
        self.visu = visu

        self.detections = None
        self.det_boxes_prev = None  # store for next iteration

        self.caps = []
        self.is_open = []
        for _ in self.sequences:
            cap = VideoCap()
            self.caps.append(cap)
            self.is_open.append(False)

        self.current_seq_id = 0
        self.current_frame_idx = 0


    def __len__(self):
        total_len = sum(self.lens) - len(self.lens)  # first frame in sequence is skipped
        if self.DEBUG:
            print("Overall length of test dataset: {}".format(total_len))
        return total_len


    def __getitem__(self, idx):

        while True:

            if self.DEBUG:
                print("Getting item idx {}, Current sequence idx {}, Current frame idx {}".format(idx, self.current_seq_id, self.current_frame_idx))

            # when the end of the sequence is reached switch to the next one
            if self.current_frame_idx == self.lens[self.current_seq_id]:
                if self.DEBUG:
                    print("Sequence {} is being closed...".format(self.sequences[self.current_seq_id]))
                self.caps[self.current_seq_id].release()
                self.is_open[self.current_seq_id] = False
                self.current_frame_idx = 0
                self.current_seq_id += 1
                # make sure the sequence index wraps around at the number of sequences
                if self.current_seq_id == len(self.sequences):
                    self.current_seq_id = 0
                if self.DEBUG:
                    print("Updated sequence id to {} and frame index to {}".format(self.current_seq_id, self.current_frame_idx))
                continue

            # this block is executed only once when a new sequence starts
            if not self.is_open[self.current_seq_id]:
                if self.DEBUG:
                    print("Sequence {} is being opened...".format(self.sequences[self.current_seq_id]))

                # open detections files
                detections_file = os.path.join(self.root_dir, self.sequences[self.current_seq_id], "det/det.txt")
                self.detections = load_detections(detections_file, num_frames=self.lens[self.current_seq_id])
                if self.DEBUG:
                    print("Detections loaded")

                # open the video sequence and drop frame
                sequence_name = str.split(self.sequences[self.current_seq_id], "/")[-1]
                video_file = os.path.join(self.root_dir, self.sequences[self.current_seq_id], "{}-{}.mp4".format(sequence_name, self.codec))
                if self.DEBUG:
                    print("Opening video file {} of sequence {}".format(video_file, sequence_name))
                ret = self.caps[self.current_seq_id].open(video_file)
                if not ret:
                    raise RuntimeError("Could not open the video file")
                if self.DEBUG:
                    print("Opened the video file")

                # drop the first frame
                ret, _, _, _, _ = self.caps[self.current_seq_id].read()
                if not ret:  # should never happen
                    raise RuntimeError("Could not read first frame from video")

                self.det_boxes_prev = self.detections[0]

                self.is_open[self.current_seq_id] = True
                self.current_frame_idx += 1
                if self.DEBUG:
                    print("Incremented frame index to {}".format(self.current_frame_idx))
                continue

            # load, process and return the next sample
            ret, frame, motion_vectors, frame_type, _ = self.caps[self.current_seq_id].read()
            if not ret:  # should never happen
                raise RuntimeError("Could not read next frame from video")
            if self.DEBUG:
                print("got frame, frame_type {}, mvs shape: {}, frame shape: {}".format(frame_type, motion_vectors.shape, frame.shape))

            # convert motion vectors to image (for I frame black image is returned)
            motion_vectors = get_vectors_by_source(motion_vectors, "past")  # get only p vectors
            motion_vectors = normalize_vectors(motion_vectors)
            motion_vectors = get_nonzero_vectors(motion_vectors)
            motion_vectors_copy = np.copy(motion_vectors)
            motion_vectors = motion_vectors_to_image(motion_vectors, (frame.shape[1], frame.shape[0]))
            motion_vectors = torch.from_numpy(motion_vectors).float()

            if self.visu:
                frame = draw_motion_vectors(frame, motion_vectors_copy, format='numpy')
                sequence_name = str.split(self.sequences[self.current_seq_id], "/")[-1]
                cv2.putText(frame, 'Sequence: {}'.format(sequence_name), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, 'Frame Idx: {}'.format(self.current_frame_idx), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, 'Frame Type: {}'.format(frame_type), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

            # get detection boxes and update previous boxes
            det_boxes = self.detections[self.current_frame_idx]
            det_boxes_prev_ = np.copy(self.det_boxes_prev)
            self.det_boxes_prev = det_boxes

            det_boxes_prev = torch.from_numpy(det_boxes_prev_).float()
            if self.visu:
                frame = draw_boxes(frame, det_boxes_prev, None, color=(50, 255, 50))

            # insert frame index into det_boxes
            num_det_boxes = (det_boxes_prev.shape)[0]
            det_boxes_prev_tmp = torch.zeros(num_det_boxes, 5).float()
            det_boxes_prev_tmp[:, 1:5] = det_boxes_prev
            det_boxes_prev_tmp[:, 0] = torch.full((num_det_boxes,), self.current_frame_idx).float()
            det_boxes_prev = det_boxes_prev_tmp

            self.current_frame_idx += 1

            sample = {
                "motion_vectors": motion_vectors,
                "det_boxes_prev": det_boxes_prev,
            }

            if self.visu:
                sample["frame"] = frame

            return sample


# run as python -m lib.dataset.dataset from root dir
if __name__ == "__main__":
    batch_size = 1
    codec = "mpeg4"
    dataset = MotionVectorDatasetTest(root_dir='data', codec=codec, visu=True, debug=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    stats = Stats()

    transforms = torchvision.transforms.Compose([
        StandardizeMotionVectors(mean=Stats.motion_vectors["mean"], std=Stats.motion_vectors["std"]),
        #RandomScaleImage(items=["motion_vectors", "frame"], scale=600, max_size=1000),
    ])

    step_wise = False

    for batch_idx in range(batch_size):
        cv2.namedWindow("frame-{}".format(batch_idx), cv2.WINDOW_NORMAL)
        cv2.resizeWindow("frame-{}".format(batch_idx), 640, 360)
        cv2.namedWindow("motion_vectors-{}".format(batch_idx), cv2.WINDOW_NORMAL)
        cv2.resizeWindow("motion_vectors-{}".format(batch_idx), 640, 360)

    for step, sample in enumerate(dataloader):

        # apply transforms
        sample = transforms(sample)

        for batch_idx in range(batch_size):

            frames = sample["frame"][batch_idx]
            motion_vectors = sample["motion_vectors"][batch_idx]
            det_boxes_prev = sample["det_boxes_prev"][batch_idx]

            frame = frames.numpy()
            motion_vectors = motion_vectors.numpy()
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
