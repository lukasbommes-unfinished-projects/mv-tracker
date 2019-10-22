import os
import glob
import pickle
import cv2
import torch
import torchvision

from lib.transforms.transforms import StandardizeMotionVectors, \
    StandardizeVelocities, ScaleImage, RandomScaleImage, RandomFlip, RandomMotionChange
from lib.dataset.stats import Stats
from lib.visu import draw_boxes, draw_velocities


class MotionVectorDatasetPrecomputed(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.items = ["motion_vectors", "boxes_prev", "boxes", "velocities",
            "num_boxes_mask", "det_boxes_prev"]
        self.dirs = {}
        for item in self.items:
            self.dirs[item] = os.path.join(root_dir, item)
        # get dataset length
        self.length = len(glob.glob(os.path.join(self.dirs[self.items[0]], "*.pkl")))

        # prepare transforms
        self.transforms = torchvision.transforms.Compose([
            RandomFlip(directions=["x", "y"]),
            StandardizeVelocities(mean=Stats.velocities["mean"], std=Stats.velocities["std"]),
            StandardizeMotionVectors(mean=Stats.motion_vectors["mean"], std=Stats.motion_vectors["std"]),
            RandomMotionChange(scale=1.0),
            #ScaleImage(items=["motion_vectors"], scale=600, max_size=1000),
            #RandomScaleImage(items=["motion_vectors"], scales=[300, 400, 500, 600, 700, 800, 900, 1000], max_size=1920),
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx >= self.length:
            raise IndexError
        sample = {}
        for item in self.items:
            file = os.path.join(self.dirs[item], "{:08d}.pkl".format(idx))
            sample[item] = pickle.load(open(file, "rb"))

        # apply transforms to each sample
        sample = self.transforms(sample)

        # swap channel order of motion vectors from BGR to RGB
        sample["motion_vectors"] = sample["motion_vectors"][..., [2, 1, 0]]

        # swap motion vector axes so that shape is (B, C, H, W) instead of (B, H, W, C)
        sample["motion_vectors"] = sample["motion_vectors"].permute(0, 3, 1, 2)

        return sample


# run as python -m lib.dataset.dataset_precomputed from root dir
if __name__ == "__main__":
    import numpy as np
    root_dir = "data_precomputed"
    modes = ["train"]
    datasets = {x: MotionVectorDatasetPrecomputed(root_dir=os.path.join(root_dir, x)) for x in modes}
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=1, shuffle=False, num_workers=0) for x in modes}

    for mode in modes:

        step_wise = True

        batch_size = datasets[mode][0]["motion_vectors"].shape[0]
        for batch_idx in range(batch_size):
            cv2.namedWindow("motion_vectors-{}".format(batch_idx), cv2.WINDOW_NORMAL)
            cv2.resizeWindow("motion_vectors-{}".format(batch_idx), 640, 360)

        for step, sample in enumerate(dataloaders[mode]):

            # remove batch dimension as batch size is always 1
            for item in sample.values():
                item.squeeze_(0)

            motion_vectors = sample["motion_vectors"]
            boxes_prev = sample["boxes_prev"]
            velocities = sample["velocities"]
            num_boxes_mask = sample["num_boxes_mask"]
            det_boxes_prev = sample["det_boxes_prev"]

            print("Step: {}".format(step))
            print("mvs 0 x motion total:", torch.sum(motion_vectors[0, 0, :, :]))
            print("mvs 0 y motion total:", torch.sum(motion_vectors[0, 1, :, :]))

            print(motion_vectors.shape)
            print(boxes_prev.shape)
            print(velocities.shape)
            print(num_boxes_mask.shape)
            print(det_boxes_prev.shape)

            for batch_idx in range(motion_vectors.shape[0]):

                motion_vectors_ = motion_vectors[batch_idx, ...]
                boxes_prev_ = boxes_prev[batch_idx, ...]
                det_boxes_prev_ = det_boxes_prev[batch_idx, ...]
                velocities_ = velocities[batch_idx, ...]

                # (C, H, W) -> (H, W, C)
                motion_vectors_ = motion_vectors_.permute(1, 2, 0)
                motion_vectors_ = motion_vectors_[..., [2, 1, 0]]
                motion_vectors_ = motion_vectors_.numpy()
                motion_vectors_ = (motion_vectors_ - np.min(motion_vectors_)) / (np.max(motion_vectors_) - np.min(motion_vectors_))
                motion_vectors_ = draw_boxes(motion_vectors_, boxes_prev_[:, 1:], None, color=(200, 200, 200))
                motion_vectors_ = draw_boxes(motion_vectors_, det_boxes_prev_[:, 1:], None, color=(50, 255, 50))
                motion_vectors_ = draw_velocities(motion_vectors_, boxes_prev_[:, 1:], velocities_, scale=100)

                print("step: {}, MVS shape: {}".format(step, motion_vectors_.shape))
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
