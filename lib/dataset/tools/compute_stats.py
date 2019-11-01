import cv2
import numpy as np
import torch

from lib.dataset.dataset import MotionVectorDataset


class RunningStats():
    def __init__(self):
        self.existingAggregate = (0, 0, 0)

    def update(self, newValue):
        (count, mean, M2) = self.existingAggregate
        count += 1
        delta = newValue - mean
        mean += delta / count
        delta2 = newValue - mean
        M2 += delta * delta2
        self.existingAggregate = (count, mean, M2)

    # retrieve the mean, variance and sample variance from an aggregate
    def get_stats(self):
        (count, mean, M2) = self.existingAggregate
        (mean, variance) = (mean, M2/count)
        std = np.sqrt(variance)
        if count < 2:
            return (float('nan'), float('nan'), float('nan'))
        else:
            return (mean, variance, std)


# run as python -m lib.dataset.tools.compute_stats from root dir
if __name__ == "__main__":

    codec = "mpeg4"
    static_only = False
    visu = False  # whether to show graphical output (frame + motion vectors) or not

    dataset_train = MotionVectorDataset(root_dir="data", batch_size=1, codec=codec,
        static_only=static_only, visu=visu, mode="train")
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=0)

    step_wise = False

    if visu:
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("frame", 640, 360)
        cv2.namedWindow("motion_vectors", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("motion_vectors", 640, 360)

    runnings_stats_mvs_x = RunningStats()
    runnings_stats_mvs_y = RunningStats()

    runnings_stats_vel_xc = RunningStats()
    runnings_stats_vel_yc = RunningStats()
    runnings_stats_vel_w = RunningStats()
    runnings_stats_vel_h = RunningStats()

    for step, sample in enumerate(dataloader_train):

        motion_vectors = sample["motion_vectors"][0].numpy()
        runnings_stats_mvs_x.update(np.mean(motion_vectors[:, :, 2]))
        runnings_stats_mvs_y.update(np.mean(motion_vectors[:, :, 1]))

        num_boxes_mask = sample["num_boxes_mask"]
        velocities = sample["velocities"]
        velocities = velocities[num_boxes_mask]
        velocities = velocities.numpy()

        for v_xc, v_yc, v_w, v_h in velocities:
            runnings_stats_vel_xc.update(v_xc)
            runnings_stats_vel_yc.update(v_yc)
            runnings_stats_vel_w.update(v_w)
            runnings_stats_vel_h.update(v_h)

        if visu:
            print("step: {}, MVS shape: {}".format(step, motion_vectors.shape))

            cv2.imshow("frame", sample["frame"][0].numpy())
            cv2.imshow("motion_vectors", motion_vectors)

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

    # output finalized stats
    print("codec = {}, static_only = {}".format(codec, static_only))
    print("mvs x -- mean: {}, variance: {}, std: {}".format(*runnings_stats_mvs_x.get_stats()))
    print("mvs y -- mean: {}, variance: {}, std: {}".format(*runnings_stats_mvs_y.get_stats()))

    print("vel xc -- mean: {}, variance: {}, std: {}".format(*runnings_stats_vel_xc.get_stats()))
    print("vel yc -- mean: {}, variance: {}, std: {}".format(*runnings_stats_vel_yc.get_stats()))
    print("vel w -- mean: {}, variance: {}, std: {}".format(*runnings_stats_vel_w.get_stats()))
    print("vel h -- mean: {}, variance: {}, std: {}".format(*runnings_stats_vel_h.get_stats()))
