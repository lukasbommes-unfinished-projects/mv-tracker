import cv2
import numpy as np
import torch

from lib.dataset.dataset_new import MotionVectorDataset


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
    visu = False  # whether to show graphical output (frame + motion vectors) or not
    codec = "mpeg4"
    mvs_mode = "upsampled"
    static_only = False
    exclude_keyframes = True
    scales = [1.0]#, 0.75, 0.5]

    dataset_train = MotionVectorDataset(root_dir='data', transforms=None, codec=codec,
        scales=scales, mvs_mode=mvs_mode, static_only=static_only,
        exclude_keyframes=exclude_keyframes, visu=visu, debug=False, mode="train")

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
        runnings_stats_mvs_x.update(np.mean(motion_vectors[0, :, :]))
        runnings_stats_mvs_y.update(np.mean(motion_vectors[1, :, :]))

        velocities = sample["velocities"][0].numpy()
        for v_xc, v_yc, v_w, v_h in velocities:
            runnings_stats_vel_xc.update(v_xc)
            runnings_stats_vel_yc.update(v_yc)
            runnings_stats_vel_w.update(v_w)
            runnings_stats_vel_h.update(v_h)

        if visu:
            motion_vectors = torch.from_numpy(motion_vectors)
            motion_vectors = motion_vectors.permute(1, 2, 0)
            motion_vectors = motion_vectors[..., [2, 1, 0]]
            motion_vectors = motion_vectors.numpy()
            motion_vectors = (motion_vectors - np.min(motion_vectors)) / (np.max(motion_vectors) - np.min(motion_vectors))

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
    print("codec = {}, mvs_mode = {}, static_only = {}, exclude_keyframes = {}, scales = {}".format(
        codec, mvs_mode, static_only, exclude_keyframes, scales))
    print("mvs x -- mean: {}, variance: {}, std: {}".format(*runnings_stats_mvs_x.get_stats()))
    print("mvs y -- mean: {}, variance: {}, std: {}".format(*runnings_stats_mvs_y.get_stats()))

    print("vel xc -- mean: {}, variance: {}, std: {}".format(*runnings_stats_vel_xc.get_stats()))
    print("vel yc -- mean: {}, variance: {}, std: {}".format(*runnings_stats_vel_yc.get_stats()))
    print("vel w -- mean: {}, variance: {}, std: {}".format(*runnings_stats_vel_w.get_stats()))
    print("vel h -- mean: {}, variance: {}, std: {}".format(*runnings_stats_vel_h.get_stats()))
