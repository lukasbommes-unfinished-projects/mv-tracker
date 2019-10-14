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
    visu = False  # whether to show graphical output (frame + motion vectors) or not
    dataset_train = MotionVectorDataset(root_dir='data', batch_size=1, codec="mpeg4", visu=visu, mode="train")
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

        velocities = sample["velocities"][0].numpy()
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
    print("mvs x -- mean: {}, variance: {}, std: {}".format(*runnings_stats_mvs_x.get_stats()))
    print("mvs y -- mean: {}, variance: {}, std: {}".format(*runnings_stats_mvs_y.get_stats()))

    print("vel xc -- mean: {}, variance: {}, std: {}".format(*runnings_stats_vel_xc.get_stats()))
    print("vel yc -- mean: {}, variance: {}, std: {}".format(*runnings_stats_vel_yc.get_stats()))
    print("vel w -- mean: {}, variance: {}, std: {}".format(*runnings_stats_vel_w.get_stats()))
    print("vel h -- mean: {}, variance: {}, std: {}".format(*runnings_stats_vel_h.get_stats()))


    # Results on training set (h264):
    # x_channel -- mean: -0.3864056486553166, variance: 22.68331783471002, std: 4.76270068707976
    # y_channel -- mean: 0.3219420202390504, variance: 1.6311065405162777, std: 1.277147814669969
    #
    # on validation set (h264) (just for comparison, not needed later)
    # x_channel -- mean: -0.19903904891235283, variance: 6.529343563155186, std: 2.5552580228139754
    # y_channel -- mean: 0.2083611949262917, variance: 1.0411895655591639, std: 1.0203869685365272
    #
    #
    # Results on training set (mpeg4):
    # x_channel -- mean: -0.16339078281237884, variance: 7.9612272040686625, std: 2.821564673026061
    # y_channel -- mean: 0.139698425141241, variance: 0.8569101028537749, std: 0.9256943895550922
    #
    # on validation set (mpeg4)
    # x_channel -- mean: -0.12560456383521534, variance: 3.341528421828638, std: 1.8279847980299613
    # y_channel -- mean: 0.1770176594258104, variance: 0.5506366588562699, std: 0.7420489598781672
