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
            return (mean, std)


# run as python -m lib.dataset.tools.compute_stats from root dir
if __name__ == "__main__":
    visu = False  # whether to show graphical output (frame + motion vectors) or not
    codec = "h264"
    mvs_mode = "upsampled"
    static_only = False
    exclude_keyframes = True
    scales = [1.0]
    vector_type = "p+b"

    dataset_train = MotionVectorDataset(root_dir='data', transforms=None, codec=codec,
        scales=scales, mvs_mode=mvs_mode, vector_type=vector_type, static_only=static_only,
        exclude_keyframes=exclude_keyframes, visu=visu, debug=False, mode="train")

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=0)

    step_wise = False

    if visu:
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("frame", 640, 360)
        cv2.namedWindow("motion_vectors_p", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("motion_vectors_p", 640, 360)
        if vector_type == "p+b":
            cv2.namedWindow("motion_vectors_b", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("motion_vectors_b", 640, 360)

    runnings_stats_mvs_p_x = RunningStats()
    runnings_stats_mvs_p_y = RunningStats()
    if vector_type == "p+b":
        runnings_stats_mvs_b_x = RunningStats()
        runnings_stats_mvs_b_y = RunningStats()

    runnings_stats_vel_xc = RunningStats()
    runnings_stats_vel_yc = RunningStats()
    if mvs_mode == "upsampled":
        runnings_stats_vel_w = RunningStats()
        runnings_stats_vel_h = RunningStats()

    for step, sample in enumerate(dataloader_train):

        motion_vectors_p = sample["motion_vectors"][0][0].numpy()
        runnings_stats_mvs_p_x.update(np.mean(motion_vectors_p[0, :, :]))
        runnings_stats_mvs_p_y.update(np.mean(motion_vectors_p[1, :, :]))
        if vector_type == "p+b":
            motion_vectors_b = sample["motion_vectors"][1][0].numpy()
            runnings_stats_mvs_b_x.update(np.mean(motion_vectors_b[0, :, :]))
            runnings_stats_mvs_b_y.update(np.mean(motion_vectors_b[1, :, :]))

        velocities = sample["velocities"][0].numpy()
        for v in velocities:
            runnings_stats_vel_xc.update(v[0])
            runnings_stats_vel_yc.update(v[1])
            if mvs_mode == "upsampled":
                runnings_stats_vel_w.update(v[2])
                runnings_stats_vel_h.update(v[3])

        if visu:
            print("step: {}".format(step))

            motion_vectors = [motion_vectors_p]
            labels = ["motion_vectors_p"]
            if vector_type == "p+b":
                motion_vectors.append(motion_vectors_b)
                labels.append("motion_vectors_b")

            for mvs, label in zip(motion_vectors, labels):
                mvs = torch.from_numpy(mvs)
                mvs = mvs.permute(1, 2, 0)
                mvs = mvs[..., [2, 1, 0]]
                mvs = mvs.numpy()
                mvs = (mvs - np.min(mvs)) / (np.max(mvs) - np.min(mvs))
                cv2.imshow(label, mvs)
            cv2.imshow("frame", sample["frame"][0].numpy())


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
    print(("codec = {}, mvs_mode = {}, static_only = {}, exclude_keyframes = {}"
        ", scales = {}").format(codec, mvs_mode, static_only, exclude_keyframes,
        scales))

    print("motion vectors: [blue := not used, green := y motion, red := x motion]")
    if mvs_mode == "upsampled":
        print("velocities: [x, y, w, h]")
    elif mvs_mode == "dense":
        print("velocities: [x, y]")

    print('P motion vectors -- "mean": [0.0, {2}, {0}], "std": [1.0, {3}, {1}]'.format(
        *runnings_stats_mvs_p_x.get_stats(), *runnings_stats_mvs_p_y.get_stats()))
    if vector_type == "p+b":
        print('B motion vectors -- "mean": [0.0, {2}, {0}], "std": [1.0, {3}, {1}]'.format(
            *runnings_stats_mvs_b_x.get_stats(), *runnings_stats_mvs_b_y.get_stats()))

    if mvs_mode == "upsampled":
        print('velocities -- "mean": [{0}, {2}, {4}, {6}], "std": [{1}, {3}, {5}, {7}]'.format(
            *runnings_stats_vel_xc.get_stats(), *runnings_stats_vel_yc.get_stats(),
            *runnings_stats_vel_w.get_stats(), *runnings_stats_vel_h.get_stats()))
    elif mvs_mode == "dense":
        print('velocities -- "mean": [{0}, {2}], "std": [{1}, {3}]'.format(
            *runnings_stats_vel_xc.get_stats(), *runnings_stats_vel_yc.get_stats()))
