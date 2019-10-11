import random
import pickle
import copy
import numpy as np
import cv2
import torch


class StandardizeMotionVectors:
    """Subtracts mean from motion vectors and divides by standard deviation.

    motion_vectors[channel] = (motion_vectors[channel] - mean[channel]) / std[channel]

    Args:
        motion_vectors (`torch.Tensor`): Motion vector image with shape (B x H x W X C)
            and channel order BGR. Blue channel has no meaning, green channel
            corresponds to y component of motion vector and red channel to x
            component of motion vector.

        mean (`list` of `float`): Mean values for blue, green and red channel to
            be subtracted from the motion vector image.

        std (`list` of `float`): Standard deviations per channel by which to
            divide the mean subtracted motion vector image.

    Returns:
        (`torch.Tensor`) the standardized motion vector image with same shape
        as input.
    """
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def __call__(self, sample):
        sample["motion_vectors"] = (sample["motion_vectors"] - self.mean) / self.std
        return sample


class StandardizeVelocities:
    """Subtracts mean from velocities and divides by standard deviation.

    velocities[x] = (velocities[x] - mean[x]) / std[x], x = {v_xc, v_yc, v_w, v_h}

    Args:
        velocities (`torch.Tensor`): Box velocities with shape (B x K x 4)
            where B is the batch size, K the maximum number of bounding boxes in
            the dataset. The last dimenision stand for [v_cx, v_cy, v_w, v_h],
            the velocities of each box center point and velocities of box width
            and height.

        mean (`list` of `float`): Mean values for [v_cx, v_cy, v_w, v_h] to
            be subtracted from the velocities.

        std (`list` of `float`): Standard deviations for [v_cx, v_cy, v_w, v_h]
            by which to divide the mean subtracted velocities.

        inverse (`bool`): If True compute the inverse of this transform, that is
            velocities[x] = velocities[x] * std[x] + mean[x], x = {v_xc, v_yc,
            v_w, v_h}

    Returns:
        (`torch.Tensor`) the standardized velocities with same shape as input.
    """
    def __init__(self, mean, std, inverse=False):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.inverse = inverse

    def __call__(self, sample):
        if self.inverse:
            sample["velocities"] = sample["velocities"] * self.std + self.mean
        else:
            sample["velocities"] = (sample["velocities"] - self.mean) / self.std
        return sample


def scale_(image, scale=600, max_size=1000):
    # determine the scaling factor
    image = image.numpy()
    size_min = np.min(image.shape[-3:-1])
    size_max = np.max(image.shape[-3:-1])
    scaling_factor = float(scale) / float(size_min)
    if np.round(scaling_factor * size_max) > max_size:
        scaling_factor = float(max_size) / float(size_max)
     # scale the frame
    if image.ndim == 3:
        image_resized = cv2.resize(image, None, None, fx=scaling_factor,
            fy=scaling_factor, interpolation=cv2.INTER_LINEAR)
    elif image.ndim == 4:
        image_resized = []
        for batch_idx in range(image.shape[0]):
            image_resized.append(cv2.resize(image[batch_idx, ...], None, None,
                fx=scaling_factor, fy=scaling_factor,
                interpolation=cv2.INTER_LINEAR))
        image_resized = np.stack(image_resized, axis=0)
    else:
        raise ValueError("Invalid frame dimension")
    image_resized = torch.from_numpy(image_resized)
    return image_resized, scaling_factor


class ScaleImage:
    """Scale the motion vectors or frame considering a maximum size.

    Frame is scaled so that it's shortest side has the size specified by the
    `scale` parameter. The aspect ratio is constant. In case, the
    longer side would exceed the value specified by `max_size` a new
    scaling factor is computed so that the longer side matches `max_size`.
    The shorter side is then smaller than specified by `scale`.

    Args:
        frame (`torch.Tensor`): Batch of frames with shape (B, H, W, C) with
            C = 3 and arbitrary datatype. Alternatively a single frame of shape
            (H, W, C) can be provided. The frames are resized based on the
            provided scaling factors.

        scale (`int` or `float`): The desired size of the shorter
            side of the frame after scaling.

        max_size (`int` or `float`): The desired maximum size of the
            longer side of the frame after scaling.

        return_scale (`bool`): If True return the used scaling factor in the
            output sample with the key `scaling_factor`.

    Returns:
        (`tuple`): First item is the batch of resized frames (`torch.Tensor`) or
        the single resized frame depending of the input. Second item is the
        computed scaling factor (`float`) which was used to scale both height
        and width of the input frames.
    """
    def __init__(self, items=["motion_vectors"], scale=600, max_size=1000, return_scale=True):
        self.items = items
        self.scale = scale
        self.max_size = max_size
        self.return_scale = return_scale

    def __call__(self, sample):
        for item in self.items:
            image = sample[item]
            image_resized, scaling_factor = scale_(image, self.scale, self.max_size)
            sample[item] = image_resized
        if self.return_scale:
            sample["scaling_factor"] = scaling_factor
        return sample


class RandomScaleImage:
    def __init__(self, items=["motion_vectors"], scales=[300, 400, 500, 600], max_size=1000, return_scale=True):
        self.items = items
        self.scales = scales
        self.max_size = max_size
        self.return_scale = return_scale

    def __call__(self, sample):
        scale = random.choice(self.scales)
        for item in self.items:
            image = sample[item]
            image_resized, scaling_factor = scale_(image, scale, self.max_size)
            sample[item] = image_resized
        if self.return_scale:
            sample["scaling_factor"] = scaling_factor
        return sample


class RandomFlip:
    """Randomly flips motion vectors, boxes and velocities.

    Whether or not to flip is chosen randomly with equal probabilities. If a
    flip along both horizontal and vertical axis shall be performed, it is first
    dtermined whether to flip along horizontal axis (p = 0.5) and afterwards
    whether to flip along vertical axis (p = 0.5). The probability of an image
    being flipped both vertically and horizontally is thus p = 0.25.

    Args:
        directions (`list` of `str`): List can contain "horizontal" and
            "vertical" to determine along which axis a flip can occur.
    """
    def __init__(self, directions=["horizontal", "vertical"]):
        self.directions = directions

    def flip_boxes_(self, boxes, direction, image_width, image_height):
        # boxes must have shape (N, 4) with [xmin, ymin, w, h] in the last dim
        if direction == "horizontal":
            boxes[:, 0] = image_width - boxes[:, 2] - boxes[:, 0]
        elif direction == "vertical":
            boxes[:, 1] = image_height - boxes[:, 3] - boxes[:, 1]
        return boxes

    def flip_velocities_(self, velocities, direction):
        # velocities must have shape (N, 4) with [v_xc, v_yc, v_w, v_h] in the last dim
        if direction == "horizontal":
            velocities[:, 0] = -1 * velocities[:, 0]
        elif direction == "vertical":
            velocities[:, 1] = -1 * velocities[:, 1]
        return velocities

    def flip_(self, motion_vectors, boxes_prev, det_boxes_prev, velocities, direction):
        motion_vectors = motion_vectors.numpy()

        # for opencv flip function
        if direction == "horizontal":
            flip_code = 1
        if direction == "vertical":
            flip_code = 0

        # mvs have shape (H, W, C) or (B, H, W, C)
        image_width = motion_vectors.shape[-2]
        image_height = motion_vectors.shape[-3]

        if motion_vectors.ndim == 3:
            motion_vectors = cv2.flip(motion_vectors, flip_code)  # flip around x axis
            boxes_prev[:, 1:] = self.flip_boxes_(boxes_prev[:, 1:], direction,
                image_width, image_height)
            det_boxes_prev[:, 1:] = self.flip_boxes_(det_boxes_prev[:, 1:],
                direction, image_width, image_height)
            velocities = self.flip_velocities_(velocities, direction)

        elif motion_vectors.ndim == 4:
            motion_vectors_flipped = []
            for batch_idx in range(motion_vectors.shape[0]):
                motion_vectors_flipped.append(cv2.flip(
                    motion_vectors[batch_idx, ...], flip_code))
                boxes_prev[batch_idx, :, 1:] = self.flip_boxes_(
                    boxes_prev[batch_idx, :, 1:], direction, image_width,
                    image_height)
                det_boxes_prev[batch_idx, :, 1:] = self.flip_boxes_(
                    det_boxes_prev[batch_idx, :, 1:], direction, image_width,
                    image_height)
                velocities[batch_idx, ...] = self.flip_velocities_(
                    velocities[batch_idx, ...], direction)
            motion_vectors = np.stack(motion_vectors_flipped, axis=0)

        else:
            raise ValueError("Invalid dimension of motion vectors")

        motion_vectors = torch.from_numpy(motion_vectors)

        return motion_vectors, boxes_prev, det_boxes_prev, velocities

    def __call__(self, sample):
        #direction = random.choice(self.directions)
        motion_vectors = sample["motion_vectors"]
        boxes_prev = sample["boxes_prev"]
        det_boxes_prev = sample["det_boxes_prev"]
        velocities = sample["velocities"]

        # flip motion vector image
        print(motion_vectors.shape)

        motion_vectors, boxes_prev, det_boxes_prev, velocities = self.flip_(
            motion_vectors, boxes_prev, det_boxes_prev, velocities,
            direction="horizontal")

        sample["motion_vectors"] = motion_vectors
        sample["boxes_prev"] = boxes_prev
        sample["det_boxes_prev"] = det_boxes_prev
        sample["velocities"] = velocities

        return sample


# class RandomFlipMotionVectors:
#     def __init__(self, directions=["horizontal", "vertical"]):
#         self.directions = directions
#
#     def __call__(self, sample)
#         return sample


# run as python -m lib.transforms.transfms from root dir
#if __name__ == "__main__":

    # def sample_is_equal(sample, sample_transformed):
    #     all_equal = True
    #     for sample_item, transformed_item in zip(sample.values(), sample_transformed.values()):
    #         print(torch.allclose(sample_item, transformed_item))
    #         all_equal &= torch.allclose(sample_item, transformed_item)
    #     return all_equal


    # # load test data
    # motion_vectors = pickle.load(open("lib/transforms/test_data/motion_vectors.pkl", "rb"))
    # boxes_prev = pickle.load(open("lib/transforms/test_data/boxes_prev.pkl", "rb"))
    # det_boxes_prev = pickle.load(open("lib/transforms/test_data/det_boxes_prev.pkl", "rb"))
    # velocities = pickle.load(open("lib/transforms/test_data/velocities.pkl", "rb"))
    #
    # sample = {
    #     "motion_vectors": motion_vectors,
    #     "boxes_prev": boxes_prev,
    #     "det_boxes_prev": det_boxes_prev,
    #     "velocities": velocities
    # }
    #
    # transform = StandardizeMotionVectors(
    #     mean=[0.0, 0.014817102005829075, 0.0705440781107341],
    #     std=[1.0, 0.060623864350822454,  0.4022695243698158])
    #
    # sample_transformed = transform(copy.deepcopy(sample))
    #
    # cv2.imshow
