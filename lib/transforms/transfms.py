"""Data transforms.

All transforms below can be used with `torchvision.transforms.Compose`, e.g.

```
custom_transform = torchvision.transforms.Compose([
    RandomFlip(directions=["x", "y"]),
    StandardizeVelocities(mean=[1.3, 2.4, 0.2], std=[1.1, 1.5, 1.3]),
    ScaleImage(items=["motion_vectors"], scale=600, max_size=1000),
])
```

All the transforms can then be applied to a sample by calling
```
sample = custom_transform(sample)
```

The sample should be a Python dictionary with the following key values pairs:

- motion_vectors (`torch.Tensor`): Motion vector image of shape
    (B, H, W, C) or (H, W, C), dtype float32 and channel order BGR. Typically,
    C = 3. Blue channel has no meaning, green channel corresponds to y component
    of motion vector and red channel to x component of motion vector.

- boxes_prev (`torch.Tensor`): Set of bounding boxes in previous frame. Shape
    (B, N, 5) or (N, 5) with batch size B, number of boxes N and format
    [frame_idx, xmin, ymin, w, h] in the last dimension.

- velocities (`torch.Tensor`): Set of box velocities. Shape (B, N, 4) or (N, 4)
    with batch size B, number of boxes N and format [v_xc, v_yc, v_w, v_h] in
    the last dimenion. The last dimenision stand for the velocities of each box
    center point and velocities of box width and height.

- det_boxes_prev (`torch.Tensor`): Same format as `boxes_prev`, but these are
    the detection boxes in the previous frame rather than the tracked boxes.

The transforms might alter these values in place. So, if you want to keep them
you have to make a copy manually.
"""
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
        mean (`list` of `float`): Mean values for blue, green and red channel to
            be subtracted from the motion vector image.

        std (`list` of `float`): Standard deviations per channel by which to
            divide the mean subtracted motion vector image.
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
        mean (`list` of `float`): Mean values for [v_cx, v_cy, v_w, v_h] to
            be subtracted from the velocities.

        std (`list` of `float`): Standard deviations for [v_cx, v_cy, v_w, v_h]
            by which to divide the mean subtracted velocities.

        inverse (`bool`): If True compute the inverse of this transform, that is
            velocities[x] = velocities[x] * std[x] + mean[x], x = {v_xc, v_yc,
            v_w, v_h}
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
        items (`list` of `str`): Can be "motion_vectors" and/or "frame".
            Determines whether to scale motion vectors and/or frame inside
            the sample dict provided during call.

        scale (`int` or `float`): The desired size of the shorter
            side of the frame after scaling.

        max_size (`int` or `float`): The desired maximum size of the
            longer side of the frame after scaling.

        return_scale (`bool`): If True return the used scaling factor is
            inserted in the output sample with key `scaling_factor`.
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
    """Scale motion vector or frame by a randomly chosen scale.

    Args:
        items (`list` of `str`): Can be "motion_vectors" and/or "frame".
            Determines whether to scale motion vectors and/or frame inside
            the sample dict provided during call.

        scales (`list` of `int` or `list` of `float`): A set of scales one of
            which is uniformly randomly chosen during each call of this
            transform.

        max_size (`int` or `float`): The desired maximum size of the
            longer side of the frame after scaling.

        return_scale (`bool`): If True return the used scaling factor is
            inserted in the output sample with key `scaling_factor`.
    """
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


def flip_motion_vectors_(motion_vectors, direction):
    # motion_vectors must have shape (H, W, C) with BGR order (x: red, y: green)
    if direction == "y":
        motion_vectors = cv2.flip(motion_vectors, 1)
        motion_vectors[:, :, 2] = -1 * motion_vectors[:, :, 2]  # change sign of x-motion
    elif direction == "x":
        motion_vectors = cv2.flip(motion_vectors, 0)
        motion_vectors[:, :, 1] = -1 * motion_vectors[:, :, 1]  # change sign of y-motion
    return motion_vectors


def flip_boxes_(boxes, direction, image_width, image_height):
    # boxes must have shape (N, 4) with [xmin, ymin, w, h] in the last dim
    if direction == "y":
        boxes[:, 0] = image_width - boxes[:, 2] - boxes[:, 0]
    elif direction == "x":
        boxes[:, 1] = image_height - boxes[:, 3] - boxes[:, 1]
    return boxes


def flip_velocities_(velocities, direction):
    # velocities must have shape (N, 4) with [v_xc, v_yc, v_w, v_h] in the last dim
    if direction == "y":
        velocities[:, 0] = -1 * velocities[:, 0]
    elif direction == "x":
        velocities[:, 1] = -1 * velocities[:, 1]
    return velocities


def flip_(motion_vectors, boxes_prev, det_boxes_prev, velocities, direction):
    motion_vectors = motion_vectors.numpy()
    # mvs have shape (H, W, C) or (B, H, W, C)
    image_width = motion_vectors.shape[-2]
    image_height = motion_vectors.shape[-3]
    if motion_vectors.ndim == 3:
        motion_vectors = flip_motion_vectors_(motion_vectors, direction)
        boxes_prev[:, 1:] = flip_boxes_(boxes_prev[:, 1:], direction,
            image_width, image_height)
        det_boxes_prev[:, 1:] = flip_boxes_(det_boxes_prev[:, 1:],
            direction, image_width, image_height)
        velocities = flip_velocities_(velocities, direction)
    elif motion_vectors.ndim == 4:
        motion_vectors_flipped = []
        for batch_idx in range(motion_vectors.shape[0]):
            motion_vectors_flipped.append(flip_motion_vectors_(
                motion_vectors[batch_idx, ...], direction))
            boxes_prev[batch_idx, :, 1:] = flip_boxes_(
                boxes_prev[batch_idx, :, 1:], direction, image_width,
                image_height)
            det_boxes_prev[batch_idx, :, 1:] = flip_boxes_(
                det_boxes_prev[batch_idx, :, 1:], direction, image_width,
                image_height)
            velocities[batch_idx, ...] = flip_velocities_(
                velocities[batch_idx, ...], direction)
        motion_vectors = np.stack(motion_vectors_flipped, axis=0)
    else:
        raise ValueError("Invalid dimension of motion vectors")
    motion_vectors = torch.from_numpy(motion_vectors)
    return motion_vectors, boxes_prev, det_boxes_prev, velocities


class Flip:
    """Flips motion vectors, boxes and velocities along a given axis.

    This method flips the motion vector image either along the x or y axis.
    Bounding boxes and velocities are flipped accordingly. Also, the directions
    of the motion vectors inside the motion vector image are inverted
    accordingly.

    Args:
        direction (`str`): Either "x" or "y" indicating along which axis to
        flip.
    """
    def __init__(self, direction="x"):
        self.direction = direction

    def __call__(self, sample):
        motion_vectors = sample["motion_vectors"]
        boxes_prev = sample["boxes_prev"]
        det_boxes_prev = sample["det_boxes_prev"]
        velocities = sample["velocities"]
        motion_vectors, boxes_prev, det_boxes_prev, velocities = flip_(
            motion_vectors, boxes_prev, det_boxes_prev, velocities,
            direction=self.direction)
        sample["motion_vectors"] = motion_vectors
        sample["boxes_prev"] = boxes_prev
        sample["det_boxes_prev"] = det_boxes_prev
        sample["velocities"] = velocities
        return sample


class RandomFlip:
    """Randomly flips motion vectors, boxes and velocities.

    Whether or not to flip is chosen randomly with equal probabilities. If a
    flip along both y and x axis shall be performed, it is first
    dtermined whether to flip along y axis (p = 0.5) and afterwards
    whether to flip along x axis (p = 0.5). The probability of an image
    being flipped both along x and y axis is thus p = 0.25.

    Args:
        directions (`list` of `str`): List can contain "y" and
            "x" to determine along which axis a flip can occur.
    """
    def __init__(self, directions=["x", "y"]):
        self.directions = directions

    def __call__(self, sample):
        motion_vectors = sample["motion_vectors"]
        boxes_prev = sample["boxes_prev"]
        det_boxes_prev = sample["det_boxes_prev"]
        velocities = sample["velocities"]
        for direction in self.directions:
            if random.choice([True, False]):  # 50 percent chance that flip happens
                motion_vectors, boxes_prev, det_boxes_prev, velocities = flip_(
                    motion_vectors, boxes_prev, det_boxes_prev, velocities,
                    direction=direction)
        sample["motion_vectors"] = motion_vectors
        sample["boxes_prev"] = boxes_prev
        sample["det_boxes_prev"] = det_boxes_prev
        sample["velocities"] = velocities
        return sample
