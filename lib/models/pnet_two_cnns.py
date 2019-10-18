import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision

from lib.resnet_atrous import resnet18
from lib.utils import load_pretrained_weights_to_modified_resnet


class PropagationNetwork(nn.Module):
    def __init__(self):
        super(PropagationNetwork, self).__init__()

        self.POOLING_SIZE = 7  # the ROIs are split into m x m regions
        self.FIXED_BLOCKS = 1
        self.TRUNCATED = False

        # load pretrained weights
        resnet = resnet18()
        resnet_weights = model_zoo.load_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')
        load_pretrained_weights_to_modified_resnet(resnet, resnet_weights)

        input_channels = 2
        base = [
            #nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.relu,
            resnet.layer2,
            resnet.relu,
            resnet.layer3,
            resnet.relu,
            resnet.layer4,
            resnet.relu]
        self.base1 = nn.Sequential(*base)
        self.base2 = nn.Sequential(*base)

        assert (0 <= self.FIXED_BLOCKS <= 4) # set this value to 0, so we can train all blocks
        if self.FIXED_BLOCKS >= 4: # fix all blocks
            for p in self.base1[10].parameters(): p.requires_grad = False
            for p in self.base2[10].parameters(): p.requires_grad = False
        if self.FIXED_BLOCKS >= 3: # fix first 3 blocks
            for p in self.base1[8].parameters(): p.requires_grad = False
            for p in self.base2[8].parameters(): p.requires_grad = False
        if self.FIXED_BLOCKS >= 2: # fix first 2 blocks
            for p in self.base1[6].parameters(): p.requires_grad = False
            for p in self.base2[6].parameters(): p.requires_grad = False
        if self.FIXED_BLOCKS >= 1: # fix first 1 block
            for p in self.base1[4].parameters(): p.requires_grad = False
            for p in self.base2[4].parameters(): p.requires_grad = False

        self.conv1x1_1 = nn.Conv2d(512, 4*self.POOLING_SIZE*self.POOLING_SIZE, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
        self.conv1x1_2 = nn.Conv2d(512, 4*self.POOLING_SIZE*self.POOLING_SIZE, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
        self.conv1x1_3 = nn.Conv2d(8, 4, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
        self.pooling = nn.AvgPool2d(kernel_size=self.POOLING_SIZE, stride=self.POOLING_SIZE)

        print([p.requires_grad for p in self.base1.parameters()])
        print([p.requires_grad for p in self.base2.parameters()])
        print(list(self.children()))


    def forward(self, motion_vectors, boxes_prev, num_boxes_mask):
        # first branch which encodes motion inside the indivual bounding boxes
        x1 = self.base1(motion_vectors)
        x1 = self.conv1x1_1(x1)

        if num_boxes_mask is not None:
            boxes_prev = boxes_prev[num_boxes_mask]
        boxes_prev = boxes_prev.view(-1, 5)
        boxes_prev = self._change_box_format(boxes_prev)
        boxes_prev = self._frame_idx_to_batch_idx(boxes_prev)

        x1 = torchvision.ops.ps_roi_pool(x1, boxes_prev, output_size=(
            self.POOLING_SIZE, self.POOLING_SIZE), spatial_scale=1/16)

        # second branch which encodes motion in the entire motion vector image
        x2 = self.base2(motion_vectors)
        x2 = self.conv1x1_2(x2)

        frame_height = motion_vectors.shape[-2]
        frame_width = motion_vectors.shape[-1]
        num_boxes = boxes_prev.shape[0]
        full_image_boxes = boxes_prev.clone()
        full_image_boxes[:, 1:] = torch.tensor([0, 0, frame_width,
            frame_height]).repeat(num_boxes, 1)

        x2 = torchvision.ops.ps_roi_pool(x2, full_image_boxes, output_size=(
            self.POOLING_SIZE, self.POOLING_SIZE), spatial_scale=1/16)

        # concatenate branch 1 and 2
        x3 = torch.cat([x1, x2], axis=1)
        x3 = self.conv1x1_3(x3)

        # pool and reshape to get 4D velocity vector for each box
        x3 = self.pooling(x3)
        x3 = x3.squeeze()
        velocities_pred = x3.view(-1, 4)
        return velocities_pred


    def _frame_idx_to_batch_idx(self, boxes):
        """Converts unique frame_idx in first column of boxes into batch index."""
        frame_idxs = torch.unique(boxes[:, 0])
        for batch_idx, frame_idx in enumerate(frame_idxs):
            idx = torch.where(boxes == frame_idx)[0]
            boxes[idx, 0] = batch_idx
        return boxes


    def _change_box_format(self, boxes):
        """Change format of boxes from [idx, x, y, w, h] to [idx, x1, y1, x2, y2]."""
        boxes[..., 0] = boxes[..., 0]
        boxes[..., 1] = boxes[..., 1]
        boxes[..., 2] = boxes[..., 2]
        boxes[..., 3] = boxes[..., 1] + boxes[..., 3]
        boxes[..., 4] = boxes[..., 2] + boxes[..., 4]
        return boxes


# names of layer weights (excludes batch norm layers, etc.), needed for weight logging
layer_keys = [
    'base1.0.weight',
    'base1.4.0.conv1.weight',
    'base1.4.0.conv2.weight',
    'base1.4.2.conv1.weight',
    'base1.4.2.conv2.weight',
    'base1.6.0.conv1.weight',
    'base1.6.0.conv2.weight',
    'base1.6.0.downsample.0.weight',
    'base1.6.2.conv1.weight',
    'base1.6.2.conv2.weight',
    'base1.8.0.conv1.weight',
    'base1.8.0.conv2.weight',
    'base1.8.0.downsample.0.weight',
    'base1.8.2.conv1.weight',
    'base1.8.2.conv2.weight',
    'base1.10.0.conv1.weight',
    'base1.10.0.conv2.weight',
    'base1.10.0.downsample.0.weight',
    'base1.10.2.conv1.weight',
    'base1.10.2.conv2.weight',
    'base2.0.weight',
    'base2.4.0.conv1.weight',
    'base2.4.0.conv2.weight',
    'base2.4.2.conv1.weight',
    'base2.4.2.conv2.weight',
    'base2.6.0.conv1.weight',
    'base2.6.0.conv2.weight',
    'base2.6.0.downsample.0.weight',
    'base2.6.2.conv1.weight',
    'base2.6.2.conv2.weight',
    'base2.8.0.conv1.weight',
    'base2.8.0.conv2.weight',
    'base2.8.0.downsample.0.weight',
    'base2.8.2.conv1.weight',
    'base2.8.2.conv2.weight',
    'base2.10.0.conv1.weight',
    'base2.10.0.conv2.weight',
    'base2.10.0.downsample.0.weight',
    'base2.10.2.conv1.weight',
    'base2.10.2.conv2.weight',
    'conv1x1_1.weight',
    'conv1x1_2.weight',
    'conv1x1_3.weight',
]


if __name__ == "__main__":

    model = PropagationNetwork()
    print([p.requires_grad for p in model.base1.parameters()])
    print([p.requires_grad for p in model.base2.parameters()])
    print(list(model.children()))
