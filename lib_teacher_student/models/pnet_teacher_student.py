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

        # load pretrained weights
        resnet = resnet18()
        resnet_weights = model_zoo.load_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')
        load_pretrained_weights_to_modified_resnet(resnet, resnet_weights)

        base = [
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
        self.base = nn.Sequential(*base)

        assert (0 <= self.FIXED_BLOCKS <= 4) # set this value to 0, so we can train all blocks
        if self.FIXED_BLOCKS >= 4: # fix all blocks
            for p in self.base[10].parameters(): p.requires_grad = False
        if self.FIXED_BLOCKS >= 3: # fix first 3 blocks
            for p in self.base[8].parameters(): p.requires_grad = False
        if self.FIXED_BLOCKS >= 2: # fix first 2 blocks
            for p in self.base[6].parameters(): p.requires_grad = False
        if self.FIXED_BLOCKS >= 1: # fix first 1 block
            for p in self.base[4].parameters(): p.requires_grad = False

        self.conv1x1 = nn.Conv2d(512, 4*self.POOLING_SIZE*self.POOLING_SIZE, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
        self.pooling = nn.AvgPool2d(kernel_size=self.POOLING_SIZE, stride=self.POOLING_SIZE)

        print([p.requires_grad for p in self.parameters()])
        print(list(self.children()))


    def forward(self, motion_vectors, boxes_prev):
        x = self.base(motion_vectors)
        x = self.conv1x1(x)

        boxes_prev_ = boxes_prev.detach().clone()
        boxes_prev_ = boxes_prev_.view(-1, 5)
        boxes_prev_ = self._change_box_format(boxes_prev_)
        boxes_prev_ = self._frame_idx_to_batch_idx(boxes_prev_)

        # compute ratio of input size to size of base output
        x = torchvision.ops.ps_roi_pool(x, boxes_prev_, output_size=(self.POOLING_SIZE, self.POOLING_SIZE), spatial_scale=1/16)
        out_features = x.clone()
        x = self.pooling(x)
        x = x.squeeze()
        velocities_pred = x.view(-1, 4)
        return velocities_pred, out_features


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
        'base.0.weight',
        'base.4.0.conv1.weight',
        'base.4.0.conv2.weight',
        'base.4.2.conv1.weight',
        'base.4.2.conv2.weight',
        'base.6.0.conv1.weight',
        'base.6.0.conv2.weight',
        'base.6.0.downsample.0.weight',
        'base.6.2.conv1.weight',
        'base.6.2.conv2.weight',
        'base.8.0.conv1.weight',
        'base.8.0.conv2.weight',
        'base.8.0.downsample.0.weight',
        'base.8.2.conv1.weight',
        'base.8.2.conv2.weight',
        'base.10.0.conv1.weight',
        'base.10.0.conv2.weight',
        'base.10.0.downsample.0.weight',
        'base.10.2.conv1.weight',
        'base.10.2.conv2.weight',
        'conv1x1.weight',
    ]


if __name__ == "__main__":

    model = PropagationNetwork()
    print([p.requires_grad for p in model.parameters()])
    print(list(model.children()))
