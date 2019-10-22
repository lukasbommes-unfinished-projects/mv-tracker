import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision



class PropagationNetwork(nn.Module):
    def __init__(self):
        super(PropagationNetwork, self).__init__()

        self.POOLING_SIZE = 7  # the ROIs are split into m x m regions

        self.conv1 = nn.Conv2d(2, 2048, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(2048)
        self.relu1 = nn.ReLU(inplace=True)

        # base = [
        #     nn.Conv2d(input_channels, 2048, kernel_size=7, stride=2, padding=3, bias=False), #resnet.conv1,
        #     nn.BatchNorm2d(2048),  #resnet.bn1,
        #     nn.ReLU(inplace=True),  #resnet.relu,
        #
        #     resnet.layer1,
        #     resnet.relu,
        #     resnet.layer2,
        #     resnet.relu,
        #     resnet.layer3,
        #     resnet.relu,
        #     resnet.layer4,
        #     resnet.relu]
        # self.base = nn.Sequential(*base)

        self.conv1x1 = nn.Conv2d(512, 4*self.POOLING_SIZE*self.POOLING_SIZE, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
        self.pooling = nn.AvgPool2d(kernel_size=self.POOLING_SIZE, stride=self.POOLING_SIZE)

        # print([p.requires_grad for p in self.base.parameters()])
        # print(list(self.children()))


    def forward(self, motion_vectors, boxes_prev, num_boxes_mask):
        motion_vectors = motion_vectors[:, :2, :, :]   # pick out the red and green channel
        print(motion_vectors)

        x = self.conv1(motion_vectors)
        print(x.shape)
        x = self.bn1(x)
        x = self.relu1(x)

        velocities_pred = torch.zeros(size=(boxes_prev.shape[0], 4))


        #x = self.base(motion_vectors[:, :2, :, :])
        #x = self.conv1x1(x)

        # boxes_prev = boxes_prev.view(-1, 5)
        # boxes_prev = self._change_box_format(boxes_prev)
        # boxes_prev = self._frame_idx_to_batch_idx(boxes_prev)
        #
        # # compute ratio of input size to size of base output
        # x = torchvision.ops.ps_roi_pool(x, boxes_prev, output_size=(self.POOLING_SIZE, self.POOLING_SIZE), spatial_scale=1/16)
        # x = self.pooling(x)
        # x = x.squeeze()
        # velocities_pred = x.view(-1, 4)
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
    # 'base.0.weight',
    # 'base.4.0.conv1.weight',
    # 'base.4.0.conv2.weight',
    # 'base.4.2.conv1.weight',
    # 'base.4.2.conv2.weight',
    # 'base.6.0.conv1.weight',
    # 'base.6.0.conv2.weight',
    # 'base.6.0.downsample.0.weight',
    # 'base.6.2.conv1.weight',
    # 'base.6.2.conv2.weight',
    # 'base.8.0.conv1.weight',
    # 'base.8.0.conv2.weight',
    # 'base.8.0.downsample.0.weight',
    # 'base.8.2.conv1.weight',
    # 'base.8.2.conv2.weight',
    # 'base.10.0.conv1.weight',
    # 'base.10.0.conv2.weight',
    # 'base.10.0.downsample.0.weight',
    # 'base.10.2.conv1.weight',
    # 'base.10.2.conv2.weight',
    # 'conv1x1.weight',
]


if __name__ == "__main__":

    model = PropagationNetwork()
    print([p.requires_grad for p in model.parameters()])
    print(list(model.children()))
