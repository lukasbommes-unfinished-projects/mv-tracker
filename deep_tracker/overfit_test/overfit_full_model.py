import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision

import sys
sys.path.append("..")
from lib.resnet_atrous import resnet18
from lib.utils import load_pretrained_weights_to_modified_resnet


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

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

        print([p.requires_grad for p in self.base.parameters()])
        print(list(self.children()))


    def forward(self, motion_vectors, boxes_prev, motion_vector_scale):
        x = self.base(motion_vectors)
        x = self.conv1x1(x)
        #x = F.relu(x)  # when using relu the output can not become negative

        # compute ratio of input size to size of base output
        spatial_scale = 1/16 * motion_vector_scale[0, 0]
        x = torchvision.ops.ps_roi_pool(x, boxes_prev, output_size=(self.POOLING_SIZE, self.POOLING_SIZE), spatial_scale=spatial_scale)
        x = self.pooling(x)
        velocities_pred = x.squeeze()
        return velocities_pred


def _frame_idx_to_batch_idx(boxes):
    """Converts unique frame_idx in first column of boxes into batch index."""
    frame_idxs = torch.unique(boxes[:, 0])
    for batch_idx, frame_idx in enumerate(frame_idxs):
        idx = torch.where(boxes == frame_idx)[0]
        boxes[idx, 0] = batch_idx
    return boxes


def _change_box_format(boxes):
    """Change format of boxes from [idx, x, y, w, h] to [idx, x1, y1, x2, y2]."""
    boxes[..., 0] = boxes[..., 0]
    boxes[..., 1] = boxes[..., 1]
    boxes[..., 2] = boxes[..., 2]
    boxes[..., 3] = boxes[..., 1] + boxes[..., 3]
    boxes[..., 4] = boxes[..., 2] + boxes[..., 4]
    return boxes


device = torch.device('cuda')

motion_vectors = pickle.load(open("data/train/motion_vectors/00000000.pkl", "rb"))
boxes_prev = pickle.load(open("data/train/boxes_prev/00000000.pkl", "rb"))
velocities = pickle.load(open("data/train/velocities/00000000.pkl", "rb"))
num_boxes_mask = pickle.load(open("data/train/num_boxes_mask/00000000.pkl", "rb"))
motion_vector_scale = pickle.load(open("data/train/motion_vector_scale/00000000.pkl", "rb"))

# select smaller batch
motion_vectors = motion_vectors[:8, ...]
boxes_prev = boxes_prev[:8, ...]
velocities = velocities[:8, ...]
num_boxes_mask = num_boxes_mask[:8, ...]
motion_vector_scale = motion_vector_scale[:8, ...]

velocities = velocities[num_boxes_mask]
velocities = velocities.view(-1, 4)

boxes_prev = _change_box_format(boxes_prev)
boxes_prev = boxes_prev[num_boxes_mask]
boxes_prev = boxes_prev.view(-1, 5)
boxes_prev = _frame_idx_to_batch_idx(boxes_prev)

motion_vectors = motion_vectors.to(device)
boxes_prev = boxes_prev.to(device)
velocities = velocities.to(device)
motion_vector_scale = motion_vector_scale.to(device)


batch_size = motion_vectors.shape[0]

model = MyModel()
model = model.to(device)

#criterion = nn.MSELoss(reduction='sum')
criterion = nn.SmoothL1Loss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=False)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)


model.train()

for epoch in range(100000):

    velocities_pred = model(motion_vectors, boxes_prev, motion_vector_scale)

    print("num_boxes_mask.shape", num_boxes_mask.shape)
    print("velocities_pred.shape", velocities_pred.shape)
    print("velocities.shape", velocities.shape)

    loss = criterion(velocities_pred, velocities)
    print("velocities_pred.shape", velocities_pred.shape)
    print("velocities.shape", velocities.shape)

    for param_group in optimizer.param_groups:
        current_lr = param_group['lr']

    print(epoch, loss.item(), current_lr)

    optimizer.zero_grad()

    with torch.set_grad_enabled(True):
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

    if epoch % 50 == 0:
        print(velocities)
        print(velocities_pred)
