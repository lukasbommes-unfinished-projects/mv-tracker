import pickle
import torch
import torch.nn as nn
import torchvision

import sys
sys.path.append("..")

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=7*7*4, kernel_size=(3, 3), padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=7, stride=7)

    def forward(self, motion_vectors, boxes_prev, motion_vector_scale):
        print(motion_vectors.shape)
        x = self.conv1(motion_vectors)
        x = self.relu1(x)

        spatial_scale = 1. * motion_vector_scale[0, 0]
        x = torchvision.ops.ps_roi_pool(x, boxes_prev, output_size=(7, 7), spatial_scale=spatial_scale)
        x = self.pool1(x)
        return x.squeeze()

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

motion_vectors = pickle.load(open("data/motion_vectors/00000000.pkl", "rb"))
boxes_prev = pickle.load(open("data/boxes_prev/00000000.pkl", "rb"))
velocities = pickle.load(open("data/velocities/00000000.pkl", "rb"))
num_boxes_mask = pickle.load(open("data/num_boxes_mask/00000000.pkl", "rb"))
motion_vector_scale = pickle.load(open("data/motion_vector_scale/00000000.pkl", "rb"))

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

criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


model.train()

for step in range(500):

    velocities_pred = model(motion_vectors, boxes_prev, motion_vector_scale)

    print("num_boxes_mask.shape", num_boxes_mask.shape)
    print("velocities_pred.shape", velocities_pred.shape)
    print("velocities.shape", velocities.shape)

    loss = criterion(velocities_pred, velocities)
    print("velocities_pred.shape", velocities_pred.shape)
    print("velocities.shape", velocities.shape)
    print(step, loss.item())

    optimizer.zero_grad()

    with torch.set_grad_enabled(True):
        loss.backward()
        optimizer.step()
