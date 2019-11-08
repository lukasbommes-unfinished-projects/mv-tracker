import pickle
import copy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.utils.tensorboard import SummaryWriter
import torchvision

import os
import time
import sys
sys.path.append("..")
from lib.resnet_atrous import resnet18
from lib.utils import load_pretrained_weights_to_modified_resnet
from lib.dataset.dataset_precomputed import MotionVectorDatasetPrecomputed
from lib.visualize_model import Visualizer


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

        boxes_prev = self._change_box_format(boxes_prev)
        boxes_prev = boxes_prev.view(-1, 5)
        boxes_prev = self._frame_idx_to_batch_idx(boxes_prev)

        # compute ratio of input size to size of base output
        spatial_scale = 1/16 * motion_vector_scale[0, 0]
        x = torchvision.ops.ps_roi_pool(x, boxes_prev, output_size=(self.POOLING_SIZE, self.POOLING_SIZE), spatial_scale=spatial_scale)
        x = self.pooling(x)
        velocities_pred = x.squeeze()
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


device = torch.device('cuda')

model = MyModel()
model = model.to(device)

#criterion = nn.MSELoss(reduction='sum')
criterion = nn.SmoothL1Loss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)  # weight_decay=0.0001
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

visualizer = Visualizer()


def train(model, criterion, optimizer, scheduler, num_epochs=2):
    tstart = time.time()
    writer = SummaryWriter()

    best_model_wts = copy.deepcopy(model.state_dict())
    pickle.dump(best_model_wts, open("models/best_model.pkl", "wb"))
    best_loss = 99999.0
    iterations = {"train": 0, "val": 0}

    for epoch in range(num_epochs):

        # get current learning rate
        learning_rate = 0
        for param_group in optimizer.param_groups:
            learning_rate = param_group['lr']

        print("Epoch {}/{} - Learning rate: {}".format(epoch, num_epochs-1, learning_rate))
        writer.add_scalar('Learning Rate', learning_rate, epoch)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            pbar = tqdm(total=len(dataloaders[phase]))
            for step, (motion_vectors, boxes_prev, velocities, _, motion_vector_scale) in enumerate(dataloaders[phase]):

                # remove batch dimension as precomputed data is already batched
                motion_vectors.squeeze_(0)
                boxes_prev.squeeze_(0)
                velocities.squeeze_(0)
                motion_vector_scale.squeeze_(0)

                # select smaller batch
                motion_vectors = motion_vectors[:8, ...]
                boxes_prev = boxes_prev[:8, ...]
                velocities = velocities[:8, ...]
                motion_vector_scale = motion_vector_scale[:8, ...]

                # move to GPU
                motion_vectors = motion_vectors.to(device)
                boxes_prev = boxes_prev.to(device)
                velocities = velocities.to(device)
                motion_vector_scale = motion_vector_scale.to(device)

                # normalize velocities to range [0, 1]
                #vel_min = torch.min(velocities)
                #vel_max = torch.max(velocities)
                #velocities = (velocities - vel_min) / (vel_max - vel_min)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):

                    # visualize model inputs
                    visualizer.save_inputs(motion_vectors, boxes_prev, motion_vector_scale, velocities)

                    velocities = velocities.view(-1, 4)

                    velocities_pred = model(motion_vectors, boxes_prev, motion_vector_scale)
                    velocities_pred = velocities_pred.view(-1, 4)

                    # visualize model outputs
                    visualizer.save_outputs(velocities_pred)
                    visualizer.show()

                    loss = criterion(velocities_pred, velocities)

                    if phase == "train":
                        params_before_update = list(model.parameters())[0].clone()
                        loss.backward()
                        optimizer.step()
                        params_after_update = list(model.parameters())[0].clone()

                        # check if model parameters are still being updated
                        if torch.allclose(params_before_update.data, params_after_update.data):
                            raise RuntimeError("The model stopped learning. Parameters are not getting updated anymore.")

                    pbar.update()

                writer.add_scalar('Loss/{}'.format(phase), loss.item(), iterations[phase])
                iterations[phase] += 1

            pbar.close()

            epoch_loss = loss.item()
            print('{} Loss: {}'.format(phase, epoch_loss))
            writer.add_scalar('Epoch Loss/{}'.format(phase), epoch_loss, epoch)

            #if phase == "val":
            #    model_wts = copy.deepcopy(model.state_dict())
            #    pickle.dump(model_wts, open("models/model_{:04d}.pkl".format(epoch), "wb"))

            if phase == "val" and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                pickle.dump(best_model_wts, open("models/best_model.pkl", "wb"))

            #if phase == "val" and scheduler:
            #    scheduler.step(epoch_loss)

        scheduler.step()

        if epoch % 50 == 0:
            print(velocities)
            print(velocities_pred)

    time_elapsed = time.time() - tstart
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Lowest validation loss: {}'.format(best_loss))

    model.load_state_dict(best_model_wts)
    writer.close()
    return model


root_dir = "data"
#root_dir = "data_whole_video"
#root_dir = "data_10_samples"

modes = ["train", "val"]
datasets = {x: MotionVectorDatasetPrecomputed(root_dir=os.path.join(root_dir, x)) for x in modes}
dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=1, shuffle=False, num_workers=0) for x in modes}

best_model = train(model, criterion, optimizer, scheduler=scheduler, num_epochs=600)
