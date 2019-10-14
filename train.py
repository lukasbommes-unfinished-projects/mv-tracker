import os
import time
import copy
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from lib.model import PropagationNetwork
from lib.dataset.dataset_precomputed import MotionVectorDatasetPrecomputed
from lib.visualize_model import Visualizer


torch.set_printoptions(precision=10)

def train(model, criterion, optimizer, scheduler, num_epochs=2, visu=False):
    tstart = time.time()
    writer = SummaryWriter()

    best_model_wts = copy.deepcopy(model.state_dict())
    torch.save(best_model_wts, "models/tracker/best_model.pth")
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

            running_loss = []

            pbar = tqdm(total=len(dataloaders[phase]))
            for step, sample in enumerate(dataloaders[phase]):

                # remove batch dimension as batch size is always 1
                for item in sample.values():
                    item.squeeze_(0)

                motion_vectors = sample["motion_vectors"]
                boxes_prev = sample["boxes_prev"]
                velocities = sample["velocities"]
                num_boxes_mask = sample["num_boxes_mask"]

                # move to GPU
                motion_vectors = motion_vectors.to(device)
                boxes_prev = boxes_prev.to(device)
                velocities = velocities.to(device)
                num_boxes_mask = num_boxes_mask.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):

                    # visualize model inputs
                    if visu:
                        visualizer.save_inputs(motion_vectors, boxes_prev, velocities)

                    velocities = velocities[num_boxes_mask].view(-1, 4)
                    velocities_pred = model(motion_vectors, boxes_prev, num_boxes_mask)

                    # visualize model outputs
                    if visu:
                        visualizer.save_outputs(velocities_pred, num_boxes_mask)
                        visualizer.show()

                    loss = criterion(velocities_pred, velocities)

                    if phase == "train":
                        params_before_update = [p.detach().clone() for p in model.parameters()]

                        loss.backward()
                        optimizer.step()

                        # monitor magnitude of weights to weights update (should be around 0.001)
                        params_after_update = [p.detach().clone() for p in model.parameters()]
                        params_norm = torch.norm(torch.cat([p.flatten() for p in params_before_update], axis=0))
                        updates = [(pa - pb).flatten() for pa, pb in zip(params_after_update, params_before_update)]
                        updates_norm = torch.norm(torch.cat(updates, axis=0))
                        writer.add_scalar('update to weight ratio', updates_norm / params_norm, iterations["train"])

                    pbar.update()

                running_loss.append(loss.item())
                writer.add_scalar('Loss/{}'.format(phase), loss.item(), iterations[phase])
                iterations[phase] += 1

            pbar.close()

            epoch_loss = np.mean(running_loss)
            print('{} Loss: {}'.format(phase, epoch_loss))
            writer.add_scalar('Epoch Loss/{}'.format(phase), epoch_loss, epoch)

            if phase == "val" and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, "models/tracker/best_model.pth")

            #if phase == "val" and scheduler:
            #    scheduler.step(epoch_loss)

        if scheduler:
            scheduler.step()

        print(velocities)
        print(velocities_pred)

    time_elapsed = time.time() - tstart
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Lowest validation loss: {}'.format(best_loss))

    model.load_state_dict(best_model_wts)
    writer.close()
    return model


if __name__ == "__main__":
    root_dir = "data_precomputed"
    train_parallel = True
    modes = ["train", "val"]
    datasets = {x: MotionVectorDatasetPrecomputed(root_dir=os.path.join(root_dir, x)) for x in modes}
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=1, shuffle=True, num_workers=8) for x in modes}

    visualizer = Visualizer()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = PropagationNetwork()
    if train_parallel and torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    model = model.to(device)

    criterion = nn.SmoothL1Loss(reduction='mean')
    #criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0005, amsgrad=False)  # weight_decay=0.0001
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
    #    factor=0.1, patience=10, )
    best_model = train(model, criterion, optimizer, scheduler=scheduler, num_epochs=80, visu=True)
