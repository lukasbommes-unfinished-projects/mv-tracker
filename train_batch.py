import os
import time
import datetime
import copy
import argparse
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from lib.models.pnet_upsampled import PropagationNetwork, layer_keys
from lib.dataset.dataset_precomputed import MotionVectorDatasetPrecomputed
from lib.dataset.velocities import box_from_velocities
from lib.utils import compute_mean_iou
from lib.visualize_model import Visualizer


torch.set_printoptions(precision=10)

def weight_checksum(model):
    checksum = 0
    for p in model.parameters():
        s = p.data.sum().detach().cpu().numpy()
        checksum += s
    return checksum


def log_weights(model, epoch, writer):
    state_dict = model.state_dict()
    is_parallel = "module" in list(state_dict.keys())[0]
    for key in layer_keys:
        if is_parallel:
            key = "module.{}".format(key)
        weights = state_dict[key].detach().cpu().flatten().numpy()
        writer.add_histogram(key, weights, global_step=epoch, bins='tensorflow')


def train(model, criterion, optimizer, scheduler, batch_size, num_epochs,
    visu=False, write_tensorboard_log=True, save_model=True):
    tstart = time.time()
    if write_tensorboard_log:
        writer = SummaryWriter()
    if visu:
        visualizer = Visualizer()

    # create output directory
    if save_model:
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        outdir_name = os.path.join("models", "tracker", date)
        os.makedirs(outdir_name, exist_ok=True)

    best_loss = 99999.0
    best_mean_iou = 0.0
    iterations = {"train": 0, "val": 0}

    print("Weight sum before training: {}".format(weight_checksum(model)))

    for epoch in range(num_epochs):

        # get current learning rate
        learning_rate = 0
        for param_group in optimizer.param_groups:
            learning_rate = param_group['lr']

        print("Epoch {}/{} - Learning rate: {}".format(epoch, num_epochs-1, learning_rate))
        if write_tensorboard_log:
            writer.add_scalar('Learning Rate', learning_rate, epoch)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = []
            running_mean_iou = []

            optimizer.zero_grad()

            pbar = tqdm(total=len(dataloaders[phase]))
            for step, sample in enumerate(dataloaders[phase]):

                # remove batch dimension as batch size is always 1
                for item in sample.values():
                    item.squeeze_(0)

                motion_vectors = sample["motion_vectors"]
                boxes_prev = sample["boxes_prev"]
                boxes = sample["boxes"]
                velocities = sample["velocities"]
                num_boxes_mask = sample["num_boxes_mask"]

                # move to GPU
                motion_vectors = motion_vectors.to(device)
                boxes_prev = boxes_prev.to(device)
                boxes = boxes.to(device)
                velocities = velocities.to(device)
                num_boxes_mask = num_boxes_mask.to(device)

                with torch.set_grad_enabled(phase == "train"):

                    # visualize model inputs
                    if visu:
                        visualizer.save_inputs(motion_vectors, boxes_prev, boxes, velocities)

                    velocities = velocities[num_boxes_mask].view(-1, 4)
                    velocities_pred = model(motion_vectors, boxes_prev, num_boxes_mask)

                    # visualize model outputs
                    if visu:
                        visualizer.save_outputs(velocities_pred, num_boxes_mask)
                        visualizer.show()

                    loss = criterion(velocities_pred, velocities)

                    if phase == "train":
                        if write_tensorboard_log:
                            params_before_update = [p.detach().clone() for p in model.parameters()]

                        loss.backward()

                        # acumulate gradients before updating (roughly equivalent to batch size > 1)
                        if (step + 1) % batch_size == 0:
                            optimizer.step()
                            optimizer.zero_grad()

                            # monitor magnitude of weights to weights update (should be around 0.001)
                            if write_tensorboard_log:
                                params_after_update = [p.detach().clone() for p in model.parameters()]
                                params_norm = torch.norm(torch.cat([p.flatten() for p in params_before_update], axis=0))
                                updates = [(pa - pb).flatten() for pa, pb in zip(params_after_update, params_before_update)]
                                updates_norm = torch.norm(torch.cat(updates, axis=0))
                                writer.add_scalar('update to weight ratio', updates_norm / params_norm, iterations["train"])

                running_loss.append(loss.item())

                if (step + 1) % batch_size == 0:

                    # log loss and mean IoU of all predicted and ground truth boxes
                    boxes = boxes[num_boxes_mask].detach().view(-1, 5)
                    boxes_prev = boxes_prev[num_boxes_mask].detach().view(-1, 5)
                    velocities_pred = velocities_pred.detach().view(-1, 4)
                    boxes_pred = box_from_velocities(boxes_prev[:, 1:], velocities_pred)
                    mean_iou = compute_mean_iou(boxes_pred, boxes[:, 1:])
                    running_mean_iou.append(mean_iou)

                    if write_tensorboard_log:
                        writer.add_scalar('Loss/{}'.format(phase), loss.item(), iterations[phase])
                        writer.add_scalar('Mean IoU/{}'.format(phase), mean_iou, iterations[phase])

                    iterations[phase] += 1

                pbar.update()

            pbar.close()

            # epoch loss and IoU
            epoch_loss = np.mean(running_loss)
            epoch_mean_iou = np.mean(running_mean_iou)
            print('{} Loss: {}; {} Mean IoU: {}'.format(phase, epoch_loss, phase, epoch_mean_iou))
            if write_tensorboard_log:
                writer.add_scalar('Epoch Loss/{}'.format(phase), epoch_loss, epoch)
                writer.add_scalar('Epoch Mean IoU/{}'.format(phase), epoch_mean_iou, epoch)

            if phase == "val" and epoch_loss <= best_loss:
                best_loss = epoch_loss
                if save_model:
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, os.path.join(outdir_name, "model_lowest_loss_epoch_{}.pth".format(epoch)))

            if phase == "val" and epoch_mean_iou >= best_mean_iou:
                best_mean_iou = epoch_mean_iou
                if save_model:
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, os.path.join(outdir_name, "model_highest_iou_epoch_{}.pth".format(epoch)))

            #if phase == "val" and scheduler:
            #    scheduler.step(epoch_loss)

        if scheduler:
            scheduler.step()

        if write_tensorboard_log:
            log_weights(model, epoch, writer)

        print(velocities)
        print(velocities_pred)

    time_elapsed = time.time() - tstart
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Lowest validation loss: {}'.format(best_loss))

    print("Weight sum after training: {}".format(weight_checksum(model)))

    if save_model:
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, os.path.join(outdir_name, "model_final.pth"))

    if write_tensorboard_log:
        writer.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    return parser.parse_args()


# make sure batch size of data in data_precomputed is 1, otherwise we get "CUDA out of memory"
if __name__ == "__main__":

    args = parse_args()

    root_dir = "data_precomputed"
    train_parallel = False
    modes = ["train", "val"]
    datasets = {x: MotionVectorDatasetPrecomputed(root_dir=os.path.join(root_dir, x)) for x in modes}
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=1, shuffle=True, num_workers=8) for x in modes}

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using learning rate {args.learning_rate}")
    print(f"Using batch size {args.batch_size}")

    model = PropagationNetwork()
    if train_parallel and torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    model = model.to(device)

    criterion = nn.SmoothL1Loss(reduction='mean')
    #criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999),
        eps=1e-08, weight_decay=0.0005, amsgrad=False)  # weight_decay=0.0001
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
    #    factor=0.1, patience=10, )
    train(model, criterion, optimizer, scheduler=scheduler, batch_size=args.batch_size,
        num_epochs=120, visu=True, write_tensorboard_log=True,
        save_model=False)
