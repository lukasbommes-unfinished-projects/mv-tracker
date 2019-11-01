import os
import time
import datetime
import copy
import logging
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter

from lib.models.pnet_upsampled import PropagationNetwork
from lib.dataset.dataset_precomputed import MotionVectorDatasetPrecomputed
from lib.transforms.transforms import StandardizeMotionVectors, \
    StandardizeVelocities, RandomFlip, RandomMotionChange
from lib.dataset.velocities import box_from_velocities
from lib.utils import compute_mean_iou, weight_checksum, count_params
from lib.visualize_model import Visualizer

from lib.dataset.stats import StatsMpeg4StaticP as Stats


torch.set_printoptions(precision=10)
torch.autograd.set_detect_anomaly(True)


def log_weights(model, epoch, writer):
    state_dict = model.state_dict()
    is_parallel = "module" in list(state_dict.keys())[0]
    for key in model.layer_keys:
        if is_parallel:
            key = "module.{}".format(key)
        weights = state_dict[key].detach().cpu().flatten().numpy()
        writer.add_histogram(key, weights, global_step=epoch, bins='tensorflow')


def train(model, criterion, optimizer, scheduler, num_epochs, visu,
    write_tensorboard_log, save_model, outdir, logger):
    tstart = time.time()
    if write_tensorboard_log:
        writer = SummaryWriter()
    if visu:
        visualizer = Visualizer()

    best_loss = 99999.0
    best_mean_iou = 0.0
    iterations = {"train": 0, "val": 0}

    logger.info("Weight sum before training: {}".format(weight_checksum(model)))

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

                optimizer.zero_grad()

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
                        optimizer.step()

                        # monitor magnitude of weights to weights update (should be around 0.001)
                        if write_tensorboard_log:
                            params_after_update = [p.detach().clone() for p in model.parameters()]
                            params_norm = torch.norm(torch.cat([p.flatten() for p in params_before_update], axis=0))
                            updates = [(pa - pb).flatten() for pa, pb in zip(params_after_update, params_before_update)]
                            updates_norm = torch.norm(torch.cat(updates, axis=0))
                            writer.add_scalar('update to weight ratio', updates_norm / params_norm, iterations["train"])

                    pbar.update()

                # log loss
                running_loss.append(loss.item())
                if write_tensorboard_log:
                    writer.add_scalar('Loss/{}'.format(phase), loss.item(), iterations[phase])

                # log mean IoU of all predicted and ground truth boxes
                if write_tensorboard_log:
                    boxes_ = boxes[num_boxes_mask].detach().clone().view(-1, 5)
                    boxes_prev_ = boxes_prev[num_boxes_mask].detach().clone().view(-1, 5)
                    velocities_pred_ = velocities_pred.detach().clone().view(-1, 4)
                    # denormalize velocities before predicting boxes
                    velocities_mean = torch.tensor(Stats.velocities["mean"]).to(device)
                    velocities_std = torch.tensor(Stats.velocities["std"]).to(device)
                    velocities_pred_ = velocities_pred_ * velocities_std + velocities_mean
                    boxes_pred_ = box_from_velocities(boxes_prev_[:, 1:], velocities_pred_)
                    mean_iou = compute_mean_iou(boxes_pred_, boxes_[:, 1:])
                    running_mean_iou.append(mean_iou)
                    writer.add_scalar('Mean IoU/{}'.format(phase), mean_iou, iterations[phase])

                iterations[phase] += 1

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
                    logger.info("Saving model with lowest loss so far")
                    torch.save(best_model_wts, os.path.join(outdir, "model_lowest_loss.pth"))

            if phase == "val" and epoch_mean_iou >= best_mean_iou:
                best_mean_iou = epoch_mean_iou
                if save_mode:
                    best_model_wts = copy.deepcopy(model.state_dict())
                    logger.info("Saving model with highest IoU so far")
                    torch.save(best_model_wts, os.path.join(outdir, "model_highest_iou.pth"))

            # store every model during the last 10 epochs of training
            if save_mode and phase == "val" and (num_epochs - epoch) <= 10:
                model_wts = copy.deepcopy(model.state_dict())
                torch.save(model_wts, os.path.join(outdir, "model_epoch_{}.pth".format(epoch)))


        if scheduler:
            scheduler.step()

        if write_tensorboard_log:
            log_weights(model, epoch, writer)

    time_elapsed = time.time() - tstart
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Lowest validation loss: {}'.format(best_loss))

    logger.info("Weight sum after training: {}".format(weight_checksum(model)))

    if save_model:
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, os.path.join(outdir, "model_final.pth"))
        torch.save({
            'epoch': epoch,
            'model_state_dict': copy.deepcopy(model.state_dict()),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': best_loss,
            'best_mean_iou': best_mean_iou,
            'last_epoch_loss': epoch_loss,
            'last_epoch_mean_iou': epoch_mean_iou,
            'iterations': iterations
        }, os.path.join(outdir, "final_training_state.pth"))


    if write_tensorboard_log:
        writer.close()


if __name__ == "__main__":

    root_dir = "data_precomputed_mpeg4_static_p"
    codec = "mpeg4"
    static_only = True
    shuffle = True
    learning_rate = 1e-4
    num_epochs = 160
    weight_decay = 0.0005
    scheduler_frequency = 40
    scheduler_factor = 0.1
    gpu = 1

    log_to_file = True
    save_model = True
    write_tensorboard_log = True
    save_normalization_stats = True

    # create output directory
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outdir = os.path.join("models", "tracker", date)
    if log_to_file or save_model or save_normalization_stats:
        os.makedirs(outdir, exist_ok=True)

    if save_normalization_stats:
        pickle.dump(Stats(), open(os.path.join(outdir, "stats.pkl"), "wb"))

    # setup logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    if log_to_file:
        fh = logging.FileHandler(os.path.join(outdir, 'train.log'))
        logger.addHandler(fh)

    logger.info("Model will be trained with the following options")
    logger.info(f"outdir: {outdir}")
    logger.info(f"root_dir: {root_dir}")
    logger.info(f"codec: {codec}")
    logger.info(f"static_only: {static_only}")
    logger.info(f"shuffle: {shuffle}")
    logger.info(f"learning_rate: {learning_rate}")
    logger.info(f"num_epochs: {num_epochs}")
    logger.info(f"weight_decay: {weight_decay}")
    logger.info(f"scheduler_frequency: {scheduler_frequency}")
    logger.info(f"scheduler_factor: {scheduler_factor}")
    logger.info(f"gpu: {gpu}")

    transforms = {
        "train": torchvision.transforms.Compose([
            RandomFlip(directions=["x", "y"]),
            StandardizeVelocities(mean=Stats.velocities["mean"], std=Stats.velocities["std"]),
            StandardizeMotionVectors(mean=Stats.motion_vectors["mean"], std=Stats.motion_vectors["std"]),
            RandomMotionChange(scale=1.0),
        ]),
        "val": torchvision.transforms.Compose([
            StandardizeVelocities(mean=Stats.velocities["mean"], std=Stats.velocities["std"]),
            StandardizeMotionVectors(mean=Stats.motion_vectors["mean"], std=Stats.motion_vectors["std"]),
        ])
    }

    modes = ["train", "val"]
    datasets = {x: MotionVectorDatasetPrecomputed(root_dir=os.path.join(root_dir, x),
        transforms=transforms[x]) for x in modes}
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=1,
        shuffle=shuffle, num_workers=8) for x in modes}

    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")

    model = PropagationNetwork()
    model = model.to(device)

    criterion = nn.SmoothL1Loss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999),
        eps=1e-08, weight_decay=weight_decay, amsgrad=False)  # weight_decay=0.0001
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_frequency,
        gamma=scheduler_factor)

    logger.info(f"model: {model}")
    logger.info(f"model requires_grad: {[p.requires_grad for p in model.parameters()]}")
    logger.info(f"model param count: {count_params(model)}")

    logger.info(f"optimizer: {optimizer}")
    logger.info(f"transforms: {transforms['train']}")

    train(model, criterion, optimizer, scheduler=scheduler, num_epochs=num_epochs,
        visu=False, write_tensorboard_log=write_tensorboard_log,
        save_model=save_model, outdir=outdir, logger=logger)
