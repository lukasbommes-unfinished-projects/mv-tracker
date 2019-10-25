import os
import time
import datetime
import copy
import argparse
from tqdm import tqdm
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision

from lib.models.pnet_dense import PropagationNetwork, layer_keys
from lib.dataset.dataset_new import MotionVectorDataset
from lib.dataset.stats import StatsMpeg4DenseStaticSinglescale as Stats
from lib.transforms.transforms import StandardizeMotionVectors, \
    StandardizeVelocities, RandomFlip, RandomMotionChange
from lib.losses.losses import IouLoss
from lib.dataset.velocities import box_from_velocities
from lib.utils import compute_mean_iou, weight_checksum, count_params


torch.set_printoptions(precision=10)
torch.autograd.set_detect_anomaly(True)


def log_weights(model, epoch, writer):
    state_dict = model.state_dict()
    is_parallel = "module" in list(state_dict.keys())[0]
    for key in layer_keys:
        if is_parallel:
            key = "module.{}".format(key)
        weights = state_dict[key].detach().cpu().flatten().numpy()
        writer.add_histogram(key, weights, global_step=epoch, bins='tensorflow')


def train(model, criterion, optimizer, scheduler, batch_size, num_epochs,
    write_tensorboard_log, save_model, outdir, logger):
    tstart = time.time()
    if write_tensorboard_log:
        writer = SummaryWriter()

    best_loss = 99999.0
    best_mean_iou = 0.0
    iterations = {"train": 0, "val": 0}

    logger.info("Weight sum before training: {}".format(weight_checksum(model)))

    for epoch in range(num_epochs):

        # get current learning rate
        learning_rate = 0
        for param_group in optimizer.param_groups:
            learning_rate = param_group['lr']

        logger.info("Epoch {}/{} - Learning rate: {}".format(epoch, num_epochs-1, learning_rate))
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

                motion_vectors = sample["motion_vectors"]
                boxes_prev = sample["boxes_prev"]
                boxes = sample["boxes"]
                velocities = sample["velocities"]

                # move to GPU
                motion_vectors = motion_vectors.to(device)
                boxes_prev = boxes_prev.to(device)
                boxes = boxes.to(device)
                velocities = velocities.to(device)

                with torch.set_grad_enabled(phase == "train"):

                    boxes_prev = boxes_prev.view(-1, 5)
                    boxes = boxes.view(-1, 5)
                    velocities = velocities.view(-1, 4)

                    velocities_pred = model(motion_vectors, boxes_prev)

                    criterion_iou, criterion_velocity = criterion

                    # compute loss for velocities
                    loss_velocities = criterion_velocity(velocities_pred, velocities)

                    # undo normalization of velocity and predict boxes
                    velocities_mean = torch.tensor(Stats.velocities["mean"]).to(device)
                    velocities_std = torch.tensor(Stats.velocities["std"]).to(device)
                    velocities_pred = velocities_pred * velocities_std + velocities_mean
                    boxes_pred = box_from_velocities(boxes_prev[:, 1:], velocities_pred)

                    # compute loss for box IoU
                    loss_iou = criterion_iou(boxes_pred, boxes[:, 1:])
                    loss = loss_velocities + loss_iou

                    # log loss ratio
                    loss_ratio = loss_velocities.item() / loss_iou.item()

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
                    boxes = boxes.detach().cpu().view(-1, 5)
                    boxes_prev = boxes_prev.detach().cpu().view(-1, 5)
                    velocities_pred = velocities_pred.detach().cpu().view(-1, 4)
                    # velocities are already denormlized so we can directly predict boxes
                    boxes_pred = box_from_velocities(boxes_prev[:, 1:], velocities_pred)
                    mean_iou = compute_mean_iou(boxes_pred, boxes[:, 1:])
                    running_mean_iou.append(mean_iou)

                    if write_tensorboard_log:
                        writer.add_scalar('Loss/{}'.format(phase), loss.item(), iterations[phase])
                        writer.add_scalar('Mean IoU/{}'.format(phase), mean_iou, iterations[phase])
                        writer.add_scalar('Loss Ratio/{}'.format(phase), loss_ratio, iterations[phase])

                    iterations[phase] += 1

                pbar.update()

            pbar.close()

            # epoch loss and IoU
            epoch_loss = np.mean(running_loss)
            epoch_mean_iou = np.mean(running_mean_iou)
            logger.info('{} Loss: {}; {} Mean IoU: {}'.format(phase, epoch_loss, phase, epoch_mean_iou))
            if write_tensorboard_log:
                writer.add_scalar('Epoch Loss/{}'.format(phase), epoch_loss, epoch)
                writer.add_scalar('Epoch Mean IoU/{}'.format(phase), epoch_mean_iou, epoch)

            if phase == "val" and epoch_loss <= best_loss:
                best_loss = epoch_loss
                if save_model:
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, os.path.join(outdir, "model_lowest_loss.pth"))

            if phase == "val" and epoch_mean_iou >= best_mean_iou:
                best_mean_iou = epoch_mean_iou
                if save_model:
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, os.path.join(outdir, "model_highest_iou.pth"))

        if scheduler:
            scheduler.step()

        logger.info("velocities {}".format(velocities))
        velocities_mean = torch.tensor(Stats.velocities["mean"]).to(device)
        velocities_std = torch.tensor(Stats.velocities["std"]).to(device)
        velocities_pred = (velocities_pred - velocities_mean) / velocities_std
        logger.info("velocities_pred {}".format(velocities_pred))
        logger.info("boxes_prev {}".format(boxes_prev))
        logger.info("boxes {}".format(boxes))
        logger.info("boxes_pred {}".format(boxes_pred))

        if write_tensorboard_log:
            log_weights(model, epoch, writer)

    time_elapsed = time.time() - tstart
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info('Lowest validation loss: {}'.format(best_loss))

    logger.info("Weight sum after training: {}".format(weight_checksum(model)))

    if save_model:
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, os.path.join(outdir, "model_final.pth"))

    if write_tensorboard_log:
        writer.close()


def parse_args():
    parser = argparse.ArgumentParser()
    # dataset params
    parser.add_argument('--root_dir', type=str, default="data")
    parser.add_argument('--codec', type=str, default="mpeg4")
    parser.add_argument('--mvs_mode', type=str, default="dense")
    parser.add_argument('--scales', nargs='+', type=float, default=1.0)
    parser.add_argument('--static_only', dest='static_only', action='store_true')
    parser.set_defaults(static_only=False)
    parser.add_argument('--with_keyframes', dest='with_keyframes', action='store_true')
    parser.set_defaults(with_keyframes=False)
    parser.add_argument('--no_shuffle', dest='no_shuffle', action='store_true')
    parser.set_defaults(no_shuffle=False)
    parser.add_argument('--batch_size', type=int, default=8)
    # training params
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--scheduler_frequency', type=int, default=20)
    parser.add_argument('--scheduler_factor', type=float, default=0.1)
    parser.add_argument('--gpus', nargs='+', type=int, default=0)
    return parser.parse_args()


# make sure batch size of data in data_precomputed is 1, otherwise we get "CUDA out of memory"
if __name__ == "__main__":

    log_to_file = False
    save_model = False
    write_tensorboard_log = False

    args = parse_args()
    if isinstance(args.scales, float):
        args.scales = [args.scales]
    if isinstance(args.gpus, int):
        args.gpus = [args.gpus]

    # create output directory
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outdir = os.path.join("models", "tracker", date)
    if log_to_file or save_model:
        os.makedirs(outdir, exist_ok=True)

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
    logger.info(f"root_dir: {args.root_dir}")
    logger.info(f"codec: {args.codec}")
    logger.info(f"mvs_mode: {args.mvs_mode}")
    logger.info(f"scales: {args.scales}")
    logger.info(f"static_only: {args.static_only}")
    logger.info(f"with_keyframes: {args.with_keyframes}")
    logger.info(f"no_shuffle: {args.no_shuffle}")
    logger.info(f"batch_size: {args.batch_size}")
    logger.info(f"learning_rate: {args.learning_rate}")
    logger.info(f"num_epochs: {args.num_epochs}")
    logger.info(f"weight_decay: {args.weight_decay}")
    logger.info(f"scheduler_frequency: {args.scheduler_frequency}")
    logger.info(f"scheduler_factor: {args.scheduler_factor}")
    logger.info(f"gpus: {args.gpus}")

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
    datasets = {x: MotionVectorDataset(root_dir=args.root_dir, transforms=transforms[x],
        codec=args.codec, scales=args.scales, mvs_mode=args.mvs_mode, static_only=args.static_only,
        exclude_keyframes=(not args.with_keyframes), visu=False, debug=False,
        mode=x) for x in modes}

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=1,
        shuffle=(not args.no_shuffle), num_workers=12) for x in modes}

    device = torch.device("cuda:{}".format(args.gpus[0]) if torch.cuda.is_available() else "cpu")

    model = PropagationNetwork()
    if len(args.gpus) > 1 and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=args.gpus)
    model = model.to(device)

    criterion = (IouLoss(), nn.SmoothL1Loss(reduction='mean'))
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999),
        eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_frequency, gamma=args.scheduler_factor)

    logger.info(f"model: {model}")
    logger.info(f"model requires_grad: {[p.requires_grad for p in model.parameters()]}")
    logger.info(f"model param count: {count_params(model)}")

    logger.info(f"loss criterion: {criterion}")
    logger.info(f"optimizer: {optimizer}")
    logger.info(f"transforms: {transforms['train']}")

    train(model, criterion, optimizer, scheduler=scheduler, batch_size=args.batch_size,
        num_epochs=args.num_epochs, write_tensorboard_log=write_tensorboard_log,
        save_model=save_model, outdir=outdir, logger=logger)