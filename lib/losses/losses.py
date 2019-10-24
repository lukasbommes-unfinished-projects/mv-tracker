import torch
import torch.nn as nn
import torchvision
from lib.utils import change_box_format


class IouLoss(nn.Module):
    def __init__(self):
        super(IouLoss, self).__init__()

    def forward(self, boxes_pred, boxes):
        boxes_pred_ = change_box_format(boxes_pred)
        boxes_ = change_box_format(boxes)
        mean_iou = torchvision.ops.boxes.box_iou(boxes_pred_, boxes_).diag().mean()
        loss = -1 * torch.log(mean_iou)
        return loss

    def __repr__(self):
        return "IouLoss ()"
