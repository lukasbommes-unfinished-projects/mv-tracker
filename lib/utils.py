import torch
from torch.nn.parameter import Parameter
import torchvision


def load_pretrained_weights_to_modified_resnet(cnn_model, pretrained_weights):
    pre_dict = cnn_model.state_dict()
    for key, val in pretrained_weights.items():
        if key[0:5] == 'layer':
            key_list = key.split('.')
            tmp = int(int(key_list[1]) * 2)
            key_list[1] = str(tmp)
            tmp_key = ''
            for i in range(len(key_list)):
                tmp_key = tmp_key + key_list[i] + '.'
            key = tmp_key[:-1]
        if isinstance(val, Parameter):
            val = val.data
        pre_dict[key].copy_(val)
    cnn_model.load_state_dict(pre_dict)


def count_params(model):
    """Return the number of parameters in the model."""
    params = list(model.parameters())
    count = 0
    for p in params:
        count += p.flatten().shape[0]
    return count


def weight_checksum(model):
    """Return the sum of all model parameters."""
    checksum = 0
    for p in model.parameters():
        s = p.data.sum().detach().cpu().numpy()
        checksum += s
    return checksum


def change_box_format(boxes):
    """Alters the box format from [(idx), xmin, ymin, w, h] to [(idx), x1, y1, x2, y2]."""
    boxes_ = boxes.clone()
    boxes_[:, -2] = boxes_[:, -2] + boxes_[:, -4]
    boxes_[:, -1] = boxes_[:, -1] + boxes_[:, -3]
    return boxes_


def compute_mean_iou(boxes_pred, boxes):
    """Compute the mean IoUs of all predicted and ground truth boxes

    Args:
        boxes_pred (`torch.tensor`): Predicted bounding boxes with shape (N, 4)
            where N is the number of boxes and each row has the format
            (xmin, xmax, w, h).

        boxes_pred (`torch.tensor`): Grount truth bounding boxes with same
            shape and format as predicted boxes. The box in the ith row of
            `boxes` is matched with the predicted box in the ith row of
            `boxes_pred`.

    Returns:
        (`float`) The mean of the IoUs of all N boxes.
    """
    # change format from (xmin, ymin, w, h) to (xmin, ymin, xmax, ymax)
    boxes_pred_ = change_box_format(boxes_pred)
    boxes_ = change_box_format(boxes)
    mean_iou = torchvision.ops.boxes.box_iou(boxes_pred_, boxes_).diag().mean()
    return float(mean_iou)
