from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import torch
from timm.loss import LabelSmoothingCrossEntropy
from torch import nn
from torch.nn import BCEWithLogitsLoss, NLLLoss2d, MSELoss, BCELoss
import torch.nn.functional as F



class LossCalculator(ABC):

    @abstractmethod
    def calculate_loss(self, outputs, sample):
        pass

class BCELossCalculator(LossCalculator):

    def __init__(self, pred_field="cls", **kwargs):
        super().__init__()
        self.pred_field = pred_field
        self.bce = BCEWithLogitsLoss(reduction='none')


    def calculate_loss(self, outputs, sample):
        neg = torch.from_numpy(np.array([7, 1, 1, 1, 1, 1, 1, 1])).cuda().float()
        pos = torch.from_numpy(np.array([14, 2, 2, 2, 2, 2, 2, 2])).cuda().float()

        targets = sample["label"].cuda().float()
        pred = outputs[self.pred_field]
        loss = self.bce(pred, targets)

        weights = pos * targets +  neg * (1 - targets)
        loss = (loss * weights).sum(axis=1)
        loss = loss / weights.sum(axis=1)
        return loss.sum()/targets.shape[0]

class BcePureLossCalc(LossCalculator):

    def __init__(self, pred_field="cls", **kwargs):
        super().__init__()
        self.pred_field = pred_field
        pos_weight = kwargs.get("pos_weight", None)
        if pos_weight:
            pos_weight = torch.ones((8,)).float() * pos_weight
            pos_weight = pos_weight.cuda().cuda()
        self.bce = BCEWithLogitsLoss(pos_weight=pos_weight)


    def calculate_loss(self, outputs, sample):
        targets = sample["label"].cuda().float()
        pred = outputs[self.pred_field]
        loss = self.bce(pred, targets)
        return loss


class CCELossCalculator(LossCalculator):
    def __init__(self, pred_field="cls", **kwargs):
        super().__init__()
        self.pred_field = pred_field
        self.bce = LabelSmoothingCrossEntropy(**kwargs)

    def calculate_loss(self, outputs, sample):
        targets = sample["label"].cuda().long()
        pred = outputs[self.pred_field]
        return self.bce(pred, targets)

def multiclass_jaccard(
        outputs,
        targets,
        min_pixels: int = 16,
        ignore_index: int = 255,
        eps: float = 1e-4,
        apply_softmax: bool = True
):
    batch_size = outputs.size()[0]
    if apply_softmax:
        probs = F.softmax(outputs, dim=1)
    else:
        probs = outputs
    num_classes = probs.size(1)
    # One hot encode targets
    dice_target = targets.contiguous().view(batch_size, -1).float()
    non_ignored_target = torch.zeros_like(dice_target).cuda().long()
    non_ignored_target[dice_target != ignore_index] = 1
    dice_target = F.one_hot((dice_target * non_ignored_target).long(), num_classes).moveaxis(2, 1)

    dice_output = probs.contiguous().view(batch_size, num_classes, -1)
    # zero out ignored probs
    dice_output = dice_output * non_ignored_target.unsqueeze(1).repeat(1, num_classes, 1).float()
    # do not compute for background
    dice_output = dice_output[:, 1:].contiguous()
    dice_target = dice_target[:, 1:].contiguous()
    # calculate soft jaccard loss per each class in image
    intersection = torch.sum(dice_output * dice_target, dim=2)
    losses = 1 - (intersection + eps) / (torch.sum(dice_output + dice_target, dim=2) - intersection + eps)
    # prepare pixel count per class for filtering
    pixels_per_class = torch.sum(dice_target, dim=2)
    none_empty_classes = pixels_per_class > min_pixels
    # compute loss per image for each non empty target class
    loss_per_image = torch.sum(losses * none_empty_classes, dim=1)

    num_classes_per_image = none_empty_classes.sum(dim=1)
    if torch.count_nonzero(num_classes_per_image).item() == 0:
        return 0
    loss = loss_per_image[num_classes_per_image > 0] / num_classes_per_image[num_classes_per_image > 0]
    # average batch loss
    return loss.mean()


class FocalLossWithJaccardOptimized(nn.Module):
    def __init__(
            self, ignore_index=255, gamma=2, ce_weight=1., d_weight=0.1, weight=None,
            size_average=True
    ):
        super().__init__()
        self.d_weight = d_weight
        self.ce_w = ce_weight
        self.gamma = gamma
        self.nll_loss = NLLLoss2d(weight, size_average, ignore_index=ignore_index)
        self.ignore_index = ignore_index

    def forward(self, outputs, targets):
        ce_loss = self.nll_loss((1 - F.softmax(outputs, dim=1)) ** self.gamma * F.log_softmax(outputs, dim=1), targets)
        d_loss = multiclass_jaccard(outputs=outputs, targets=targets, ignore_index=self.ignore_index)
        return self.ce_w * ce_loss + self.d_weight * d_loss


class SegFastLossCalculator(LossCalculator):

    def __init__(self, field: str = "mask", **kwargs):
        super().__init__()
        self.focal_dice = FocalLossWithJaccardOptimized(**kwargs)
        self.field = field

    def calculate_loss(self, outputs: Dict[str, torch.Tensor], sample: Dict[str, torch.Tensor]):
        target = sample[self.field].cuda().long()
        mask = outputs[self.field]
        return self.focal_dice(mask, target)


def dice_round(preds, trues, t=0.5):
    preds = (preds > t).float()
    return 1 - soft_dice_loss(preds, trues)


def soft_dice_loss(outputs, targets, per_patient=False):
    eps = 1e-5
    if per_patient:
        dice_target = targets.contiguous().float()
        dice_output = outputs.contiguous().float()
        dim = (1, 2, 3)
    else:
        dice_target = targets.view(-1).contiguous().float()
        dice_output = outputs.view(-1).contiguous().float()
        dim=(-1,)
    intersection = torch.sum(dice_output * dice_target, dim=dim)
    union = torch.sum(dice_output, dim=dim) + torch.sum(dice_target, dim=dim) + eps
    if union.sum().item() < 64:
        return 0
    loss = (1 - (2 * intersection + eps) / union)
    return loss.mean()