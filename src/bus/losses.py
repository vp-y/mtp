"""Segmentation loss used by the paper: 0.4 * BCE + 0.5 * (1 - Dice) + 0.5 * (1 - IoU)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompositeSegLoss(nn.Module):
    def __init__(self, bce: float = 0.4, dice: float = 0.5, iou: float = 0.5, eps: float = 1.0):
        super().__init__()
        self.w_bce, self.w_dice, self.w_iou, self.eps = bce, dice, iou, eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits, target = logits.float(), target.float()
        loss = self.w_bce * F.binary_cross_entropy_with_logits(logits, target)
        p = torch.sigmoid(logits)
        dims = tuple(range(1, p.ndim))
        inter = (p * target).sum(dims)
        total = p.sum(dims) + target.sum(dims)
        dice = (2 * inter + self.eps) / (total + self.eps)
        iou = (inter + self.eps) / (total - inter + self.eps)
        return loss + self.w_dice * (1 - dice).mean() + self.w_iou * (1 - iou).mean()
