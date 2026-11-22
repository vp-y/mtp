"""Segmentation and classification metrics.

Segmentation metrics are reported two ways:
  per-image mean (dice, iou, pixel_acc)       standard for medical segmentation
  pooled over the whole set (dice_pooled, ...) sums pixels first, as some papers do
The paper's own Dice/IoU pairs look pooled while its baselines look per-image, so we report
both for every method.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve


def seg_scores(pred: np.ndarray, gt: np.ndarray) -> dict:
    p, g = pred.astype(bool), gt.astype(bool)
    inter = int(np.logical_and(p, g).sum())
    ps, gs = int(p.sum()), int(g.sum())
    union = ps + gs - inter
    return {
        "dice": 1.0 if ps + gs == 0 else 2.0 * inter / (ps + gs),
        "iou": 1.0 if union == 0 else inter / union,
        "pixel_acc": float((p == g).mean()),
        "inter_px": inter,
        "pred_px": ps,
        "gt_px": gs,
    }


@torch.no_grad()
def batch_dice_iou(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
    """Per-image Dice and IoU for a batch of logits (B,1,H,W)."""
    p = (torch.sigmoid(logits.float()) >= threshold).float()
    t = target.float()
    dims = tuple(range(1, p.ndim))
    inter = (p * t).sum(dims)
    ps, ts = p.sum(dims), t.sum(dims)
    union = ps + ts - inter
    one = torch.ones_like(inter)
    dice = torch.where(ps + ts > 0, 2 * inter / (ps + ts).clamp_min(1e-9), one)
    iou = torch.where(union > 0, inter / union.clamp_min(1e-9), one)
    return dice, iou


def summarize_seg(df: pd.DataFrame) -> dict:
    inter, ps, gs = df["inter_px"].sum(), df["pred_px"].sum(), df["gt_px"].sum()
    return {
        "dice": float(df["dice"].mean()),
        "iou": float(df["iou"].mean()),
        "pixel_acc": float(df["pixel_acc"].mean()),
        "dice_std": float(df["dice"].std(ddof=0)),
        "iou_std": float(df["iou"].std(ddof=0)),
        "dice_pooled": float(2 * inter / (ps + gs)) if ps + gs else 1.0,
        "iou_pooled": float(inter / (ps + gs - inter)) if ps + gs - inter else 1.0,
        "n_empty_pred": int((df["pred_px"] == 0).sum()),
        "n": int(len(df)),
    }


def cls_scores(y_true, prob, threshold: float = 0.5) -> dict:
    """Binary metrics with malignant = positive class."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob, dtype=float)
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    return {
        "accuracy": (tp + tn) / max(len(y), 1),
        "precision": precision,
        "recall": recall,            # = sensitivity
        "specificity": specificity,
        "f1": f1,
        "auc": auc,
        "balanced_accuracy": (recall + specificity) / 2,
        "threshold": float(threshold),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "n": int(len(y)),
    }


def threshold_for_sensitivity(y_true, prob, target: float = 0.95) -> float:
    """Highest threshold whose sensitivity (recall) is >= target. Tune on VAL, never on test."""
    fpr, tpr, thr = roc_curve(np.asarray(y_true).astype(int), np.asarray(prob, dtype=float))
    ok = np.nonzero(tpr >= target)[0]
    if len(ok) == 0:
        return 0.0
    return float(min(thr[ok[0]], 1.0))
