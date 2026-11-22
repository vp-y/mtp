import numpy as np
import pandas as pd
import torch

from bus.metrics import batch_dice_iou, cls_scores, seg_scores, summarize_seg, threshold_for_sensitivity


def test_seg_scores_perfect_and_half():
    g = np.zeros((10, 10), bool)
    g[:, :5] = True
    s = seg_scores(g, g)
    assert s["dice"] == 1.0 and s["iou"] == 1.0 and s["pixel_acc"] == 1.0
    p = np.zeros((10, 10), bool)
    p[:, :10] = True  # predicts everything: inter 50, pred 100, gt 50
    s = seg_scores(p, g)
    assert abs(s["dice"] - 100 / 150) < 1e-9
    assert abs(s["iou"] - 0.5) < 1e-9


def test_seg_scores_both_empty():
    z = np.zeros((5, 5), bool)
    assert seg_scores(z, z)["dice"] == 1.0


def test_pooled_differs_from_mean():
    a = seg_scores(np.ones((10, 10), bool), np.ones((10, 10), bool))  # big, perfect
    small_gt = np.zeros((10, 10), bool)
    small_gt[0, 0] = True
    b = seg_scores(np.zeros((10, 10), bool), small_gt)                # tiny, missed
    s = summarize_seg(pd.DataFrame([a, b]))
    assert abs(s["dice"] - 0.5) < 1e-9
    assert s["dice_pooled"] > 0.99
    assert s["n_empty_pred"] == 1


def test_batch_dice_iou_matches_numpy():
    t = torch.zeros(2, 1, 8, 8)
    t[0, 0, :4] = 1
    logits = torch.full((2, 1, 8, 8), -10.0)
    logits[0, 0, :4] = 10
    d, i = batch_dice_iou(logits, t)
    assert torch.allclose(d, torch.tensor([1.0, 1.0])) and torch.allclose(i, torch.tensor([1.0, 1.0]))


def test_cls_scores():
    y = [0, 0, 1, 1]
    p = [0.1, 0.6, 0.4, 0.9]
    s = cls_scores(y, p, 0.5)
    assert (s["tp"], s["tn"], s["fp"], s["fn"]) == (1, 1, 1, 1)
    assert s["accuracy"] == 0.5 and s["auc"] == 0.75


def test_threshold_for_sensitivity():
    y = np.array([0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.7, 0.3, 0.6, 0.8, 0.9])
    thr = threshold_for_sensitivity(y, p, 1.0)
    assert ((p >= thr) & (y == 1)).sum() == 4
