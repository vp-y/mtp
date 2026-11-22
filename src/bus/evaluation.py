"""Shared evaluation loop: run a predictor on the rows of a scenario/role, score, save, plot.

Every evaluated model writes the same files into its run directory:
  per_image_<role>.csv   one row per image (scores, probability, ...)
  metrics_<role>.json    summary (segmentation and/or classification)
  pred_masks/<role>/     predicted masks (<id>.png)
  cases_<role>.png       2 best + 2 worst segmentations (paper Fig. 3 / 4 style)
"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .data.datasets import read_gray, read_mask  # noqa: E402
from .data.manifest import LABEL_TO_INT, load_manifest, select, smoke_subset  # noqa: E402
from .metrics import cls_scores, seg_scores, summarize_seg  # noqa: E402
from .utils import save_json  # noqa: E402


def save_mask_png(mask: np.ndarray, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)


def finalize(records, out_dir, role, meta, cls_threshold=0.5):
    """Summarise per-image records, save CSV + JSON, print one line. Returns (summary, DataFrame)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    res = {**meta, "role": role, "n": int(len(df))}
    if len(df) and "dice" in df.columns:
        res["segmentation"] = summarize_seg(df)
    if len(df) and "prob" in df.columns:
        lab = df[df["y"].isin([0, 1])]
        if len(lab):
            res["classification"] = cls_scores(lab["y"], lab["prob"], cls_threshold)
    if "lesion_found" in df.columns:
        res["n_lesion_not_found"] = int((~df["lesion_found"].astype(bool)).sum())
    df.to_csv(out_dir / f"per_image_{role}.csv", index=False)
    save_json(res, out_dir / f"metrics_{role}.json")
    _print_summary(res)
    return res, df


def _print_summary(res: dict) -> None:
    parts = [f"[{res.get('task')}] {res.get('name')} {res.get('scenario')}/{res.get('role')} n={res.get('n')}"]
    s, c = res.get("segmentation"), res.get("classification")
    if s:
        parts.append(f"Dice={s['dice']:.3f} IoU={s['iou']:.3f} Acc={s['pixel_acc']:.3f} "
                     f"(pooled Dice={s['dice_pooled']:.3f} IoU={s['iou_pooled']:.3f})")
    if c:
        parts.append(f"AUC={c['auc']:.3f} F1={c['f1']:.3f} P={c['precision']:.3f} "
                     f"R={c['recall']:.3f} Acc={c['accuracy']:.3f}")
    print(" | ".join(parts), flush=True)


def cases_figure(df: pd.DataFrame, data_root, pred_dir, out_path, k: int = 2, title: str = "") -> None:
    """Raw | ground truth | prediction overlay for the k best and k worst images by IoU."""
    if df is None or len(df) == 0 or "iou" not in df.columns:
        return
    data_root, pred_dir = Path(data_root), Path(pred_dir)
    d = df.sort_values("iou")
    picks = list(d.tail(k).itertuples(index=False))[::-1] + list(d.head(k).itertuples(index=False))
    fig, axes = plt.subplots(len(picks), 3, figsize=(9, 3 * len(picks)), squeeze=False)
    for row, r in zip(axes, picks):
        img = read_gray(data_root / r.image_path)
        gt = read_mask(data_root / r.mask_path)
        pr = read_mask(pred_dir / f"{r.id}.png")
        row[0].imshow(img, cmap="gray")
        row[1].imshow(gt, cmap="gray")
        row[2].imshow(img, cmap="gray")
        row[2].imshow(np.ma.masked_where(~pr, pr.astype(float)), cmap="autumn", alpha=0.45, vmin=0, vmax=1)
        row[2].text(0.02, 0.98, f"IoU={r.iou:.3f}\nDice={r.dice:.3f}", transform=row[2].transAxes,
                    va="top", fontsize=8, color="white", bbox=dict(facecolor="black", alpha=0.5, pad=2))
        row[0].set_ylabel(str(r.id), fontsize=7)
        for a in row:
            a.set_xticks([])
            a.set_yticks([])
    for a, t in zip(axes[0], ("Raw", "GT", "Pred")):
        a.set_title(t)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def evaluate_model(predict, meta: dict, data_root, out_dir, scenario: str, roles,
                   include_normal: bool = False, smoke: bool = False, cls_threshold: float = 0.5) -> dict:
    """Run `predict(gray) -> {'mask': bool HxW, optional 'prob', other scalars}` over each role."""
    root, out_dir = Path(data_root), Path(out_dir)
    manifest = load_manifest(root)
    results = {}
    for role in roles:
        rows = select(manifest, scenario, role, include_normal)
        if smoke:
            rows = smoke_subset(rows, 8)
        pred_dir = out_dir / "pred_masks" / role
        records = []
        for r in rows.itertuples(index=False):
            out = dict(predict(read_gray(root / r.image_path)))
            mask = out.pop("mask")
            save_mask_png(mask, pred_dir / f"{r.id}.png")
            rec = {"id": r.id, "dataset": r.dataset, "label": r.label, "y": LABEL_TO_INT.get(r.label, -1),
                   "image_path": r.image_path, "mask_path": r.mask_path}
            rec.update(seg_scores(mask, read_mask(root / r.mask_path)))
            rec.update(out)
            records.append(rec)
        res, df = finalize(records, out_dir, role, {**meta, "scenario": scenario}, cls_threshold)
        cases_figure(df, root, pred_dir, out_dir / f"cases_{role}.png",
                     title=f"{meta.get('name')} | scenario {scenario} | {role}")
        results[role] = res
    return results
