"""Stage 2: train a benign/malignant classifier on lesion crops, then evaluate it.

  python -m bus.train_cls --config configs/cls/base.yaml
  python -m bus.train_cls --config configs/cls/base.yaml --set model.backbone=efficientnet_b0 model.head=linear
  python -m bus.train_cls --config configs/cls/base.yaml --set data.mask_dir=runs/pred_masks/A   # predicted crops
  python -m bus.train_cls --config configs/cls/base.yaml --smoke
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .config import base_parser, config_from_args, resolve_data_root
from .data.augment import build_augment
from .data.datasets import ClsDataset
from .data.manifest import load_manifest, select, smoke_subset
from .evaluation import finalize
from .inference import ClsPredictor
from .models.classification import Classifier, set_trainable
from .utils import (Amp, EarlyStopping, History, balanced_sampler, get_device, make_loader, make_run_dir,
                    save_yaml, seed_everything)


def describe(cfg: dict) -> str:
    m = cfg["model"]
    return f"{m['backbone']}/{m.get('head')}/k{m.get('unfreeze_k')}/d{m.get('dropout')}"


@torch.no_grad()
def _predict(model, loader, device, amp):
    model.eval()
    ys, ps = [], []
    for b in loader:
        with amp.autocast():
            logit = model(b["image"].to(device))
        ps.append(torch.sigmoid(logit.float()).squeeze(1).cpu())
        ys.append(b["label"])
    return torch.cat(ys).numpy().astype(int), torch.cat(ps).numpy()


def train_cls(cfg: dict, data_root, out_root="runs", smoke: bool = False) -> Path:
    d, t, m = cfg["data"], cfg["train"], cfg["model"]
    seed = int(t.get("seed", 42))
    seed_everything(seed)
    device = get_device()
    rd = make_run_dir(out_root, "cls", d["scenario"], cfg["name"], seed)
    save_yaml(cfg, rd / "config.yaml")

    manifest = load_manifest(data_root)
    tr, va = select(manifest, d["scenario"], "train"), select(manifest, d["scenario"], "val")
    if smoke:
        tr, va = smoke_subset(tr, 16), smoke_subset(va, 8)
    print(f"[cls] {cfg['name']} ({describe(cfg)}) scenario={d['scenario']} "
          f"train={len(tr)} val={len(va)} crops={'predicted' if d.get('mask_dir') else 'ground truth'} "
          f"device={device}", flush=True)

    size, mode, crop_cfg, mask_dir = d["image_size"], d["resize_mode"], cfg.get("crop", {}), d.get("mask_dir")
    ds_tr = ClsDataset(tr, data_root, crop_cfg, size, mode, build_augment(cfg.get("augment")), mask_dir)
    ds_va = ClsDataset(va, data_root, crop_cfg, size, mode, None, mask_dir)
    nw = 0 if smoke else int(t.get("num_workers", 2))
    sampler = balanced_sampler(ds_tr.labels) if t.get("oversample", True) else None
    dl_tr = make_loader(ds_tr, t["batch_size"], True, nw, sampler)
    dl_va = make_loader(ds_va, t["batch_size"], False, nw)

    model = Classifier(m["backbone"], m.get("head", "linear"), float(m.get("dropout", 0.0)),
                       pretrained=m.get("pretrained", True), img_size=size)
    n_trainable = set_trainable(model, m.get("unfreeze_k", 0))
    print(f"  trainable parameters: {n_trainable:,}", flush=True)
    model.to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=float(t["lr"]),
                           weight_decay=float(t.get("weight_decay", 0.0)))
    sch = t.get("scheduler", {})
    scheduler = ReduceLROnPlateau(opt, mode="max", factor=sch.get("factor", 0.1), patience=sch.get("patience", 3))
    stopper = EarlyStopping(t.get("early_stopping", 10), "max")
    loss_fn = nn.BCEWithLogitsLoss()
    amp = Amp(device, t.get("amp", True))
    history = History(rd / "history.csv")

    for epoch in range(1, (1 if smoke else int(t["epochs"])) + 1):
        t0 = time.time()
        model.train()
        total, n = 0.0, 0
        for b in dl_tr:
            x, y = b["image"].to(device, non_blocking=True), b["label"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with amp.autocast():
                logit = model(x).squeeze(1)
            loss = loss_fn(logit.float(), y)
            amp.backward_step(loss, opt)
            total += loss.item() * len(x)
            n += len(x)
        y_va, p_va = _predict(model, dl_va, device, amp)
        val_loss = F.binary_cross_entropy(torch.tensor(p_va).clamp(1e-7, 1 - 1e-7),
                                          torch.tensor(y_va, dtype=torch.float32)).item()
        val_auc = roc_auc_score(y_va, p_va) if len(np.unique(y_va)) == 2 else float("nan")
        score = -val_loss if math.isnan(val_auc) else val_auc
        scheduler.step(score)
        improved = stopper.step(score)
        if improved:
            torch.save({"kind": "cls", "config": cfg, "model": model.state_dict(), "epoch": epoch}, rd / "best.pt")
        history.log(epoch=epoch, lr=opt.param_groups[0]["lr"], train_loss=total / max(n, 1),
                    val_loss=val_loss, val_auc=val_auc, best=improved, sec=round(time.time() - t0, 1))
        if stopper.stop:
            print(f"  early stopping at epoch {epoch}", flush=True)
            break

    del model, opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    evaluate_cls(rd / "best.pt", data_root, rd, cfg.get("eval", {}).get("roles", ["val", "test"]), smoke=smoke)
    return rd


def evaluate_cls(ckpt, data_root, out_dir, roles, scenario=None, smoke=False) -> dict:
    """Classification of crops taken from ground-truth (or data.mask_dir) masks: paper Table 2."""
    pred = ClsPredictor(ckpt)
    cfg = pred.cfg
    d = cfg["data"]
    scenario = scenario or d["scenario"]
    amp = Amp(pred.device, cfg["train"].get("amp", True))
    manifest = load_manifest(data_root)
    meta = {"task": "cls", "name": cfg["name"], "seed": cfg["train"].get("seed"), "desc": describe(cfg),
            "scenario": scenario, "checkpoint": str(ckpt)}
    results = {}
    for role in roles:
        rows = select(manifest, scenario, role)
        if smoke:
            rows = smoke_subset(rows, 8)
        ds = ClsDataset(rows, data_root, cfg.get("crop", {}), d["image_size"], d["resize_mode"], None,
                        d.get("mask_dir"))
        y, p = _predict(pred.model, make_loader(ds, cfg["train"]["batch_size"], False, 0), pred.device, amp)
        records = [{"id": r.id, "dataset": r.dataset, "label": r.label, "y": int(yy), "prob": float(pp),
                    "used_full_image": bool(full)}
                   for r, yy, pp, full in zip(rows.itertuples(index=False), y, p, ds.used_full_image)]
        results[role], _ = finalize(records, out_dir, role, meta, pred.threshold)
    return results


def main():
    p = base_parser("Stage 2: train / evaluate a lesion classifier")
    p.add_argument("--ckpt", default=None, help="skip training; evaluate this checkpoint or run dir")
    a = p.parse_args()
    cfg = config_from_args(a)
    root = resolve_data_root(a.data_root)
    if a.ckpt:
        rd = make_run_dir(a.out, "cls", cfg["data"]["scenario"], cfg["name"], cfg["train"].get("seed", 42))
        evaluate_cls(a.ckpt, root, rd, cfg.get("eval", {}).get("roles", ["test"]),
                     scenario=cfg["data"]["scenario"], smoke=a.smoke)
    else:
        rd = train_cls(cfg, root, a.out, smoke=a.smoke)
    print(f"run dir: {rd}")


if __name__ == "__main__":
    main()
