"""Stage 1: train a lesion segmentation model, then evaluate it.

  python -m bus.train_seg --config configs/seg/base.yaml
  python -m bus.train_seg --config configs/seg/base.yaml --set model.arch=unet train.lr=3e-4
  python -m bus.train_seg --config configs/seg/base.yaml --smoke                      # quick code check
  python -m bus.train_seg --config RUN/config.yaml --ckpt RUN --set data.scenario=C "eval.roles=[test]"  # eval only
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .config import base_parser, config_from_args, resolve_data_root
from .data.augment import build_augment
from .data.datasets import SegDataset
from .data.manifest import load_manifest, select, smoke_subset
from .evaluation import evaluate_model
from .inference import SegPredictor
from .losses import CompositeSegLoss
from .metrics import batch_dice_iou
from .models.segmentation import build_seg_model
from .utils import (Amp, EarlyStopping, History, balanced_sampler, get_device, make_loader, make_run_dir,
                    save_yaml, seed_everything)


def describe(cfg: dict) -> str:
    m = cfg["model"]
    enc = "" if "vanilla" in m["arch"] else f"/{m.get('encoder')}"
    return f"{m['arch']}{enc}/c{m.get('base_channels')}"


@torch.no_grad()
def _validate(model, loader, loss_fn, device, amp, threshold):
    model.eval()
    total, n, dices, ious = 0.0, 0, [], []
    for b in loader:
        x, y = b["image"].to(device), b["mask"].to(device)
        with amp.autocast():
            logits = model(x)
        total += loss_fn(logits, y).item() * len(x)
        n += len(x)
        d, i = batch_dice_iou(logits, y, threshold)
        dices.append(d.cpu())
        ious.append(i.cpu())
    return total / max(n, 1), torch.cat(dices).mean().item(), torch.cat(ious).mean().item()


def train_seg(cfg: dict, data_root, out_root="runs", smoke: bool = False) -> Path:
    d, t = cfg["data"], cfg["train"]
    seed = int(t.get("seed", 42))
    seed_everything(seed)
    device = get_device()
    rd = make_run_dir(out_root, "seg", d["scenario"], cfg["name"], seed)
    save_yaml(cfg, rd / "config.yaml")

    manifest = load_manifest(data_root)
    inc = d.get("include_normal", False)
    tr, va = select(manifest, d["scenario"], "train", inc), select(manifest, d["scenario"], "val", inc)
    if smoke:
        tr, va = smoke_subset(tr, 16), smoke_subset(va, 8)
    print(f"[seg] {cfg['name']} ({describe(cfg)}) scenario={d['scenario']} "
          f"train={len(tr)} val={len(va)} device={device}", flush=True)

    size, mode, ch = d["image_size"], d["resize_mode"], d["in_channels"]
    ds_tr = SegDataset(tr, data_root, size, mode, ch, augment=build_augment(cfg.get("augment")))
    ds_va = SegDataset(va, data_root, size, mode, ch)
    nw = 0 if smoke else int(t.get("num_workers", 2))
    sampler = balanced_sampler(ds_tr.labels) if t.get("oversample") else None
    dl_tr = make_loader(ds_tr, t["batch_size"], True, nw, sampler)
    dl_va = make_loader(ds_va, t["batch_size"], False, nw)

    model = build_seg_model(cfg["model"], ch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(t["lr"]), weight_decay=float(t.get("weight_decay", 0.0)))
    monitor = t.get("monitor", "dice")
    mode_ = "min" if monitor == "loss" else "max"
    sch = t.get("scheduler", {})
    scheduler = ReduceLROnPlateau(opt, mode=mode_, factor=sch.get("factor", 0.1), patience=sch.get("patience", 3))
    stopper = EarlyStopping(t.get("early_stopping", 10), mode_)
    loss_fn = CompositeSegLoss(**cfg.get("loss", {}))
    amp = Amp(device, t.get("amp", True))
    history = History(rd / "history.csv")
    threshold = cfg.get("eval", {}).get("threshold", 0.5)

    for epoch in range(1, (1 if smoke else int(t["epochs"])) + 1):
        t0 = time.time()
        model.train()
        total, n = 0.0, 0
        for b in dl_tr:
            x = b["image"].to(device, non_blocking=True)
            y = b["mask"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with amp.autocast():
                logits = model(x)
            loss = loss_fn(logits, y)
            amp.backward_step(loss, opt)
            total += loss.item() * len(x)
            n += len(x)
        val_loss, val_dice, val_iou = _validate(model, dl_va, loss_fn, device, amp, threshold)
        score = val_loss if monitor == "loss" else val_dice
        scheduler.step(score)
        improved = stopper.step(score)
        if improved:
            torch.save({"kind": "seg", "config": cfg, "model": model.state_dict(), "epoch": epoch}, rd / "best.pt")
        history.log(epoch=epoch, lr=opt.param_groups[0]["lr"], train_loss=total / max(n, 1),
                    val_loss=val_loss, val_dice=val_dice, val_iou=val_iou, best=improved,
                    sec=round(time.time() - t0, 1))
        if stopper.stop:
            print(f"  early stopping at epoch {epoch}", flush=True)
            break

    del model, opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    evaluate_seg(rd / "best.pt", data_root, rd, cfg.get("eval", {}).get("roles", ["val", "test"]),
                 smoke=smoke)
    return rd


def evaluate_seg(ckpt, data_root, out_dir, roles, scenario=None, smoke=False) -> dict:
    pred = SegPredictor(ckpt)
    cfg = pred.cfg
    scenario = scenario or cfg["data"]["scenario"]
    meta = {"task": "seg", "name": cfg["name"], "seed": cfg["train"].get("seed"), "desc": describe(cfg),
            "checkpoint": str(ckpt)}
    return evaluate_model(lambda g: {"mask": pred.predict_mask(g)}, meta, data_root, out_dir, scenario, roles,
                          include_normal=cfg["data"].get("include_normal", False), smoke=smoke)


def main():
    p = base_parser("Stage 1: train / evaluate a segmentation model")
    p.add_argument("--ckpt", default=None, help="skip training; evaluate this checkpoint or run dir")
    a = p.parse_args()
    cfg = config_from_args(a)
    root = resolve_data_root(a.data_root)
    if a.ckpt:
        rd = make_run_dir(a.out, "seg", cfg["data"]["scenario"], cfg["name"], cfg["train"].get("seed", 42))
        evaluate_seg(a.ckpt, root, rd, cfg.get("eval", {}).get("roles", ["test"]),
                     scenario=cfg["data"]["scenario"], smoke=a.smoke)
    else:
        rd = train_seg(cfg, root, a.out, smoke=a.smoke)
    print(f"run dir: {rd}")


if __name__ == "__main__":
    main()
