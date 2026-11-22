"""Multi-task baseline: one shared Swin Transformer encoder, a light segmentation decoder and a
classification head trained jointly (paper section "Multi-task Swin Transformer Baseline").

  python -m bus.baselines.multitask_swin --config configs/baselines/multitask_swin.yaml
  python -m bus.baselines.multitask_swin --config configs/baselines/multitask_swin.yaml \
      --ckpt runs/mtswin/A/mtswin_tiny/seed42 --set data.scenario=C "eval.roles=[test]"   # eval only
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

from ..config import base_parser, config_from_args, resolve_data_root
from ..data.augment import build_augment
from ..data.datasets import SegDataset
from ..data.manifest import load_manifest, select, smoke_subset
from ..data.preprocess import resize_to_square, restore, to_tensor
from ..evaluation import evaluate_model
from ..inference import load_ckpt
from ..metrics import batch_dice_iou
from ..utils import (Amp, EarlyStopping, History, balanced_sampler, get_device, make_loader, make_run_dir,
                     save_yaml, seed_everything)


def _conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class MultiTaskSwin(nn.Module):
    def __init__(self, name="swin_tiny_patch4_window7_224", pretrained=True, img_size=224,
                 num_classes=2, dropout=0.3):
        super().__init__()
        import timm

        self.encoder = timm.create_model(name, pretrained=pretrained, features_only=True,
                                         out_indices=(2, 3), img_size=img_size)
        c2, c3 = self.encoder.feature_info.channels()
        self.chs = (c2, c3)
        self.up1 = _conv_block(c3 + c2, 256)   # upsample stage-4 features, fuse stage-3, conv
        self.up2 = _conv_block(256, 64)        # upsample again, conv
        self.seg_head = nn.Conv2d(64, 1, 1)
        self.cls_head = nn.Sequential(nn.Linear(c3, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
                                      nn.Linear(256, num_classes))

    @staticmethod
    def _nchw(f, c):
        # some timm versions return Swin features channels-last (N, H, W, C)
        return f.permute(0, 3, 1, 2).contiguous() if f.shape[1] != c and f.shape[-1] == c else f

    def forward(self, x):
        f2, f3 = (self._nchw(f, c) for f, c in zip(self.encoder(x), self.chs))
        z = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        z = self.up1(torch.cat([z, f2], dim=1))
        z = self.up2(F.interpolate(z, scale_factor=2, mode="bilinear", align_corners=False))
        seg = F.interpolate(self.seg_head(z), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return seg, self.cls_head(f3.mean(dim=(2, 3)))


def _build(cfg, pretrained):
    mc = cfg["model"]
    return MultiTaskSwin(mc.get("name", "swin_tiny_patch4_window7_224"), pretrained,
                         cfg["data"]["image_size"], 2, float(mc.get("dropout", 0.3)))


class MTSwinRunner:
    def __init__(self, ckpt, device=None):
        ck = load_ckpt(ckpt)
        self.cfg = ck["config"]
        self.size, self.mode = self.cfg["data"]["image_size"], self.cfg["data"]["resize_mode"]
        self.threshold = self.cfg.get("eval", {}).get("threshold", 0.5)
        self.device = device or get_device()
        self.model = _build(self.cfg, pretrained=False)
        self.model.load_state_dict(ck["model"])
        self.model.to(self.device).eval()

    @torch.no_grad()
    def __call__(self, gray):
        img, g = resize_to_square(gray, self.size, self.mode)
        seg, cls = self.model(to_tensor(img, 3)[None].to(self.device))
        prob = torch.sigmoid(seg[0, 0].float()).cpu().numpy()
        return {"mask": restore(prob, g) >= self.threshold,
                "prob": float(torch.softmax(cls.float(), dim=1)[0, 1])}


def train_mtswin(cfg, data_root, out_root="runs", smoke=False) -> Path:
    d, t = cfg["data"], cfg["train"]
    seed = int(t.get("seed", 42))
    seed_everything(seed)
    device = get_device()
    rd = make_run_dir(out_root, "mtswin", d["scenario"], cfg["name"], seed)
    save_yaml(cfg, rd / "config.yaml")
    manifest = load_manifest(data_root)
    tr, va = select(manifest, d["scenario"], "train"), select(manifest, d["scenario"], "val")
    if smoke:
        tr, va = smoke_subset(tr, 16), smoke_subset(va, 8)
    print(f"[mtswin] {cfg['name']} scenario={d['scenario']} train={len(tr)} val={len(va)} device={device}",
          flush=True)
    size, mode = d["image_size"], d["resize_mode"]
    ds_tr = SegDataset(tr, data_root, size, mode, 3, build_augment(cfg.get("augment")), with_label=True)
    ds_va = SegDataset(va, data_root, size, mode, 3, with_label=True)
    nw = 0 if smoke else int(t.get("num_workers", 2))
    sampler = balanced_sampler(ds_tr.labels) if t.get("oversample", True) else None
    dl_tr = make_loader(ds_tr, t["batch_size"], True, nw, sampler)
    dl_va = make_loader(ds_va, t["batch_size"], False, nw)

    model = _build(cfg, pretrained=cfg["model"].get("pretrained", True)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(t["lr"]), weight_decay=float(t.get("weight_decay", 1e-4)))
    sch = t.get("scheduler", {})
    scheduler = ReduceLROnPlateau(opt, mode="max", factor=sch.get("factor", 0.1), patience=sch.get("patience", 3))
    stopper = EarlyStopping(t.get("early_stopping", 10), "max")
    w = cfg.get("loss", {})
    w_seg, w_cls = float(w.get("seg", 1.0)), float(w.get("cls", 1.0))
    bce, ce = nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()
    amp = Amp(device, t.get("amp", True))
    history = History(rd / "history.csv")

    for epoch in range(1, (1 if smoke else int(t["epochs"])) + 1):
        t0 = time.time()
        model.train()
        total, n = 0.0, 0
        for b in dl_tr:
            x, y, lab = b["image"].to(device), b["mask"].to(device), b["label"].to(device)
            opt.zero_grad(set_to_none=True)
            with amp.autocast():
                seg, cls = model(x)
            loss = w_seg * bce(seg.float(), y) + w_cls * ce(cls.float(), lab)
            amp.backward_step(loss, opt)
            total += loss.item() * len(x)
            n += len(x)
        model.eval()
        dices, probs, labels = [], [], []
        with torch.no_grad():
            for b in dl_va:
                with amp.autocast():
                    seg, cls = model(b["image"].to(device))
                dices.append(batch_dice_iou(seg, b["mask"].to(device))[0].cpu())
                probs.append(torch.softmax(cls.float(), dim=1)[:, 1].cpu())
                labels.append(b["label"])
        val_dice = torch.cat(dices).mean().item()
        y_va, p_va = torch.cat(labels).numpy(), torch.cat(probs).numpy()
        val_auc = roc_auc_score(y_va, p_va) if len(np.unique(y_va)) == 2 else float("nan")
        score = val_dice if math.isnan(val_auc) else (val_dice + val_auc) / 2
        scheduler.step(score)
        improved = stopper.step(score)
        if improved:
            torch.save({"kind": "mtswin", "config": cfg, "model": model.state_dict(), "epoch": epoch}, rd / "best.pt")
        history.log(epoch=epoch, lr=opt.param_groups[0]["lr"], train_loss=total / max(n, 1), val_dice=val_dice,
                    val_auc=val_auc, best=improved, sec=round(time.time() - t0, 1))
        if stopper.stop:
            print(f"  early stopping at epoch {epoch}", flush=True)
            break

    del model, opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    evaluate_mtswin(rd / "best.pt", data_root, rd, cfg.get("eval", {}).get("roles", ["val", "test"]), smoke=smoke)
    return rd


def evaluate_mtswin(ckpt, data_root, out_dir, roles, scenario=None, smoke=False):
    runner = MTSwinRunner(ckpt)
    cfg = runner.cfg
    meta = {"task": "mtswin", "name": cfg["name"], "seed": cfg["train"].get("seed"),
            "desc": f"Multi-task {cfg['model'].get('name')}", "checkpoint": str(ckpt)}
    return evaluate_model(runner, meta, data_root, out_dir, scenario or cfg["data"]["scenario"], roles, smoke=smoke)


def main():
    p = base_parser("Baseline: multi-task Swin Transformer")
    p.add_argument("--ckpt", default=None, help="skip training; evaluate this checkpoint or run dir")
    a = p.parse_args()
    cfg = config_from_args(a)
    root = resolve_data_root(a.data_root)
    if a.ckpt:
        rd = make_run_dir(a.out, "mtswin", cfg["data"]["scenario"], cfg["name"], cfg["train"].get("seed", 42))
        evaluate_mtswin(a.ckpt, root, rd, cfg.get("eval", {}).get("roles", ["test"]),
                        scenario=cfg["data"]["scenario"], smoke=a.smoke)
    else:
        rd = train_mtswin(cfg, root, a.out, smoke=a.smoke)
    print(f"run dir: {rd}")


if __name__ == "__main__":
    main()
