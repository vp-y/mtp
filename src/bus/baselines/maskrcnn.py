"""Single-step baseline: Mask R-CNN (ResNet-50-FPN, COCO-pretrained) segments and classifies at once.

  python -m bus.baselines.maskrcnn --config configs/baselines/maskrcnn.yaml                    # scenario A
  python -m bus.baselines.maskrcnn --config configs/baselines/maskrcnn.yaml --set data.scenario=B
  python -m bus.baselines.maskrcnn --config configs/baselines/maskrcnn.yaml \
      --ckpt runs/maskrcnn/A/maskrcnn_r50_fpn/seed42 --set data.scenario=C "eval.roles=[test]"   # eval only

Classes: 0 background, 1 benign, 2 malignant. P(malignant) = s_mal / (s_mal + s_ben) using the
highest detection score of each class, so AUC is computed from a continuous score (the paper's
Mask R-CNN AUC values look like they came from hard labels).
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import base_parser, config_from_args, resolve_data_root
from ..data.augment import build_augment
from ..data.datasets import SegDataset
from ..data.manifest import load_manifest, select, smoke_subset
from ..data.preprocess import resize_to_square, restore, to_tensor
from ..evaluation import evaluate_model
from ..inference import load_ckpt
from ..utils import History, get_device, make_loader, make_run_dir, save_yaml, seed_everything

NUM_CLASSES = 3  # background, benign, malignant


def build_maskrcnn(min_size=512, max_size=512, pretrained=True):
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    if pretrained:
        model = maskrcnn_resnet50_fpn(weights="DEFAULT", min_size=min_size, max_size=max_size)
    else:
        model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, min_size=min_size, max_size=max_size)
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, NUM_CLASSES)
    in_m = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_m, 256, NUM_CLASSES)
    return model


class DetDataset(Dataset):
    """Each connected lesion blob becomes one instance labelled with the image class."""

    def __init__(self, df, root, size, resize_mode, augment=None):
        self.base = SegDataset(df, root, size, resize_mode, channels=3, augment=augment,
                               with_label=True, normalize=False)  # torchvision normalises internally

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        s = self.base[i]
        m = s["mask"][0].numpy().astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        masks, boxes = [], []
        for k in range(1, n):
            x, y, w, h, area = stats[k]
            if area < 8 or w < 2 or h < 2:
                continue
            masks.append(labels == k)
            boxes.append([x, y, x + w, y + h])
        cls = int(s["label"]) + 1
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.full((len(boxes),), cls, dtype=torch.int64),
            "masks": (torch.from_numpy(np.stack(masks).astype(np.uint8)) if masks
                      else torch.zeros((0,) + m.shape, dtype=torch.uint8)),
        }
        return s["image"], target


def _collate(batch):
    return tuple(zip(*batch))


class MaskRCNNRunner:
    def __init__(self, ckpt, device=None):
        ck = load_ckpt(ckpt)
        self.cfg = cfg = ck["config"]
        mc = cfg["model"]
        self.size, self.mode = cfg["data"]["image_size"], cfg["data"]["resize_mode"]
        self.score_thr = mc.get("score_threshold", 0.5)
        self.device = device or get_device()
        self.model = build_maskrcnn(mc.get("min_size", 512), mc.get("max_size", 512), pretrained=False)
        self.model.load_state_dict(ck["model"])
        self.model.to(self.device).eval()

    @torch.no_grad()
    def __call__(self, gray):
        img, g = resize_to_square(gray, self.size, self.mode)
        out = self.model([to_tensor(img, 3, normalize=False).to(self.device)])[0]
        scores = out["scores"].cpu().numpy()
        labels = out["labels"].cpu().numpy()
        masks = out["masks"][:, 0].cpu().numpy() if len(scores) else np.zeros((0, self.size, self.size))
        keep = scores >= self.score_thr
        seg = (masks[keep] > 0.5).any(axis=0) if keep.any() else np.zeros((self.size, self.size), bool)
        s_ben = float(scores[labels == 1].max(initial=0.0))
        s_mal = float(scores[labels == 2].max(initial=0.0))
        p_mal = s_mal / (s_mal + s_ben) if s_mal + s_ben > 0 else 0.5
        mask = restore(seg.astype(np.uint8), g, cv2.INTER_NEAREST).astype(bool)
        return {"mask": mask, "prob": p_mal, "n_detections": int(keep.sum())}


def train_maskrcnn(cfg, data_root, out_root="runs", smoke=False) -> Path:
    d, t, mc = cfg["data"], cfg["train"], cfg["model"]
    seed = int(t.get("seed", 42))
    seed_everything(seed)
    device = get_device()
    rd = make_run_dir(out_root, "maskrcnn", d["scenario"], cfg["name"], seed)
    save_yaml(cfg, rd / "config.yaml")
    manifest = load_manifest(data_root)
    tr = select(manifest, d["scenario"], "train")
    if smoke:
        tr = smoke_subset(tr, 8)
    print(f"[maskrcnn] {cfg['name']} scenario={d['scenario']} train={len(tr)} device={device}", flush=True)
    ds = DetDataset(tr, data_root, d["image_size"], d["resize_mode"], build_augment(cfg.get("augment")))
    dl = make_loader(ds, t["batch_size"], True, 0 if smoke else int(t.get("num_workers", 2)), collate_fn=_collate)

    model = build_maskrcnn(mc.get("min_size", 512), mc.get("max_size", 512), mc.get("pretrained", True)).to(device)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=float(t["lr"]),
                          momentum=float(t.get("momentum", 0.9)), weight_decay=float(t.get("weight_decay", 5e-4)))
    history = History(rd / "history.csv")
    for epoch in range(1, (1 if smoke else int(t["epochs"])) + 1):
        t0 = time.time()
        model.train()
        total, n = 0.0, 0
        for imgs, targets in dl:
            imgs = [im.to(device) for im in imgs]
            targets = [{k: v.to(device) for k, v in tg.items()} for tg in targets]
            loss = sum(model(imgs, targets).values())
            if not math.isfinite(loss.item()):
                raise FloatingPointError("Mask R-CNN loss is not finite; lower train.lr")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(imgs)
            n += len(imgs)
        history.log(epoch=epoch, train_loss=total / max(n, 1), sec=round(time.time() - t0, 1))
    torch.save({"kind": "maskrcnn", "config": cfg, "model": model.state_dict()}, rd / "last.pt")
    del model, opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    evaluate_maskrcnn(rd / "last.pt", data_root, rd, cfg.get("eval", {}).get("roles", ["val", "test"]), smoke=smoke)
    return rd


def evaluate_maskrcnn(ckpt, data_root, out_dir, roles, scenario=None, smoke=False):
    runner = MaskRCNNRunner(ckpt)
    cfg = runner.cfg
    meta = {"task": "maskrcnn", "name": cfg["name"], "seed": cfg["train"].get("seed"),
            "desc": "Mask R-CNN R50-FPN", "checkpoint": str(ckpt)}
    return evaluate_model(runner, meta, data_root, out_dir, scenario or cfg["data"]["scenario"], roles, smoke=smoke)


def main():
    p = base_parser("Baseline: Mask R-CNN (single-step)")
    p.add_argument("--ckpt", default=None, help="skip training; evaluate this checkpoint or run dir")
    a = p.parse_args()
    cfg = config_from_args(a)
    root = resolve_data_root(a.data_root)
    if a.ckpt:
        rd = make_run_dir(a.out, "maskrcnn", cfg["data"]["scenario"], cfg["name"], cfg["train"].get("seed", 42))
        evaluate_maskrcnn(a.ckpt, root, rd, cfg.get("eval", {}).get("roles", ["test"]),
                          scenario=cfg["data"]["scenario"], smoke=a.smoke)
    else:
        rd = train_maskrcnn(cfg, root, a.out, smoke=a.smoke)
    print(f"run dir: {rd}")


if __name__ == "__main__":
    main()
