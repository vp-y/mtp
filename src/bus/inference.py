"""Loading trained checkpoints and predicting on ORIGINAL-resolution grayscale images.

This module is what the deployed app will import: DualStagePipeline(seg_ckpt, cls_ckpt)(image).
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from .data.preprocess import resize_to_square, restore, to_tensor
from .models.classification import Classifier
from .models.segmentation import build_seg_model
from .roi import crop, mask_to_box, postprocess
from .utils import get_device


def resolve_ckpt(path) -> Path:
    """Accept a checkpoint file or a run directory (uses best.pt, else last.pt)."""
    p = Path(path)
    if p.is_dir():
        for name in ("best.pt", "last.pt"):
            if (p / name).exists():
                return p / name
        raise FileNotFoundError(f"no best.pt / last.pt in {p}")
    return p


def load_ckpt(path) -> dict:
    return torch.load(resolve_ckpt(path), map_location="cpu", weights_only=False)


class SegPredictor:
    def __init__(self, ckpt, device=None, threshold=None, post=None, min_area=None):
        ck = load_ckpt(ckpt)
        cfg = copy.deepcopy(ck["config"])
        cfg["model"]["encoder_weights"] = None  # weights come from the checkpoint
        self.cfg = cfg
        d, e = cfg["data"], cfg.get("eval", {})
        self.size, self.mode, self.channels = d["image_size"], d["resize_mode"], d["in_channels"]
        self.threshold = e.get("threshold", 0.5) if threshold is None else threshold
        self.post = e.get("post", "none") if post is None else post
        self.min_area = e.get("min_area", 0) if min_area is None else min_area
        self.device = device or get_device()
        self.model = build_seg_model(cfg["model"], self.channels)
        self.model.load_state_dict(ck["model"])
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict_prob(self, gray: np.ndarray) -> np.ndarray:
        img, g = resize_to_square(gray, self.size, self.mode)
        x = to_tensor(img, self.channels)[None].to(self.device)
        prob = torch.sigmoid(self.model(x)[0, 0].float()).cpu().numpy()
        return restore(prob, g)

    def predict_mask(self, gray: np.ndarray) -> np.ndarray:
        return postprocess(self.predict_prob(gray) >= self.threshold, self.post, self.min_area)


class ClsPredictor:
    def __init__(self, ckpt, device=None):
        ck = load_ckpt(ckpt)
        cfg = ck["config"]
        m = cfg["model"]
        self.cfg = cfg
        self.size, self.mode = cfg["data"]["image_size"], cfg["data"]["resize_mode"]
        self.crop_cfg = cfg.get("crop", {})
        self.threshold = cfg.get("eval", {}).get("threshold", 0.5)
        self.device = device or get_device()
        self.model = Classifier(m["backbone"], m.get("head", "linear"), m.get("dropout", 0.0),
                                pretrained=False, img_size=self.size)
        self.model.load_state_dict(ck["model"])
        self.model.to(self.device).eval()

    def box(self, mask: np.ndarray):
        return mask_to_box(mask, **self.crop_cfg)

    @torch.no_grad()
    def predict_patch(self, patch: np.ndarray) -> float:
        img, _ = resize_to_square(patch, self.size, self.mode)
        x = to_tensor(img, 3)[None].to(self.device)
        return float(torch.sigmoid(self.model(x)[0, 0].float()))


class DualStagePipeline:
    """segment -> clean mask -> bounding box + margin -> crop -> classify.

    empty_policy decides what happens when no lesion is segmented:
      'full_image' classify the whole image (default, keeps every case scorable)
      'benign'     report P(malignant) = 0
    """

    def __init__(self, seg_ckpt, cls_ckpt, device=None, empty_policy: str = "full_image", **seg_kwargs):
        self.seg = SegPredictor(seg_ckpt, device=device, **seg_kwargs)
        self.cls = ClsPredictor(cls_ckpt, device=device)
        if empty_policy not in ("full_image", "benign"):
            raise ValueError("empty_policy must be 'full_image' or 'benign'")
        self.empty_policy = empty_policy

    def __call__(self, gray: np.ndarray) -> dict:
        prob_map = self.seg.predict_prob(gray)
        mask = postprocess(prob_map >= self.seg.threshold, self.seg.post, self.seg.min_area)
        box = self.cls.box(mask)
        if box is not None:
            p = self.cls.predict_patch(crop(gray, box))
        elif self.empty_policy == "full_image":
            p = self.cls.predict_patch(gray)
        else:
            p = 0.0
        return {"prob_map": prob_map, "mask": mask, "box": box,
                "lesion_found": box is not None, "prob_malignant": p}
