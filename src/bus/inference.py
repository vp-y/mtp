"""Loading trained checkpoints and predicting on ORIGINAL-resolution grayscale images.

This module is what the deployed app imports:

    pipe = DualStagePipeline(seg_ckpts, cls_ckpts, tta=True)
    out = pipe(gray_uint8_image)   # prob_map, mask, lesion_box, lesion_found, prob_malignant

`seg_ckpts` / `cls_ckpts` may be one checkpoint (file or run dir) or a list: several models are
averaged (an ensemble, e.g. the same configuration trained with 3 seeds). `tta=True` also averages
each model's prediction with its prediction on the horizontally flipped image.
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


def _as_list(x) -> list:
    if x is None:
        return []
    return [x] if isinstance(x, (str, Path)) else list(x)


def _load_state(model: torch.nn.Module, state: dict) -> None:
    # exported bundles store float16 weights; cast back to the model's dtype
    ref = model.state_dict()
    model.load_state_dict({k: (v.to(ref[k].dtype) if k in ref and torch.is_tensor(v) else v)
                           for k, v in state.items()})


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
        _load_state(self.model, ck["model"])
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict_prob(self, gray: np.ndarray, tta: bool = False) -> np.ndarray:
        img, g = resize_to_square(gray, self.size, self.mode)
        x = to_tensor(img, self.channels)[None].to(self.device)
        if tta:
            x = torch.cat([x, torch.flip(x, dims=[3])])
        prob = torch.sigmoid(self.model(x)[:, 0].float())
        if tta:
            prob = (prob[0] + torch.flip(prob[1], dims=[1])) / 2
        else:
            prob = prob[0]
        return restore(prob.cpu().numpy(), g)

    def predict_mask(self, gray: np.ndarray, tta: bool = False) -> np.ndarray:
        return postprocess(self.predict_prob(gray, tta) >= self.threshold, self.post, self.min_area)


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
        _load_state(self.model, ck["model"])
        self.model.to(self.device).eval()

    def box(self, mask: np.ndarray):
        return mask_to_box(mask, **self.crop_cfg)

    @torch.no_grad()
    def predict_patch(self, patch: np.ndarray, tta: bool = False) -> float:
        img, _ = resize_to_square(patch, self.size, self.mode)
        x = to_tensor(img, 3)[None].to(self.device)
        if tta:
            x = torch.cat([x, torch.flip(x, dims=[3])])
        return float(torch.sigmoid(self.model(x)[:, 0].float()).mean())


class DualStagePipeline:
    """segment -> clean mask -> bounding box + margin -> crop -> classify.

    seg_ckpt / cls_ckpt: one checkpoint or a list (ensemble: probabilities are averaged).
    empty_policy decides what happens when no lesion is segmented:
      'full_image' classify the whole image (default, keeps every case scorable)
      'benign'     report P(malignant) = 0
    threshold / post / min_area: segmentation settings (default: from the first seg checkpoint).
    """

    def __init__(self, seg_ckpt, cls_ckpt, device=None, empty_policy: str = "full_image",
                 tta: bool = False, threshold=None, post=None, min_area=None):
        if empty_policy not in ("full_image", "benign"):
            raise ValueError("empty_policy must be 'full_image' or 'benign'")
        device = device or get_device()
        self.segs = [SegPredictor(c, device=device, threshold=threshold, post=post, min_area=min_area)
                     for c in _as_list(seg_ckpt)]
        self.clss = [ClsPredictor(c, device=device) for c in _as_list(cls_ckpt)]
        if not self.segs or not self.clss:
            raise ValueError("need at least one segmentation and one classification checkpoint")
        self.seg, self.cls = self.segs[0], self.clss[0]  # settings / names come from the first model
        self.empty_policy, self.tta = empty_policy, bool(tta)

    def segment(self, gray: np.ndarray):
        prob_map = np.mean([s.predict_prob(gray, self.tta) for s in self.segs], axis=0)
        mask = postprocess(prob_map >= self.seg.threshold, self.seg.post, self.seg.min_area)
        return prob_map, mask

    def classify(self, gray: np.ndarray, mask: np.ndarray) -> float:
        probs = []
        for c in self.clss:
            box = c.box(mask)
            if box is not None:
                probs.append(c.predict_patch(crop(gray, box), self.tta))
            elif self.empty_policy == "full_image":
                probs.append(c.predict_patch(gray, self.tta))
            else:
                probs.append(0.0)
        return float(np.mean(probs))

    def __call__(self, gray: np.ndarray) -> dict:
        prob_map, mask = self.segment(gray)
        lesion_box = mask_to_box(mask, margin=0.0, min_size=1)
        return {"prob_map": prob_map, "mask": mask,
                "lesion_box": lesion_box,          # tight box around the (largest) lesion
                "box": self.cls.box(mask),         # region the (first) classifier looked at
                "lesion_found": lesion_box is not None,
                "prob_malignant": self.classify(gray, mask)}
