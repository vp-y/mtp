"""Deterministic geometry (resize / pad / undo) and tensor conversion.

Every prediction is mapped back to the ORIGINAL image grid with `restore`, so metrics,
crops and the deployed app all work on original-resolution images.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

GRAY_MEAN, GRAY_STD = 0.449, 0.226  # ImageNet mean/std averaged over RGB
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


@dataclass(frozen=True)
class Geometry:
    orig_h: int
    orig_w: int
    new_h: int
    new_w: int
    top: int
    left: int
    size: int
    mode: str


def resize_to_square(img: np.ndarray, size: int, mode: str = "pad", interp: int = cv2.INTER_LINEAR):
    """Resize to size x size. mode='pad' keeps the aspect ratio (zero padding), 'stretch' does not."""
    h, w = img.shape[:2]
    if mode == "stretch":
        out = cv2.resize(img, (size, size), interpolation=interp)
        return out, Geometry(h, w, size, size, 0, 0, size, mode)
    if mode != "pad":
        raise ValueError(f"resize_mode must be 'pad' or 'stretch', got {mode!r}")
    s = size / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.zeros((size, size) + img.shape[2:], dtype=img.dtype)
    out[top:top + nh, left:left + nw] = resized
    return out, Geometry(h, w, nh, nw, top, left, size, mode)


def restore(arr: np.ndarray, g: Geometry, interp: int = cv2.INTER_LINEAR) -> np.ndarray:
    """Inverse of resize_to_square: size x size map -> original (orig_h, orig_w)."""
    region = arr[g.top:g.top + g.new_h, g.left:g.left + g.new_w]
    return cv2.resize(region, (g.orig_w, g.orig_h), interpolation=interp)


def to_tensor(img: np.ndarray, channels: int = 1, normalize: bool = True) -> torch.Tensor:
    """uint8 HxW grayscale -> float tensor (C, H, W). channels=3 replicates the gray channel."""
    x = img.astype(np.float32) / 255.0
    if channels == 3:
        x = np.repeat(x[None], 3, axis=0)
        if normalize:
            x = (x - IMAGENET_MEAN) / IMAGENET_STD
    elif channels == 1:
        x = x[None]
        if normalize:
            x = (x - GRAY_MEAN) / GRAY_STD
    else:
        raise ValueError("channels must be 1 or 3")
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
