"""Mask post-processing and region-of-interest (ROI) extraction: paper workflow steps 3-4."""
from __future__ import annotations

import cv2
import numpy as np


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 2:  # background + at most one blob
        return m.astype(bool)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == k


def remove_small(mask: np.ndarray, min_area: int) -> np.ndarray:
    m = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


def postprocess(mask: np.ndarray, mode: str = "none", min_area: int = 0) -> np.ndarray:
    """mode: 'none' (raw) or 'largest' (keep only the biggest blob)."""
    mask = mask.astype(bool)
    if min_area and min_area > 0:
        mask = remove_small(mask, int(min_area))
    if mode == "largest":
        mask = keep_largest_component(mask)
    elif mode != "none":
        raise ValueError(f"post must be 'none' or 'largest', got {mode!r}")
    return mask


def mask_to_box(mask: np.ndarray, margin: float = 0.1, min_size: int = 16,
                square: bool = False, component: str = "largest"):
    """Bounding box (x0, y0, x1, y1) of the lesion plus a margin, clamped to the image.

    Returns None if the mask is empty (no lesion found).
    """
    m = mask.astype(bool)
    if component == "largest":
        m = keep_largest_component(m)
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return None
    H, W = m.shape
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    cy, cx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
    h = max((y1 - y0) * (1 + 2 * margin), min_size)
    w = max((x1 - x0) * (1 + 2 * margin), min_size)
    if square:
        h = w = max(h, w)
    y0, y1 = int(round(cy - h / 2)), int(round(cy + h / 2))
    x0, x1 = int(round(cx - w / 2)), int(round(cx + w / 2))
    y0, x0 = max(0, y0), max(0, x0)
    y1, x1 = min(H, y1), min(W, x1)
    return (x0, y0, x1, y1)


def crop(img: np.ndarray, box) -> np.ndarray:
    x0, y0, x1, y1 = box
    return img[y0:y1, x0:x1]
