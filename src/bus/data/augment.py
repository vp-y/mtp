"""Training-time augmentation (applied jointly to image and mask)."""
from __future__ import annotations

import albumentations as A


def build_augment(cfg):
    """Paper: rotation (20 deg), translation (20 %), zoom, horizontal flip. Returns None if disabled."""
    if not cfg or not cfg.get("enabled", True):
        return None
    ops = []
    if cfg.get("hflip", 0) > 0:
        ops.append(A.HorizontalFlip(p=cfg["hflip"]))
    rot = float(cfg.get("rotate", 0))
    tr = float(cfg.get("translate", 0))
    sc = tuple(float(v) for v in cfg.get("scale", (1.0, 1.0)))
    if rot or tr or sc != (1.0, 1.0):
        ops.append(A.Affine(rotate=(-rot, rot),
                            translate_percent={"x": (-tr, tr), "y": (-tr, tr)},
                            scale=sc,
                            p=cfg.get("p_affine", 0.8)))
    if cfg.get("brightness_contrast", 0) > 0:  # optional, not in the paper
        ops.append(A.RandomBrightnessContrast(p=cfg["brightness_contrast"]))
    return A.Compose(ops) if ops else None
