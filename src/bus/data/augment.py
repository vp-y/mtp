"""Training-time augmentation.

Geometric (paper): rotation 20 deg, translation 20 %, zoom, horizontal flip; applied jointly to
image and mask with albumentations.

Photometric (tuning phase, not in the paper): brightness / contrast, gamma, multiplicative speckle
noise and blur, applied to the IMAGE ONLY. These imitate differences between ultrasound machines
and settings (gain, dynamic range, speckle, resolution), which the error analysis showed to be the
main cause of the drop on other hospitals' images. Implemented in numpy / OpenCV so that the
behaviour does not depend on the albumentations version.
"""
from __future__ import annotations

import os

import albumentations as A
import cv2
import numpy as np


class Photometric:
    """Random intensity changes for a uint8 grayscale image. Every op fires with probability p."""

    def __init__(self, p: float = 0.5, brightness: float = 0.2, contrast: float = 0.2,
                 gamma=(0.7, 1.5), speckle: float = 0.15, blur: float = 0.2, seed=None):
        self.p, self.brightness, self.contrast = float(p), float(brightness), float(contrast)
        self.gamma = tuple(float(g) for g in gamma) if gamma else None
        self.speckle, self.blur = float(speckle), float(blur)
        self.seed, self._rng, self._pid = seed, None, None

    @property
    def rng(self) -> np.random.Generator:
        # Created lazily in each DataLoader worker (new pid) from that worker's torch seed, so
        # workers do not all draw the same "random" numbers.
        if self._rng is None or self._pid != os.getpid():
            import torch
            s = self.seed if self.seed is not None else torch.initial_seed() % (2 ** 32)
            self._rng, self._pid = np.random.default_rng(s), os.getpid()
        return self._rng

    def _fire(self) -> bool:
        return self.rng.random() < self.p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        x = img.astype(np.float32) / 255.0
        if self.contrast and self._fire():
            c = 1.0 + self.rng.uniform(-self.contrast, self.contrast)
            x = (x - x.mean()) * c + x.mean()
        if self.brightness and self._fire():
            x = x + self.rng.uniform(-self.brightness, self.brightness)
        x = np.clip(x, 0.0, 1.0)
        if self.gamma and self._fire():
            x = x ** self.rng.uniform(*self.gamma)
        if self.speckle and self._fire():  # multiplicative noise, as ultrasound speckle
            x = x * (1.0 + self.rng.normal(0.0, self.rng.uniform(0, self.speckle), x.shape).astype(np.float32))
        if self.blur and self._fire():
            sigma = self.rng.uniform(0.3, 1.0 + 4 * self.blur)
            x = cv2.GaussianBlur(x, (0, 0), sigma)
        return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


class Augment:
    """Geometric (image + mask) then photometric (image only). Called like albumentations:
    aug(image=img) or aug(image=img, mask=msk) -> dict with the same keys."""

    def __init__(self, geometric=None, photometric=None):
        self.geometric, self.photometric = geometric, photometric

    def __call__(self, image, mask=None):
        out = {"image": image} if mask is None else {"image": image, "mask": mask}
        if self.geometric is not None:
            out = self.geometric(**out)
        if self.photometric is not None:
            out["image"] = self.photometric(out["image"])
        return out


def build_augment(cfg):
    """Build the training augmentation from the `augment` config section. Returns None if disabled."""
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
    geometric = A.Compose(ops) if ops else None

    ph = cfg.get("photometric")
    photometric = None
    if ph and (ph is True or ph.get("enabled", True)):
        ph = {} if ph is True else {k: v for k, v in ph.items() if k != "enabled"}
        photometric = Photometric(**ph)
    if geometric is None and photometric is None:
        return None
    return Augment(geometric, photometric)
