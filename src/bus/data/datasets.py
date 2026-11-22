"""PyTorch datasets built from manifest rows. Images are cached in memory (datasets are small)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..roi import crop, mask_to_box
from .manifest import LABEL_TO_INT
from .preprocess import resize_to_square, to_tensor


def read_gray(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def read_mask(path) -> np.ndarray:
    return read_gray(path) > 127


class SegDataset(Dataset):
    """(image, mask[, label]) at size x size. Used for segmentation and the joint baselines."""

    def __init__(self, df, root, size=256, resize_mode="pad", channels=1, augment=None,
                 with_label=False, normalize=True):
        self.df = df.reset_index(drop=True)
        root = Path(root)
        self.channels, self.augment = channels, augment
        self.with_label, self.normalize = with_label, normalize
        self.images, self.masks = [], []
        for r in self.df.itertuples(index=False):
            img, _ = resize_to_square(read_gray(root / r.image_path), size, resize_mode, cv2.INTER_LINEAR)
            msk, _ = resize_to_square(read_mask(root / r.mask_path).astype(np.uint8), size, resize_mode,
                                      cv2.INTER_NEAREST)
            self.images.append(img)
            self.masks.append(msk)
        self.labels = np.array([LABEL_TO_INT.get(lbl, -1) for lbl in self.df["label"]])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img, msk = self.images[i], self.masks[i]
        if self.augment is not None:
            out = self.augment(image=img, mask=msk)
            img, msk = out["image"], out["mask"]
        sample = {
            "image": to_tensor(img, self.channels, self.normalize),
            "mask": torch.from_numpy(np.ascontiguousarray(msk, dtype=np.float32))[None],
            "index": i,
        }
        if self.with_label:
            sample["label"] = torch.tensor(int(self.labels[i]), dtype=torch.long)
        return sample


class ClsDataset(Dataset):
    """Lesion crops for benign (0) / malignant (1) classification.

    The ROI comes from the ground-truth mask, or from predicted masks in `mask_dir`
    (files named <id>.png, written by `python -m bus.predict_masks`). If a mask is empty
    the whole image is used and `used_full_image` records it.
    """

    def __init__(self, df, root, crop_cfg, size=224, resize_mode="pad", augment=None, mask_dir=None):
        self.df = df.reset_index(drop=True)
        root = Path(root)
        self.augment = augment
        self.crops, self.used_full_image = [], []
        for r in self.df.itertuples(index=False):
            if r.label not in LABEL_TO_INT:
                raise ValueError(f"ClsDataset needs benign/malignant rows, got {r.label!r} ({r.id})")
            img = read_gray(root / r.image_path)
            mpath = Path(mask_dir) / f"{r.id}.png" if mask_dir else root / r.mask_path
            box = mask_to_box(read_mask(mpath), **(crop_cfg or {}))
            patch = crop(img, box) if box is not None else img
            patch, _ = resize_to_square(patch, size, resize_mode, cv2.INTER_LINEAR)
            self.crops.append(patch)
            self.used_full_image.append(box is None)
        self.labels = np.array([LABEL_TO_INT[lbl] for lbl in self.df["label"]])

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, i):
        patch = self.crops[i]
        if self.augment is not None:
            patch = self.augment(image=patch)["image"]
        return {"image": to_tensor(patch, 3), "label": torch.tensor(float(self.labels[i])), "index": i}
