"""Tests for the tuning-phase features (run on Kaggle in the notebook's first section)."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from bus.data.augment import Augment, Photometric, build_augment
from bus.grid import expand
from bus.reporting import compare
from bus.roi import mask_to_box
from bus.stats import compare as stats_compare
from bus.threshold import pick
from bus.utils import EarlyStopping


def test_photometric_keeps_shape_and_dtype():
    img = (np.arange(64 * 80) % 256).astype(np.uint8).reshape(64, 80)
    out = Photometric(p=1.0, seed=0)(img)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert not np.array_equal(out, img)


def test_augment_wrapper_leaves_mask_alone():
    img = np.full((32, 32), 100, np.uint8)
    msk = np.zeros((32, 32), np.uint8)
    msk[8:16, 8:16] = 1
    out = Augment(None, Photometric(p=1.0, seed=1))(image=img, mask=msk)
    assert np.array_equal(out["mask"], msk)
    assert set(Augment(None, Photometric(seed=2))(image=img)) == {"image"}


def test_build_augment_with_photometric_config():
    aug = build_augment({"hflip": 0.5, "rotate": 10, "photometric": {"p": 1.0, "speckle": 0.1}})
    img = np.random.default_rng(0).integers(0, 255, (40, 40), dtype=np.uint8)
    out = aug(image=img, mask=np.zeros((40, 40), np.uint8))
    assert out["image"].shape == (40, 40) and out["mask"].shape == (40, 40)
    assert build_augment({"enabled": False}) is None


def test_full_image_box():
    m = np.zeros((50, 60), bool)
    m[10:20, 10:20] = True
    assert mask_to_box(m, full_image=True) == (0, 0, 60, 50)
    assert mask_to_box(np.zeros((5, 5), bool), full_image=True) is None


def test_grid_axes_expansion():
    grid = {"axes": [[{"name": "a", "set": {"x": 1}}, {"name": "b", "set": {"x": 2}}],
                     [{"name": "m1", "set": {"y": 1, "z": 0}}, {"name": "full", "set": {}}]]}
    runs = expand(grid)
    assert [r["name"] for r in runs] == ["a_m1", "a_full", "b_m1", "b_full"]
    assert runs[0]["set"] == {"x": 1, "y": 1, "z": 0}


def test_early_stopping_min_epochs():
    es = EarlyStopping(patience=2, mode="max", min_epochs=5)
    for _ in range(4):
        es.step(0.0)
    assert not es.stop          # patience exhausted, but only 4 epochs
    es.step(0.0)
    assert es.stop


def test_threshold_pick(tmp_path):
    y = [0] * 50 + [1] * 50
    p = list(np.linspace(0, 0.6, 50)) + list(np.linspace(0.3, 1, 50))
    f = tmp_path / "val.csv"
    pd.DataFrame({"y": y, "prob": p}).to_csv(f, index=False)
    r = pick(f, [0.9])["thresholds"]["0.90"]
    assert r["val_sensitivity"] >= 0.9 and 0 < r["threshold"] < 0.5


def test_compare_table(tmp_path):
    m = {"role": "test", "n": 10, "segmentation": {"dice": 0.7, "iou": 0.6},
         "classification": {"auc": 0.9, "recall": 0.8, "specificity": 0.85, "f1": 0.8, "accuracy": 0.82,
                            "threshold": 0.3},
         "by_dataset": {"BUSI": {"classification": {"auc": 0.91, "recall": 0.7}}},
         "normal_scans": {"n": 5, "lesion_wrongly_found": 1, "correctly_clear_rate": 0.8}}
    f = tmp_path / "metrics_test.json"
    f.write_text(json.dumps(m))
    df = compare([("final", f)])
    assert df.loc[0, "auc[BUSI]"] == 0.91 and df.loc[0, "normal_clear_rate"] == 0.8


def test_stats_separate_thresholds():
    a = pd.DataFrame({"id": list("abcd"), "label": ["benign", "malignant", "benign", "normal"],
                      "y": [0, 1, 0, -1], "prob": [0.2, 0.4, 0.1, 0.9], "dice": [0.8, 0.7, 0.9, 1.0]})
    b = a.assign(prob=[0.2, 0.4, 0.1, 0.9], dice=[0.7, 0.6, 0.8, 0.0])
    r = stats_compare(a, b, threshold=0.3, threshold_b=0.5, n_boot=50)
    assert r["n_paired"] == 3                                  # normal scan left out
    assert r["classification"]["only_a_correct"] == 1          # malignant caught only at 0.3


def test_vanilla_unet_he_init_gives_signal():
    from bus.models.segmentation import VanillaUNet
    torch.manual_seed(0)
    out = VanillaUNet(1, base=8)(torch.randn(2, 1, 64, 64))
    assert out.std().item() > 1e-2   # default init made this almost constant -> stuck at "background"


def test_bundle_roundtrip_with_ensemble_and_tta(tmp_path):
    pytest.importorskip("timm")
    from bus.bundle import export_bundle, load_bundle
    from bus.models.classification import Classifier
    from bus.models.segmentation import VanillaUNet

    seg_cfg = {"name": "tiny_seg", "data": {"image_size": 64, "resize_mode": "pad", "in_channels": 1},
               "model": {"arch": "vanilla_unet", "base_channels": 8}, "eval": {"threshold": 0.5},
               "train": {"seed": 1}}
    cls_cfg = {"name": "tiny_cls", "data": {"image_size": 64, "resize_mode": "pad"},
               "model": {"backbone": "resnet18", "head": "linear", "dropout": 0.0},
               "crop": {"margin": 0.5}, "eval": {"threshold": 0.5}, "train": {"seed": 1}}
    seg_ck, cls_ck = [], []
    for i in range(2):
        torch.save({"kind": "seg", "config": seg_cfg, "model": VanillaUNet(1, base=8).state_dict()},
                   tmp_path / f"s{i}.pt")
        torch.save({"kind": "cls", "config": cls_cfg,
                    "model": Classifier("resnet18", "linear", pretrained=False, img_size=64).state_dict()},
                   tmp_path / f"c{i}.pt")
        seg_ck.append(tmp_path / f"s{i}.pt")
        cls_ck.append(tmp_path / f"c{i}.pt")
    out = export_bundle(seg_ck, cls_ck, tmp_path / "bundle", threshold=0.3, post="largest", tta=True)
    pipe, settings = load_bundle(out, device=torch.device("cpu"))
    assert settings["threshold"] == 0.3 and len(pipe.segs) == 2 and len(pipe.clss) == 2 and pipe.tta
    gray = np.random.default_rng(0).integers(0, 255, (90, 120), dtype=np.uint8)
    o = pipe(gray)
    assert o["prob_map"].shape == gray.shape and o["mask"].shape == gray.shape
    assert 0.0 <= o["prob_malignant"] <= 1.0
