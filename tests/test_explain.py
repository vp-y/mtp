"""Explainability (Grad-CAM / LIME / Kernel SHAP), report, plain-language summary and dashboard API."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from bus import explain as xai
from bus import narrate


# ------------------------------------------------------------------------------------- method maths

def test_lime_recovers_a_linear_model():
    rng = np.random.default_rng(0)
    w = np.array([0.30, -0.20, 0.0, 0.10, 0.05])
    Z = rng.integers(0, 2, size=(600, 5)).astype(np.uint8)
    Z[0] = 1
    y = 0.2 + Z @ w
    fit = xai.lime_weights(Z, y, alpha=1e-6)
    assert np.allclose(fit["weights"], w, atol=1e-6) and fit["fit_r2"] > 0.999


def test_kernel_shap_is_exact_for_additive_models_and_sums_up():
    rng = np.random.default_rng(1)
    M = 8
    w = rng.normal(0, 0.1, M)
    Z = xai.shap_samples(M, 200, rng)
    assert Z.shape[1] == M and len(Z) >= 2 and set(np.unique(Z)) <= {0, 1}
    assert not ((Z.sum(1) == 0) | (Z.sum(1) == M)).any()      # empty / full coalitions are added separately
    f = lambda z: 0.4 + z @ w                                 # noqa: E731
    out = xai.shap_weights(Z, f(Z), f(np.ones(M)), f(np.zeros(M)))
    assert np.allclose(out["weights"], w, atol=1e-8)          # Shapley values of an additive model = its weights
    assert out["additivity_error"] < 1e-9


def test_superpixels_split_on_the_lesion_outline():
    patch = (np.random.default_rng(2).random((60, 80)) * 255).astype(np.uint8)
    mask = np.zeros((60, 80), bool)
    mask[15:45, 20:60] = True
    lab = xai.superpixels(patch, 12, mask)
    assert lab.shape == patch.shape and lab.min() == 0
    assert set(np.unique(lab)) == set(range(lab.max() + 1))   # consecutive labels
    for s in range(lab.max() + 1):                            # every region mostly inside or mostly outside
        frac = mask[lab == s].mean()
        assert frac <= 0.15 or frac >= 0.85 or (lab == s).sum() < 40


def test_region_map_and_lesion_measurements():
    gray = np.full((100, 120), 150, np.uint8)
    yy, xx = np.mgrid[:100, :120]
    mask = (yy - 50) ** 2 + (xx - 60) ** 2 <= 20 ** 2        # dark round "lesion"
    gray[mask] = 60
    reg = xai.region_map(mask)
    assert set(np.unique(reg)) == {0, 1, 2} and (reg[mask] <= 1).all()
    m = xai.lesion_measurements(gray, mask)
    assert m["circularity"] > 0.8 and m["solidity"] > 0.95
    assert 0.9 < m["depth_to_width"] < 1.1 and m["brightness_vs_surrounding"] < 0.6
    assert m["separate_regions_outlined"] == 1 and not m["touches_image_border"]
    assert xai.lesion_measurements(gray, np.zeros_like(mask)) is None


# ---------------------------------------------------------------------------- tiny end-to-end bundle

def _tiny_bundle(tmp_path):
    pytest.importorskip("timm")
    from bus.bundle import export_bundle
    from bus.models.classification import Classifier
    from bus.models.segmentation import VanillaUNet

    # segmentation threshold 0 -> every pixel is "lesion", so the classification stage always runs
    seg_cfg = {"name": "tiny_seg", "data": {"image_size": 64, "resize_mode": "pad", "in_channels": 1},
               "model": {"arch": "vanilla_unet", "base_channels": 8}, "eval": {"threshold": 0.0},
               "train": {"seed": 1}}
    cls_cfg = {"name": "tiny_cls", "data": {"image_size": 64, "resize_mode": "pad"},
               "model": {"backbone": "resnet18", "head": "linear", "dropout": 0.0},
               "crop": {"margin": 0.1}, "eval": {"threshold": 0.5}, "train": {"seed": 1}}
    seg, cls = [], []
    for i in range(2):
        torch.manual_seed(i)
        torch.save({"kind": "seg", "config": seg_cfg, "model": VanillaUNet(1, base=8).state_dict()}, tmp_path / f"s{i}.pt")
        torch.save({"kind": "cls", "config": cls_cfg,
                    "model": Classifier("resnet18", "linear", pretrained=False, img_size=64).state_dict()},
                   tmp_path / f"c{i}.pt")
        seg.append(tmp_path / f"s{i}.pt")
        cls.append(tmp_path / f"c{i}.pt")
    return export_bundle(seg, cls, tmp_path / "bundle", threshold=0.5, post="largest", tta=False)


def _image():
    g = (np.random.default_rng(3).random((90, 110)) * 200).astype(np.uint8)
    g[30:60, 40:75] //= 3
    return g


def test_explainer_end_to_end(tmp_path):
    from bus.bundle import load_bundle
    pipe, settings = load_bundle(_tiny_bundle(tmp_path), device=torch.device("cpu"))
    gray = _image()
    a = xai.analyse(pipe, gray)
    ref = pipe(gray)
    assert a["lesion_found"] and abs(a["prob_malignant"] - ref["prob_malignant"]) < 1e-6
    assert len(a["member_probs"]) == 2 and len(a["member_maps"]) == 2

    cam = xai.gradcam(pipe, a, settings["threshold"])
    assert cam is not None, "Grad-CAM must work for timm CNN backbones"
    assert cam["map"].shape == gray.shape and cam["map"].min() >= 0 and cam["map"].max() <= 1 + 1e-6
    sh = cam["metrics"]["evidence_by_region"]
    assert set(sh) == set(xai.REGIONS)

    exps = {"gradcam": {"metrics": cam["metrics"]}}
    for method in ("lime", "shap"):
        r = xai.attribution(pipe, a, settings["threshold"], method, n_samples=24, n_segments=8, batch_size=8)
        assert r["map"].shape == gray.shape and len(r["weights"]) == r["metrics"]["n_regions"]
        assert r["metrics"]["deletion_check"]["score_before"] == pytest.approx(r["metrics"]["score_explained"])
        if method == "shap":
            m = r["metrics"]
            assert abs(sum(r["weights"]) - (m["score_explained"] - m["score_all_regions_hidden"])) < 1e-6
        exps[method] = {"metrics": r["metrics"]}
        assert xai.render_attribution(gray, r["map"], r["region_box"], a["mask"]).shape == gray.shape + (3,)
    assert xai.render_cam(gray, cam["map"], a["box"]).shape == gray.shape + (3,)

    report = xai.build_report(pipe, a, settings, exps)
    json.dumps(report, allow_nan=False)                       # JSON-safe: no NaN, no numpy types
    assert set(report["explanations"]) == {"gradcam", "lime", "shap"}
    assert report["prediction"]["score_is_calibrated_probability"] is False


# ------------------------------------------------------------------------------------------ narration

def _report(found=True):
    return {"prediction": {"lesion_found": found, "decision": "x", "level": "high" if found else "none",
                           "score_malignant": 0.71 if found else None, "threshold": 0.3,
                           "distance_from_threshold": 0.41 if found else None, "models_above_threshold": "3/3"},
            "segmentation": {"threshold": 0.5, "max_lesion_probability": 0.2},
            "lesion_measurements": {"area_pct_of_image": 2.1, "depth_to_width": 1.2, "solidity": 0.8,
                                    "brightness_vs_surrounding": 0.5,
                                    "position": {"vertical": "middle", "horizontal": "centre"}} if found else None,
            "explanations": {"gradcam": {"focus_inside_or_on_lesion": 0.8},
                             "lime": {"top_regions": [{"location": "on the lesion edge", "direction": "towards malignant",
                                                       "effect": 0.12}],
                                      "deletion_check": {"regions_hidden": 3, "score_before": 0.71, "score_after": 0.3}}}
            if found else {},
            "flags": [{"code": "small_lesion", "text": "The outlined lesion is small."}] if found else [],
            "reference_performance": {"A/external": {"description": "BUS-BRA images", "images": 1875,
                                                     "sensitivity_cancers_flagged": 0.86,
                                                     "specificity_benign_not_flagged": 0.41}}}


@pytest.mark.parametrize("found", [True, False])
def test_template_summary_has_all_sections(found):
    text = narrate.template_summary(_report(found))
    for s in narrate.SECTIONS:
        assert f"## {s}" in text
    assert "not be used for diagnosis" in text


def test_summarize_without_key_uses_template(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    out = narrate.summarize(_report(), language="Hindi", audience="clinician")
    assert out["source"] == "template" and "## Summary" in out["text"] and "English" in out["note"]
    assert narrate.status()["enabled"] is False


def test_llm_prompt_contains_only_the_report():
    msg = narrate._user_message(_report(), "English", "patient")
    assert "```json" in msg and "patient" in msg
    assert "lesion_measurements" in msg and "data:image" not in msg


# ------------------------------------------------------------------------------------- dashboard API

def test_dashboard_api(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    try:  # file uploads need python-multipart (module name differs between versions)
        import python_multipart  # noqa: F401
    except ImportError:
        pytest.importorskip("multipart")
    import cv2
    import importlib
    from fastapi.testclient import TestClient

    monkeypatch.setenv("BUS_BUNDLE", str(_tiny_bundle(tmp_path)))
    monkeypatch.setenv("BUS_DEVICE", "cpu")
    monkeypatch.setenv("BUS_XAI_SAMPLES", "16")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    server = importlib.reload(importlib.import_module("app.server"))

    with TestClient(server.app) as client:
        assert client.get("/api/health").json()["models_loaded"]
        info = client.get("/api/info").json()
        assert info["llm"]["enabled"] is False and info["explanations"]["samples"] == 16
        ok, png = cv2.imencode(".png", _image())
        r = client.post("/api/predict", files={"file": ("x.png", png.tobytes(), "image/png")})
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["lesion_found"] and d["gradcam"].startswith("data:image/png") and d["report"]["prediction"]
        cid = d["case_id"]
        for method in ("lime", "shap"):
            e = client.post(f"/api/explain/{cid}?method={method}&segments=8")
            assert e.status_code == 200, e.text
            assert e.json()["image"].startswith("data:image/png") and method in e.json()["report"]["explanations"]
        assert client.post(f"/api/explain/{cid}?method=nope").status_code == 422
        s = client.post(f"/api/summary/{cid}", json={"language": "English", "audience": "patient"})
        assert s.status_code == 200 and s.json()["source"] == "template" and "## Summary" in s.json()["text"]
        assert client.get(f"/api/report/{cid}").json()["summary"]["source"] == "template"
        assert client.post("/api/summary/doesnotexist", json={}).status_code == 404
        assert client.get("/").status_code == 200
