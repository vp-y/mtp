"""Quantitative evaluation of the explanations on labelled test images (thesis chapter "Explainability").

  python -m bus.explain_eval --bundle deploy_bundle --scenario A --role external --n 120 \
      --methods gradcam lime shap --out runs/xai/external

For a stratified sample of benign / malignant images it measures:
  gradcam_energy_in_gt       share of the Grad-CAM attention inside the GROUND-TRUTH lesion
  gradcam_energy_in_gt_band  ... inside the ground-truth lesion plus a margin band around it
  gt_share_of_region         share of the classifier's region covered by the lesion (= chance level)
  gradcam_peak_in_gt_band    "pointing game": the attention maximum lies on the lesion or its margin
  <m>_deletion_change        score change when the 3 most supporting regions are blurred (faithfulness:
                             a larger drop towards the other class = the explanation found what mattered)
  <m>_top_in_gt              the most influential LIME / SHAP region lies on the true lesion
  lime_fit_r2                how well the local LIME model reproduces the classifier
and compares correct vs wrong decisions. Writes per_image.csv, summary.json and examples.png.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import explain as xai
from .config import resolve_data_root
from .data.datasets import read_gray, read_mask
from .data.manifest import LABEL_TO_INT, load_manifest, select
from .utils import save_json


def _band(mask: np.ndarray) -> np.ndarray:
    area = int(mask.sum())
    if area == 0:
        return mask
    k = max(2, int(round(0.12 * math.sqrt(area))))
    return cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
                      ).astype(bool)


def _sample(rows: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or n >= len(rows):
        return rows
    per = max(1, n // rows["label"].nunique())
    return pd.concat([g.sample(min(per, len(g)), random_state=seed) for _, g in rows.groupby("label")]
                     ).reset_index(drop=True)


def evaluate(bundle, data_root, scenario="A", role="external", n=120, methods=("gradcam", "lime"),
             samples=None, segments=30, out="runs/xai", seed=0, device=None, examples=6) -> dict:
    from .bundle import load_bundle

    pipe, settings = load_bundle(bundle, device=device)
    thr = float(settings["threshold"])
    root, out = Path(data_root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = _sample(select(load_manifest(root), scenario, role), n, seed)
    recs, shown = [], []
    for i, r in enumerate(rows.itertuples(index=False)):
        gray, gt = read_gray(root / r.image_path), read_mask(root / r.mask_path)
        a = xai.analyse(pipe, gray)
        y = LABEL_TO_INT[r.label]
        rec = {"id": r.id, "dataset": r.dataset, "label": r.label, "y": y, "lesion_found": a["lesion_found"],
               "score": a["prob_malignant"], "correct": (a["prob_malignant"] >= thr) == bool(y)}
        if not a["lesion_found"]:
            recs.append(rec)
            continue
        band = _band(gt)
        x0, y0, x1, y1 = xai._union(xai._member_boxes(pipe, a))
        rec["gt_share_of_region"] = float(gt[y0:y1, x0:x1].mean())
        maps = {}
        if "gradcam" in methods:
            cam = xai.gradcam(pipe, a, thr)
            if cam is not None:
                c = cam["map"]
                tot = c.sum()
                rec["gradcam_energy_in_gt"] = float(c[gt].sum() / tot) if tot > 0 else np.nan
                rec["gradcam_energy_in_gt_band"] = float(c[band].sum() / tot) if tot > 0 else np.nan
                rec["gradcam_peak_in_gt_band"] = bool(band.flat[int(np.argmax(c))]) if tot > 0 else np.nan
                maps["gradcam"] = c
        for m in ("lime", "shap"):
            if m not in methods:
                continue
            res = xai.attribution(pipe, a, thr, m, n_samples=samples, n_segments=segments, seed=seed)
            met = res["metrics"]
            d = met["deletion_check"]
            # towards the other class: positive = the hidden regions really supported the decision
            sign = -1.0 if a["prob_malignant"] >= thr else 1.0
            rec[f"{m}_deletion_change"] = sign * d["change"]
            rec[f"{m}_deletion_flips"] = bool((d["score_after"] >= thr) != (d["score_before"] >= thr))
            j = int(np.argmax(np.abs(res["weights"])))
            rx0, ry0, rx1, ry1 = res["region_box"]
            seg = res["labels"] == j
            rec[f"{m}_top_in_gt"] = bool(band[ry0:ry1, rx0:rx1][seg].mean() > 0.5)
            if m == "lime":
                rec["lime_fit_r2"] = met["fit_r2"]
            maps[m] = (res["map"], res["region_box"])
        recs.append(rec)
        if len(shown) < examples and (len(shown) % 2 == int(not rec["correct"]) or i > 3 * examples):
            shown.append((r, gray, gt, a, maps, rec))
        print(f"[{i + 1}/{len(rows)}] {r.id} {r.label} score={a['prob_malignant']:.2f} "
              + " ".join(f"{k}={v:.2f}" for k, v in rec.items() if isinstance(v, float) and k != "score"))

    df = pd.DataFrame(recs)
    df.to_csv(out / "per_image.csv", index=False)
    metric_cols = [c for c in df.columns if c.startswith(("gradcam_", "lime_", "shap_", "gt_share"))]

    def agg(g):
        return {c: float(pd.to_numeric(g[c], errors="coerce").mean()) for c in metric_cols} | {"n": int(len(g))}

    found = df[df["lesion_found"]]
    summary = {"bundle": str(bundle), "scenario": scenario, "role": role, "threshold": thr,
               "n_images": int(len(df)), "n_lesion_found": int(len(found)), "methods": list(methods),
               "samples": samples or xai.default_samples(pipe.cls.device), "segments": segments,
               "all": agg(found) if len(found) else {},
               "correct": agg(found[found["correct"]]) if found["correct"].any() else {},
               "wrong": agg(found[~found["correct"]]) if (~found["correct"]).any() else {},
               "by_label": {k: agg(g) for k, g in found.groupby("label")}}
    save_json(summary, out / "summary.json")
    if shown:
        _figure(shown, methods, out / "examples.png")
    print("summary:", {k: round(v, 3) for k, v in summary["all"].items() if isinstance(v, float)})
    return summary


def _figure(shown, methods, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = ["image + ground truth", "prediction"] + [m for m in ("gradcam", "lime", "shap") if m in methods]
    fig, axes = plt.subplots(len(shown), len(cols), figsize=(3 * len(cols), 3 * len(shown)), squeeze=False)
    for row, (r, gray, gt, a, maps, rec) in zip(axes, shown):
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        gt_vis = rgb.copy()
        cs, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(gt_vis, cs, -1, (0, 255, 0), max(1, round(max(gray.shape) / 300)))
        panels = [gt_vis]
        pred = rgb.copy()
        cs, _ = cv2.findContours(a["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(pred, cs, -1, (255, 200, 0), max(1, round(max(gray.shape) / 300)))
        panels.append(pred)
        for m in cols[2:]:
            if m == "gradcam" and "gradcam" in maps:
                panels.append(cv2.cvtColor(xai.render_cam(gray, maps["gradcam"]), cv2.COLOR_BGR2RGB))
            elif m in maps:
                amap, box = maps[m]
                panels.append(cv2.cvtColor(xai.render_attribution(gray, amap, box, a["mask"]), cv2.COLOR_BGR2RGB))
            else:
                panels.append(np.zeros_like(rgb))
        titles = [f"{r.label} ({r.dataset})", f"score {rec['score']:.2f} {'correct' if rec['correct'] else 'WRONG'}"] + cols[2:]
        for ax, img, t in zip(row, panels, titles):
            ax.imshow(img)
            ax.set_title(t, fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", required=True)
    p.add_argument("--data-root", default=None)
    p.add_argument("--scenario", default="A", choices=["A", "B", "C", "deploy"])
    p.add_argument("--role", default="external", choices=["val", "test", "external"])
    p.add_argument("--n", type=int, default=120, help="images (stratified by label); 0 = all")
    p.add_argument("--methods", nargs="+", default=["gradcam", "lime"], choices=["gradcam", "lime", "shap"])
    p.add_argument("--samples", type=int, default=None, help="perturbations per image for LIME / SHAP")
    p.add_argument("--segments", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/xai")
    a = p.parse_args()
    evaluate(a.bundle, resolve_data_root(a.data_root), a.scenario, a.role, a.n, a.methods, a.samples, a.segments,
             a.out, a.seed)


if __name__ == "__main__":
    main()
