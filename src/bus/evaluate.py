"""Evaluate the full dual-stage pipeline (paper Table 3, "Our Approach").

  python -m bus.evaluate --seg runs/seg/A/<seg>/seed42 --cls runs/cls/A/<cls>/seed42 --scenario A --role test
  python -m bus.evaluate ... --scenario C --role test        # cross-dataset (Breast-Lesions-USG)
  python -m bus.evaluate ... --scenario A --role external    # BUS-BRA, 1,064 unseen patients

Tuning-phase options:
  --seg A B C --cls X Y Z     ensembles (probabilities averaged), e.g. one configuration x 3 seeds
  --tta                       also average with the prediction on the horizontally flipped image
  --threshold 0.23            P(malignant) decision threshold (pick it on VAL with bus.threshold)
  --post largest              keep only the largest segmented region
  --include-normal            also run on normal scans (reports how often a lesion is wrongly found)

Results go to <out>/pipeline/<scenario>/<name>/ (metrics_<role>.json, per_image_<role>.csv, ...).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import resolve_data_root
from .evaluation import evaluate_model
from .inference import DualStagePipeline, _as_list


def evaluate_pipeline(seg, cls, data_root, scenario, role, out_root="runs", name="dualstage",
                      empty_policy="full_image", threshold=None, seg_threshold=None, post=None,
                      smoke=False, tta=False, include_normal=False) -> dict:
    pipe = DualStagePipeline(seg, cls, empty_policy=empty_policy, tta=tta, threshold=seg_threshold, post=post)
    thr = pipe.cls.threshold if threshold is None else threshold
    out_dir = Path(out_root) / "pipeline" / scenario / name
    segs, clss = _as_list(seg), _as_list(cls)
    desc = f"{pipe.seg.cfg['name']}x{len(segs)} -> {pipe.cls.cfg['name']}x{len(clss)}" \
        if len(segs) > 1 or len(clss) > 1 else f"{pipe.seg.cfg['name']} -> {pipe.cls.cfg['name']}"
    meta = {"task": "pipeline", "name": name, "seed": None,
            "desc": desc + (" +TTA" if tta else ""),
            "seg_checkpoint": [str(s) for s in segs] if len(segs) > 1 else str(segs[0]),
            "cls_checkpoint": [str(c) for c in clss] if len(clss) > 1 else str(clss[0]),
            "empty_policy": empty_policy, "tta": bool(tta),
            "seg_post": pipe.seg.post, "seg_threshold": pipe.seg.threshold}

    def predict(gray):
        o = pipe(gray)
        return {"mask": o["mask"], "prob": o["prob_malignant"], "lesion_found": o["lesion_found"]}

    return evaluate_model(predict, meta, data_root, out_dir, scenario, [role], include_normal=include_normal,
                          smoke=smoke, cls_threshold=thr)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seg", required=True, nargs="+", help="segmentation checkpoint(s) or run dir(s)")
    p.add_argument("--cls", required=True, nargs="+", help="classification checkpoint(s) or run dir(s)")
    p.add_argument("--scenario", default="A", choices=["A", "B", "C", "deploy"])
    p.add_argument("--role", default="test", choices=["val", "test", "external"])
    p.add_argument("--data-root", default=None)
    p.add_argument("--out", default="runs")
    p.add_argument("--name", default="dualstage")
    p.add_argument("--empty-policy", default="full_image", choices=["full_image", "benign"])
    p.add_argument("--threshold", type=float, default=None, help="P(malignant) decision threshold")
    p.add_argument("--seg-threshold", type=float, default=None, help="mask probability threshold")
    p.add_argument("--post", default=None, choices=["none", "largest"], help="mask post-processing")
    p.add_argument("--tta", action="store_true", help="horizontal-flip test-time augmentation")
    p.add_argument("--include-normal", action="store_true", help="also evaluate normal (lesion-free) scans")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    evaluate_pipeline(a.seg, a.cls, resolve_data_root(a.data_root), a.scenario, a.role, a.out, a.name,
                      a.empty_policy, a.threshold, a.seg_threshold, a.post, a.smoke, a.tta, a.include_normal)


if __name__ == "__main__":
    main()
