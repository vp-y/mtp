"""Pick the P(malignant) decision threshold on the VALIDATION set for a target sensitivity.

  python -m bus.threshold runs/pipeline/deploy/final/per_image_val.csv --sensitivity 0.90
  python -m bus.threshold VAL.csv --sensitivity 0.90 0.95 --out threshold.json

For early detection, missing a cancer (false negative) costs far more than a false alarm, so the
default 0.5 threshold is replaced by the highest threshold that still reaches the target sensitivity
on validation. Then evaluate test / external with `bus.evaluate --threshold T` and report the
sensitivity actually reached there (it is usually a little lower than on validation).
"""
from __future__ import annotations

import argparse

import pandas as pd

from .metrics import cls_scores, threshold_for_sensitivity
from .utils import save_json


def pick(csv_path, targets=(0.90,)) -> dict:
    df = pd.read_csv(csv_path)
    df = df[df["y"].isin([0, 1])]
    out = {"source": str(csv_path), "n": int(len(df)), "thresholds": {}}
    for t in targets:
        thr = threshold_for_sensitivity(df["y"], df["prob"], float(t))
        s = cls_scores(df["y"], df["prob"], thr)
        out["thresholds"][f"{float(t):.2f}"] = {"threshold": thr, "val_sensitivity": s["recall"],
                                                "val_specificity": s["specificity"]}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", help="per_image_val.csv of the model / pipeline")
    p.add_argument("--sensitivity", nargs="+", type=float, default=[0.90])
    p.add_argument("--out", default=None, help="optional JSON output")
    a = p.parse_args()
    res = pick(a.csv, a.sensitivity)
    for t, r in res["thresholds"].items():
        print(f"target sensitivity {t}: threshold={r['threshold']:.4f} "
              f"(val sensitivity={r['val_sensitivity']:.3f}, specificity={r['val_specificity']:.3f})")
    if a.out:
        save_json(res, a.out)


if __name__ == "__main__":
    main()
