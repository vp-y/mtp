"""Paired statistical comparison of two methods evaluated on the same images.

  python -m bus.stats --a runs/pipeline/A/dualstage/per_image_test.csv \
                      --b runs/maskrcnn/A/maskrcnn_r50_fpn/seed42/per_image_test.csv --names ours maskrcnn

Segmentation: Wilcoxon signed-rank test on per-image Dice and IoU (the test named in the paper).
Classification: exact McNemar test on per-image correctness, and a paired bootstrap 95 % CI of the AUC difference.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
from sklearn.metrics import roc_auc_score


def compare(a: pd.DataFrame, b: pd.DataFrame, threshold: float = 0.5, n_boot: int = 2000, seed: int = 0) -> dict:
    m = a.merge(b, on="id", suffixes=("_a", "_b"))
    out = {"n_paired": int(len(m))}
    for k in ("dice", "iou"):
        if f"{k}_a" in m and f"{k}_b" in m:
            entry = {"mean_a": float(m[f"{k}_a"].mean()), "mean_b": float(m[f"{k}_b"].mean())}
            try:
                entry["p_wilcoxon"] = float(wilcoxon(m[f"{k}_a"], m[f"{k}_b"]).pvalue)
            except ValueError:  # all differences are zero
                entry["p_wilcoxon"] = 1.0
            out[k] = entry
    if "prob_a" in m and "prob_b" in m:
        m = m[m["y_a"].isin([0, 1])]
        y = m["y_a"].to_numpy().astype(int)
        pa, pb = m["prob_a"].to_numpy(), m["prob_b"].to_numpy()
        ca, cb = (pa >= threshold) == y, (pb >= threshold) == y
        only_a, only_b = int((ca & ~cb).sum()), int((~ca & cb).sum())
        p_mcnemar = binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
        entry = {"acc_a": float(ca.mean()), "acc_b": float(cb.mean()), "only_a_correct": only_a,
                 "only_b_correct": only_b, "p_mcnemar": float(p_mcnemar)}
        if len(np.unique(y)) == 2:
            entry["auc_a"], entry["auc_b"] = float(roc_auc_score(y, pa)), float(roc_auc_score(y, pb))
            rng = np.random.default_rng(seed)
            diffs = []
            for _ in range(n_boot):
                idx = rng.integers(0, len(y), len(y))
                if len(np.unique(y[idx])) == 2:
                    diffs.append(roc_auc_score(y[idx], pa[idx]) - roc_auc_score(y[idx], pb[idx]))
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            entry["auc_diff_ci95"] = [float(lo), float(hi)]
        out["classification"] = entry
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="per_image_<role>.csv of method A")
    p.add_argument("--b", required=True, help="per_image_<role>.csv of method B")
    p.add_argument("--names", nargs=2, default=["A", "B"])
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--out", default=None, help="optional JSON output path")
    a = p.parse_args()
    res = {"a": a.names[0], "b": a.names[1], **compare(pd.read_csv(a.a), pd.read_csv(a.b), a.threshold)}
    text = json.dumps(res, indent=2)
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
