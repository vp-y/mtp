"""Pick the best run by VALIDATION score, and collect all results into paper-style tables.

  python -m bus.reporting best runs/seg/A --metric segmentation.dice
  python -m bus.reporting collect runs --out reports

In a notebook:  from bus.reporting import best_run;  best_run("runs/seg/A", "segmentation.dice")
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from .utils import load_json


def _get(d, dotted: str):
    for k in dotted.split("."):
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def best_run(root, metric: str = "segmentation.dice", role: str = "val", mode: str = "max",
             verbose: bool = True) -> str:
    """Run directory of the configuration with the best mean `metric` on `role` (default: val).

    Model selection never looks at the test or external sets. With several seeds the
    configuration is ranked by its MEAN over seeds, and its best single seed is returned.
    """
    return best_runs(root, metric, role, mode, verbose)[0]


def best_runs(root, metric: str = "segmentation.dice", role: str = "val", mode: str = "max",
              verbose: bool = True) -> list:
    """All seed run directories of the best configuration (best seed first): an ensemble."""
    df = _scan(root, metric, role)
    asc = mode == "min"
    agg = df.groupby("name")["value"].agg(["mean", "std", "count"]).sort_values("mean", ascending=asc)
    best = agg.index[0]
    runs = df[df["name"] == best].sort_values("value", ascending=asc)["run"].tolist()
    if verbose:
        print(f"top configurations by {role} {metric} (mean over seeds):")
        print(agg.head(10).round(4).to_string())
        print(f"-> best: {best} ({len(runs)} seed run(s)); first: {runs[0]}")
    return runs


def rank(root, metric: str = "segmentation.dice", role: str = "val") -> pd.DataFrame:
    """Mean / std / count of `metric` per configuration name (no printing)."""
    df = _scan(root, metric, role)
    return df.groupby("name")["value"].agg(["mean", "std", "count"]).sort_values("mean", ascending=False)


def _scan(root, metric: str, role: str) -> pd.DataFrame:
    rows = []
    for p in Path(root).rglob(f"metrics_{role}.json"):
        r = load_json(p)
        v = _get(r, metric)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        rows.append({"name": r.get("name"), "seed": r.get("seed"), "value": float(v), "run": str(p.parent)})
    if not rows:
        raise FileNotFoundError(f"no metrics_{role}.json with '{metric}' under {root}")
    return pd.DataFrame(rows)


def compare(entries) -> pd.DataFrame:
    """Side-by-side table of evaluations. entries: list of (label, metrics_json_path).

    Includes the per-dataset AUC / recall when the evaluation covered several datasets.
    """
    rows = []
    for label, path in entries:
        r = load_json(path)
        s, c = r.get("segmentation") or {}, r.get("classification") or {}
        row = {"system": label, "role": r.get("role"), "n": r.get("n"),
               "dice": s.get("dice"), "iou": s.get("iou"),
               "auc": c.get("auc"), "sensitivity": c.get("recall"), "specificity": c.get("specificity"),
               "f1": c.get("f1"), "accuracy": c.get("accuracy"), "threshold": c.get("threshold")}
        for ds, sub in (r.get("by_dataset") or {}).items():
            sc = sub.get("classification") or {}
            row[f"auc[{ds}]"] = sc.get("auc")
            row[f"sens[{ds}]"] = sc.get("recall")
        ns = r.get("normal_scans")
        if ns:
            row["normal_clear_rate"] = ns["correctly_clear_rate"]
        rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    return _md_table(df)


def _flatten(path: Path) -> dict:
    r = load_json(path)
    row = {k: r.get(k) for k in ("task", "scenario", "name", "seed", "role", "desc", "n")}
    row["run"] = str(path.parent)
    for k, v in (r.get("segmentation") or {}).items():
        row[f"seg_{k}"] = v
    for k, v in (r.get("classification") or {}).items():
        row[f"cls_{k}"] = v
    if "n_lesion_not_found" in r:
        row["n_lesion_not_found"] = r["n_lesion_not_found"]
    return row


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_no results_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in r.tolist()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _mean_std(df: pd.DataFrame, keys, metrics) -> pd.DataFrame:
    metrics = [m for m in metrics if m in df.columns]
    if df.empty or not metrics:
        return pd.DataFrame()
    g = df.groupby(keys, dropna=False)
    out = g[metrics].mean()
    if (g.size() > 1).any():  # show the spread across seeds when there is one
        std = g[metrics].std(ddof=0)
        for m in metrics:
            out[m] = [f"{a:.3f} ± {b:.3f}" if pd.notna(b) and b > 0 else f"{a:.3f}"
                      for a, b in zip(out[m], std[m])]
    out["seeds"] = g.size()
    return out.reset_index()


SEG_COLS = ["seg_dice", "seg_iou", "seg_pixel_acc", "seg_dice_pooled", "seg_iou_pooled"]
CLS_COLS = ["cls_auc", "cls_f1", "cls_precision", "cls_recall", "cls_specificity", "cls_accuracy"]


def collect(runs_root, out_dir) -> pd.DataFrame:
    rows = [_flatten(p) for p in Path(runs_root).rglob("metrics_*.json")]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"no metrics found under {runs_root}")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.to_csv(out / "all_results.csv", index=False)

    md = ["# Results\n", "All numbers are on held-out data. Model selection used the validation split only.\n"]
    t1 = _mean_std(df[(df.task == "seg") & (df.role == "test")], ["scenario", "name", "desc"], SEG_COLS)
    if not t1.empty:
        t1 = t1.sort_values("seg_dice", ascending=False, key=lambda s: s.astype(str))
    md += ["\n## Table 1: segmentation (test)\n", _md_table(t1)]
    t2 = _mean_std(df[(df.task == "cls") & (df.role == "test")], ["scenario", "name", "desc"], CLS_COLS)
    if not t2.empty:
        t2 = t2.sort_values("cls_auc", ascending=False, key=lambda s: s.astype(str))
    md += ["\n## Table 2: classification of ground-truth crops (test)\n", _md_table(t2)]
    joint = df[df.task.isin(["pipeline", "maskrcnn", "mtswin"])]
    for role, title in (("test", "Table 3: full systems (test)"), ("external", "External test: BUS-BRA")):
        t3 = _mean_std(joint[joint.role == role], ["task", "scenario", "name"], SEG_COLS + CLS_COLS)
        if not t3.empty:
            t3 = t3.sort_values(["scenario", "task"])
        md += [f"\n## {title}\n", _md_table(t3)]
    (out / "results.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out / 'results.md'} and {out / 'all_results.csv'} ({len(df)} result files)")
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("best", help="print the best run directory (by validation score)")
    b.add_argument("root")
    b.add_argument("--metric", default="segmentation.dice")
    b.add_argument("--role", default="val")
    b.add_argument("--mode", default="max", choices=["max", "min"])
    c = sub.add_parser("collect", help="gather every metrics_*.json into reports/results.md")
    c.add_argument("runs")
    c.add_argument("--out", default="reports")
    a = p.parse_args()
    if a.cmd == "best":
        best_run(a.root, a.metric, a.role, a.mode)
    else:
        collect(a.runs, a.out)


if __name__ == "__main__":
    main()
