"""Tuning phase: the whole protocol as resumable stages (run on Kaggle, see notebooks/kaggle_tuning.ipynb).

  python -m bus.tune all --out /kaggle/working/tuning            # everything, in order
  python -m bus.tune cls seg --out /kaggle/working/tuning        # selected stages
  python -m bus.tune all --out /kaggle/working/tune_smoke --smoke   # code check: tiny data, 1 epoch

Stages (every stage skips work that is already finished, so a stopped session can just be re-run):
  vanilla  re-run the paper's plain U-Net with the fix (He init, min. 30 epochs, monitor val loss)
  cls      classifier grid on Scenario A: backbone x crop context x fine-tuning depth, 3 seeds
  seg      segmentation grid on Scenario A: architecture x photometric augmentation, 3 seeds
  deploy   retrain the best seg / cls configuration on BUSI + Breast-Lesions-USG + BUS-UCLM, 3 seeds
           (classifier with and without photometric augmentation)
  select   compare candidate systems on the common VALIDATION set (all three datasets), choose the
           final one, its mask post-processing and its decision threshold (target sensitivity 90 %)
  final    evaluate baseline (= reproduction) and final system on the test sets and BUS-BRA, + statistics
  export   write the deployment bundle (models + settings) for the app
  report   tuning_report.md + results.md

Selection only ever uses validation data. Test and external sets are evaluated once, at the end.
"""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import pandas as pd

from .bundle import export_bundle
from .config import load_config, resolve_data_root
from .evaluate import evaluate_pipeline
from .grid import REPO, run_grid
from .reporting import _md_table, _scan, best_runs, collect, compare
from .stats import compare as stats_compare
from .threshold import pick
from .utils import load_json, run_dir_path, save_json

STAGES = ["vanilla", "cls", "seg", "deploy", "select", "final", "export", "report"]
SEEDS = [42, 43, 44]
BASE_SEG = "unet_resnet34_c64_geo"         # == reproduction's best segmentation configuration
BASE_CLS = "resnet50_mlp_d0.5_m0.1_k10"    # == reproduction's best classifier configuration
PHOTO = {"p": 0.5, "brightness": 0.2, "contrast": 0.2, "gamma": [0.7, 1.5], "speckle": 0.15, "blur": 0.2}
TARGET_SENSITIVITY = 0.90
GRIDS = REPO / "configs" / "grids"
# where the final systems are tested; Scenario C is left out on purpose: the deploy-trained models
# have seen Breast-Lesions-USG training images, so "all of Breast-Lesions-USG" is not unseen for them
FINAL_EVALS = [("deploy", "test", True), ("A", "test", False), ("A", "external", False)]


class Ctx:
    def __init__(self, out, data_root=None, smoke=False, seeds=None):
        self.out = Path(out)
        self.root = resolve_data_root(data_root)
        self.smoke = smoke
        self.seeds = list(seeds or (SEEDS[:2] if smoke else SEEDS))
        self.max_configs = 2 if smoke else None
        self.out.mkdir(parents=True, exist_ok=True)

    @property
    def state_path(self) -> Path:
        return self.out / "tuning_state.json"

    def state(self) -> dict:
        return load_json(self.state_path) if self.state_path.exists() else {}

    def save(self, **kw) -> None:
        s = self.state()
        s.update(kw)
        save_json(s, self.state_path)


# ----------------------------------------------------------------------------- training stages
def stage_vanilla(ctx: Ctx) -> None:
    run_grid(GRIDS / "table1_seg.yaml", ctx.out / "repro_fix", ctx.root, seeds=[42], only=["vanilla"],
             smoke=ctx.smoke)


def stage_cls(ctx: Ctx) -> None:
    run_grid(GRIDS / "tune_cls.yaml", ctx.out, ctx.root, seeds=ctx.seeds, smoke=ctx.smoke,
             max_configs=ctx.max_configs)


def stage_seg(ctx: Ctx) -> None:
    run_grid(GRIDS / "tune_seg.yaml", ctx.out, ctx.root, seeds=ctx.seeds, smoke=ctx.smoke,
             max_configs=ctx.max_configs)


def _train_seeds(ctx: Ctx, task: str, cfg: dict, scenario: str, name: str) -> None:
    from .train_cls import train_cls
    from .train_seg import train_seg
    train = train_seg if task == "seg" else train_cls
    for seed in ctx.seeds:
        c = copy.deepcopy(cfg)
        c["name"], c["data"]["scenario"], c["train"]["seed"] = name, scenario, seed
        c.setdefault("eval", {})["roles"] = ["val", "test"]
        rd = run_dir_path(ctx.out, task, scenario, name, seed)
        if (rd / "metrics_test.json").exists():
            print(f"skip {task}/{scenario}/{name}/seed{seed} (finished)", flush=True)
            continue
        print(f"\n[deploy] {task} {name} seed{seed}", flush=True)
        train(c, ctx.root, ctx.out, smoke=ctx.smoke)


def stage_deploy(ctx: Ctx) -> None:
    seg_best = best_runs(ctx.out / "seg" / "A", "segmentation.dice")
    cls_best = best_runs(ctx.out / "cls" / "A", "classification.auc")
    seg_cfg = load_config(Path(seg_best[0]) / "config.yaml")
    _train_seeds(ctx, "seg", seg_cfg, "deploy", seg_cfg["name"])
    cls_cfg = load_config(Path(cls_best[0]) / "config.yaml")
    _train_seeds(ctx, "cls", cls_cfg, "deploy", cls_cfg["name"])
    photo = copy.deepcopy(cls_cfg)
    photo.setdefault("augment", {})["photometric"] = dict(PHOTO)
    _train_seeds(ctx, "cls", photo, "deploy", cls_cfg["name"] + "_photo")


# ----------------------------------------------------------------------------- selection
def _run(ctx: Ctx, task: str, scenario: str, name: str, seed: int = 42) -> str:
    rd = run_dir_path(ctx.out, task, scenario, name, seed)
    if not (rd / "best.pt").exists():
        raise FileNotFoundError(f"{rd} has no best.pt - run the '{task}' stage first")
    return str(rd)


def candidates(ctx: Ctx) -> dict:
    a_seg = best_runs(ctx.out / "seg" / "A", "segmentation.dice", verbose=False)
    a_cls = best_runs(ctx.out / "cls" / "A", "classification.auc", verbose=False)
    d_seg = best_runs(ctx.out / "seg" / "deploy", "segmentation.dice", verbose=False)
    d_cls = best_runs(ctx.out / "cls" / "deploy", "classification.auc", verbose=False)
    return {
        "baseline": {"seg": [_run(ctx, "seg", "A", BASE_SEG)], "cls": [_run(ctx, "cls", "A", BASE_CLS)],
                     "tta": False},
        "tunedA_single": {"seg": a_seg[:1], "cls": a_cls[:1], "tta": False},
        "tunedA_ensemble": {"seg": a_seg, "cls": a_cls, "tta": False},
        "deploy_ensemble": {"seg": d_seg, "cls": d_cls, "tta": False},
        "deploy_ensemble_tta": {"seg": d_seg, "cls": d_cls, "tta": True},
    }


def _eval(ctx: Ctx, system: dict, scenario: str, role: str, name: str, post="none", threshold=None,
          include_normal=False) -> dict:
    path = ctx.out / "pipeline" / scenario / name / f"metrics_{role}.json"
    if path.exists():
        print(f"skip pipeline {scenario}/{name}/{role} (finished)", flush=True)
        return load_json(path)
    res = evaluate_pipeline(system["seg"], system["cls"], ctx.root, scenario, role, ctx.out, name,
                            threshold=threshold, post=post, smoke=ctx.smoke, tta=system["tta"],
                            include_normal=include_normal)
    return res[role]


def _num(x) -> float:
    return -math.inf if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def stage_select(ctx: Ctx) -> None:
    cands = candidates(ctx)
    scores = {}
    for key, system in cands.items():
        r = _eval(ctx, system, "deploy", "val", f"cand_{key}")
        scores[key] = {"auc": (r.get("classification") or {}).get("auc"),
                       "dice": (r.get("segmentation") or {}).get("dice")}
    print(pd.DataFrame(scores).T.to_string())
    chosen = max(cands, key=lambda k: _num(scores[k]["auc"]))  # ties: first (simplest) candidate wins
    final = dict(cands[chosen])
    # mask post-processing: keep only the largest region if that helps Dice without hurting AUC
    r = _eval(ctx, final, "deploy", "val", f"cand_{chosen}_largest", post="largest")
    d_l, a_l = _num((r.get("segmentation") or {}).get("dice")), _num((r.get("classification") or {}).get("auc"))
    post = "largest" if d_l > _num(scores[chosen]["dice"]) and a_l >= _num(scores[chosen]["auc"]) - 0.005 \
        else "none"
    val_dir = ctx.out / "pipeline" / "deploy" / (f"cand_{chosen}_largest" if post == "largest" else f"cand_{chosen}")
    thr = pick(val_dir / "per_image_val.csv", [TARGET_SENSITIVITY, 0.95])
    threshold = thr["thresholds"][f"{TARGET_SENSITIVITY:.2f}"]["threshold"]
    final.update(post=post, threshold=threshold, candidate=chosen)
    baseline = dict(cands["baseline"], post="none", threshold=0.5, candidate="baseline")
    print(f"\nchosen: {chosen} | post={post} | threshold={threshold:.4f} "
          f"(validation sensitivity target {TARGET_SENSITIVITY:.0%})")
    ctx.save(candidates_val=scores, chosen=chosen, final=final, baseline=baseline, thresholds=thr,
             post_check={"dice_largest": d_l, "auc_largest": a_l})


# ----------------------------------------------------------------------------- final evaluation
def stage_final(ctx: Ctx) -> None:
    st = ctx.state()
    if "final" not in st:
        raise RuntimeError("run the 'select' stage first")
    for sysname in ("baseline", "final"):
        system = st[sysname]
        for scenario, role, normal in FINAL_EVALS:
            _eval(ctx, system, scenario, role, sysname, post=system["post"], threshold=system["threshold"],
                  include_normal=normal)
    stats = {}
    for scenario, role, _ in FINAL_EVALS:
        fa = ctx.out / "pipeline" / scenario / "final" / f"per_image_{role}.csv"
        fb = ctx.out / "pipeline" / scenario / "baseline" / f"per_image_{role}.csv"
        res = stats_compare(pd.read_csv(fa), pd.read_csv(fb), threshold=st["final"]["threshold"],
                            threshold_b=st["baseline"]["threshold"])
        stats[f"{scenario}/{role}"] = {"a": "final", "b": "baseline", **res}
    save_json(stats, ctx.out / "stats_final_vs_baseline.json")
    print("statistics written to", ctx.out / "stats_final_vs_baseline.json")


def stage_export(ctx: Ctx) -> None:
    st = ctx.state()
    f = st["final"]
    metrics = [ctx.out / "pipeline" / s / "final" / f"metrics_{r}.json" for s, r, _ in FINAL_EVALS]
    export_bundle(f["seg"], f["cls"], ctx.out / "deploy_bundle", f["threshold"], f["post"], f["tta"],
                  metrics=[m for m in metrics if m.exists()],
                  notes=f"candidate={f['candidate']}; threshold for {TARGET_SENSITIVITY:.0%} validation sensitivity")


# ----------------------------------------------------------------------------- report
def _rank_with_test(root, metric) -> pd.DataFrame:
    try:
        v, t = _scan(root, metric, "val"), _scan(root, metric, "test")
    except FileNotFoundError:
        return pd.DataFrame()
    g = v.groupby("name")["value"]
    out = pd.DataFrame({"val_mean": g.mean(), "val_std": g.std(ddof=0), "seeds": g.size(),
                        "test_mean": t.groupby("name")["value"].mean(),
                        "test_std": t.groupby("name")["value"].std(ddof=0)})
    return out.sort_values("val_mean", ascending=False).reset_index()


def _context_summary(cls_rank: pd.DataFrame) -> pd.DataFrame:
    if cls_rank.empty:
        return cls_rank
    df = cls_rank.copy()
    parts = df["name"].str.rsplit("_", n=2, expand=True)
    if parts.shape[1] < 3:
        return pd.DataFrame()
    df["backbone"], df["context"], df["finetune"] = parts[0], parts[1], parts[2]
    return df.pivot_table(index="context", columns="finetune", values="val_mean", aggfunc="mean").reset_index()


def stage_report(ctx: Ctx) -> None:
    st = ctx.state()
    md = ["# Tuning report\n",
          "Selection used validation data only; test and BUS-BRA were evaluated once at the end.\n"]

    fix = ctx.out / "repro_fix" / "seg" / "A"
    rows = [load_json(p) for p in fix.rglob("metrics_test.json")] if fix.exists() else []
    if rows:
        md += ["\n## 1. Plain U-Net, fixed (reproduction Table 1, Scenario A test)\n",
               _md_table(pd.DataFrame([{"name": r["name"], "dice": r["segmentation"]["dice"],
                                        "iou": r["segmentation"]["iou"],
                                        "dice_pooled": r["segmentation"]["dice_pooled"]} for r in rows]))]

    cls_rank = _rank_with_test(ctx.out / "cls" / "A", "classification.auc")
    md += ["\n## 2. Classifier grid (Scenario A, ground-truth crops): AUC, mean over seeds\n",
           "Ranked by validation AUC. Context m0.1 = paper-style crop, m0.5 / m1.0 = box x2 / x3, "
           "full = whole image. k10 = last 10 tensors trainable (paper), kall = full fine-tuning.\n",
           _md_table(cls_rank),
           "\nMean validation AUC by crop context and fine-tuning depth (averaged over backbones):\n",
           _md_table(_context_summary(cls_rank))]
    md += ["\n## 3. Segmentation grid (Scenario A): Dice, mean over seeds\n",
           _md_table(_rank_with_test(ctx.out / "seg" / "A", "segmentation.dice"))]
    md += ["\n## 4. Retrained on BUSI + Breast-Lesions-USG + BUS-UCLM (deploy)\n",
           _md_table(_rank_with_test(ctx.out / "cls" / "deploy", "classification.auc")),
           _md_table(_rank_with_test(ctx.out / "seg" / "deploy", "segmentation.dice"))]

    cand_dir = ctx.out / "pipeline" / "deploy"
    entries = [(p.parent.name.replace("cand_", ""), p) for p in sorted(cand_dir.glob("cand_*/metrics_val.json"))]
    if entries:
        md += ["\n## 5. Candidate systems on the common validation set (threshold 0.5)\n",
               _md_table(compare(entries))]
    if "final" in st:
        f = st["final"]
        md += [f"\n**Chosen:** `{st['chosen']}` | mask post-processing `{f['post']}` | test-time augmentation "
               f"`{f['tta']}` | decision threshold `{f['threshold']:.4f}` "
               f"(highest threshold with >= {TARGET_SENSITIVITY:.0%} sensitivity on validation)\n"]
        for scenario, role, _ in FINAL_EVALS:
            e = [(s, ctx.out / "pipeline" / scenario / s / f"metrics_{role}.json") for s in ("baseline", "final")]
            e = [x for x in e if x[1].exists()]
            if e:
                title = {"deploy/test": "BUSI + Breast-Lesions-USG + BUS-UCLM test (+ normal scans)",
                         "A/test": "BUSI test (Scenario A, as in the paper)",
                         "A/external": "BUS-BRA external test (other hospital, other machines)"}[f"{scenario}/{role}"]
                md += [f"\n## 6. Final vs baseline: {title}\n", _md_table(compare(e))]
        sp = ctx.out / "stats_final_vs_baseline.json"
        if sp.exists():
            md += ["\n## 7. Paired statistics, final vs baseline\n"]
            rows = []
            for k, v in load_json(sp).items():
                c = v.get("classification", {})
                rows.append({"test set": k, "n": v.get("n_paired"),
                             "dice final": v.get("dice", {}).get("mean_a"),
                             "dice baseline": v.get("dice", {}).get("mean_b"),
                             "p Wilcoxon (dice)": v.get("dice", {}).get("p_wilcoxon"),
                             "auc final": c.get("auc_a"), "auc baseline": c.get("auc_b"),
                             "AUC diff 95% CI": str([round(x, 3) for x in c.get("auc_diff_ci95", [])]),
                             "p McNemar": c.get("p_mcnemar")})
            md += [_md_table(pd.DataFrame(rows))]
    rep = ctx.out / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "tuning_report.md").write_text("\n".join(md), encoding="utf-8")
    collect(ctx.out, rep)
    print(f"wrote {rep / 'tuning_report.md'}")


RUNNERS = {"vanilla": stage_vanilla, "cls": stage_cls, "seg": stage_seg, "deploy": stage_deploy,
           "select": stage_select, "final": stage_final, "export": stage_export, "report": stage_report}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stages", nargs="+", choices=STAGES + ["all"])
    p.add_argument("--out", default="tuning")
    p.add_argument("--data-root", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--smoke", action="store_true", help="tiny data, 1 epoch, 2 configs per grid, 2 seeds")
    a = p.parse_args()
    ctx = Ctx(a.out, a.data_root, a.smoke, a.seeds)
    stages = STAGES if "all" in a.stages else a.stages
    for s in stages:
        print(f"\n{'=' * 20} stage: {s} {'=' * 20}", flush=True)
        RUNNERS[s](ctx)


if __name__ == "__main__":
    main()
