"""Deployment bundle: the chosen models + the decision settings, in one folder the app can load.

  python -m bus.bundle export --seg RUN1 RUN2 --cls RUN3 RUN4 --threshold 0.23 --post largest --tta \
      --metrics runs/pipeline/deploy/final/metrics_test.json --out deploy_bundle
  python -m bus.bundle info deploy_bundle

Layout:
  deploy_bundle/bundle.json   settings, model list, reference metrics
  deploy_bundle/seg_0.pt ...  segmentation checkpoints (float16 weights + config)
  deploy_bundle/cls_0.pt ...  classification checkpoints (float16 weights + config)
"""
from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path

import torch

from .inference import DualStagePipeline, _as_list, load_ckpt
from .utils import load_json, save_json

BUNDLE_VERSION = 1


def _versions() -> dict:
    """Library versions used at export: install the same ones where the app runs."""
    out = {"torch": torch.__version__}
    for mod, key in (("timm", "timm"), ("segmentation_models_pytorch", "segmentation-models-pytorch")):
        try:
            out[key] = __import__(mod).__version__
        except Exception:  # not installed / no version attribute
            pass
    return out


def _slim(ckpt_path, half: bool = True) -> dict:
    ck = load_ckpt(ckpt_path)
    state = {k: (v.half() if half and torch.is_tensor(v) and v.is_floating_point() else v)
             for k, v in ck["model"].items()}
    return {"kind": ck.get("kind"), "config": ck["config"], "model": state, "epoch": ck.get("epoch")}


def export_bundle(seg, cls, out_dir, threshold: float, post: str = "none", tta: bool = False,
                  empty_policy: str = "full_image", metrics=None, notes: str = "", half: bool = True) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    entry = {"version": BUNDLE_VERSION, "created": _dt.datetime.now().isoformat(timespec="seconds"),
             "threshold": float(threshold), "post": post, "tta": bool(tta), "empty_policy": empty_policy,
             "seg": [], "cls": [], "notes": notes, "versions": _versions(), "metrics": {}}
    for kind, paths in (("seg", _as_list(seg)), ("cls", _as_list(cls))):
        for i, p in enumerate(paths):
            fname = f"{kind}_{i}.pt"
            ck = _slim(p, half)
            torch.save(ck, out / fname)
            entry[kind].append({"file": fname, "name": ck["config"].get("name"),
                                "seed": ck["config"].get("train", {}).get("seed"), "source": str(p)})
    for m in metrics or []:
        r = load_json(m)
        entry["metrics"][f"{r.get('scenario')}/{r.get('role')}"] = {
            k: r.get(k) for k in ("n", "segmentation", "classification", "by_dataset", "normal_scans")
            if r.get(k) is not None}
    save_json(entry, out / "bundle.json")
    size = sum(f.stat().st_size for f in out.glob("*.pt")) / 2 ** 20
    print(f"bundle written to {out} ({len(entry['seg'])} seg + {len(entry['cls'])} cls models, {size:.0f} MB)")
    return out


def load_bundle(path, device=None):
    """-> (DualStagePipeline, settings dict). The decision threshold is settings['threshold']."""
    path = Path(path)
    cfg = load_json(path / "bundle.json")
    pipe = DualStagePipeline([path / s["file"] for s in cfg["seg"]], [path / c["file"] for c in cfg["cls"]],
                             device=device, empty_policy=cfg.get("empty_policy", "full_image"),
                             tta=cfg.get("tta", False), post=cfg.get("post"))
    return pipe, cfg


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--seg", nargs="+", required=True)
    e.add_argument("--cls", nargs="+", required=True)
    e.add_argument("--threshold", type=float, required=True)
    e.add_argument("--post", default="none", choices=["none", "largest"])
    e.add_argument("--tta", action="store_true")
    e.add_argument("--empty-policy", default="full_image", choices=["full_image", "benign"])
    e.add_argument("--metrics", nargs="*", default=[], help="metrics_*.json files to store as reference")
    e.add_argument("--notes", default="")
    e.add_argument("--full-precision", action="store_true", help="keep float32 weights")
    e.add_argument("--out", default="deploy_bundle")
    i = sub.add_parser("info")
    i.add_argument("path")
    a = p.parse_args()
    if a.cmd == "export":
        export_bundle(a.seg, a.cls, a.out, a.threshold, a.post, a.tta, a.empty_policy, a.metrics, a.notes,
                      half=not a.full_precision)
    else:
        cfg = load_json(Path(a.path) / "bundle.json")
        print({k: v for k, v in cfg.items() if k != "metrics"})
        for k, v in cfg.get("metrics", {}).items():
            c = v.get("classification", {})
            print(f"{k}: n={v.get('n')} AUC={c.get('auc')} sensitivity={c.get('recall')} "
                  f"specificity={c.get('specificity')}")


if __name__ == "__main__":
    main()
