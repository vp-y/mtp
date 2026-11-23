"""Run a grid of experiments described in a YAML file. Finished runs are skipped, so an interrupted
grid (e.g. a Kaggle session timeout) can simply be started again.

  python -m bus.grid configs/grids/table1_seg.yaml --out runs
  python -m bus.grid configs/grids/table2_cls.yaml --out runs --only mobilenet efficientnet
  python -m bus.grid configs/grids/table1_seg.yaml --out runs --seeds 42 43 44
  python -m bus.grid configs/grids/tune_cls.yaml --out tuning --set data.scenario=deploy
"""
from __future__ import annotations

import argparse
import copy
import itertools
import traceback
from pathlib import Path

import yaml

from .config import apply_overrides, load_config, resolve_data_root, set_nested
from .utils import run_dir_path

REPO = Path(__file__).resolve().parents[2]


def expand(grid: dict) -> list:
    """Turn the grid into run specs {name, set}.

    product:  {key: [values]}               cartesian product of single settings (+ name_template)
    axes:     [[{name, set}, ...], ...]      cartesian product of NAMED variants; each variant may set
                                             several keys; names are joined with '_'
    runs:     [{name, set}, ...]             explicit list
    """
    runs = []
    axes = grid.get("axes")
    if axes:
        for combo in itertools.product(*axes):
            settings = {}
            for variant in combo:
                settings.update(variant.get("set") or {})
            runs.append({"name": "_".join(str(v["name"]) for v in combo), "set": settings})
    prod = grid.get("product")
    if prod:
        keys = list(prod)
        template = grid.get("name_template")
        for combo in itertools.product(*(prod[k] for k in keys)):
            settings = dict(zip(keys, combo))
            if template:
                name = template.format(**{k.replace(".", "_"): v for k, v in settings.items()})
            else:
                name = "_".join(str(v) for v in combo)
            runs.append({"name": name, "set": settings})
    runs.extend(grid.get("runs", []) or [])
    return runs


def _trainer(task: str):
    if task == "seg":
        from .train_seg import train_seg
        return train_seg
    if task == "cls":
        from .train_cls import train_cls
        return train_cls
    raise ValueError(f"grid task must be 'seg' or 'cls', got {task!r}")


def run_grid(grid_path, out="runs", data_root=None, overrides=(), seeds=None, only=None, force=False,
             smoke=False, max_configs=None) -> dict:
    """Run every configuration x seed of a grid file. Returns {'done', 'skipped', 'failed'}.

    max_configs: run only the first N configurations (used by --smoke checks of long grids).
    """
    grid_path = Path(grid_path)
    if not grid_path.is_absolute() and not grid_path.exists():
        grid_path = REPO / grid_path
    grid = yaml.safe_load(grid_path.read_text(encoding="utf-8"))
    task = grid["task"]
    train = _trainer(task)
    base_path = Path(grid["base"])
    if not base_path.is_absolute() and not base_path.exists():
        base_path = REPO / base_path
    base = load_config(base_path)
    for k, v in (grid.get("set") or {}).items():
        set_nested(base, k, v)
    root = resolve_data_root(data_root)
    seeds = list(seeds or grid.get("seeds") or [42])

    runs = expand(grid)
    if only:
        runs = [r for r in runs if any(s in r["name"] for s in only)]
    if max_configs:
        runs = runs[:int(max_configs)]
    print(f"{len(runs)} configurations x {len(seeds)} seed(s) from {grid_path}", flush=True)

    done, skipped, failed = 0, 0, []
    for i, run in enumerate(runs, 1):
        for seed in seeds:
            cfg = copy.deepcopy(base)
            for k, v in (run.get("set") or {}).items():
                set_nested(cfg, k, v)
            apply_overrides(cfg, overrides)
            cfg["name"] = run["name"]
            cfg.setdefault("train", {})["seed"] = seed
            rd = run_dir_path(out, task, cfg["data"]["scenario"], cfg["name"], seed)
            if (rd / "metrics_test.json").exists() and not force:
                skipped += 1
                print(f"[{i}/{len(runs)}] skip {run['name']} seed{seed} (finished)", flush=True)
                continue
            print(f"\n[{i}/{len(runs)}] {run['name']} seed{seed}", flush=True)
            try:
                train(cfg, root, out, smoke=smoke)
                done += 1
            except Exception:  # keep the grid going; report at the end
                traceback.print_exc()
                failed.append(f"{run['name']}/seed{seed}")
            finally:
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
    print(f"\ngrid finished: {done} trained, {skipped} skipped, {len(failed)} failed")
    for f in failed:
        print(f"  FAILED {f}")
    return {"done": done, "skipped": skipped, "failed": failed}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("grid", help="grid YAML file")
    p.add_argument("--data-root", default=None)
    p.add_argument("--out", default="runs")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="overrides applied to every run")
    p.add_argument("--seeds", nargs="*", type=int, default=None, help="default: grid 'seeds' or [42]")
    p.add_argument("--only", nargs="*", default=None, help="run only names containing one of these strings")
    p.add_argument("--force", action="store_true", help="re-run finished runs")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    run_grid(a.grid, a.out, a.data_root, a.set, a.seeds, a.only, a.force, a.smoke)


if __name__ == "__main__":
    main()
