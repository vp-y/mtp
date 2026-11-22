"""Configuration: YAML files (optional `_base_` inheritance) plus `key.sub=value` overrides."""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import yaml


def deep_update(base: dict, upd: dict) -> dict:
    """Recursively merge `upd` into `base` (in place) and return `base`."""
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def load_config(path) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = cfg.pop("_base_", None)
    if base:
        cfg = deep_update(load_config(path.parent / base), cfg)
    return cfg


def parse_value(text: str):
    """YAML-parse a CLI value; also accept `1e-4` style floats (plain YAML 1.1 reads them as str)."""
    value = yaml.safe_load(text)
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            pass
    return value


def set_nested(cfg: dict, dotted_key: str, value) -> None:
    node = cfg
    *parents, leaf = dotted_key.strip().split(".")
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = copy.deepcopy(value)


def apply_overrides(cfg: dict, overrides) -> dict:
    """Apply CLI overrides like ["train.lr=3e-4", "data.scenario=B", "eval.roles=[test]"]."""
    for item in overrides or []:
        key, sep, val = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like key.sub=value, got {item!r}")
        set_nested(cfg, key, parse_value(val))
    return cfg


def resolve_data_root(arg=None) -> Path:
    """Find the folder that contains manifest.csv.

    Order: explicit argument, $BUS_DATA_ROOT, ./dataset. If the folder itself has no
    manifest.csv it is searched recursively (Kaggle sometimes nests uploaded folders).
    """
    root = Path(arg or os.environ.get("BUS_DATA_ROOT") or "dataset")
    if (root / "manifest.csv").exists():
        return root
    if root.exists():
        hits = sorted(root.rglob("manifest.csv"))
        if hits:
            return hits[0].parent
    raise FileNotFoundError(
        f"manifest.csv not found under {root!s}. Pass --data-root or set BUS_DATA_ROOT.")


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=True, help="YAML config file")
    p.add_argument("--data-root", default=None,
                   help="folder with manifest.csv (default: $BUS_DATA_ROOT or ./dataset)")
    p.add_argument("--out", default="runs", help="output root folder")
    p.add_argument("--name", default=None, help="run name (default: config 'name')")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="config overrides, e.g. train.lr=3e-4 data.scenario=B")
    p.add_argument("--smoke", action="store_true",
                   help="tiny data subset and 1 epoch: checks that the whole code path runs")
    return p


def config_from_args(args) -> dict:
    cfg = apply_overrides(load_config(args.config), args.set)
    if args.name:
        cfg["name"] = args.name
    cfg.setdefault("name", Path(args.config).stem)
    return cfg
