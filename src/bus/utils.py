"""Small utilities shared by all trainers."""
from __future__ import annotations

import json
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_dir_path(out_root, task, scenario, name, seed) -> Path:
    """runs/<task>/<scenario>/<name>/seed<seed>"""
    return Path(out_root) / task / str(scenario) / str(name) / f"seed{seed}"


def make_run_dir(out_root, task, scenario, name, seed) -> Path:
    d = run_dir_path(out_root, task, scenario, name, seed)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _to_builtin(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def save_json(obj, path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=_to_builtin), encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_yaml(obj, path) -> None:
    Path(path).write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")


class EarlyStopping:
    """Tracks the best value of a monitored metric; `stop` turns True after `patience` bad epochs."""

    def __init__(self, patience, mode: str = "max"):
        self.patience, self.mode = patience, mode
        self.best, self.bad = None, 0

    def step(self, value: float) -> bool:
        better = self.best is None or (value > self.best if self.mode == "max" else value < self.best)
        if better:
            self.best, self.bad = value, 0
        else:
            self.bad += 1
        return better

    @property
    def stop(self) -> bool:
        return self.patience is not None and self.bad >= self.patience


class History:
    """Appends one row per epoch to a CSV and prints it."""

    def __init__(self, path):
        self.path, self.rows = Path(path), []

    def log(self, **row) -> None:
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.path, index=False)
        print("  " + " | ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in row.items()), flush=True)


class Amp:
    """Mixed precision on CUDA, a no-op on CPU."""

    def __init__(self, device: torch.device, enabled: bool = True):
        self.enabled = bool(enabled) and device.type == "cuda"
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.enabled)
        except (AttributeError, TypeError):  # older torch
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.enabled)

    def autocast(self):
        return torch.autocast(device_type="cuda", dtype=torch.float16) if self.enabled else nullcontext()

    def backward_step(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer) -> None:
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()


def balanced_sampler(labels) -> WeightedRandomSampler:
    """Oversampling: every class is drawn equally often (paper: 'oversampling to equalize classes')."""
    labels = np.asarray(labels)
    counts = {c: int((labels == c).sum()) for c in np.unique(labels)}
    weights = np.array([1.0 / counts[c] for c in labels], dtype=np.float64)
    return WeightedRandomSampler(torch.as_tensor(weights), num_samples=len(labels), replacement=True)


def make_loader(dataset, batch_size: int, train: bool, num_workers: int = 2,
                sampler=None, collate_fn=None) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train and sampler is None,
        sampler=sampler if train else None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train and len(dataset) > batch_size,
        collate_fn=collate_fn,
    )
