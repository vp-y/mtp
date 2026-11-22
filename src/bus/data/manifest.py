"""Reading manifest.csv and selecting the rows of each experimental scenario.

Scenarios (paper section "Experimental Tuning and Optimization"):
  A       train / val / test on BUSI
  B       train / val / test on BUSI + Breast-Lesions-USG
  C       train / val on BUSI, test on ALL of Breast-Lesions-USG (cross-dataset)
  deploy  BUSI + Breast-Lesions-USG + BUS-UCLM (for the deployable system, incl. normal scans)
Role "external" is BUS-BRA (1,064 unseen patients) for every scenario.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LABEL_TO_INT = {"benign": 0, "malignant": 1}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}

BUSI, BLU, UCLM, BRA = "BUSI", "Breast-Lesions-USG", "BUS-UCLM", "BUS-BRA"

# role -> list of (dataset, split); split None = every row of that dataset
SCENARIOS = {
    "A": {"train": [(BUSI, "train")], "val": [(BUSI, "val")], "test": [(BUSI, "test")]},
    "B": {"train": [(BUSI, "train"), (BLU, "train")],
          "val": [(BUSI, "val"), (BLU, "val")],
          "test": [(BUSI, "test"), (BLU, "test")]},
    "C": {"train": [(BUSI, "train")], "val": [(BUSI, "val")], "test": [(BLU, None)]},
    "deploy": {"train": [(BUSI, "train"), (BLU, "train"), (UCLM, "train")],
               "val": [(BUSI, "val"), (BLU, "val"), (UCLM, "val")],
               "test": [(BUSI, "test"), (BLU, "test"), (UCLM, "test")]},
}
EXTERNAL = [(BRA, "external_test")]
ROLES = ("train", "val", "test", "external")


def load_manifest(root) -> pd.DataFrame:
    return pd.read_csv(Path(root) / "manifest.csv", dtype={"patient_id": str, "group_id": str})


def select(df: pd.DataFrame, scenario: str, role: str, include_normal: bool = False) -> pd.DataFrame:
    if role == "external":
        parts = EXTERNAL
    else:
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}; choose from {list(SCENARIOS)}")
        if role not in SCENARIOS[scenario]:
            raise ValueError(f"unknown role {role!r}; choose from {ROLES}")
        parts = SCENARIOS[scenario][role]
    keep = np.zeros(len(df), dtype=bool)
    for dataset, split in parts:
        m = (df["dataset"] == dataset).to_numpy()
        if split is not None:
            m &= (df["split"] == split).to_numpy()
        keep |= m
    out = df[keep]
    if not include_normal:
        out = out[out["label"] != "normal"]
    return out.reset_index(drop=True)


def smoke_subset(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """A few rows of every label (used by --smoke runs)."""
    if len(df) == 0:
        return df
    per = max(1, n // df["label"].nunique())
    return pd.concat([g.head(per) for _, g in df.groupby("label")]).reset_index(drop=True)
