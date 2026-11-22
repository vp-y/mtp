"""Build leakage-free train/val/test splits for all breast-ultrasound datasets.

Writes into OUT_DIR:
  manifest.csv              one row per image, with label, mask paths, group and split
  busi_duplicate_groups.csv BUSI near-duplicate clusters (size > 1)

Split policy
  BUSI                50/15/35 train/val/test (as in the paper), grouped by near-duplicate
                      cluster because BUSI has no patient IDs. Clusters whose images carry
                      conflicting labels are excluded.
  Breast-Lesions-USG  50/15/35, one image per patient -> patient-level.
  BUS-UCLM            70/15/15 by patient (38 patients).
  BUS-BRA             100% external_test: never used for training or model selection.

Usage:  python make_splits.py RAW_DATASET_DIR OUT_DIR
"""
from __future__ import annotations

from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

import sys

SEED = 42
DS = Path(sys.argv[1]).resolve()    # extracted raw downloads
HERE = Path(sys.argv[2]).resolve()  # output folder
HERE.mkdir(parents=True, exist_ok=True)

# BUSI near-duplicate rule (chosen by visual inspection of pairs).
PHASH_STRICT = 10                 # link if pHash distance <= 10
PHASH_LOOSE, DHASH_LOOSE = 14, 15  # or pHash <= 14 and dHash <= 15


def rel(p: Path) -> str:
    return p.relative_to(DS).as_posix()


def _one_split(df, label_col, group_col, ratios, seed, n_folds=20):
    """Cut into n_folds stratified group folds, then hand whole folds to
    train/val/test according to `ratios` (in units of 1/n_folds)."""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.empty(len(df), dtype=int)
    for k, (_, idx) in enumerate(sgkf.split(df, df[label_col], df[group_col])):
        fold[idx] = k
    n_tr, n_va, _ = ratios
    return np.where(fold < n_tr, "train", np.where(fold < n_tr + n_va, "val", "test"))


def assign_splits(df, label_col, group_col, ratios, n_trials=300):
    """Group-aware split. Tries `n_trials` seeds (deterministic, starting at SEED) and
    keeps the one whose per-split class mix and split sizes best match the targets.
    Every split must contain every class that has >= 3 groups."""
    import warnings
    warnings.filterwarnings("ignore", message="The least populated class")
    overall = df[label_col].value_counts(normalize=True)
    target = np.array(ratios) / sum(ratios)
    must_have = [c for c in overall.index if df[df[label_col] == c][group_col].nunique() >= 3]
    best = None
    for s in range(SEED, SEED + n_trials):
        sp = _one_split(df, label_col, group_col, ratios, seed=s)
        names = ("train", "val", "test")
        mix = [df[sp == k][label_col].value_counts(normalize=True).reindex(overall.index, fill_value=0)
               for k in names]
        if any(m[c] == 0 for m in mix for c in must_have):
            continue
        size = np.array([(sp == k).mean() for k in names])
        err = sum(((m - overall) ** 2).sum() for m in mix) + ((size - target) ** 2).sum()
        if best is None or err < best[0]:
            best = (err, sp)
    return best[1]


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


# --------------------------------------------------------------------- BUSI
def build_busi():
    root = DS / "BUSI" / "Dataset_BUSI_with_GT"
    rows, ph, dh = [], [], []
    for cls in ["benign", "malignant", "normal"]:
        for img in sorted((root / cls).glob("*.png")):
            if "_mask" in img.name:
                continue
            masks = sorted(root.joinpath(cls).glob(f"{img.stem}_mask*.png"))
            g = Image.open(img).convert("L")
            ph.append(imagehash.phash(g).hash.flatten())
            dh.append(imagehash.dhash(g).hash.flatten())
            rows.append(dict(dataset="BUSI", image_path=rel(img),
                             mask_paths=";".join(rel(m) for m in masks),
                             label=cls, patient_id=pd.NA,
                             notes="multi-mask" if len(masks) > 1 else ""))
    df = pd.DataFrame(rows)
    ph, dh = np.array(ph), np.array(dh)
    Dp = (ph[:, None] != ph[None]).sum(-1)
    Dd = (dh[:, None] != dh[None]).sum(-1)
    link = (Dp <= PHASH_STRICT) | ((Dp <= PHASH_LOOSE) & (Dd <= DHASH_LOOSE))
    uf = UnionFind(len(df))
    for a, b in zip(*np.where(np.triu(link, 1))):
        uf.union(a, b)
    roots = [uf.find(i) for i in range(len(df))]
    gid = {r: f"BUSI_g{k:04d}" for k, r in enumerate(dict.fromkeys(roots))}
    df["group_id"] = [gid[r] for r in roots]

    n_labels = df.groupby("group_id").label.nunique()
    conflict = set(n_labels[n_labels > 1].index)
    df["split"], df["exclude_reason"] = "", ""
    bad = df.group_id.isin(conflict)
    df.loc[bad, ["split", "exclude_reason"]] = ["excluded", "duplicate-label-conflict"]
    ok = ~bad
    df.loc[ok, "split"] = assign_splits(df[ok], "label", "group_id", (10, 3, 7))

    sizes = df.group_id.map(df.group_id.value_counts())
    dup = df[sizes > 1].sort_values(["group_id", "image_path"])
    dup[["group_id", "image_path", "label", "split", "exclude_reason"]].to_csv(
        HERE / "busi_duplicate_groups.csv", index=False)
    return df


# ------------------------------------------------------- Breast-Lesions-USG
def build_breast_lesions_usg():
    base = DS / "Breast-Lesions-USG"
    folder = base / "BrEaST-Lesions_USG-images_and_masks"
    meta = pd.read_excel(next(base.glob("*.xlsx")))
    rows = []
    for _, r in meta.iterrows():
        masks = []
        for col in ["Mask_tumor_filename", "Mask_other_filename"]:
            v = r.get(col)
            if isinstance(v, str) and v.strip():
                masks += [folder / s.strip() for s in v.replace(";", "&").split("&") if s.strip()]
        tumor = [m for m in masks if "_tumor" in m.name]
        rows.append(dict(dataset="Breast-Lesions-USG",
                         image_path=rel(folder / r.Image_filename),
                         mask_paths=";".join(rel(m) for m in tumor),
                         label=str(r.Classification).strip().lower(),
                         patient_id=f"BLU_{r.CaseID}",
                         notes="has-other-masks" if len(masks) > len(tumor) else ""))
    df = pd.DataFrame(rows)
    df["group_id"] = df.patient_id
    df["exclude_reason"] = ""
    df["split"] = assign_splits(df, "label", "group_id", (10, 3, 7))
    return df


# ----------------------------------------------------------------- BUS-UCLM
def build_bus_uclm():
    base = DS / "BUS-UCLM"
    info = pd.read_csv(base / "INFO.csv", sep=";", encoding="utf-8-sig")
    rows = []
    for _, r in info.iterrows():
        notes = []
        if str(r.Doppler).strip().upper().startswith("Y"):
            notes.append("doppler")
        if str(r.Marks).strip().upper().startswith("Y"):
            notes.append("marks")
        if str(r.Combined).strip().upper().startswith("Y"):
            notes.append("combined")
        notes.append("rgb-mask")
        rows.append(dict(dataset="BUS-UCLM", image_path=rel(base / "images" / r.Image),
                         mask_paths=rel(base / "masks" / r.Image),
                         label=r.Label.strip().lower(),
                         patient_id=f"UCLM_{r.Image.split('_')[0]}",
                         notes=";".join(notes)))
    df = pd.DataFrame(rows)
    df["group_id"] = df.patient_id
    df["exclude_reason"] = ""
    df["split"] = assign_splits(df, "label", "group_id", (14, 3, 3))
    return df


# ------------------------------------------------------------------ BUS-BRA
def build_bus_bra():
    base = DS / "BUS-BRA" / "BUSBRA"
    d = pd.read_csv(base / "bus_data.csv")
    df = pd.DataFrame(dict(
        dataset="BUS-BRA",
        image_path=[rel(base / "Images" / f"{i}.png") for i in d.ID],
        mask_paths=[rel(base / "Masks" / f"{i.replace('bus_', 'mask_')}.png") for i in d.ID],
        label=d.Pathology.str.lower(),
        patient_id=[f"BRA_{c:04d}" for c in d.Case],
        notes=[f"BIRADS={b};device={v}" for b, v in zip(d.BIRADS, d.Device)],
    ))
    df["group_id"] = df.patient_id
    df["exclude_reason"] = ""
    df["split"] = "external_test"
    return df


def main():
    df = pd.concat([build_busi(), build_breast_lesions_usg(), build_bus_uclm(), build_bus_bra()],
                   ignore_index=True)
    cols = ["dataset", "image_path", "mask_paths", "label", "patient_id", "group_id",
            "split", "exclude_reason", "notes"]
    df = df[cols]

    # Integrity checks
    for p in df.image_path:
        assert (DS / p).exists(), p
    for ms in df.mask_paths.dropna():
        for m in filter(None, ms.split(";")):
            assert (DS / m).exists(), m
    leak = df[df.split.isin(["train", "val", "test"])].groupby("group_id").split.nunique()
    assert (leak == 1).all(), f"groups spanning splits: {list(leak[leak > 1].index)}"

    df.to_csv(HERE / "manifest.csv", index=False)

    order = ["train", "val", "test", "external_test", "excluded"]
    tab = pd.crosstab([df.dataset, df.split], df.label, margins=True)
    print(tab.reindex(sorted(tab.index, key=lambda t: (t[0], order.index(t[1]) if t[1] in order else 9))))
    print("\nGroups/patients per split:")
    print(df.groupby(["dataset", "split"]).group_id.nunique().unstack(fill_value=0))
    b = df[df.dataset == "BUSI"]
    gs = b.group_id.value_counts()
    print(f"\nBUSI: {len(gs)} groups, {int((gs > 1).sum())} with >1 image "
          f"({int(gs[gs > 1].sum())} images), largest={int(gs.max())}, "
          f"excluded={int((b.split == 'excluded').sum())}")


if __name__ == "__main__":
    main()
