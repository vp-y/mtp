"""Save predicted lesion masks (<id>.png) so the classifier can be trained/evaluated on predicted crops.

  python -m bus.predict_masks --seg runs/seg/A/<name>/seed42 --scenario A \
      --roles train val test external --out runs/pred_masks/A
  python -m bus.train_cls --config configs/cls/base.yaml --set data.mask_dir=runs/pred_masks/A
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import resolve_data_root
from .data.datasets import read_gray
from .data.manifest import load_manifest, select, smoke_subset
from .evaluation import save_mask_png
from .inference import SegPredictor


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seg", required=True, help="segmentation checkpoint or run dir")
    p.add_argument("--scenario", default="A")
    p.add_argument("--roles", nargs="+", default=["train", "val", "test"])
    p.add_argument("--out", required=True)
    p.add_argument("--data-root", default=None)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()

    root = resolve_data_root(a.data_root)
    pred = SegPredictor(a.seg)
    manifest = load_manifest(root)
    out = Path(a.out)
    n_empty = n = 0
    for role in a.roles:
        rows = select(manifest, a.scenario, role)
        if a.smoke:
            rows = smoke_subset(rows, 8)
        for r in rows.itertuples(index=False):
            m = pred.predict_mask(read_gray(root / r.image_path))
            save_mask_png(m, out / f"{r.id}.png")
            n += 1
            n_empty += int(not m.any())
    print(f"wrote {n} masks to {out} ({n_empty} empty)")


if __name__ == "__main__":
    main()
