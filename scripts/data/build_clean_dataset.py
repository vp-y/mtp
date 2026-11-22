"""Turn the four raw breast-ultrasound downloads into one clean, uniform dataset.

Input : RAW  = folder with the extracted downloads (layout as produced on 2026-11-22):
          BUSI/Dataset_BUSI_with_GT/{benign,malignant,normal}/
          Breast-Lesions-USG/BrEaST-Lesions_USG-images_and_masks/ + clinical xlsx
          BUS-UCLM/{images,masks,INFO.csv}
          BUS-BRA/BUSBRA/{Images,Masks,bus_data.csv,LICENSE.txt}
        SPLITS = manifest.csv written by make_splits.py (decides split / exclusion)
Output: OUT/
          images/<dataset>/<id>.png   8-bit grayscale, original resolution
          masks/<dataset>/<id>.png    8-bit, 0 = background, 255 = lesion (all masks of
                                      an image merged; all-zero for normal images)
          metadata/                   original clinical/label files + licence
          manifest.csv                one row per kept image

Removed: BUSI duplicate-label conflicts (19), BUS-UCLM Doppler images (37),
         BUS-UCLM HESN_002-004 whose masks do not align with the cropped images.

Usage: python build_clean_dataset.py RAW SPLITS_MANIFEST OUT
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

RAW, SPLITS, OUT = (Path(a) for a in sys.argv[1:4])
SHORT = {"BUSI": "busi", "Breast-Lesions-USG": "blu", "BUS-UCLM": "uclm", "BUS-BRA": "bra"}


def new_id(ds, image_path):
    stem = Path(image_path).stem
    if ds == "BUSI":  # "benign (12)" -> busi_benign_0012
        cls, num = stem.split(" (")
        return f"busi_{cls}_{int(num.rstrip(')')):04d}"
    if ds == "BUS-BRA":  # bus_0001-l -> bra_0001-l
        return "bra_" + stem.removeprefix("bus_")
    return f"{SHORT[ds]}_{stem}"


def load_mask(paths, size, ds):
    out = np.zeros(size[::-1], dtype=bool)
    for p in paths:
        m = Image.open(RAW / p)
        if m.size != size:
            raise ValueError(f"mask size {m.size} != image size {size}: {p}")
        a = np.asarray(m.convert("RGB")) if ds == "BUS-UCLM" else np.asarray(m.convert("L"))
        out |= (a.max(-1) > 0) if a.ndim == 3 else (a > 0)
    return out


def uclm_mask_colour(path):
    a = np.asarray(Image.open(RAW / path).convert("RGB")).reshape(-1, 3)
    red, green = ((a[:, 0] > 127) & (a[:, 1] < 128)).sum(), ((a[:, 1] > 127) & (a[:, 0] < 128)).sum()
    return "malignant" if red > green else "benign" if green > 0 else "normal"


man = pd.read_csv(SPLITS, dtype=str).fillna("")
removed = []
keep = man.split != "excluded"
removed += [(r.dataset, r.image_path, r.exclude_reason) for r in man[~keep].itertuples()]
man = man[keep]
uclm = man.dataset == "BUS-UCLM"
dop = uclm & man.notes.str.contains("doppler")
removed += [(r.dataset, r.image_path, "doppler") for r in man[dop].itertuples()]
man = man[~dop]

rows, colour_mismatch = [], []
for r in man.itertuples():
    img = Image.open(RAW / r.image_path)
    size = img.size
    masks = [m for m in r.mask_paths.split(";") if m]
    if r.dataset == "BUS-UCLM" and r.label == "normal":
        masks = []  # all-black masks; some HESN ones have a different size than the image
    try:
        mask = load_mask(masks, size, r.dataset)
    except ValueError:
        removed.append((r.dataset, r.image_path, "mask-image-misaligned"))
        continue
    if r.label != "normal" and not mask.any():
        removed.append((r.dataset, r.image_path, "empty-lesion-mask"))
        continue
    if r.dataset == "BUS-UCLM" and r.label != "normal":
        c = uclm_mask_colour(r.mask_paths)
        if c != r.label:
            colour_mismatch.append((r.image_path, r.label, c))

    iid = new_id(r.dataset, r.image_path)
    sub = SHORT[r.dataset]
    ip, mp = OUT / "images" / sub / f"{iid}.png", OUT / "masks" / sub / f"{iid}.png"
    ip.parent.mkdir(parents=True, exist_ok=True)
    mp.parent.mkdir(parents=True, exist_ok=True)
    img.convert("L").save(ip, optimize=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(mp, optimize=True)

    flags = []
    if "marks" in r.notes:
        flags.append("marks")
    if "has-other-masks" in r.notes:
        flags.append("other-abnormal-areas")
    if "multi-mask" in r.notes:
        flags.append("merged-masks")
    extra = ";".join(x for x in r.notes.split(";") if x.startswith(("BIRADS", "device")))
    rows.append(dict(id=iid, dataset=r.dataset, image_path=ip.relative_to(OUT).as_posix(),
                     mask_path=mp.relative_to(OUT).as_posix(), label=r.label,
                     patient_id=r.patient_id, group_id=r.group_id, split=r.split,
                     width=size[0], height=size[1], lesion_area_pct=round(100 * mask.mean(), 3),
                     has_marks="marks" in flags, flags=";".join(flags), extra=extra,
                     source_file=r.image_path))

out = pd.DataFrame(rows)
assert out.id.is_unique
out.to_csv(OUT / "manifest.csv", index=False)

md = OUT / "metadata"
md.mkdir(exist_ok=True)
for src, dst in [
    (next((RAW / "Breast-Lesions-USG").glob("*.xlsx")), "breast_lesions_usg_clinical.xlsx"),
    (RAW / "BUS-BRA/BUSBRA/bus_data.csv", "bus_bra_bus_data.csv"),
    (RAW / "BUS-BRA/BUSBRA/LICENSE.txt", "bus_bra_LICENSE.txt"),
    (RAW / "BUS-UCLM/INFO.csv", "bus_uclm_INFO.csv"),
    (SPLITS.parent / "busi_duplicate_groups.csv", "busi_duplicate_groups.csv"),
]:
    shutil.copy2(src, md / dst)
pd.DataFrame(removed, columns=["dataset", "source_file", "reason"]).to_csv(md / "removed_images.csv", index=False)

print(pd.crosstab([out.dataset, out.split], out.label, margins=True))
print("\nremoved:", pd.Series([x[2] for x in removed]).value_counts().to_dict())
print("UCLM mask-colour vs label mismatches:", len(colour_mismatch), colour_mismatch[:5])
print("images with marks:", int(out.has_marks.sum()))
