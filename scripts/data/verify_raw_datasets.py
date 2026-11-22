"""Inventory + sanity checks for the downloaded breast-ultrasound datasets."""
import hashlib
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# Dataset root: optional first argument, otherwise the folder this script lives in.
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


def pixhash(p):
    a = np.asarray(Image.open(p).convert("L"))
    return hashlib.md5(a.tobytes()).hexdigest()


def sizes(paths, n=None):
    c = Counter()
    for p in paths[:n] if n else paths:
        with Image.open(p) as im:
            c[(im.size, im.mode)] += 1
    return c


print("=" * 70, "\nBUSI")
busi = ROOT / "BUSI" / "Dataset_BUSI_with_GT"
all_imgs = []
for cls in ["benign", "malignant", "normal"]:
    files = sorted((busi / cls).glob("*.png"))
    imgs = [f for f in files if "_mask" not in f.name]
    masks = [f for f in files if "_mask" in f.name]
    per_img = Counter()
    for m in masks:
        per_img[re.sub(r"_mask(_\d+)?\.png$", "", m.name)] += 1
    missing = [i.name for i in imgs if per_img[i.stem] == 0]
    multi = sum(1 for k, v in per_img.items() if v > 1)
    print(f"  {cls:9s} images={len(imgs):4d} masks={len(masks):4d} "
          f"imgs_with_>1_mask={multi} imgs_without_mask={len(missing)}")
    all_imgs += [(cls, i) for i in imgs]
print("  total images:", len(all_imgs))
print("  size/mode (top 5):", sizes([p for _, p in all_imgs]).most_common(5))
groups = defaultdict(list)
for cls, p in all_imgs:
    groups[pixhash(p)].append(f"{cls}/{p.name}")
dups = [g for g in groups.values() if len(g) > 1]
conflict = [g for g in dups if len({x.split('/')[0] for x in g}) > 1]
print(f"  EXACT pixel-duplicate groups: {len(dups)} "
      f"(images involved: {sum(len(g) for g in dups)}), cross-class conflicts: {len(conflict)}")
for g in dups[:8]:
    print("    ", g)

print("=" * 70, "\nBreast-Lesions-USG")
bl = ROOT / "Breast-Lesions-USG" / "BrEaST-Lesions_USG-images_and_masks"
files = sorted(bl.glob("*.png"))
imgs = [f for f in files if re.fullmatch(r"case\d+\.png", f.name)]
masks = [f for f in files if f not in imgs]
print(f"  images={len(imgs)} mask-files={len(masks)} mask suffixes:",
      Counter(re.sub(r"case\d+_", "", m.stem) for m in masks))
print("  size/mode (top 5):", sizes(imgs).most_common(5))
xl = next((ROOT / "Breast-Lesions-USG").glob("*.xlsx"))
df = pd.read_excel(xl)
print("  xlsx shape:", df.shape)
print("  columns:", list(df.columns))
for col in df.columns:
    if col.lower() in {"classification", "birads", "verification", "pathology"}:
        print(f"  {col}:", df[col].value_counts(dropna=False).to_dict())

print("=" * 70, "\nBUS-BRA")
bb = ROOT / "BUS-BRA" / "BUSBRA"
d = pd.read_csv(bb / "bus_data.csv")
imgs = sorted((bb / "Images").glob("*.png"))
masks = sorted((bb / "Masks").glob("*.png"))
print(f"  csv rows={len(d)} images={len(imgs)} masks={len(masks)} patients(Case)={d.Case.nunique()}")
print("  images per Pathology:", d.Pathology.value_counts().to_dict())
print("  patients per Pathology:", d.groupby("Pathology").Case.nunique().to_dict())
print("  BIRADS:", d.BIRADS.value_counts().sort_index().to_dict())
print("  Device:", d.Device.value_counts().to_dict())
print("  mask names sample:", [m.name for m in masks[:2]])
print("  size/mode (sample 200):", sizes(imgs, 200).most_common(3))

print("=" * 70, "\nBUS-UCLM")
bu = ROOT / "BUS-UCLM"
info = pd.read_csv(bu / "INFO.csv", sep=";", encoding="utf-8-sig")
imgs = sorted((bu / "images").glob("*"))
masks = sorted((bu / "masks").glob("*"))
info["patient"] = info.Image.str.split("_").str[0]
print(f"  csv rows={len(info)} images={len(imgs)} masks={len(masks)} patients={info.patient.nunique()}")
print("  Label:", info.Label.value_counts().to_dict())
print("  patients with malignant:", info[info.Label == "Malignant"].patient.nunique(),
      "| with benign:", info[info.Label == "Benign"].patient.nunique())
for c in ["Doppler", "Marks", "Combined", "Resolution"]:
    print(f"  {c}:", info[c].value_counts().to_dict())
m0 = np.asarray(Image.open(masks[0]).convert("RGB"))
print("  mask sample unique colours:", np.unique(m0.reshape(-1, 3), axis=0)[:6].tolist())
