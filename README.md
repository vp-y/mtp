# Dual-Stage Breast Ultrasound Lesion Segmentation & Classification

This repository reproduces and extends
**Bruno, Macrì & Dodaro, "A Dual-stage Deep Learning Framework for Breast Ultrasound Image Segmentation and Classification", *J. Med. Syst.* 49:162 (2025)**.

```
Ultrasound image ─► 1. Segmentation ─► 2. Mask → ROI box ─► 3. Crop ─► 4. Classification ─► Benign / Malignant
                     DeepLabV3+/ResNet34                                  MobileNetV3-Small     + probability
```

**Workflow:** write code on your PC → push to **GitHub** → train on **Kaggle** (free GPU) by cloning the repo.
Nothing is trained on your own computer.

**Status:** the code for the full paper reproduction is written (Tables 1–3, both baselines, statistics,
external test) and is run on Kaggle with `notebooks/kaggle_reproduce.ipynb`.
Next phases: hyper-parameter tuning, then deployment (API + web UI).

---

## Contents
1. [Repository layout](#1-repository-layout)
2. [Step 1: Push the code to GitHub](#2-step-1-push-the-code-to-github)
3. [Step 2: Upload the dataset to Kaggle (once)](#3-step-2-upload-the-dataset-to-kaggle-once)
4. [Step 3: Create the Kaggle notebook](#4-step-3-create-the-kaggle-notebook)
5. [Step 4: Run the experiments](#5-step-4-run-the-experiments)
6. [Step 5: Download and read the results](#6-step-5-download-and-read-the-results)
7. [Updating the code later](#7-updating-the-code-later)
8. [Command reference](#8-command-reference)
9. [Troubleshooting](#9-troubleshooting)
10. [Data and citations](#10-data-and-citations)

---

## 1. Repository layout

```
├── configs/
│   ├── seg/base.yaml              stage-1 settings (default = paper's best: DeepLabV3+ / ResNet34 / 64)
│   ├── cls/base.yaml              stage-2 settings (default = paper's best: MobileNetV3-Small / MLP / K=10 / dropout 0.5)
│   ├── baselines/                 Mask R-CNN and multi-task Swin
│   └── grids/                     Table 1 (13 runs) and Table 2 (56 runs)
├── src/bus/
│   ├── data/                      manifest + scenarios, preprocessing, augmentation, datasets
│   ├── models/                    segmentation and classification networks
│   ├── roi.py                     mask clean-up and ROI cropping
│   ├── losses.py, metrics.py      paper loss; Dice/IoU (per-image + pooled), AUC/F1/...
│   ├── train_seg.py               stage 1 training + evaluation
│   ├── train_cls.py               stage 2 training + evaluation
│   ├── inference.py               DualStagePipeline (also used later by the deployed app)
│   ├── evaluate.py                full pipeline evaluation (Table 3)
│   ├── baselines/                 maskrcnn.py, multitask_swin.py
│   ├── grid.py                    runs a grid of experiments (resumable)
│   ├── reporting.py               picks best runs (on validation), builds result tables
│   └── stats.py                   Wilcoxon / McNemar / bootstrap AUC comparison
├── notebooks/kaggle_reproduce.ipynb   the notebook you run on Kaggle
├── scripts/data/                  how dataset/ was built from the raw downloads
├── tests/                         unit tests (run on Kaggle first)
├── docs/IMPLEMENTATION_NOTES.md   paper → code mapping, interpretations, deviations
└── dataset/                       NOT in git (uploaded to Kaggle separately)
```

---

## 2. Step 1: Push the code to GitHub

`dataset/`, `runs/`, checkpoints and the paper PDF are in `.gitignore`, so only code is uploaded.

1. Create an **empty** repository on github.com (no README, no .gitignore), e.g. `breast-ultrasound-dualstage`.
   *Public* is easiest for Kaggle. For *private*, see the token note in Step 3.
2. In a terminal in `E:\MTP`:
   ```bash
   git init
   git add .
   git commit -m "Dual-stage breast ultrasound pipeline"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo>.git
   git push -u origin main
   ```
3. On GitHub, check that `src/`, `configs/` and `notebooks/` are there and `dataset/` is **not**.

---

## 3. Step 2: Upload the dataset to Kaggle (once)

1. Zip the dataset folder: in Explorer, right-click `E:\MTP\dataset` → *Send to* → *Compressed (zipped) folder*.
   The zip must contain `manifest.csv`, `images/`, `masks/`, `metadata/` (about 380 MB).
2. Go to **kaggle.com → Create → New Dataset**, upload the zip, title it e.g. `bus-dataset`,
   keep it **Private**, and click **Create**. Kaggle unzips it automatically.

The notebook finds `manifest.csv` by itself, wherever Kaggle places it under `/kaggle/input/`.

> Keep the dataset private. Some source datasets (e.g. BUSI) do not clearly allow redistribution.

---

## 4. Step 3: Create the Kaggle notebook

1. **kaggle.com → Create → New Notebook → File → Import Notebook → GitHub**, paste the link to
   `notebooks/kaggle_reproduce.ipynb` in your repo. (Or upload the `.ipynb` file directly.)
2. Right-hand panel:
   - **Session options → Accelerator → GPU T4 x2** (or P100)
   - **Session options → Internet → On** (needed for `git clone`, `pip install` and pretrained weights; requires a phone-verified Kaggle account)
   - **Add Input → Datasets → Your Work →** `bus-dataset`
3. In the first code cell, set `REPO_URL = "https://github.com/<your-username>/<repo>.git"`.
4. **Private repository only:** create a GitHub *fine-grained token* with read access to the repo, add it in
   Kaggle under **Add-ons → Secrets** with the name `GITHUB_TOKEN`, attach it to the notebook, and uncomment the
   three token lines in the first cell.

---

## 5. Step 4: Run the experiments

Run the notebook **section by section**. Every run saves its results in `/kaggle/working/runs/...`.

| Section | What it does | Approx. time on a T4* |
|---|---|---|
| 0. Setup | clone repo, install `segmentation-models-pytorch`, find data | 1–2 min |
| 1. Check the code | unit tests + 1-epoch *smoke* run of every component | 3–5 min |
| 2. Table 1 | 13 segmentation models on Scenario A, picks the best on **validation** Dice | 1–2.5 h |
| 3. Table 2 | 56 classifier configurations on Scenario A, picks the best on **validation** AUC | 1.5–3 h |
| 4. Pipeline | full system on Scenario A, Scenario C and **BUS-BRA** (external) | 10–15 min |
| 5. Scenario B | retrain the best models on BUSI + Breast-Lesions-USG and evaluate | 15–30 min |
| 6. Baselines | Mask R-CNN and multi-task Swin on A, B, C | 1–2 h |
| 7. Stats & tables | Wilcoxon / McNemar tests, `reports/results.md` | 1 min |
| 8. Package | `results.zip` + `best_models.zip` to download | 1 min |

\* Rough estimates. Measure the first runs and adjust. Kaggle sessions last at most **12 h**, and the free GPU quota is about **30 h/week**.

**Always run Section 1 first.** If the smoke runs finish without errors, the full runs will too.

**Long runs:** use **Save Version → Save & Run All (Commit)**. The notebook then runs in the background
(up to 12 h) and its `/kaggle/working` output is kept. To continue in a new session, attach that output as an
input and run the *Resume* cell. The grids skip runs that already finished.

**Faster first pass:** run a subset of Table 2, e.g.
`!python -m bus.grid configs/grids/table2_cls.yaml --out {RUNS} --only mobilenet efficientnet`.

---

## 6. Step 5: Download and read the results

From the notebook's **Output** tab, download `results.zip` (metrics, figures, logs) and `best_models.zip`
(chosen checkpoints, needed later for deployment).

Each run folder `runs/<task>/<scenario>/<name>/seed42/` contains:

| File | Meaning |
|---|---|
| `config.yaml` | exact settings of the run (re-runnable with `--config`) |
| `history.csv` | loss and validation metrics per epoch |
| `metrics_val.json`, `metrics_test.json`, `metrics_external.json` | summary scores per split |
| `per_image_<role>.csv` | one row per image (Dice, IoU, probability, ...) |
| `cases_<role>.png` | 2 best and 2 worst segmentations (like the paper's Fig. 3 / 4) |
| `best.pt` / `last.pt` | model weights |

`reports/results.md` gathers everything into the paper's tables. Segmentation is reported **per-image**
(`dice`, `iou`) and **pooled** (`dice_pooled`, `iou_pooled`). See `docs/IMPLEMENTATION_NOTES.md` for why.

Expect somewhat **lower numbers than the paper**. Our BUSI split removes duplicate images that leak between
train and test in a random split.

---

## 7. Updating the code later

```bash
# on your PC, after changing code
git add .
git commit -m "describe the change"
git push
```
On Kaggle, re-run the **Setup** cell. It deletes the old clone and pulls the latest code.

---

## 8. Command reference

All commands accept `--data-root` (otherwise `$BUS_DATA_ROOT` or `./dataset`), `--out` (default `runs`),
`--set key.sub=value ...` to override any config value, and `--smoke` for a quick check.

```bash
# Stage 1: segmentation
python -m bus.train_seg --config configs/seg/base.yaml
python -m bus.train_seg --config configs/seg/base.yaml --set model.arch=unet model.encoder=efficientnet-b0 train.lr=3e-4

# Stage 2: classification (ground-truth crops by default)
python -m bus.train_cls --config configs/cls/base.yaml
python -m bus.train_cls --config configs/cls/base.yaml --set model.backbone=efficientnet_b0 model.head=linear model.dropout=0

# Train the classifier on PREDICTED crops instead
python -m bus.predict_masks --seg runs/seg/A/<seg>/seed42 --scenario A --roles train val test --out runs/pred_masks/A
python -m bus.train_cls --config configs/cls/base.yaml --set data.mask_dir=runs/pred_masks/A

# Full pipeline (Table 3). Role: val | test | external (BUS-BRA)
python -m bus.evaluate --seg runs/seg/A/<seg>/seed42 --cls runs/cls/A/<cls>/seed42 --scenario A --role test
python -m bus.evaluate ... --post largest --threshold 0.3        # pipeline-level options (for tuning)

# Grids (resumable; --only filters by name; --seeds 42 43 44 for repeated runs)
python -m bus.grid configs/grids/table1_seg.yaml --out runs
python -m bus.grid configs/grids/table2_cls.yaml --out runs --only mobilenet

# Baselines (+ evaluation-only mode with --ckpt)
python -m bus.baselines.maskrcnn --config configs/baselines/maskrcnn.yaml
python -m bus.baselines.multitask_swin --config configs/baselines/multitask_swin.yaml
python -m bus.baselines.maskrcnn --config configs/baselines/maskrcnn.yaml --ckpt <run_dir> --set data.scenario=C "eval.roles=[test]"

# Best run (by VALIDATION score), result tables, statistics
python -m bus.reporting best runs/seg/A --metric segmentation.dice
python -m bus.reporting collect runs --out reports
python -m bus.stats --a <per_image_test.csv of A> --b <per_image_test.csv of B> --names ours maskrcnn

# Unit tests
python -m pytest -q tests
```

**Scenarios** (`data.scenario`): `A` BUSI only · `B` BUSI + Breast-Lesions-USG · `C` train on BUSI,
test on all of Breast-Lesions-USG · `deploy` all three training datasets incl. normal scans.
Role `external` = BUS-BRA in every scenario.

**Useful config switches:** `data.resize_mode=stretch` (exactly as in the paper), `eval.roles=[val,test,external]`,
`model.unfreeze_k=all`, `crop.margin=0.2`, `eval.post=largest`.

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `git clone` fails | Internet is off, `REPO_URL` is wrong, or the repo is private (see Step 3.4) |
| `manifest.csv not found` | The dataset is not attached (Add Input), or the zip does not contain `manifest.csv` |
| Downloading pretrained weights fails | Internet must be **On** (phone-verified Kaggle account) |
| `CUDA out of memory` | `--set train.batch_size=8` (Swin / ViT / Mask R-CNN are the largest) |
| Session ended mid-grid | Attach the previous output, run the *Resume* cell, rerun the grid (finished runs are skipped) |
| `No module named bus` in the notebook | Re-run the second setup cell (it adds `src/` to `sys.path`) |
| A grid run failed | The grid continues and prints `FAILED ...` at the end. Re-run just that one with `--only <name>` |

---

## 10. Data and citations

The dataset (built in `scripts/data/`, documented in `dataset/README.md`) combines four public collections:

| Dataset | Use here | Cite |
|---|---|---|
| BUSI | Scenarios A, B, C (train/val/test) | Al-Dhabyani et al., *Data in Brief* 28:104863, 2020 |
| Breast-Lesions-USG (TCIA) | Scenarios B, C | Pawłowska et al., *Sci. Data* 11:148, 2024; TCIA doi:10.7937/9WKK-Q141 |
| BUS-UCLM | deployment (normal scans) | Mendeley Data doi:10.17632/7fvgj4jsp7.3 |
| BUS-BRA | external test only | Gómez-Flores et al., *Med. Phys.*, 2024 |

Splits are by patient (or by near-duplicate cluster for BUSI), so no patient appears in more than one split.

> Research code. Not a medical device and not for clinical diagnosis.
