# Dual-Stage Breast Ultrasound Lesion Segmentation & Classification

This repository reproduces, improves and deploys
**Bruno, Macrì & Dodaro, "A Dual-stage Deep Learning Framework for Breast Ultrasound Image Segmentation and Classification", *J. Med. Syst.* 49:162 (2025)**.

```
Ultrasound image ─► 1. Segmentation ─► 2. Mask → ROI box ─► 3. Crop ─► 4. Classification ─► Benign / Malignant
                     (U-Net / DeepLabV3+)                                 (CNN backbone)         + probability
```

**Workflow:** write code on your PC → push to **GitHub** → train on **Kaggle** (free GPU) by cloning the repo →
download the trained model bundle → run the web app (Hugging Face Spaces or any Docker host).
Nothing is trained on your own computer.

| Phase | Notebook / folder | Status |
|---|---|---|
| 1. Reproduce the paper (Tables 1–3, baselines, statistics, external test) | `notebooks/kaggle_reproduce.ipynb` | done |
| 2. Tuning: fix the plain U-Net, crop context, fine-tuning depth, more data, ensembles, threshold | `notebooks/kaggle_tuning.ipynb` | ready to run |
| 3. Deployment: web API + browser UI, model bundle | `app/`, `Dockerfile`, `docs/DEPLOYMENT.md` | ready (needs the bundle from phase 2) |

---

## Contents
1. [Repository layout](#1-repository-layout)
2. [One-time setup: GitHub and the Kaggle dataset](#2-one-time-setup-github-and-the-kaggle-dataset)
3. [Phase 1: reproduction on Kaggle](#3-phase-1-reproduction-on-kaggle)
4. [Phase 2: tuning on Kaggle](#4-phase-2-tuning-on-kaggle)
5. [Phase 3: deployment (web app)](#5-phase-3-deployment-web-app)
6. [Updating the code](#6-updating-the-code)
7. [Command reference](#7-command-reference)
8. [Troubleshooting](#8-troubleshooting)
9. [Data and citations](#9-data-and-citations)

---

## 1. Repository layout

```
├── configs/
│   ├── seg/base.yaml, cls/base.yaml   stage-1 / stage-2 settings (defaults = paper's best models)
│   ├── baselines/                     Mask R-CNN and multi-task Swin
│   └── grids/                         table1_seg (14 runs), table2_cls (56), tune_cls (24 x 3 seeds), tune_seg (4 x 3)
├── src/bus/
│   ├── data/                          manifest + scenarios, preprocessing, augmentation (incl. photometric), datasets
│   ├── models/                        segmentation and classification networks
│   ├── roi.py                         mask clean-up and ROI cropping (margin / whole-image context)
│   ├── train_seg.py, train_cls.py     training + evaluation of each stage
│   ├── inference.py                   DualStagePipeline: ensembles + flip test-time augmentation (used by the app)
│   ├── evaluate.py                    full-pipeline evaluation (Table 3, final tests)
│   ├── tune.py                        the whole tuning protocol, as resumable stages
│   ├── threshold.py                   decision threshold for a target sensitivity (on validation)
│   ├── bundle.py                      export / load the deployment model bundle
│   ├── baselines/                     maskrcnn.py, multitask_swin.py
│   ├── grid.py, reporting.py, stats.py
├── app/                               web app: server.py (FastAPI) + static/index.html (UI)
├── Dockerfile, requirements-app.txt   container for the web app
├── deploy/hf_space_README.md          header file for a Hugging Face Space
├── notebooks/                         kaggle_reproduce.ipynb (phase 1), kaggle_tuning.ipynb (phase 2)
├── docs/                              IMPLEMENTATION_NOTES.md, TUNING.md, DEPLOYMENT.md
├── scripts/data/                      how dataset/ was built from the raw downloads
├── tests/                             unit tests (run on Kaggle first)
└── dataset/, kaggle_outputs/, deploy_bundle/    NOT in git
```

---

## 2. One-time setup: GitHub and the Kaggle dataset

**GitHub.** The code lives at `https://github.com/vp-y/mtp`. `dataset/`, run outputs, checkpoints, zips and the
paper PDF are in `.gitignore`, so only code is uploaded. To publish changes, see [Updating the code](#6-updating-the-code).

**Dataset on Kaggle (once).**
1. Zip `E:\MTP\dataset` (right-click → *Send to* → *Compressed (zipped) folder*). It must contain `manifest.csv`,
   `images/`, `masks/`, `metadata/` (about 380 MB).
2. **kaggle.com → Create → New Dataset**, upload the zip, keep it **Private**, click **Create**.

The notebooks find `manifest.csv` wherever Kaggle places it under `/kaggle/input/`.
Keep the dataset private: some sources (e.g. BUSI) do not clearly allow redistribution.

**Creating a notebook from the repo (both phases).**
1. **kaggle.com → Create → New Notebook → File → Import Notebook → GitHub**, enter `vp-y/mtp`, pick the notebook.
2. Right-hand panel: **Accelerator → GPU T4 x2** (or P100) · **Internet → On** (phone-verified account) ·
   **Add Input → Datasets → Your Work →** your dataset.
3. Private repo only: create a GitHub fine-grained token (read access), add it under **Add-ons → Secrets** as
   `GITHUB_TOKEN`, and uncomment the two token lines in the first cell.

Run the notebook top to bottom. For long runs use **Save Version → Save & Run All (Commit)**: it keeps running
after you close the browser (max 12 h per session, about 30 GPU-hours per week).

---

## 3. Phase 1: reproduction on Kaggle

Notebook: `notebooks/kaggle_reproduce.ipynb` (results in `/kaggle/working/runs`).

| Section | What it does | Time on a T4 |
|---|---|---|
| 0–1 | setup, unit tests, 1-epoch smoke run of every component | 5 min |
| 2 | Table 1: 13 segmentation models (Scenario A), best by **validation** Dice | 1–2.5 h |
| 3 | Table 2: 56 classifiers (Scenario A), best by **validation** AUC | 1.5–3 h |
| 4 | full pipeline on Scenario A, Scenario C and BUS-BRA (external) | 15 min |
| 5 | Scenario B (BUSI + Breast-Lesions-USG) | 30 min |
| 6 | baselines: Mask R-CNN and multi-task Swin | 1–2 h |
| 7–8 | statistics, `reports/results.md`, downloads `results.zip` + `best_models.zip` | 2 min |

Each run folder `runs/<task>/<scenario>/<name>/seed42/` holds `config.yaml`, `history.csv`,
`metrics_<role>.json`, `per_image_<role>.csv`, `cases_<role>.png` (2 best / 2 worst segmentations) and `best.pt`.
Segmentation is reported per image (`dice`) and pooled (`dice_pooled`); see `docs/IMPLEMENTATION_NOTES.md`.

---

## 4. Phase 2: tuning on Kaggle

Notebook: `notebooks/kaggle_tuning.ipynb` (results in `/kaggle/working/tuning`). Every change is motivated by
the error analysis of phase 1; the reasoning and the protocol are in **`docs/TUNING.md`**.

| Section | What it does | Time on a T4 |
|---|---|---|
| 1 | unit tests + the **whole** tuning protocol on 8 images (code check) | 10 min |
| 2 | plain U-Net with the fix (He init, ≥ 30 epochs, schedule on val loss) + BatchNorm variant | 15 min |
| 3 | classifier: 3 backbones × crop context (paper crop, ×2, ×3, whole image) × fine-tuning (K=10, all) × 3 seeds | 1.5–2.5 h |
| 4 | segmentation: U-Net / DeepLabV3+ × photometric augmentation × 3 seeds | 1 h |
| 5 | retrain the best models on BUSI + Breast-Lesions-USG + BUS-UCLM, 3 seeds | 1 h |
| 6 | choose the final system on validation; decision threshold for 90 % sensitivity | 15 min |
| 7 | baseline (= reproduction) vs final on all test sets and BUS-BRA, paired statistics | 20 min |
| 8 | export the deployment bundle, run it on an image, test the web API | 5 min |
| 9 | `reports/tuning_report.md`; downloads `tuning_results.zip` + `deploy_bundle.zip` | 2 min |

Every cell runs one stage of `python -m bus.tune`; finished work is skipped, so after a timeout attach the
previous output, run the *Resume* cell and run the remaining cells again.

**Download** `tuning_results.zip` (reports, metrics, figures) and `deploy_bundle.zip` (the final models, needed
for phase 3). Put them in `E:\MTP\kaggle_outputs\` (ignored by git).

---

## 5. Phase 3: deployment (web app)

The app (`app/server.py`) loads the bundle and serves a browser UI: upload an ultrasound image → lesion outline,
probability map, probability of malignancy with the decision threshold, and a decision text.

Short version (step-by-step details and the API in **`docs/DEPLOYMENT.md`**):

- **Free public URL, nothing installed on your PC: Hugging Face Spaces.** Create a Space (SDK *Docker*), upload
  `Dockerfile`, `requirements-app.txt`, `app/`, `src/`, the unzipped `deploy_bundle/`, and
  `deploy/hf_space_README.md` renamed to `README.md`. The Space builds and gives you a URL.
- **Any machine with Docker:** unzip the bundle to `deploy_bundle/` in the repo, then
  `docker build -t bus-app .` and `docker run -p 7860:7860 bus-app` → http://localhost:7860.

> Research prototype. Not a medical device, not for diagnosis or treatment decisions.

---

## 6. Updating the code

```bash
# on your PC, in E:\MTP, after changing code
git add -A
git commit -m "describe the change"
git push
```
On Kaggle, re-run the **Setup** cell: it deletes the old clone and pulls the latest code.

---

## 7. Command reference

All training commands accept `--data-root` (otherwise `$BUS_DATA_ROOT` or `./dataset`), `--out`,
`--set key.sub=value ...` to override any config value, and `--smoke` for a quick check.

```bash
# Stage 1 / stage 2 training
python -m bus.train_seg --config configs/seg/base.yaml --set model.arch=unet train.lr=3e-4
python -m bus.train_cls --config configs/cls/base.yaml --set model.backbone=resnet50 crop.margin=0.5 model.unfreeze_k=all

# Full pipeline. Role: val | test | external (BUS-BRA). Several --seg / --cls = ensemble
python -m bus.evaluate --seg SEG_RUN --cls CLS_RUN --scenario A --role test
python -m bus.evaluate --seg S1 S2 S3 --cls C1 C2 C3 --tta --post largest --threshold 0.21 \
                       --scenario deploy --role test --include-normal --name final

# Decision threshold for a target sensitivity, on VALIDATION predictions
python -m bus.threshold runs/pipeline/deploy/final/per_image_val.csv --sensitivity 0.90 0.95

# Grids (resumable; --only filters by name; --seeds for repeated runs)
python -m bus.grid configs/grids/table1_seg.yaml --out runs
python -m bus.grid configs/grids/tune_cls.yaml --out tuning --only resnet50

# Tuning protocol (phase 2), any subset of stages, or all
python -m bus.tune all --out tuning
python -m bus.tune vanilla cls seg deploy select final export report --out tuning

# Deployment bundle
python -m bus.bundle export --seg S1 S2 --cls C1 C2 --threshold 0.21 --post largest --tta --out deploy_bundle
python -m bus.bundle info deploy_bundle

# Baselines, best run, tables, statistics
python -m bus.baselines.maskrcnn --config configs/baselines/maskrcnn.yaml
python -m bus.baselines.multitask_swin --config configs/baselines/multitask_swin.yaml
python -m bus.reporting best runs/seg/A --metric segmentation.dice
python -m bus.reporting collect runs --out reports
python -m bus.stats --a A/per_image_test.csv --b B/per_image_test.csv --names ours maskrcnn

# Web app (inference only)
uvicorn app.server:app --host 0.0.0.0 --port 7860

# Unit tests
python -m pytest -q tests
```

**Scenarios** (`data.scenario`): `A` BUSI · `B` BUSI + Breast-Lesions-USG · `C` train on BUSI, test on all of
Breast-Lesions-USG · `deploy` BUSI + Breast-Lesions-USG + BUS-UCLM. Role `external` = BUS-BRA in every scenario.
Normal (lesion-free) scans are left out unless `data.include_normal=true` / `--include-normal`.

**Useful switches:** `data.resize_mode=stretch` (as in the paper), `crop.margin=0.5`, `crop.full_image=true`,
`model.unfreeze_k=all`, `augment.photometric={p: 0.5}`, `train.min_epochs=30`, `train.monitor=loss`,
`eval.post=largest`, `eval.roles=[val,test,external]`.

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `git clone` fails | Internet is off, or the repo is private (token, see section 2) |
| `manifest.csv not found` | The dataset is not attached (Add Input), or the zip has no `manifest.csv` |
| Downloading pretrained weights fails | Internet must be **On** (phone-verified Kaggle account) |
| `No module named 'bus'` in a notebook cell | Re-run the Setup cell (it sets `PYTHONPATH` and clears Python's import cache) |
| `CUDA out of memory` | `--set train.batch_size=8` |
| Session ended mid-run | Attach the previous output, run the *Resume* cell, re-run the cells (finished work is skipped) |
| A grid run failed | The grid continues and prints `FAILED ...`; re-run it with `--only <name>` |
| App: "models not loaded" | `deploy_bundle/` missing or incomplete; check `/api/health`. If weights fail to load, install the `timm` / `segmentation-models-pytorch` versions listed in `deploy_bundle/bundle.json` |

---

## 9. Data and citations

The dataset (built in `scripts/data/`, documented in `dataset/README.md`) combines four public collections:

| Dataset | Use here | Cite |
|---|---|---|
| BUSI | Scenarios A, B, C, deploy | Al-Dhabyani et al., *Data in Brief* 28:104863, 2020 |
| Breast-Lesions-USG (TCIA) | Scenarios B, C, deploy | Pawłowska et al., *Sci. Data* 11:148, 2024; TCIA doi:10.7937/9WKK-Q141 |
| BUS-UCLM | deploy (training data, test set, normal scans) | Mendeley Data doi:10.17632/7fvgj4jsp7.3 |
| BUS-BRA | external test only | Gómez-Flores et al., *Med. Phys.*, 2024 |

Splits are by patient (or by near-duplicate cluster for BUSI), so no patient appears in more than one split.

> Research code. Not a medical device and not for clinical diagnosis.
