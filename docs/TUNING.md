# Tuning phase: what was changed, why, and how it is evaluated

The tuning phase starts from the reproduction (phase 1) and its error analysis. Each change targets a weakness
that the reproduction showed. Nothing is chosen by looking at test data.

## 1. Evidence from the reproduction

| Finding (phase 1) | Numbers |
|---|---|
| The **classifier**, not the segmentation, limits the system | same classifier: AUC 0.894 with ground-truth crops vs 0.890 with predicted crops (BUSI test); 16 of 70 cancers missed either way |
| Missed cancers are **small lesions** | median lesion area 5.8 % of the image (missed) vs 15.4 % (caught) |
| The whole-image multi-task Swin classifies better, most of all on small lesions | BUS-BRA AUC on the smallest quarter of lesions: Swin 0.700 vs ours 0.637 |
| Weak transfer to **other hospitals / machines** | AUC 0.89 (BUSI) → 0.75–0.76 (Breast-Lesions-USG, BUS-BRA); recall by machine 35–66 % |
| Selecting 1 of 56 configurations on 98 validation images is **noisy** | val-best classifier (val AUC 0.962) ranked 30 / 56 on test (0.894) |
| The 0.5 threshold **misses cancers**; probabilities are poorly calibrated | sensitivity 0.77 (BUSI), 0.51 (BUS-BRA) |
| Segmentation failures are **scattered false regions** around faint lesions | 27 / 223 BUSI test images with Dice < 0.3 |
| The paper's plain U-Net did not train | Dice 0: stopped at epoch 11; LR schedule on a metric stuck at 0 |

## 2. Changes

| # | Change | Targets | Where |
|---|---|---|---|
| 0 | Plain U-Net: He initialisation (as in Ronneberger et al. 2015), ≥ 30 epochs before early stopping, LR schedule on val loss; plus a BatchNorm variant | broken Table 1 row | `models/segmentation.py`, `utils.EarlyStopping(min_epochs)`, `grids/table1_seg.yaml` |
| 1 | **Crop context** for the classifier: margin 0.1 (paper) / 0.5 / 1.0 / whole image | small lesions, the gap to Swin | `roi.mask_to_box(margin, full_image)`, `grids/tune_cls.yaml` |
| 2 | **Fine-tuning depth**: last 10 tensors (paper, lr 1e-3) vs whole backbone (lr 1e-4, weight decay 1e-4) | domain shift | `grids/tune_cls.yaml` |
| 3 | **Photometric augmentation** (brightness, contrast, gamma, speckle, blur; image only) | other machines | `data/augment.py`, `grids/tune_seg.yaml`, deploy stage |
| 4 | **More, more varied training data**: BUSI + Breast-Lesions-USG + BUS-UCLM (`deploy` scenario) | other hospitals | `tune.stage_deploy` |
| 5 | **3 seeds** per configuration; ranking by the mean; **ensembles** of the 3 seeds | selection noise, calibration | `grid.py --seeds`, `reporting.best_runs`, `inference.DualStagePipeline` |
| 6 | **Flip test-time augmentation** | stability | `inference.py` (`--tta`) |
| 7 | **Keep the largest segmented region** (if it helps on validation) | scattered false regions | `--post largest` |
| 8 | **Decision threshold for 90 % sensitivity**, chosen on validation | missed cancers | `bus.threshold` |

## 3. Protocol (`python -m bus.tune all`)

1. `vanilla`: re-run the fixed plain U-Net (Scenario A) → completes reproduction Table 1.
2. `cls`: 3 backbones (the three best by validation AUC in the reproduction) × 4 crop contexts × 2 fine-tuning
   depths × 3 seeds on Scenario A, ground-truth crops. Ranking: mean validation AUC.
3. `seg`: U-Net / DeepLabV3+ (ResNet34, 64) × with / without photometric augmentation × 3 seeds, Scenario A.
   Ranking: mean validation Dice.
4. `deploy`: the best segmentation and classification configurations retrained on all three training sets,
   3 seeds; classifier with and without photometric augmentation.
5. `select`: five candidate systems evaluated on the **common validation set** (validation splits of all three
   datasets): baseline (= reproduction's best configurations, seed 42), best single tuned models, 3-seed
   ensembles trained on BUSI, 3-seed ensembles trained on all data, the latter + TTA. Highest validation AUC wins
   (ties → the simpler candidate). Then mask post-processing (`largest` if it improves validation Dice without
   lowering AUC by more than 0.005), then the threshold with ≥ 90 % validation sensitivity.
6. `final`: baseline (threshold 0.5) and final system (chosen threshold) evaluated **once** on
   - the test splits of all three datasets, plus their normal scans (how often a lesion is wrongly found),
   - BUSI test (Scenario A, comparable to the paper and to phase 1),
   - BUS-BRA (external; 1,064 patients, other hospital and machines).
   Paired statistics final vs baseline: Wilcoxon (Dice), McNemar (correct / wrong at each system's own
   threshold), bootstrap 95 % CI of the AUC difference.
7. `export`: deployment bundle (models in float16 + settings + reference metrics + library versions).
8. `report`: `reports/tuning_report.md` (all tables below) and `reports/results.md`.

Scenario C is not used for the final systems: models trained on all three datasets have seen
Breast-Lesions-USG training images, so "all of Breast-Lesions-USG" would not be unseen data.

## 4. What to report in the thesis

- Table 1 completed with the fixed plain U-Net (and the BatchNorm variant).
- Classifier: validation AUC (mean ± std over 3 seeds) by crop context × fine-tuning depth: does more
  context help, and how much of the Swin gap does it close?
- Segmentation: effect of photometric augmentation (mean ± std over seeds).
- Final vs baseline on each test set: AUC, sensitivity, specificity, Dice, the per-dataset breakdown,
  normal-scan clear rate, and the paired statistics.
- The operating point: sensitivity reached on test / BUS-BRA at the threshold chosen on validation.
  It is usually somewhat below the 90 % target; report it as measured.
