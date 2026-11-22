# Implementation notes: paper → code

Paper: Bruno, Macrì & Dodaro, *A Dual-stage Deep Learning Framework for Breast Ultrasound Image
Segmentation and Classification*, J. Med. Syst. 49:162 (2025), doi:10.1007/s10916-025-02298-6.

## Where each part of the paper lives

| Paper | Code |
|---|---|
| Fig. 1 step 1: read and pre-process (grayscale, 256×256) | `src/bus/data/preprocess.py`, `src/bus/data/datasets.py` |
| Augmentation: rotation 20°, translation 20 %, zoom, horizontal flip (train only) | `src/bus/data/augment.py` |
| Oversampling to balance classes | `utils.balanced_sampler` |
| Step 2: segmentation (U-Net / FPN / DeepLabV3+ × ResNet34 / EfficientNet-B0, plain U-Net) | `src/bus/models/segmentation.py` |
| Loss 0.4·BCE + 0.5·(1−Dice) + 0.5·(1−IoU) | `src/bus/losses.py` |
| Adam 1e-4, ReduceLROnPlateau (×0.1, patience 3), early stopping 10 | `src/bus/train_seg.py`, `configs/seg/base.yaml` |
| Steps 3–4: binarised mask → deterministic ROI → crop | `src/bus/roi.py` |
| Step 5: classifier = pretrained backbone + linear / MLP(128, 64) head, last K groups unfrozen | `src/bus/models/classification.py` |
| BCEWithLogits, Adam 1e-3, same scheduler / early stopping | `src/bus/train_cls.py`, `configs/cls/base.yaml` |
| Scenarios A, B, C | `src/bus/data/manifest.py` |
| Table 1 / Table 2 grids | `configs/grids/table1_seg.yaml`, `configs/grids/table2_cls.yaml` |
| Table 3 "Our approach" | `src/bus/evaluate.py` (`DualStagePipeline` in `src/bus/inference.py`) |
| Mask R-CNN baseline (R50-FPN, COCO, SGD 1e-3 / 0.9 / 5e-4, 30 epochs) | `src/bus/baselines/maskrcnn.py` |
| Multi-task Swin baseline (AdamW 1e-4 / 1e-4, dropout 0.3) | `src/bus/baselines/multitask_swin.py` |
| Wilcoxon signed-rank test, p < 0.05 | `src/bus/stats.py` |

## Interpretations (the paper does not specify these)

1. **"Base Channel" (32 / 64)** is undefined for library models. We map it to decoder width:
   U-Net decoder `(8b, 4b, 2b, b, b/2)`, FPN pyramid `8b` / segmentation `4b`, DeepLabV3+ `4b`.
   b = 32 gives the library defaults for U-Net and FPN, and b = 64 gives the default for DeepLabV3+.
2. **"Unfreeze the last K parameter groups"**: we use the last K parameter *tensors* of the backbone
   (`list(backbone.parameters())[-K:]`). Frozen BatchNorm layers stay in eval mode.
3. **ROI crop margin**: 10 % of the lesion box on each side, minimum 16 px, from the largest
   connected component (`configs/cls/base.yaml → crop`).
4. **Classifier training crops**: from ground-truth masks by default. Predicted-mask crops are available via
   `bus.predict_masks` + `data.mask_dir`.
5. **No lesion segmented**: the pipeline classifies the whole image (`--empty-policy full_image`)
   and counts these cases (`n_lesion_not_found`).
6. **Segmentation in Scenario C**: models trained on BUSI are applied unchanged to Breast-Lesions-USG.
7. **Plain U-Net**: no BatchNorm (the paper describes conv + ReLU only). Set `model.batch_norm: true` to add it.
8. **Batch size, max epochs, AMP**: 16 / 100 for segmentation, 32 / 100 for classification, with mixed
   precision on GPU. Early stopping usually ends training much earlier.

## Deliberate deviations (and why)

| Paper | Here | Reason |
|---|---|---|
| Random 50/15/35 split of BUSI | Same ratios, but **grouped by near-duplicate cluster**; 19 label-conflicting images removed | BUSI has about 19 % duplicates, so a random split leaks test images into training |
| BUSI: 487 benign / 697 lesion images | 427 benign + 203 malignant used (437 / 210 before removing conflicts) | The paper's counts do not match the dataset |
| Stretch to 256×256 | **Pad to square** by default (`data.resize_mode: stretch` restores the paper) | Keeps lesion shape, a key malignancy cue |
| Best model chosen (apparently) on test results | Chosen on the **validation** split (`bus.reporting.best_run`) | Avoids optimistic test numbers |
| Dice/IoU aggregation unclear (ours look pooled, baselines per-image) | **Both** reported for every method (`dice`, `dice_pooled`, ...) | Makes comparisons fair |
| Mask R-CNN AUC apparently from hard labels | AUC from a continuous score `s_mal / (s_mal + s_ben)` | AUC needs a ranking score |
| Swin input 256×256 | 224×224 | `swin_tiny_patch4_window7_224` needs sizes divisible by 32 × 7 |
| BUS-BRA not used | Held-out **external test** (1,064 patients) | A test on patients from another hospital and other scanners |

Because of the de-duplicated split, **expect lower scores than the paper**, especially classification
AUC. That difference is a finding in its own right.
