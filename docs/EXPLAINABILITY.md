# Explainability and plain-language summaries

The dashboard answers three questions for every uploaded image:

1. **What did the system decide?** Lesion outline and malignancy score versus the decision threshold
   (`bus.inference`, unchanged from phases 1–2).
2. **Why?** Explanation methods, turned into a small set of numbers (`bus.explain`).
3. **What does that mean for a non-expert?** A language model writes a plain-language summary from those
   numbers (`bus.narrate`).

```
image ─► segmentation ─► outline ─► crop ─► classifier ─► score
             │                                  │
             ▼                                  ▼
  probability map, uncertainty,       Grad-CAM · LIME · Kernel SHAP
  ensemble agreement                  + deletion check
             └──────────────┬───────────────────┘
                            ▼
          report (JSON: numbers, labels, flags, reference performance)
                            ▼
          language model (Claude) ─► plain-language explanation
          (or the built-in template text if no API key is set)
```

## 1. Explanation methods

All methods explain the **classifier** (stage 2), because that is where the benign / malignant decision is
made. The error analysis of phase 1 showed that the classifier, not the segmentation, limits performance.
Every map is shown on the original image. Where the classifier is an ensemble, the explanation is averaged
over its members.

| Method | Idea | Cost | Output |
|---|---|---|---|
| **Grad-CAM** (Selvaraju et al., ICCV 2017) | Gradient of the predicted-class logit with respect to the last convolutional feature map; channel weights = mean gradient; ReLU of the weighted sum | 1 forward + 1 backward pass per model | Heat map of where the network looked |
| **LIME** (Ribeiro et al., KDD 2016) | Cut the classifier's region into superpixels, blur random subsets, fit a locally weighted linear model to the scores | `BUS_XAI_SAMPLES` forward passes (default 120 on CPU, 300 on GPU) | Signed effect of every region |
| **Kernel SHAP** (Lundberg & Lee, NeurIPS 2017) | Same perturbations, sampled with the Shapley kernel; the regression is constrained so that the effects add up to score(image) − score(all blurred) | Same as LIME | Shapley value of every region |

Implementation details (`src/bus/explain.py`):

- **Grad-CAM** uses timm's `forward_features` / `forward_head` split. The split is checked once against the
  normal forward pass, and Grad-CAM is skipped if it does not reproduce it, for example for transformers. It
  explains the *predicted* class: malignant if the score is at or above the threshold, otherwise benign.
- **Superpixels:** SLIC (scikit-image, 30 segments, compactness 0.2, σ = 2), or a grid if scikit-image is
  missing. Superpixels that straddle the predicted lesion outline are split in two, so each region is clearly
  inside or outside the lesion.
- **Hidden regions** are replaced by a strongly blurred copy (σ = 8 % of the region size), as in shap's image
  masker. They are not replaced with black, because in ultrasound a black area looks like an anechoic lesion.
- **LIME** follows `lime_image`: uniform binary samples with the first sample unperturbed, cosine distance,
  kernel width 0.25, ridge regression with α = 1, and the weighted R² reported as `fit_r2`.
- **Kernel SHAP** follows shap's `KernelExplainer`: coalition sizes drawn with p(s) ∝ (M−1)/(s(M−s)), paired
  complements, and the efficiency constraint solved by eliminating one variable. The SHAP values of an
  additive model are recovered exactly (`tests/test_explain.py`).
- LIME and SHAP score the ensemble mean without flip averaging, which halves the cost. `score_explained`
  is shown next to the pipeline score.
- **Deletion check** (faithfulness): the three regions that most support the decision are blurred and the
  classifier is run again. A large change means that the explanation found what actually drove the decision.

## 2. Numbers derived for every image (`build_report`)

| Group | Fields |
|---|---|
| Prediction | `score_malignant`, `threshold`, `distance_from_threshold`, `member_scores`, `models_above_threshold`, `score_is_calibrated_probability: false` |
| Segmentation | `mean_probability_inside_outline`, `uncertain_pixels_vs_lesion_area` (probability 0.3–0.7 near the outline), `member_agreement_dice` (pairwise Dice between ensemble members) |
| Grad-CAM | `evidence_by_region` (share of the attention in the lesion interior, on the margin band, and in the surrounding tissue, each next to its area share), `focus_inside_or_on_lesion`, `peak_region` |
| LIME / SHAP | `top_regions` (effect, direction, location such as "on the lesion edge" or "tissue below (behind) the lesion"), `by_region` totals, `deletion_check`, `fit_r2` / `additivity_error` |
| Outline measurements | area, depth-to-width ratio, circularity, solidity, brightness relative to the surrounding tissue, brightness behind the lesion, position. These are **descriptive only**: not model inputs and not a radiological assessment |
| Flags | borderline score, model disagreement, small lesion (< 3 % of the image: most cancers missed in testing were small), several regions, lesion touching the border, uncertain outline, low resolution |
| Reference performance | AUC, sensitivity, specificity and normal-scan clear rate stored in the bundle (test sets and external BUS-BRA) |

## 3. Plain-language summary (`src/bus/narrate.py`)

- **Model:** Claude Opus 5.5 (`claude-opus-5-5`, Anthropic API, official `anthropic` Python SDK), with effort
  `medium`. If the model declines a request, the server-side fallback (`fallbacks: "default"`) re-runs it on
  another model. Model and effort can be changed with `BUS_LLM_MODEL` / `BUS_LLM_EFFORT`.
- **Input:** only the JSON report above. The model sees no image, no file name and no patient data, so
  nothing identifying leaves the server.
- **Grounding rules** in the system prompt:
  - use only numbers from the report;
  - the score is not a calibrated probability, so never write "X % chance of cancer";
  - never say that the person has or does not have cancer;
  - explain each method in one simple sentence;
  - mention every flag and the reference performance, including the external test;
  - outline measurements are not a diagnosis and are never turned into a BI-RADS category;
  - always end by saying that a clinician must review the images.
- **Output:** five fixed sections (Summary · What the system found · Why the system decided this · How sure the
  system is · What happens next). The reader is chosen as *patient* or *clinician*, and the language as
  English, Hindi or several others.
- **Fallback:** if no `ANTHROPIC_API_KEY` is set, or the call fails (network, rate limit, invalid key), a
  rule-based template produces the same five sections in English, and the panel says why.
- **Cost:** one summary is about 3–4 k input tokens and 0.5–1 k output tokens, so a few US cents with Opus 5.5.

## 4. Evaluating the explanations (`python -m bus.explain_eval`)

`notebooks/kaggle_dashboard.ipynb` runs this on 120 BUS-BRA (external) and 120 internal test images. The
images are stratified by label, and the results go into `xai_results.zip`.

| Metric | Meaning | Good if |
|---|---|---|
| `gradcam_energy_in_gt` / `_in_gt_band` | share of the Grad-CAM attention inside the ground-truth lesion (plus margin) | well above `gt_share_of_region` (chance level) |
| `gradcam_peak_in_gt_band` | pointing game: the attention peak lies on the true lesion or its margin | close to 1 |
| `lime_/shap_deletion_change` | score change towards the other class after blurring the 3 most supporting regions | clearly > 0 |
| `lime_/shap_deletion_flips` | that deletion flips the decision | — (shows how concentrated the evidence is) |
| `lime_/shap_top_in_gt` | the most influential region lies on the true lesion | high |
| `lime_fit_r2` | local fidelity of LIME | close to 1 |

Each metric is reported for all images, for correct versus wrong decisions, and per label. Comparing correct
and wrong decisions tests whether wrong decisions come with attention outside the lesion. `examples.png`
shows typical cases (image + ground truth, prediction, Grad-CAM, LIME, SHAP) for the thesis.

## 5. Limitations

- Explanations describe the model's behaviour, not biology. A heat map on the lesion edge does not mean that
  the edge is malignant.
- LIME and SHAP depend on the superpixels, the blur baseline and the number of samples. With 120 samples on a
  CPU they are approximations, and `fit_r2` and the deletion check show how far to trust them.
- Grad-CAM has the resolution of the last feature map (7 × 7 for ResNet-50 at 224 px).
- The language model can still phrase things imperfectly. The text is labelled as automatically generated,
  and the numbers it is based on are shown next to it.
- Research prototype: not a medical device, not for diagnosis or treatment decisions.
