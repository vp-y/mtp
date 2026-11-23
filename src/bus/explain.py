"""Explainability for the dual-stage pipeline: why did the system decide what it decided?

Three explanation methods for the CLASSIFIER (stage 2), all expressed on the original image:

  Grad-CAM    (Selvaraju et al., ICCV 2017) gradient-weighted activation map of the last convolutional
              feature map: where the network "looked" for the predicted class. One forward + backward
              pass per model, so it runs for every prediction.
  LIME        (Ribeiro et al., KDD 2016) the classifier region is cut into superpixels; random subsets
              are blurred out and a locally weighted linear model is fitted to the classifier's
              responses. Weight > 0: the region pushes the score towards "malignant".
  Kernel SHAP (Lundberg & Lee, NeurIPS 2017) same perturbations, but the regression uses the Shapley
              kernel and is constrained so that the region contributions add up exactly to
              score(image) - score(everything blurred).

Both perturbation methods follow the reference implementations (lime_image: cosine distance, kernel
width 0.25, ridge alpha 1; shap KernelExplainer: Shapley-kernel coalition sampling with paired
complements, efficiency constraint). They are implemented here directly (numpy only) so the deployed
app needs no extra packages and the code works with any number of ensemble members. "Hidden" regions
are replaced by a strongly blurred copy of the image (as shap's image masker does) rather than black,
because black would look like an (anechoic) lesion in ultrasound.

For SEGMENTATION (stage 1) the explanation is its own probability map plus uncertainty measures:
mean confidence inside the outline, the share of uncertain pixels along the border, and the agreement
between ensemble members.

`build_report` turns all of this into a compact, JSON-able set of plain numbers and flags. The web app
shows it, and `bus.narrate` gives it to the language model, which writes the plain-language summary.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import torch

from .data.preprocess import resize_to_square, restore
from .roi import keep_largest_component, mask_to_box

REGIONS = ("interior", "margin", "surrounding")
UNCERTAIN = (0.3, 0.7)      # segmentation probabilities treated as "unsure"
BORDERLINE = 0.10           # |score - threshold| below this -> borderline
SMALL_LESION_PCT = 3.0      # missed cancers in our tests were mostly small (median 5.8 % of the image)


# ----------------------------------------------------------------------------------------------- analysis

def analyse(pipe, gray: np.ndarray) -> dict:
    """Run the pipeline once and keep every ensemble member's output (same result as pipe(gray))."""
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("expected a 2-D uint8 grayscale image")
    member_maps = pipe.segment_members(gray)
    prob_map, mask = pipe.combine(member_maps)
    lesion_box = mask_to_box(mask, margin=0.0, min_size=1)
    member_probs = [float(p) for p in pipe.classify_members(gray, mask)]
    return {"gray": gray, "prob_map": prob_map, "mask": mask, "member_maps": member_maps,
            "lesion_box": lesion_box, "box": pipe.cls.box(mask), "lesion_found": lesion_box is not None,
            "member_probs": member_probs, "prob_malignant": float(np.mean(member_probs))}


def _member_boxes(pipe, a: dict) -> list:
    """Box each classifier cropped (None -> whole image)."""
    H, W = a["gray"].shape
    return [c.box(a["mask"]) or (0, 0, W, H) for c in pipe.clss]


def _union(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def region_map(mask: np.ndarray) -> np.ndarray:
    """0 = lesion interior, 1 = lesion margin (band around the outline), 2 = surrounding tissue."""
    m = keep_largest_component(mask).astype(np.uint8)
    out = np.full(mask.shape, 2, dtype=np.uint8)
    area = int(m.sum())
    if area == 0:
        return out
    k = max(2, int(round(0.12 * math.sqrt(area))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    inner = cv2.erode(m, kernel).astype(bool)
    outer = cv2.dilate(m, kernel).astype(bool)
    out[outer] = 1
    out[inner] = 0
    return out


def _region_shares(weights: np.ndarray, regions: np.ndarray) -> dict:
    """Share of a non-negative map that falls in each region, and each region's share of the area."""
    w = np.clip(weights, 0, None).astype(np.float64)
    tot, n = w.sum(), regions.size
    return {r: {"evidence": float(w[regions == i].sum() / tot) if tot > 0 else 0.0,
                "area": float((regions == i).sum() / n)} for i, r in enumerate(REGIONS)}


# ----------------------------------------------------------------------------------------------- Grad-CAM

def _gradcam_patch(c, patch: np.ndarray, sign: float):
    """Grad-CAM of one classifier on one patch -> map with the patch's shape, or None if unsupported."""
    model = c.model
    x = c.to_input(patch)[None].to(c.device)
    bb = model.backbone
    if not (hasattr(bb, "forward_features") and hasattr(bb, "forward_head")):
        return None
    with torch.enable_grad():
        f = bb.forward_features(x)
        if f.ndim != 4:  # transformer tokens: no spatial map
            return None
        f = f.detach().float().requires_grad_(True)
        logit = model.head(bb.forward_head(f))[:, 0]
        if getattr(c, "_cam_checked", None) is None:  # the split must reproduce the normal forward pass
            with torch.no_grad():
                ref = model(x)[:, 0].float()
            c._cam_checked = bool(torch.allclose(logit.detach(), ref, atol=1e-3, rtol=1e-3))
        if not c._cam_checked:
            return None
        (g,) = torch.autograd.grad((sign * logit).sum(), f)
    cam = torch.relu((g.mean(dim=(2, 3), keepdim=True) * f).sum(1))[0].detach().cpu().numpy()
    cam = cv2.resize(cam.astype(np.float32), (c.size, c.size), interpolation=cv2.INTER_LINEAR)
    _, geo = resize_to_square(patch, c.size, c.mode)
    return restore(cam, geo)


def gradcam(pipe, a: dict, threshold: float):
    """Grad-CAM for the PREDICTED class, averaged over the classifier ensemble, on the full image.

    Returns {"map": HxW in [0, 1], "target": "malignant" | "benign", "metrics": {...}} or None.
    """
    if not a["lesion_found"]:
        return None
    target = "malignant" if a["prob_malignant"] >= threshold else "benign"
    sign = 1.0 if target == "malignant" else -1.0
    H, W = a["gray"].shape
    total, used = np.zeros((H, W), np.float32), 0
    for c, (x0, y0, x1, y1) in zip(pipe.clss, _member_boxes(pipe, a)):
        cam = _gradcam_patch(c, a["gray"][y0:y1, x0:x1], sign)
        if cam is None:
            continue
        used += 1
        if cam.max() > 0:
            total[y0:y1, x0:x1] += cam / cam.max()
    if used == 0:
        return None
    cam = total / used
    if cam.max() > 0:
        cam = cam / cam.max()
    regions = region_map(a["mask"])
    shares = _region_shares(cam, regions)
    peak = np.unravel_index(int(np.argmax(cam)), cam.shape)
    return {"map": cam, "target": target, "metrics": {
        "method": "gradcam", "explains": target, "models_used": used,
        "evidence_by_region": shares,
        "peak_region": REGIONS[int(regions[peak])] if cam.max() > 0 else None,
        "focus_inside_or_on_lesion": shares["interior"]["evidence"] + shares["margin"]["evidence"]}}


# -------------------------------------------------------------------------------------- LIME / Kernel SHAP

def superpixels(patch: np.ndarray, n_segments: int = 30, mask=None, seed: int = 0) -> np.ndarray:
    """Superpixel labels 0..M-1 (SLIC if scikit-image is installed, else a grid).

    If a lesion mask is given, superpixels that straddle the lesion outline are split in two, so every
    region is clearly inside or outside the lesion.
    """
    h, w = patch.shape
    lab = None
    try:
        from skimage.segmentation import slic
        img = patch.astype(np.float64) / 255.0
        kw = dict(n_segments=int(n_segments), compactness=0.2, sigma=2.0, start_label=0)
        try:
            lab = slic(img, channel_axis=None, **kw)
        except TypeError:  # scikit-image < 0.19
            lab = slic(img, multichannel=False, **kw)
    except ImportError:
        pass
    if lab is None or lab.max() < 1:
        g = max(2, int(round(math.sqrt(n_segments))))
        ys = np.minimum(np.arange(h) * g // max(h, 1), g - 1)
        xs = np.minimum(np.arange(w) * g // max(w, 1), g - 1)
        lab = ys[:, None] * g + xs[None, :]
    lab = lab.astype(np.int64)
    if mask is not None and mask.any() and not mask.all():
        inside = mask.astype(bool)
        new = lab * 2
        for s in np.unique(lab):
            seg = lab == s
            frac = inside[seg].mean()
            if 0.15 < frac < 0.85:
                new[seg & inside] += 1
        lab = new
    _, inv = np.unique(lab, return_inverse=True)
    return inv.reshape(h, w)


def lime_weights(Z: np.ndarray, y: np.ndarray, kernel_width: float = 0.25, alpha: float = 1.0) -> dict:
    """Weighted ridge regression of y on binary Z (as lime_image). Row 0 must be the unperturbed image."""
    Z = Z.astype(np.float64)
    ref = np.ones(Z.shape[1])
    norms = np.linalg.norm(Z, axis=1) * np.linalg.norm(ref)
    cos = np.divide(Z @ ref, norms, out=np.zeros(len(Z)), where=norms > 0)
    d = 1.0 - cos
    w = np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))
    sw = w / w.sum()
    zm, ym = sw @ Z, sw @ y
    Zc, yc = Z - zm, y - ym
    A = Zc.T @ (Zc * w[:, None]) + alpha * np.eye(Z.shape[1])
    coef = np.linalg.solve(A, Zc.T @ (w * yc))
    intercept = ym - zm @ coef
    pred = intercept + Z @ coef
    ss_res = float(w @ (y - pred) ** 2)
    ss_tot = float(w @ (y - ym) ** 2)
    return {"weights": coef, "intercept": float(intercept),
            "fit_r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0}


def shap_samples(M: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Coalitions drawn from the Shapley kernel (size s with p ~ (M-1)/(s(M-s))), plus their complements."""
    if M < 2:
        return np.zeros((0, M), dtype=np.uint8)
    sizes = np.arange(1, M)
    p = (M - 1) / (sizes * (M - sizes))
    p = p / p.sum()
    half = max(1, n // 2)
    Z = np.zeros((half, M), dtype=np.uint8)
    for i, s in enumerate(rng.choice(sizes, size=half, p=p)):
        Z[i, rng.choice(M, size=int(s), replace=False)] = 1
    return np.concatenate([Z, 1 - Z])


def shap_weights(Z: np.ndarray, y: np.ndarray, f_full: float, f_empty: float) -> dict:
    """Kernel SHAP regression with the efficiency constraint sum(phi) = f_full - f_empty.

    Z must be sampled with `shap_samples` (the kernel is in the sampling, so the regression is unweighted).
    """
    Z = Z.astype(np.float64)
    M = Z.shape[1]
    delta = f_full - f_empty
    if M == 1:
        return {"weights": np.array([delta]), "base": f_empty, "additivity_error": 0.0}
    y_adj = y - f_empty - Z[:, -1] * delta
    X = Z[:, :-1] - Z[:, -1:]
    phi = np.linalg.lstsq(X, y_adj, rcond=None)[0]
    phi = np.append(phi, delta - phi.sum())
    return {"weights": phi, "base": float(f_empty), "additivity_error": float(abs(phi.sum() - delta))}


def default_samples(device) -> int:
    return 300 if getattr(device, "type", str(device)) == "cuda" else 120


def attribution(pipe, a: dict, threshold: float, method: str = "lime", n_samples=None, n_segments: int = 30,
                seed: int = 0, batch_size: int = 32) -> dict:
    """LIME or Kernel SHAP over superpixels of the region the classifiers look at.

    Scores are the ensemble mean P(malignant) WITHOUT flip-averaging (to halve the cost); the explained
    score `score_explained` is reported next to the pipeline's score.
    """
    if method not in ("lime", "shap"):
        raise ValueError("method must be 'lime' or 'shap'")
    if not a["lesion_found"]:
        raise ValueError("no lesion found: there is no classification to explain")
    n_samples = int(n_samples or default_samples(pipe.cls.device))
    gray = a["gray"]
    boxes = _member_boxes(pipe, a)
    R = _union(boxes)
    rx0, ry0, rx1, ry1 = R
    region = gray[ry0:ry1, rx0:rx1]
    rmask = a["mask"][ry0:ry1, rx0:rx1]
    labels = superpixels(region, n_segments, rmask, seed)
    M = int(labels.max()) + 1
    sigma = max(3.0, 0.08 * max(region.shape))
    blurred = cv2.GaussianBlur(region, (0, 0), sigma)
    rng = np.random.default_rng(seed)

    if method == "lime":
        Z = rng.integers(0, 2, size=(n_samples, M), dtype=np.uint8)
        Z[0] = 1
    else:
        Z = shap_samples(M, max(2, n_samples - 2), rng)
        Z = np.concatenate([np.ones((1, M), np.uint8), np.zeros((1, M), np.uint8), Z])

    def score(rows: np.ndarray) -> np.ndarray:
        out = np.zeros(len(rows), dtype=np.float64)
        for i in range(0, len(rows), batch_size):
            zs = rows[i:i + batch_size]
            imgs = [np.where(z[labels].astype(bool), region, blurred) for z in zs]
            member = []
            for c, (x0, y0, x1, y1) in zip(pipe.clss, boxes):
                patches = [im[y0 - ry0:y1 - ry0, x0 - rx0:x1 - rx0] for im in imgs]
                member.append(c.predict_patches(patches, batch_size=batch_size))
            out[i:i + len(zs)] = np.mean(member, axis=0)
        return out

    y = score(Z)
    f_full = float(y[0])
    if method == "lime":
        fit = lime_weights(Z, y)
        quality = {"fit_r2": fit["fit_r2"]}
        base = fit["intercept"]
    else:
        f_empty = float(y[1])
        fit = shap_weights(Z[2:], y[2:], f_full, f_empty)
        quality = {"additivity_error": fit["additivity_error"]}
        base = fit["base"]
    weights = fit["weights"]

    # deletion check: blur the (up to) 3 regions that most support the predicted class
    predicted_malignant = a["prob_malignant"] >= threshold
    support = weights if predicted_malignant else -weights
    order = np.argsort(-support)
    top = [int(j) for j in order[:3] if support[j] > 0]
    z = np.ones(M, np.uint8)
    z[top] = 0
    f_deleted = float(score(z[None])[0]) if top else f_full

    regions = region_map(rmask)
    seg_region = np.array([np.bincount(regions[labels == j], minlength=3).argmax() for j in range(M)])
    seg_area = np.bincount(labels.ravel(), minlength=M) / labels.size
    pos, neg = np.clip(weights, 0, None), np.clip(-weights, 0, None)
    by_region = {r: {"towards_malignant": float(pos[seg_region == i].sum()),
                     "towards_benign": float(neg[seg_region == i].sum())} for i, r in enumerate(REGIONS)}
    cy, cx = _centroid(rmask)
    ranked = []
    for j in np.argsort(-np.abs(weights))[:5]:
        ys, xs = np.nonzero(labels == j)
        ranked.append({"rank": len(ranked) + 1, "effect": float(weights[j]),
                       "direction": "towards malignant" if weights[j] > 0 else "towards benign",
                       "location": _describe(int(seg_region[j]), ys.mean() - cy, xs.mean() - cx),
                       "area_pct_of_region": float(100 * seg_area[j])})

    # full-image signed map (for display): each superpixel painted with its weight
    H, W = gray.shape
    amap = np.zeros((H, W), np.float32)
    amap[ry0:ry1, rx0:rx1] = weights[labels]
    return {"map": amap, "region_box": [int(v) for v in R], "labels": labels, "weights": weights,
            "metrics": {
                "method": method, "n_samples": int(len(Z)), "n_regions": M,
                "score_explained": f_full, "score_all_regions_hidden": base if method == "shap" else None,
                "total_towards_malignant": float(pos.sum()), "total_towards_benign": float(neg.sum()),
                "by_region": by_region, "top_regions": ranked,
                "deletion_check": {"regions_hidden": len(top), "score_before": f_full, "score_after": f_deleted,
                                   "change": f_deleted - f_full},
                **quality}}


def _centroid(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return mask.shape[0] / 2, mask.shape[1] / 2
    return ys.mean(), xs.mean()


def _describe(region: int, dy: float, dx: float) -> str:
    if region == 0:
        return "inside the lesion"
    if region == 1:
        return "on the lesion edge"
    # image rows grow with depth: "below" = deeper = behind the lesion
    if abs(dy) >= abs(dx):
        return "tissue below (behind) the lesion" if dy > 0 else "tissue above the lesion (towards the skin)"
    return "tissue to the right of the lesion" if dx > 0 else "tissue to the left of the lesion"


# ------------------------------------------------------------------------------------- lesion description

def lesion_measurements(gray: np.ndarray, mask: np.ndarray):
    """Simple, descriptive measurements of the OUTLINED region (not model inputs, not a diagnosis)."""
    raw = mask.astype(bool)
    m = keep_largest_component(raw)
    area = int(m.sum())
    if area == 0:
        return None
    H, W = m.shape
    n_regions = int(cv2.connectedComponents(raw.astype(np.uint8), connectivity=8)[0] - 1)
    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    bh, bw = y1 - y0, x1 - x0
    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea)
    perim = cv2.arcLength(c, True)
    hull = cv2.contourArea(cv2.convexHull(c))
    k = max(3, int(round(0.25 * math.sqrt(area))))
    ring = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
                      ).astype(bool) & ~raw
    g = gray.astype(np.float64)
    inside_mean = float(g[m].mean())
    ring_mean = float(g[ring].mean()) if ring.any() else None
    below = g[y1:min(H, y1 + max(1, bh // 2)), x0:x1]
    posterior = float(below.mean() / ring_mean) if below.size and ring_mean else None
    cy, cx = ys.mean() / H, xs.mean() / W
    return {
        "area_pct_of_image": float(100 * area / m.size),
        "width_px": int(bw), "depth_px": int(bh),
        "depth_to_width": float(bh / bw),
        "circularity": float(min(1.0, 4 * math.pi * area / perim ** 2)) if perim > 0 else None,
        "solidity": float(min(1.0, area / hull)) if hull > 0 else None,
        "brightness_vs_surrounding": float(inside_mean / ring_mean) if ring_mean else None,
        "brightness_behind_vs_surrounding": posterior,
        "position": {"horizontal": "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "centre",
                     "vertical": "upper (near the skin)" if cy < 1 / 3 else "lower (deep)" if cy > 2 / 3 else "middle"},
        "separate_regions_outlined": n_regions,
        "touches_image_border": bool(y0 == 0 or x0 == 0 or y1 == H or x1 == W),
    }


def _pairwise_dice(masks) -> float | None:
    if len(masks) < 2:
        return None
    vals = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            a, b = masks[i], masks[j]
            s = a.sum() + b.sum()
            vals.append(2 * (a & b).sum() / s if s else 1.0)
    return float(np.mean(vals))


def segmentation_confidence(pipe, a: dict) -> dict:
    p, m = a["prob_map"], a["mask"]
    thr = pipe.seg.threshold
    out = {"models": len(a["member_maps"]), "threshold": float(thr),
           "max_lesion_probability": float(p.max())}
    if m.any():
        band = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        unsure = (p >= UNCERTAIN[0]) & (p <= UNCERTAIN[1]) & band
        out["mean_probability_inside_outline"] = float(p[m].mean())
        out["uncertain_pixels_vs_lesion_area"] = float(unsure.sum() / m.sum())
    out["member_agreement_dice"] = _pairwise_dice([mm >= thr for mm in a["member_maps"]])
    return out


# ------------------------------------------------------------------------------------------------ report

def _reference(settings: dict) -> dict:
    names = {"deploy/test": "held-out test images (BUSI + Breast-Lesions-USG + BUS-UCLM)",
             "A/test": "BUSI test images", "A/external": "BUS-BRA images from another hospital (external)"}
    out = {}
    for k, v in (settings.get("metrics") or {}).items():
        c = v.get("classification") or {}
        entry = {"description": names.get(k, k)}
        if c:
            entry.update({"images": c.get("n"), "auc": c.get("auc"),
                          "sensitivity_cancers_flagged": c.get("recall"),
                          "specificity_benign_not_flagged": c.get("specificity")})
        ns = v.get("normal_scans")
        if ns:
            entry["normal_scans_correctly_clear"] = ns.get("correctly_clear_rate")
        out[k] = entry
    return out


def build_report(pipe, a: dict, settings: dict, explanations: dict | None = None) -> dict:
    """Everything the UI and the language model need, as plain JSON-able numbers, labels and flags."""
    thr = float(settings.get("threshold", 0.5))
    explanations = explanations or {}
    found = bool(a["lesion_found"])
    p = a["prob_malignant"]
    probs = a["member_probs"]
    votes = sum(q >= thr for q in probs)
    seg = segmentation_confidence(pipe, a)
    lesion = lesion_measurements(a["gray"], a["mask"]) if found else None

    if not found:
        decision, level = "No lesion detected", "none"
    elif p >= thr:
        decision, level = "Suspicious for malignancy: further assessment recommended", "high"
    else:
        decision, level = "Lesion with benign-looking features", "low"

    flags = []
    if not found:
        flags.append(("no_lesion", "The segmentation models did not outline any lesion."))
    else:
        if abs(p - thr) < BORDERLINE:
            flags.append(("borderline_score", "The score is close to the decision threshold."))
        if 0 < votes < len(probs):
            flags.append(("models_disagree", f"{votes} of {len(probs)} classification models are above the threshold."))
        if lesion and lesion["area_pct_of_image"] < SMALL_LESION_PCT:
            flags.append(("small_lesion", "The outlined lesion is small; in testing, most missed cancers were small lesions."))
        if lesion and lesion["area_pct_of_image"] > 50:
            flags.append(("very_large_region", "The outlined region covers more than half of the image (unusual)."))
        if lesion and lesion["separate_regions_outlined"] > 1:
            flags.append(("several_regions", "More than one region was outlined; only the largest was classified."))
        if lesion and lesion["touches_image_border"]:
            flags.append(("touches_border", "The lesion touches the image border; part of it may be outside the image."))
        if seg.get("uncertain_pixels_vs_lesion_area", 0) > 0.5:
            flags.append(("uncertain_outline", "The lesion border is not sharply defined in the segmentation."))
        if seg.get("member_agreement_dice") is not None and seg["member_agreement_dice"] < 0.7:
            flags.append(("outline_disagreement", "The segmentation models disagree about the lesion outline."))
    if min(a["gray"].shape) < 128:
        flags.append(("small_image", "The image resolution is low."))

    return _finite({
        "prediction": {
            "lesion_found": found, "decision": decision, "level": level,
            "score_malignant": p if found else None, "threshold": thr,
            "distance_from_threshold": (p - thr) if found else None,
            "score_is_calibrated_probability": False,
            "member_scores": probs if found else None,
            "models_above_threshold": f"{votes}/{len(probs)}" if found else None,
            "threshold_rule": settings.get("notes") or "threshold chosen on validation data",
        },
        "segmentation": seg,
        "lesion_measurements": lesion,
        "explanations": {k: v["metrics"] for k, v in explanations.items() if v and "metrics" in v},
        "flags": [{"code": c, "text": t} for c, t in flags],
        "reference_performance": _reference(settings),
        "system": {"segmentation_models": len(pipe.segs), "classification_models": len(pipe.clss),
                   "flip_averaging": bool(pipe.tta),
                   "training_data": "BUSI, Breast-Lesions-USG and BUS-UCLM (public research datasets)"},
    })


def _finite(o):
    """NaN / inf -> None and numpy scalars -> Python (JSON cannot hold NaN)."""
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ---------------------------------------------------------------------------------------------- pictures

def render_cam(gray: np.ndarray, cam: np.ndarray, box=None) -> np.ndarray:
    """Grad-CAM heat map over the image (BGR uint8)."""
    heat = cv2.applyColorMap((np.clip(cam, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    alpha = (0.15 + 0.45 * np.clip(cam, 0, 1))[..., None]
    out = (base * (1 - alpha) + heat * alpha).astype(np.uint8)
    if box is not None:
        x0, y0, x1, y1 = box
        cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 255), max(1, round(max(gray.shape) / 400)))
    return out


def render_attribution(gray: np.ndarray, amap: np.ndarray, region_box, mask=None) -> np.ndarray:
    """Signed region map (BGR uint8): red = towards malignant, blue = towards benign."""
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    scale = float(np.abs(amap).max())
    if scale > 0:
        v = np.clip(amap / scale, -1, 1)
        red = np.array([40, 40, 230], np.float32)    # BGR
        blue = np.array([230, 120, 30], np.float32)
        color = np.where(v[..., None] > 0, red, blue)
        alpha = (0.65 * np.abs(v))[..., None]
        base = base * (1 - alpha) + color * alpha
    out = base.astype(np.uint8)
    thick = max(1, round(max(gray.shape) / 400))
    if mask is not None and mask.any():
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (255, 255, 255), thick)
    x0, y0, x1, y1 = region_box
    cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), (200, 200, 200), thick)
    return out


__all__ = ["analyse", "gradcam", "attribution", "build_report", "lesion_measurements", "superpixels",
           "lime_weights", "shap_samples", "shap_weights", "region_map", "render_cam", "render_attribution"]
