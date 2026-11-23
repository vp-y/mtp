"""Web app: upload a breast ultrasound image -> lesion outline + probability of malignancy.

Run (from the repository root, with the app requirements installed):
    uvicorn app.server:app --host 0.0.0.0 --port 7860
then open http://localhost:7860

Environment variables:
    BUS_BUNDLE   folder exported by `python -m bus.bundle export` (default: ./deploy_bundle)
    BUS_DEVICE   cpu | cuda (default: cuda if available)

API:
    GET  /api/health    is the model loaded?
    GET  /api/info      model settings and the reference metrics stored in the bundle
    POST /api/predict   multipart form field "file" (PNG / JPG / BMP / TIFF ultrasound image)

RESEARCH PROTOTYPE - not a medical device, not for clinical decisions.
"""
from __future__ import annotations

import base64
import math
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # allow running without `pip install -e .`
    sys.path.insert(0, str(ROOT / "src"))

from bus.bundle import load_bundle  # noqa: E402

BUNDLE = Path(os.environ.get("BUS_BUNDLE", ROOT / "deploy_bundle"))
STATIC = Path(__file__).resolve().parent / "static"
MAX_BYTES = 20 * 2 ** 20
MAX_SIDE = 1600  # larger images are downscaled first (the models work at 224-256 px anyway)
DISCLAIMER = ("Research prototype for a Master's thesis. Not a medical device. Results must not be used "
              "for diagnosis or treatment decisions; every image needs review by a qualified clinician.")

_lock = threading.Lock()
_state = {"pipe": None, "settings": None, "error": None}


def _load():
    if _state["pipe"] is None and _state["error"] is None:
        try:
            import torch
            dev = os.environ.get("BUS_DEVICE")
            device = torch.device(dev) if dev else None
            _state["pipe"], _state["settings"] = load_bundle(BUNDLE, device=device)
        except Exception as e:  # report on /api/health instead of crashing the server
            _state["error"] = f"{type(e).__name__}: {e}"
    return _state["pipe"]


@asynccontextmanager
async def _lifespan(_app):
    _load()  # load the models once, when the server starts
    yield


app = FastAPI(title="Breast ultrasound lesion analysis", version="1.0", lifespan=_lifespan,
              description="Dual-stage segmentation + classification (research prototype).")


def _public_settings() -> dict:
    s = _state["settings"] or {}
    return {"threshold": s.get("threshold"), "tta": s.get("tta"), "post": s.get("post"),
            "segmentation_models": [m.get("name") for m in s.get("seg", [])],
            "classification_models": [m.get("name") for m in s.get("cls", [])],
            "created": s.get("created"), "notes": s.get("notes"), "versions": s.get("versions")}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok" if _state["pipe"] is not None else "error", "models_loaded": _state["pipe"] is not None,
            "bundle": str(BUNDLE), "error": _state["error"]}


@app.get("/api/info")
def info():
    if _load() is None:
        raise HTTPException(503, f"models not loaded: {_state['error']}")
    return _finite({**_public_settings(), "reference_metrics": (_state["settings"] or {}).get("metrics", {}),
                    "disclaimer": DISCLAIMER})


def _finite(o):
    """NaN / inf -> None (JSON responses cannot contain them; e.g. AUC of a one-class subset)."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    return o


def _png_data_url(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise HTTPException(500, "could not encode image")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _overlay(gray: np.ndarray, mask: np.ndarray, box, malignant: bool) -> np.ndarray:
    color = (60, 60, 230) if malignant else (80, 200, 80)  # BGR: red-ish / green-ish
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if mask.any():
        fill = vis.copy()
        fill[mask] = color
        vis = cv2.addWeighted(fill, 0.25, vis, 0.75, 0)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        thick = max(1, round(max(gray.shape) / 300))
        cv2.drawContours(vis, contours, -1, color, thick)
        if box is not None:
            x0, y0, x1, y1 = box
            cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), (0, 215, 255), max(1, thick // 2))
    return vis


def _heatmap(gray: np.ndarray, prob_map: np.ndarray) -> np.ndarray:
    heat = cv2.applyColorMap((np.clip(prob_map, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.addWeighted(heat, 0.45, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 0.55, 0)


@app.post("/api/predict")
def predict(file: UploadFile = File(...)):
    pipe = _load()
    if pipe is None:
        raise HTTPException(503, f"models not loaded: {_state['error']}")
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "image larger than 20 MB")
    gray = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise HTTPException(400, "could not read the file as an image (use PNG, JPG, BMP or TIFF)")
    h, w = gray.shape
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        gray = cv2.resize(gray, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)

    t0 = time.time()
    with _lock:  # one inference at a time keeps memory use predictable on small servers
        out = pipe(gray)
    elapsed = (time.time() - t0) * 1000
    thr = float(_state["settings"]["threshold"])
    p = float(out["prob_malignant"])
    found = bool(out["lesion_found"])
    if not found:
        decision, level = "No lesion detected", "none"
    elif p >= thr:
        decision, level = "Suspicious for malignancy: further assessment recommended", "high"
    else:
        decision, level = "Lesion with benign-looking features", "low"
    mask = out["mask"]
    return {
        "lesion_found": found,
        "prob_malignant": p if found else None,
        "threshold": thr,
        "decision": decision,
        "level": level,
        "lesion_box": [int(v) for v in out["lesion_box"]] if found else None,
        "lesion_area_pct": round(100.0 * float(mask.mean()), 2),
        "image_size": [int(gray.shape[1]), int(gray.shape[0])],
        "overlay": _png_data_url(_overlay(gray, mask, out["lesion_box"], found and p >= thr)),
        "heatmap": _png_data_url(_heatmap(gray, out["prob_map"])),
        "elapsed_ms": round(elapsed),
        "model": _public_settings(),
        "disclaimer": DISCLAIMER,
    }
