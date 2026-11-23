"""Dashboard backend: upload a breast ultrasound image -> prediction, explanations, plain-language summary.

Run (from the repository root, with the app requirements installed):
    uvicorn app.server:app --host 0.0.0.0 --port 7860
then open http://localhost:7860

Environment variables:
    BUS_BUNDLE         folder exported by `python -m bus.bundle export` (default: ./deploy_bundle)
    BUS_DEVICE         cpu | cuda (default: cuda if available)
    BUS_XAI_SAMPLES    perturbations for LIME / SHAP (default 120 on CPU, 300 on GPU)
    ANTHROPIC_API_KEY  enables the language-model summary (see bus/narrate.py); without it a template is used
    BUS_LLM, BUS_LLM_MODEL, BUS_LLM_EFFORT   see bus/narrate.py

API (one "case" per uploaded image, kept in memory only, never written to disk):
    GET  /api/health                  is the model loaded?
    GET  /api/info                    model settings, reference metrics, language-model status
    POST /api/predict                 form field "file" -> prediction + Grad-CAM + report, returns case_id
    POST /api/explain/{case_id}       ?method=lime|shap -> region-based explanation
    POST /api/summary/{case_id}       JSON {"language": "English", "audience": "patient"|"clinician"}
    GET  /api/report/{case_id}        the full JSON report of a case

RESEARCH PROTOTYPE - not a medical device, not for clinical decisions.
"""
from __future__ import annotations

import base64
import math
import os
import sys
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # allow running without `pip install -e .`
    sys.path.insert(0, str(ROOT / "src"))

from bus import explain as xai  # noqa: E402
from bus import narrate  # noqa: E402
from bus.bundle import load_bundle  # noqa: E402

BUNDLE = Path(os.environ.get("BUS_BUNDLE", ROOT / "deploy_bundle"))
STATIC = Path(__file__).resolve().parent / "static"
MAX_BYTES = 20 * 2 ** 20
MAX_SIDE = 1600  # larger images are downscaled first (the models work at 224-256 px anyway)
MAX_CASES = 16   # most recent cases kept in memory for the explain / summary steps
DISCLAIMER = ("Research prototype for a Master's thesis. Not a medical device. Results must not be used "
              "for diagnosis or treatment decisions; every image needs review by a qualified clinician.")

_lock = threading.Lock()          # one model computation at a time (predictable memory on small servers)
_cases_lock = threading.Lock()
_cases: "OrderedDict[str, dict]" = OrderedDict()
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


def _pipe():
    pipe = _load()
    if pipe is None:
        raise HTTPException(503, f"models not loaded: {_state['error']}")
    return pipe


@asynccontextmanager
async def _lifespan(_app):
    _load()  # load the models once, when the server starts
    yield


app = FastAPI(title="Breast ultrasound lesion analysis", version="2.0", lifespan=_lifespan,
              description="Dual-stage segmentation + classification with explanations (research prototype).")


def _threshold() -> float:
    return float((_state["settings"] or {}).get("threshold", 0.5))


def _xai_samples():
    v = os.environ.get("BUS_XAI_SAMPLES")
    return int(v) if v else None


def _public_settings() -> dict:
    s = _state["settings"] or {}
    return {"threshold": s.get("threshold"), "tta": s.get("tta"), "post": s.get("post"),
            "segmentation_models": [m.get("name") for m in s.get("seg", [])],
            "classification_models": [m.get("name") for m in s.get("cls", [])],
            "created": s.get("created"), "notes": s.get("notes"), "versions": s.get("versions")}


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


# --------------------------------------------------------------------------------------------- cases

def _store(case: dict) -> str:
    cid = uuid.uuid4().hex
    with _cases_lock:
        _cases[cid] = case
        while len(_cases) > MAX_CASES:
            _cases.popitem(last=False)
    return cid


def _case(cid: str) -> dict:
    with _cases_lock:
        case = _cases.get(cid)
        if case is not None:
            _cases.move_to_end(cid)
    if case is None:
        raise HTTPException(404, "case not found (cases are kept in memory only; upload the image again)")
    return case


def _refresh_report(case: dict) -> dict:
    case["report"] = xai.build_report(_pipe(), case["analysis"], _state["settings"] or {}, case["explanations"])
    return case["report"]


# -------------------------------------------------------------------------------------------- routes

@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok" if _state["pipe"] is not None else "error", "models_loaded": _state["pipe"] is not None,
            "bundle": str(BUNDLE), "error": _state["error"]}


@app.get("/api/info")
def info():
    pipe = _pipe()
    return _finite({**_public_settings(), "reference_metrics": (_state["settings"] or {}).get("metrics", {}),
                    "llm": narrate.status(),
                    "explanations": {"methods": ["gradcam", "lime", "shap"],
                                     "samples": _xai_samples() or xai.default_samples(pipe.cls.device)},
                    "disclaimer": DISCLAIMER})


@app.post("/api/predict")
def predict(file: UploadFile = File(...), gradcam: bool = Query(True)):
    pipe = _pipe()
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

    thr = _threshold()
    t0 = time.time()
    with _lock:
        a = xai.analyse(pipe, gray)
        t_pred = time.time()
        cam = xai.gradcam(pipe, a, thr) if gradcam else None
    elapsed = (t_pred - t0) * 1000
    case = {"analysis": a, "explanations": {}, "created": time.time(), "name": file.filename}
    images = {"overlay": None, "heatmap": None, "gradcam": None}
    if cam is not None:
        case["explanations"]["gradcam"] = {"metrics": cam["metrics"]}
        images["gradcam"] = _png_data_url(xai.render_cam(gray, cam["map"], a["box"]))
    report = _refresh_report(case)
    cid = _store(case)

    p, found = float(a["prob_malignant"]), bool(a["lesion_found"])
    pr = report["prediction"]
    mask = a["mask"]
    images["overlay"] = _png_data_url(_overlay(gray, mask, a["lesion_box"], found and p >= thr))
    images["heatmap"] = _png_data_url(_heatmap(gray, a["prob_map"]))
    return {
        "case_id": cid,
        "lesion_found": found,
        "prob_malignant": p if found else None,
        "threshold": thr,
        "decision": pr["decision"],
        "level": pr["level"],
        "lesion_box": [int(v) for v in a["lesion_box"]] if found else None,
        "lesion_area_pct": round(100.0 * float(mask.mean()), 2),
        "image_size": [int(gray.shape[1]), int(gray.shape[0])],
        "overlay": images["overlay"],
        "heatmap": images["heatmap"],
        "gradcam": images["gradcam"],
        "elapsed_ms": round(elapsed),
        "explain_ms": round((time.time() - t_pred) * 1000),
        "report": report,
        "model": _public_settings(),
        "disclaimer": DISCLAIMER,
    }


@app.post("/api/explain/{case_id}")
def explain(case_id: str, method: str = Query("lime", pattern="^(lime|shap)$"),
            samples: Optional[int] = Query(None, ge=16, le=2000), segments: int = Query(30, ge=6, le=120)):
    pipe = _pipe()
    case = _case(case_id)
    a = case["analysis"]
    if not a["lesion_found"]:
        raise HTTPException(409, "no lesion found: there is no classification to explain")
    t0 = time.time()
    with _lock:
        res = xai.attribution(pipe, a, _threshold(), method, n_samples=samples or _xai_samples(),
                              n_segments=segments)
    case["explanations"][method] = {"metrics": res["metrics"]}
    report = _refresh_report(case)
    return {"case_id": case_id, "method": method,
            "image": _png_data_url(xai.render_attribution(a["gray"], res["map"], res["region_box"], a["mask"])),
            "metrics": report["explanations"][method], "report": report,
            "elapsed_ms": round((time.time() - t0) * 1000)}


@app.post("/api/summary/{case_id}")
def summary(case_id: str, body: dict = Body(default={})):
    case = _case(case_id)
    language = str(body.get("language") or "English")
    audience = str(body.get("audience") or "patient")
    out = narrate.summarize(case["report"], language=language, audience=audience)
    case["summary"] = out
    return {"case_id": case_id, **out, "disclaimer": DISCLAIMER}


@app.get("/api/report/{case_id}")
def report(case_id: str):
    case = _case(case_id)
    return {"case_id": case_id, "report": case["report"], "summary": case.get("summary")}
