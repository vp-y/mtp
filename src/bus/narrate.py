"""Plain-language explanation of one analysis, written by a language model (Claude) from the numbers.

The model receives ONLY the JSON report produced by `bus.explain.build_report` (scores, explanation
metrics, measurements, flags, reference performance). It never sees the image or any patient data,
and it is instructed to use nothing but those numbers.

Configuration (environment variables):
    ANTHROPIC_API_KEY  API key from console.anthropic.com. Without it the app uses the built-in
                       template text below (no network calls).
    BUS_LLM            "off" disables the language model even if a key is set.
    BUS_LLM_MODEL      default "claude-opus-5-5"
    BUS_LLM_EFFORT     low | medium | high (default "medium")

Usage:
    from bus.narrate import summarize
    out = summarize(report, language="English", audience="patient")
    out["text"]    Markdown with five sections
    out["source"]  "llm" or "template"
"""
from __future__ import annotations

import json
import os
import time

DEFAULT_MODEL = "claude-opus-5-5"
AUDIENCES = {
    "patient": "a patient or family member with no medical training. Use everyday words, short sentences, "
               "and explain any technical term the first time you use it.",
    "clinician": "a clinician (radiologist, surgeon or GP). Technical terms such as Grad-CAM, sensitivity, "
                 "specificity, echogenicity and posterior features may be used without explanation.",
}
SECTIONS = ("Summary", "What the system found", "Why the system decided this",
            "How sure the system is", "What happens next")

SYSTEM_PROMPT = """You write the plain-language explanation panel of a research prototype that analyses breast \
ultrasound images with AI. The system has two stages: segmentation models outline a lesion, then \
classification models look at the region around the outline and give a score between 0 and 1; a score at \
or above the decision threshold is labelled "suspicious for malignancy", below it "benign-looking".

You receive one JSON report. Explain it faithfully to the reader described in the request.

Ground rules:
- Use only facts in the JSON. Never invent numbers, findings, history or diagnoses. If something is not in the \
data, do not guess it. Round numbers sensibly (two significant digits, or whole percentages).
- The score is NOT a calibrated probability ("score_is_calibrated_probability": false). Never say "an X% \
chance of cancer". Say how the score compares with the threshold instead.
- Never tell the reader that they have or do not have cancer. Describe what the system flagged. The system is \
a research prototype, not a medical device, and is not a diagnosis.
- Explain the explanation methods in simple terms when they are present:
  * Grad-CAM: a heat map of the parts of the image the classifier relied on most for its decision. \
"focus_inside_or_on_lesion" is the share of that attention on the lesion or its edge.
  * LIME and SHAP: "hide and test" checks. Small areas of the image are blurred out in many combinations and \
the system measures how the score changes. Positive effects push the score towards suspicious, negative ones \
towards benign-looking. SHAP effects add up to the difference between the score of the full image and the \
score with every area hidden.
  * deletion_check: the score after the most influential areas were hidden. A large change means those areas \
really drove the decision; a small change means the decision is spread over many areas.
  * fit_r2 (LIME): how well the simple LIME model reproduces the system's behaviour (closer to 1 = more \
trustworthy explanation).
- "lesion_measurements" are simple measurements of the outlined region (shape, orientation, brightness \
compared with the tissue around it, brightness behind it). They are NOT inputs to the AI and NOT a \
radiologist's assessment. You may mention that radiologists look at properties like these, but never turn them \
into a diagnosis or a BI-RADS category. depth_to_width > 1 means the outline is taller than wide; circularity \
and solidity close to 1 mean a round, smooth outline; lower values mean an irregular outline; \
brightness_vs_surrounding < 1 means darker than the surrounding tissue.
- Be honest about reliability: mention every item in "flags", the agreement between models, and the \
reference performance (for example: "in testing on N images, the system flagged about 9 in 10 cancers, but \
also flagged about 6 in 10 benign lesions"). Use the external (other hospital) result when it exists.
- If no lesion was found, explain that no classification was made, what that can mean (no visible lesion, or \
a lesion the system missed), and use the normal-scan reference results if present.
- Calm, kind, non-alarming tone. No exclamation marks. No emojis.

Format: Markdown. Use exactly these five level-2 headings in this order, translated into the requested \
language: "Summary", "What the system found", "Why the system decided this", "How sure the system is", \
"What happens next". Under "Summary" write 2-3 sentences. Bullet points are allowed. 250-420 words in total. \
"What happens next" must say that a qualified clinician has to review the images and that this tool must not \
be used for diagnosis or treatment decisions. Output only the explanation."""


def status() -> dict:
    """Is the language model configured? (shown on /api/info)"""
    model = os.environ.get("BUS_LLM_MODEL", DEFAULT_MODEL)
    if os.environ.get("BUS_LLM", "").lower() in ("off", "0", "false", "no"):
        return {"enabled": False, "model": None, "reason": "disabled with BUS_LLM=off"}
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return {"enabled": False, "model": None, "reason": "no ANTHROPIC_API_KEY set: using the built-in template"}
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return {"enabled": False, "model": None, "reason": "package 'anthropic' not installed: pip install anthropic"}
    return {"enabled": True, "model": model, "reason": None}


def _user_message(report: dict, language: str, audience: str) -> str:
    return (f"Reader: {AUDIENCES.get(audience, AUDIENCES['patient'])}\n"
            f"Language: write the whole explanation in {language}.\n\n"
            f"Analysis report (JSON):\n```json\n{json.dumps(report, indent=1, ensure_ascii=False)}\n```")


def _call_llm(report: dict, language: str, audience: str, model: str) -> dict:
    import anthropic

    client = anthropic.Anthropic(timeout=120.0, max_retries=2)
    kwargs = dict(
        model=model,
        max_tokens=16000,
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        output_config={"effort": os.environ.get("BUS_LLM_EFFORT", "medium")},
        messages=[{"role": "user", "content": _user_message(report, language, audience)}],
    )
    try:  # server-side fallback model if the main model declines (needs a recent SDK)
        resp = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
    except (TypeError, anthropic.BadRequestError):  # SDK or model without the fallback option
        resp = client.messages.create(**kwargs)
    if resp.stop_reason == "refusal":
        raise RuntimeError("the language model declined this request")
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    if not text:
        raise RuntimeError(f"empty response (stop reason: {resp.stop_reason})")
    if resp.stop_reason == "max_tokens":
        text += "\n\n*(The explanation was cut off.)*"
    usage = getattr(resp, "usage", None)
    return {"text": text, "model": getattr(resp, "model", model),
            "usage": {"input_tokens": getattr(usage, "input_tokens", None),
                      "output_tokens": getattr(usage, "output_tokens", None)}}


def summarize(report: dict, language: str = "English", audience: str = "patient") -> dict:
    """-> {"text", "source": "llm" | "template", "model", "note", "elapsed_ms", "usage"}.

    Falls back to the template whenever the language model is not configured or the call fails, so the
    panel always shows an explanation; `note` says why.
    """
    audience = audience if audience in AUDIENCES else "patient"
    language = (language or "English").strip()[:40] or "English"
    t0 = time.time()
    st = status()
    note = st["reason"]
    if st["enabled"]:
        import anthropic
        try:
            out = _call_llm(report, language, audience, st["model"])
            return {"text": out["text"], "source": "llm", "model": out["model"], "note": None,
                    "usage": out["usage"], "elapsed_ms": round(1000 * (time.time() - t0))}
        except anthropic.AuthenticationError:
            note = "the API key was rejected (check ANTHROPIC_API_KEY)"
        except anthropic.PermissionDeniedError:
            note = "the API key has no access to this model"
        except anthropic.NotFoundError:
            note = f"model '{st['model']}' not found (check BUS_LLM_MODEL)"
        except anthropic.RateLimitError:
            note = "rate limit reached, try again in a minute"
        except anthropic.APIStatusError as e:
            note = f"language model error {e.status_code}"
        except anthropic.APIConnectionError:
            note = "could not reach the language model (no internet connection?)"
        except (RuntimeError, TypeError, ValueError) as e:
            note = f"language model unavailable: {e}"
        note = f"Template text shown because {note}."
    elif note:
        note = f"Template text shown: {note}."
    if language.lower() not in ("english", "en"):
        note = (note + " " if note else "") + "The template is only available in English."
    return {"text": template_summary(report, audience), "source": "template", "model": None, "note": note,
            "usage": None, "elapsed_ms": round(1000 * (time.time() - t0))}


# -------------------------------------------------------------------------------------------- template

def _pct(x, digits=0):
    return "n/a" if x is None else f"{100 * x:.{digits}f} %"


def _ref_sentence(report: dict) -> str | None:
    ref = report.get("reference_performance") or {}
    for key in ("A/external", "deploy/test", "A/test"):
        r = ref.get(key) or {}
        if r.get("sensitivity_cancers_flagged") is not None:
            return (f"In testing on {r.get('images', 'n/a')} {r['description']}, the system flagged "
                    f"{_pct(r['sensitivity_cancers_flagged'])} of cancers and correctly left "
                    f"{_pct(r.get('specificity_benign_not_flagged'))} of benign lesions unflagged.")
    return None


def template_summary(report: dict, audience: str = "patient") -> str:
    """Rule-based explanation with the same five sections (used when no language model is available)."""
    pr = report.get("prediction", {})
    seg = report.get("segmentation", {})
    les = report.get("lesion_measurements") or {}
    ex = report.get("explanations", {})
    flags = report.get("flags", [])
    ref = _ref_sentence(report)
    L = []

    L.append("## Summary")
    if not pr.get("lesion_found"):
        L.append("The system did not outline any lesion in this image, so it did not assess whether a lesion "
                 "looks benign or suspicious. This is not proof that the breast is normal: the system can miss "
                 "lesions, especially small ones.")
    else:
        s, t = pr["score_malignant"], pr["threshold"]
        word = "above" if pr["level"] == "high" else "below"
        label = "suspicious" if pr["level"] == "high" else "benign-looking"
        L.append(f"The system outlined one lesion and rated it **{label}**. Its score was {s:.2f}, {word} the "
                 f"decision threshold of {t:.2f}. This is an automatic research result, not a diagnosis.")

    L.append("\n## What the system found")
    if pr.get("lesion_found"):
        L.append(f"- An outlined region covering about {les.get('area_pct_of_image', 0):.1f} % of the image, "
                 f"in the {les.get('position', {}).get('vertical', '?')} {les.get('position', {}).get('horizontal', '?')} "
                 "part of the image.")
        if les.get("depth_to_width") is not None:
            shape = "taller than it is wide" if les["depth_to_width"] > 1 else "wider than it is tall"
            L.append(f"- The outline is {shape} (depth-to-width ratio {les['depth_to_width']:.2f}).")
        if les.get("solidity") is not None:
            smooth = "smooth and regular" if les["solidity"] > 0.9 else "somewhat irregular"
            L.append(f"- Its border is {smooth} (solidity {les['solidity']:.2f}; 1 = perfectly convex).")
        if les.get("brightness_vs_surrounding") is not None:
            b = les["brightness_vs_surrounding"]
            L.append(f"- It appears {'darker' if b < 1 else 'brighter'} than the tissue around it "
                     f"(brightness ratio {b:.2f}).")
        L.append("- These are simple measurements of the outline, not part of the AI decision.")
    else:
        L.append(f"- The highest lesion probability anywhere in the image was {seg.get('max_lesion_probability', 0):.2f}, "
                 f"below the {seg.get('threshold', 0.5):.2f} needed to outline a region.")

    L.append("\n## Why the system decided this")
    if not pr.get("lesion_found"):
        L.append("Without an outlined lesion there is no classification to explain.")
    else:
        g = ex.get("gradcam")
        if g:
            L.append(f"- **Attention map (Grad-CAM):** about {_pct(g.get('focus_inside_or_on_lesion'))} of the "
                     "classifier's attention was on the lesion or its edge.")
        for name in ("lime", "shap"):
            m = ex.get(name)
            if not m:
                continue
            top = (m.get("top_regions") or [None])[0]
            title = "LIME" if name == "lime" else "SHAP"
            if top:
                L.append(f"- **{title} (hide-and-test):** the most influential area was {top['location']}, "
                         f"pushing the score {top['direction']} ({top['effect']:+.2f}).")
            d = m.get("deletion_check") or {}
            if d.get("regions_hidden"):
                L.append(f"  Hiding the {d['regions_hidden']} most influential areas changed the score from "
                         f"{d['score_before']:.2f} to {d['score_after']:.2f}.")
        if not (g or ex.get("lime") or ex.get("shap")):
            L.append("No explanation methods were run for this image.")

    L.append("\n## How sure the system is")
    if pr.get("lesion_found"):
        L.append(f"- {pr.get('models_above_threshold', '?')} classification models were above the threshold.")
        if pr.get("distance_from_threshold") is not None:
            d = pr["distance_from_threshold"]
            L.append(f"- The score is {abs(d):.2f} {'above' if d >= 0 else 'below'} the threshold"
                     + (" (borderline)." if abs(d) < 0.1 else "."))
    for f in flags:
        L.append(f"- Note: {f['text']}")
    if ref:
        L.append(f"- {ref}")
    L.append("- The score is a model score, not the chance of cancer.")

    L.append("\n## What happens next")
    L.append("A qualified clinician must review the images together with the clinical findings. This tool is a "
             "research prototype and must not be used for diagnosis or treatment decisions.")
    return "\n".join(L)
