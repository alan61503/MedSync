"""
MedSync LLM Service – Groq-powered 2-Factor AI Verification
Sends inference results to LLaMA 3.3 70B via Groq for secondary clinical validation.
"""

import os
import json
import time
import requests as _requests


GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"


def _get_api_key() -> str:
    """Read Groq API key from environment, supporting .env files."""
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        # Try loading from .env manually
        env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
        env_path = os.path.normpath(env_path)
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line.startswith("GROQ_API_KEY="):
                        key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
    return key


def _build_verification_prompt(predictions: dict, findings: dict, meta: dict) -> str:
    """Build a structured clinical verification prompt for LLaMA."""
    ost_score = predictions.get("osteoporosis", 0)
    ost_pct = round(ost_score * 100, 1)
    t_score = round(-1.0 - (ost_score * 2.2), 1)
    cortical = round(findings.get("Cortical Bone Thinning", 0) * 100, 1)
    trabecular = round(findings.get("Trabecular Microarchitecture Degradation", 0) * 100, 1)
    bmd = round(findings.get("Bone Mineral Density (BMD) Attenuation", 0) * 100, 1)
    fracture = round(findings.get("Fragility Fracture Indicator", 0) * 100, 1)

    if ost_score >= 0.62:
        risk = "High Risk (Osteoporosis)"
    elif ost_score >= 0.35:
        risk = "Moderate Risk (Osteopenia)"
    else:
        risk = "Low Risk (Normal BMD)"

    return f"""You are a senior radiologist performing a 2-factor clinical verification of an AI-generated osteoporosis diagnosis.

## AI Primary Diagnosis (to be verified)
- Target Condition: Osteoporosis
- AI Probability Score: {ost_pct}%
- Estimated DEXA T-Score: {t_score} SD
- Risk Classification: {risk}

## Neural Network Attribution Metrics
- Cortical Bone Thinning: {cortical}%
- Trabecular Microarchitecture Degradation: {trabecular}%
- Bone Mineral Density (BMD) Attenuation: {bmd}%
- Fragility Fracture Indicator: {fracture}%

## Verification Task
Evaluate whether the above metrics are internally consistent and clinically plausible.
Apply standard WHO osteoporosis criteria (T-score ≤ -2.5 = Osteoporosis, -2.5 to -1.0 = Osteopenia, > -1.0 = Normal).

Return ONLY a valid JSON object with this exact structure (no markdown, no extra text):
{{
  "verified": true or false,
  "confidence": 0.0 to 1.0,
  "verdict": "CONFIRMED" or "FLAGGED" or "UNCERTAIN",
  "t_score_consistent": true or false,
  "clinical_alignment": "short explanation of whether metrics align with the diagnosis",
  "checklist": [
    {{"item": "Cortical Thinning vs Risk Level", "pass": true or false, "note": "brief note"}},
    {{"item": "Trabecular Degradation vs BMD", "pass": true or false, "note": "brief note"}},
    {{"item": "T-Score vs Fracture Risk", "pass": true or false, "note": "brief note"}},
    {{"item": "WHO Criteria Compliance", "pass": true or false, "note": "brief note"}}
  ],
  "recommendation": "one-sentence clinical follow-up recommendation"
}}"""


def verify_inference(inference: dict) -> dict:
    """
    Main entry point: call Groq LLaMA 3.3 70B to perform 2-factor verification.
    Returns a structured verification result dict.
    """
    api_key = _get_api_key()
    if not api_key:
        return _offline_fallback(inference, reason="GROQ_API_KEY not set")

    predictions = inference.get("predictions", {})
    findings = inference.get("supporting_findings", {})
    meta = {}

    prompt = _build_verification_prompt(predictions, findings, meta)

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert medical AI that validates radiology AI outputs. "
                    "You respond ONLY with valid JSON and nothing else. "
                    "Never include markdown code fences or extra text."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.15,
        "max_tokens": 800,
        "response_format": {"type": "json_object"},
    }

    start = time.time()
    try:
        resp = _requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "MedSync/1.0 (Medical AI Verification)",
            },
            json=payload,
            timeout=20,
            verify=False,  # bypass Windows SSL cert issues in dev
        )
        resp.raise_for_status()
        raw = resp.text
    except _requests.exceptions.RequestException as exc:
        return _offline_fallback(inference, reason=str(exc))

    elapsed = round(time.time() - start, 2)

    try:
        groq_resp = json.loads(raw)
        content = groq_resp["choices"][0]["message"]["content"]
        result = json.loads(content)
        result["_meta"] = {
            "model": GROQ_MODEL,
            "latency_s": elapsed,
            "source": "groq",
        }
        return result
    except Exception as parse_err:
        return _offline_fallback(inference, reason=f"Parse error: {parse_err}")


def analyse(predictions: dict, confidence_scores: dict, extra: dict) -> dict:
    """
    Legacy entry point referenced in patients.py osteoporosis-report endpoint.
    Builds a minimal inference dict and delegates to verify_inference.
    """
    inference = {
        "predictions": predictions,
        "supporting_findings": confidence_scores,
    }
    return verify_inference(inference)


def _offline_fallback(inference: dict, reason: str = "unavailable") -> dict:
    """Return a deterministic rule-based fallback when the API is unavailable."""
    predictions = inference.get("predictions", {})
    findings = inference.get("supporting_findings", {})
    ost_score = predictions.get("osteoporosis", 0)
    cortical = findings.get("Cortical Bone Thinning", 0)
    trabecular = findings.get("Trabecular Microarchitecture Degradation", 0)
    bmd = findings.get("Bone Mineral Density (BMD) Attenuation", 0)
    fracture = findings.get("Fragility Fracture Indicator", 0)

    t_score = -1.0 - (ost_score * 2.2)
    who_check = (ost_score >= 0.62 and t_score <= -2.5) or (0.35 <= ost_score < 0.62 and -2.5 < t_score <= -1.0) or (ost_score < 0.35 and t_score > -1.0)
    consistent = cortical > 0.4 and trabecular > 0.3 and bmd > 0.35
    verified = who_check and consistent

    return {
        "verified": verified,
        "confidence": round(0.55 + (0.2 if who_check else 0) + (0.15 if consistent else 0), 2),
        "verdict": "CONFIRMED" if verified else "UNCERTAIN",
        "t_score_consistent": who_check,
        "clinical_alignment": "Rule-based offline verification (Groq API unavailable).",
        "checklist": [
            {"item": "Cortical Thinning vs Risk Level", "pass": cortical > 0.4, "note": f"{round(cortical*100,1)}% thinning recorded"},
            {"item": "Trabecular Degradation vs BMD", "pass": trabecular > 0.3, "note": f"{round(trabecular*100,1)}% degradation"},
            {"item": "T-Score vs Fracture Risk", "pass": fracture > 0.3 if ost_score >= 0.35 else True, "note": f"T-Score {round(t_score,1)} SD"},
            {"item": "WHO Criteria Compliance", "pass": who_check, "note": "Based on DEXA T-score estimation"},
        ],
        "recommendation": "Manual physician review recommended due to offline verification mode.",
        "_meta": {"model": "offline-rules", "latency_s": 0, "source": "fallback", "reason": reason},
    }
