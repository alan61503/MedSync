"""LLM service integration (Groq).

Loads `GROQ_API_KEY` from environment (or .env) and, when configured,
makes a minimal request to a configurable Groq inference endpoint.

If no key is present the service returns a safe stubbed response so callers
can continue to work during development.
"""
import os
from typing import Dict, Any

import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Configurable endpoint; set `GROQ_API_URL` in .env if your provider differs
GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.ai/v1/predict")


def _build_prompt(predictions: dict, confidence: dict, activated_regions: dict) -> str:
    return (
        "Review the model predictions and provide: agreement (True/False),\n"
        "reasoning (short), and a concise report.\n\n"
        f"Predictions: {predictions}\n"
        f"Confidence: {confidence}\n"
        f"Activated regions: {activated_regions}\n"
    )


def _call_groq(prompt: str) -> Any:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set")
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"input": prompt}
    resp = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def analyse(predictions: dict, confidence: dict, activated_regions: dict) -> Dict[str, Any]:
    """Analyse model outputs using Groq (if configured), otherwise return a stub.

    Returns a dict with keys: `agreement` (bool), `reasoning` (str), `report` (str),
    and `raw` (provider response when available).
    """
    prompt = _build_prompt(predictions, confidence, activated_regions)
    try:
        if GROQ_API_KEY:
            resp = _call_groq(prompt)
            if isinstance(resp, dict):
                agreement = resp.get("agreement", True)
                reasoning = resp.get("reasoning", "")
                report = resp.get("report", "")
            else:
                # Fallback: embed full response into reasoning
                agreement = True
                reasoning = str(resp)
                report = ""
            return {"agreement": agreement, "reasoning": reasoning, "report": report, "raw": resp}
        else:
            return {"agreement": True, "reasoning": "GROQ_API_KEY not configured; stubbed response.", "report": ""}
    except Exception as e:
        # produce an offline textual report when network/LLM fails
        try:
            score = None
            if isinstance(predictions, dict) and "osteoporosis" in predictions:
                score = float(predictions["osteoporosis"])
            elif isinstance(confidence, dict) and "osteoporosis" in confidence:
                score = float(confidence["osteoporosis"])

            if score is not None:
                severity = "high" if score >= 0.7 else "moderate" if score >= 0.4 else "low"
                report = (
                    f"Automated osteoporosis assessment: likelihood={score:.2f} ({severity}). "
                    "Recommendation: correlate clinically and consider DEXA for definitive diagnosis."
                )
                return {"agreement": True if score < 0.9 else False, "reasoning": f"Offline report generated; score={score:.2f}", "report": report}
        except Exception:
            pass
        return {"agreement": False, "reasoning": f"LLM call failed: {e}", "report": ""}

