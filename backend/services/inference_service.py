from __future__ import annotations

from pathlib import Path
from typing import Any

from .ct_bone_service import run_ct_bmd
from .dxa_service import run_dxa_bmd
from .file_service import BASE_UPLOAD_DIR, detect_modality
from .xray_service import run_inference as run_xray_inference


def resolve_upload_path(image_url_or_path: str) -> Path:
    """Resolve a client-provided upload URL or path to a local filesystem path."""
    if not image_url_or_path:
        raise ValueError("image_url required")

    raw = str(image_url_or_path).strip().replace("\\", "/")

    if "/uploads/" in raw:
        raw = raw.split("/uploads/", 1)[1]
    elif raw.startswith("uploads/"):
        raw = raw[len("uploads/") :]
    elif raw.startswith("backend/uploads/"):
        raw = raw[len("backend/uploads/") :]

    path = Path(raw)
    if path.is_absolute() and path.exists():
        return path
    return BASE_UPLOAD_DIR / path



def _infer_source_kind(file_path: Path) -> str:
    """Choose the model family from the stored path first, then modality hints."""
    path_parts = {part.lower() for part in file_path.parts}

    if "ct" in path_parts:
        return "ct"
    if "dxa" in path_parts:
        return "dxa"
    if "xrays" in path_parts or "xray" in path_parts:
        return "xray"

    modality = detect_modality(str(file_path)).upper()
    if modality == "CT":
        return "ct"
    if modality == "DXA":
        return "dxa"
    return "xray"


def _score_from_t_score(t_score: Any) -> float:
    try:
        t_value = float(t_score)
    except (TypeError, ValueError):
        return 0.5

    score = (-1.0 - t_value) / 2.2
    return max(0.0, min(1.0, score))


def _normalize_bmd_result(raw_result: dict, source_kind: str) -> dict:
    score = _score_from_t_score(raw_result.get("t_score"))
    risk_level = raw_result.get("risk_level") or "Unknown"
    risk_color = raw_result.get("risk_color") or "gray"
    clinical_notes = raw_result.get("clinical_notes") or raw_result.get("diagnostic") or ""

    supporting_findings: dict[str, Any] = {
        "Cortical Bone Thinning": score,
        "Trabecular Microarchitecture Degradation": min(1.0, score * 0.95),
        "Bone Mineral Density (BMD) Attenuation": min(1.0, score),
        "Fragility Fracture Indicator": min(1.0, max(0.0, score * 0.85)),
    }

    normalized = dict(raw_result)
    normalized.setdefault("disease", "Osteoporosis")
    normalized["osteoporosis"] = {
        "score": score,
        "percentage": round(score * 100, 1),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "clinical_notes": clinical_notes,
    }
    normalized["predictions"] = {"osteoporosis": score}
    normalized["supporting_findings"] = supporting_findings
    normalized["measurements"] = {
        "bmd": raw_result.get("bmd"),
        "t_score": raw_result.get("t_score"),
        "z_score": raw_result.get("z_score"),
        "fracture_assessment": raw_result.get("fracture_assessment"),
    }
    normalized["heatmap_path"] = normalized.get("heatmap_path", "") or ""
    normalized["overlay_path"] = normalized.get("overlay_path", normalized["heatmap_path"]) or ""
    normalized.setdefault(
        "xai_status",
        "Explainable AI heatmap not generated for this modality." if source_kind != "xray" else "Explainable AI Grad-CAM generated successfully",
    )
    normalized["_meta"] = {
        **(raw_result.get("_meta") or {}),
        "source": source_kind,
        "normalized": True,
    }
    return normalized


def run_routed_inference(image_url_or_path: str) -> dict:
    """Route inference requests to the model that matches the stored file path."""
    file_path = resolve_upload_path(image_url_or_path)
    source_kind = _infer_source_kind(file_path)

    if source_kind == "ct":
        return _normalize_bmd_result(run_ct_bmd(str(file_path)), source_kind)
    if source_kind == "dxa":
        return _normalize_bmd_result(run_dxa_bmd(str(file_path)), source_kind)
    return run_xray_inference(str(file_path))