import os
import shutil
from pathlib import Path
from typing import Tuple
import pydicom


BASE_UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads"
BASE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def save_file(patient_id: str, category: str, filename: str, fileobj) -> str:
    target_dir = BASE_UPLOAD_DIR / patient_id / category
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename
    with open(target_path, "wb") as f:
        shutil.copyfileobj(fileobj, f)
    return str(target_path)


def is_dicom_file(path_or_bytes) -> bool:
    try:
        if isinstance(path_or_bytes, (str, Path)):
            pydicom.dcmread(str(path_or_bytes), stop_before_pixels=True)
        else:
            pydicom.dcmread(path_or_bytes, stop_before_pixels=True)
        return True
    except Exception:
        return False


def detect_modality(file_path: str) -> str:
    try:
        ds = pydicom.dcmread(file_path, stop_before_pixels=True)
        return getattr(ds, "Modality", "UNKNOWN")
    except Exception:
        # fallback based on extension
        ext = Path(file_path).suffix.lower()
        if ext in [".png", ".jpg", ".jpeg"]:
            return "XRAY"
        return "UNKNOWN"
