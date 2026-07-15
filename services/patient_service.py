"""
Lightweight service interface for patient operations.
This module is a placeholder showing how AI services should consume patient data.
"""
from typing import Any


def get_structured_patient(patient_id: str, api_base: str = "http://localhost:8000/api") -> Any:
    """Fetch structured data from backend API for downstream AI modules."""
    import requests

    r = requests.get(f"{api_base}/patients/{patient_id}/structured-data")
    r.raise_for_status()
    return r.json()
