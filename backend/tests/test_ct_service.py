import os
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

@pytest.fixture(autouse=True)
def enable_debug():
    """Enable debug endpoint for the duration of the test."""
    os.environ["DEBUG_CT"] = "1"
    yield
    os.environ.pop("DEBUG_CT", None)

def test_run_ct_bmd_debug():
    """Ensure the debug endpoint returns expected fields."""
    payload = {"ct_path": "nonexistent/file.nii"}
    response = client.post("/api/run-ct-bmd-debug", json=payload)
    assert response.status_code == 200, f"Unexpected status {response.status_code}"
    data = response.json()
    # Core result keys
    for key in ("bmd", "t_score", "risk_level"):
        assert key in data, f"Missing {key} in response"
    # Debug block
    assert "_debug" in data, "Debug block not present"
    debug = data["_debug"]
    assert "raw_bmd" in debug and isinstance(debug["raw_bmd"], float)
    assert "model_output_summary" in debug
    assert "tensor shape" in debug["model_output_summary"]
