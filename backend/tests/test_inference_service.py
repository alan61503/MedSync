from fastapi.testclient import TestClient

from backend.api.patients import run_inference_endpoint as patient_run_inference_endpoint
from backend.main import app
from backend.services import inference_service


client = TestClient(app)


def test_run_routed_inference_dispatches_by_folder(monkeypatch):
    calls = []

    def fake_ct(path):
        calls.append(("ct", path))
        return {"source": "ct"}

    def fake_dxa(path):
        calls.append(("dxa", path))
        return {"source": "dxa"}

    def fake_xray(path):
        calls.append(("xray", path))
        return {"source": "xray"}

    monkeypatch.setattr(inference_service, "run_ct_bmd", fake_ct)
    monkeypatch.setattr(inference_service, "run_dxa_bmd", fake_dxa)
    monkeypatch.setattr(inference_service, "run_xray_inference", fake_xray)

    assert inference_service.run_routed_inference("/uploads/p1/ct/scan.nii")["source"] == "ct"
    assert inference_service.run_routed_inference("/uploads/p1/dxa/scan.dcm")["source"] == "dxa"
    assert inference_service.run_routed_inference("/uploads/p1/xrays/scan.png")["source"] == "xray"

    assert calls == [
        ("ct", str(inference_service.resolve_upload_path("/uploads/p1/ct/scan.nii"))),
        ("dxa", str(inference_service.resolve_upload_path("/uploads/p1/dxa/scan.dcm"))),
        ("xray", str(inference_service.resolve_upload_path("/uploads/p1/xrays/scan.png"))),
    ]


def test_fastapi_run_inference_endpoint_exists(monkeypatch):
    def fake_router(image_url):
        return {"routed": image_url}

    monkeypatch.setattr("backend.api.patients.run_routed_inference", fake_router)

    response = client.post("/api/run-inference", json={"image_url": "/uploads/p1/xrays/scan.png"})
    assert response.status_code == 200
    assert response.json() == {"routed": "/uploads/p1/xrays/scan.png"}