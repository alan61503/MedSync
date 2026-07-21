import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from backend.services.xray_service import run_inference

def test_osteoporosis_model():
    sample_img = repo_root / "backend" / "uploads" / "testpatient" / "xrays" / "sample_xray.png"
    if not sample_img.exists():
        print(f"Sample image not found at {sample_img}")
        return

    print(f"Testing Osteoporosis inference & XAI Grad-CAM on sample: {sample_img}")
    result = run_inference(str(sample_img))

    print("\n--- Diagnostic Results ---")
    print(f"Disease Focus: {result.get('disease')}")
    osteo = result.get("osteoporosis", {})
    print(f"Osteoporosis Risk Score: {osteo.get('score')} ({osteo.get('percentage')}%)")
    print(f"Clinical Risk Level: {osteo.get('risk_level')}")
    print(f"Notes: {osteo.get('clinical_notes')}")

    print("\n--- Pathologies & Findings ---")
    for k, v in result.get("supporting_findings", {}).items():
        print(f"  {k}: {v:.4f}")

    print("\n--- XAI Explainability Heatmaps ---")
    print(f"Grad-CAM Heatmap Path: {result.get('heatmap_path')}")
    print(f"Visual Overlay Path:   {result.get('overlay_path')}")
    print(f"Status: {result.get('xai_status')}")

if __name__ == "__main__":
    test_osteoporosis_model()
