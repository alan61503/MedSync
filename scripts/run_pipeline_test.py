import os
from pathlib import Path
import sys
import numpy as np
from skimage.io import imsave
from pathlib import Path

# Ensure repo root is on sys.path so packages like `backend` and `services` import
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from backend.services.xray_service import run_inference
from services.llm_service import analyse


def make_sample_image(path: Path):
    img = np.zeros((256, 256), dtype=np.uint8)
    # simple gradient + circle
    for i in range(img.shape[0]):
        img[i] = np.linspace(0, 255, img.shape[1], dtype=np.uint8)
    rr, cc = np.ogrid[:256, :256]
    mask = (rr - 128) ** 2 + (cc - 128) ** 2 <= 40 ** 2
    img[mask] = 200
    path.parent.mkdir(parents=True, exist_ok=True)
    imsave(str(path), img)


def main():
    uploads = Path(__file__).resolve().parent.parent / "backend" / "uploads"
    patient_id = "testpatient"
    img_path = uploads / patient_id / "xrays" / "sample_xray.png"
    make_sample_image(img_path)
    print("Saved sample image to", img_path)

    res = run_inference(str(img_path))
    print("Inference result (osteoporosis-focused):")
    print(res.get("predictions"))

    llm = analyse(res.get("predictions", {}), res.get("confidence_scores", {}), {"heatmap": res.get("heatmap_path")})
    print("LLM analysis result (osteoporosis):")
    print(llm)


if __name__ == "__main__":
    main()
