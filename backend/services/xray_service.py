import os
from pathlib import Path
import json

BASE_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def run_inference(image_path: str) -> dict:
    """Run inference using torchxrayvision if available, otherwise return placeholder.

    Returns: {predictions: dict, confidence_scores: dict, heatmap_path: str}
    The returned `heatmap_path` is a URL under `/uploads/...` when possible.
    """
    try:
        import numpy as np
        import torch
        import torchvision
        import skimage.io
        import torchxrayvision as xrv
        import matplotlib.pyplot as plt

        # load image and preprocess
        img = skimage.io.imread(image_path)
        img = xrv.datasets.normalize(img, 255)
        if img.ndim == 3:
            img = img.mean(2)[None, ...]

        transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])
        img = transform(img)
        tensor = torch.from_numpy(img).float()

        # lazy load model
        model = xrv.models.DenseNet(weights="densenet121-res224-all")
        model.eval()
        with torch.no_grad():
            out = model(tensor[None, ...])
            scores = out[0].detach().cpu().numpy()

        preds = dict(zip(model.pathologies, scores.tolist()))

        # determine uploads dir and patient id to save heatmap under uploads
        try:
            from .file_service import BASE_UPLOAD_DIR
            image_p = Path(image_path)
            rel = image_p.relative_to(BASE_UPLOAD_DIR)
            patient_id = rel.parts[0]
            heatmap_dir = BASE_UPLOAD_DIR / patient_id / "heatmaps"
            heatmap_dir.mkdir(parents=True, exist_ok=True)
            heatmap_filename = image_p.stem + "_heatmap.png"
            heatmap_path_fs = heatmap_dir / heatmap_filename
            # create a simple heatmap from the preprocessed tensor
            heat = tensor[0]
            heat = (heat - float(heat.min())) / (float(heat.max() - heat.min()) + 1e-8)
            heat_np = (heat.numpy() * 255).astype("uint8")
            plt.imsave(heatmap_path_fs, heat_np, cmap="jet")
            heatmap_url = f"/uploads/{patient_id}/heatmaps/{heatmap_filename}"
        except Exception:
            # fallback to saving under outputs
            heatmap_url = ""

        outobj = {"predictions": preds, "confidence_scores": preds, "heatmap_path": heatmap_url}
        out_json = BASE_OUTPUT_DIR / (Path(image_path).stem + ".json")
        _save_json(outobj, out_json)
        return outobj
    except Exception as e:
        return {"predictions": {}, "confidence_scores": {}, "heatmap_path": "", "error": str(e)}
