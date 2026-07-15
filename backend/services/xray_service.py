import os
from pathlib import Path
import json

BASE_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
BASE_HEATMAP_DIR = Path(__file__).resolve().parent.parent / "heatmaps"
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BASE_HEATMAP_DIR.mkdir(parents=True, exist_ok=True)


def _save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def run_inference(image_path: str) -> dict:
    """Run inference using torchxrayvision if available, otherwise return placeholder.

    Returns: {predictions: dict, confidence_scores: dict, heatmap_path: str}
    """
    try:
        import numpy as np
        import torch
        import torchvision
        import skimage.io
        import torchxrayvision as xrv

        # load image and preprocess
        img = skimage.io.imread(image_path)
        # xrv normalize expects uint8 image 0-255
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

        # simple heatmap placeholder: use the mean image resized as grayscale overlay
        try:
            import matplotlib.pyplot as plt
            heat = tensor[0]
            heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
            heat_np = (heat.numpy() * 255).astype('uint8')
            heat_img_path = BASE_HEATMAP_DIR / (Path(image_path).stem + "_heatmap.png")
            plt.imsave(heat_img_path, heat_np, cmap='jet')
            heatmap_path = str(heat_img_path)
        except Exception:
            heatmap_path = ""

        outobj = {"predictions": preds, "confidence_scores": preds, "heatmap_path": heatmap_path}
        # save JSON alongside outputs
        out_json = BASE_OUTPUT_DIR / (Path(image_path).stem + ".json")
        _save_json(outobj, out_json)
        return outobj
    except Exception as e:
        # fallback: return empty results with note
        return {"predictions": {}, "confidence_scores": {}, "heatmap_path": "", "error": str(e)}
