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
        # Attempt the xrv preprocessing pipeline; if it fails (various image shapes),
        # fallback to a simple PIL resize -> numpy routine for robustness.
        try:
            img = xrv.datasets.normalize(img, 255)
            if img.ndim == 3:
                img = img.mean(2)[None, ...]
            transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])
            img = transform(img)
            tensor = torch.from_numpy(img).float()
        except Exception:
            from PIL import Image
            # ensure grayscale
            if img.ndim == 3:
                img = img.mean(2).astype("uint8")
            pil = Image.fromarray(img)
            pil = pil.resize((224, 224))
            arr = np.array(pil).astype("float32") / 255.0
            tensor = torch.from_numpy(arr[None, ...])

        # lazy load model
        model = xrv.models.DenseNet(weights="densenet121-res224-all")
        model.eval()
        with torch.no_grad():
            out = model(tensor[None, ...])
            scores = out[0].detach().cpu().numpy()

        preds = dict(zip(model.pathologies, scores.tolist()))

        # Compute an `osteoporosis` score:
        # - If the pretrained model provides an "Osteoporosis" label, use it.
        # - Otherwise use a lightweight heuristic: low high-frequency content in the image
        #   (we use a Sobel gradient magnitude as a proxy for bone contrast) and
        #   map it into [0,1] where higher means more likely osteoporosis.
        osteoporosis_score = None
        if any("osteoporosis" in p.lower() for p in model.pathologies):
            for p in model.pathologies:
                if p.lower().find("osteoporosis") >= 0:
                    osteoporosis_score = float(preds.get(p, 0.0))
                    break
        if osteoporosis_score is None:
            try:
                from skimage.filters import sobel
                import numpy as _np

                # use the preprocessed tensor if available, otherwise reload image
                if 'tensor' in locals():
                    img_for_heuristic = tensor[0]
                    arr = img_for_heuristic.detach().cpu().numpy() if hasattr(img_for_heuristic, 'detach') else img_for_heuristic
                else:
                    arr = skimage.io.imread(image_path)
                    if arr.ndim == 3:
                        arr = arr.mean(2)
                    arr = arr.astype('float32') / 255.0

                grad = sobel(arr)
                hf = float(_np.mean(grad))
                # normalize heuristic using an empirical range
                # typical mean(sobel) may be in ~[0.0, 0.3]; clamp and scale
                hf_clamped = max(0.0, min(hf, 0.3)) / 0.3
                osteoporosis_score = float(1.0 - hf_clamped)
            except Exception:
                osteoporosis_score = 0.0

        # add osteoporosis heuristic into preds (preserve model outputs)
        try:
            preds["osteoporosis"] = osteoporosis_score
        except Exception:
            preds = {"osteoporosis": osteoporosis_score}

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
            # bring to 0-1 and save as heatmap
            h = heat.detach().cpu().numpy() if hasattr(heat, 'detach') else heat.numpy()
            heat = (h - float(h.min())) / (float(h.max() - h.min()) + 1e-8)
            heat_np = (heat * 255).astype("uint8")
            plt.imsave(heatmap_path_fs, heat_np, cmap="jet")

            # attempt a simple Grad-CAM explanation and save an explanatory heatmap
            try:
                import torch.nn as nn

                def find_conv_layer(mod):
                    # find a convolutional layer to hook (search reversed)
                    for m in reversed(list(mod.modules())):
                        if isinstance(m, nn.Conv2d):
                            return m
                    return None

                target_layer = find_conv_layer(model)
                if target_layer is not None:
                    activations = None
                    grads = None

                    def forward_hook(module, inp, outp):
                        nonlocal activations
                        activations = outp.detach()

                    def backward_hook(module, grad_in, grad_out):
                        nonlocal grads
                        grads = grad_out[0].detach()

                    fh = target_layer.register_forward_hook(forward_hook)
                    bh = target_layer.register_backward_hook(backward_hook)

                    model.zero_grad()
                    out = model(tensor[None, ...])
                    # choose the top predicted index
                    idx = int(out[0].argmax().item())
                    score = out[0, idx]
                    score.backward(retain_graph=True)

                    if activations is not None and grads is not None:
                        weights = grads.mean(dim=(2, 3), keepdim=True)
                        cam = (weights * activations).sum(dim=1)[0]
                        cam = cam.cpu().numpy()
                        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                        cam_img = (cam * 255).astype("uint8")
                        gradmap_path = heatmap_dir / (image_p.stem + "_gradcam.png")
                        plt.imsave(gradmap_path, cam_img, cmap="jet")
                        # overwrite heatmap_url to point to gradcam for XAI
                        heatmap_url = f"/uploads/{patient_id}/heatmaps/{gradmap_path.name}"

                    fh.remove()
                    bh.remove()
            except Exception:
                pass
            heatmap_url = f"/uploads/{patient_id}/heatmaps/{heatmap_filename}"
        except Exception:
            # fallback to saving under outputs
            heatmap_url = ""

        # Build pneumonia-focused output: main `pneumonia` score plus supporting findings
        def _get_score(d, name):
            # case-insensitive lookup
            for k in d.keys():
                if k.lower() == name.lower():
                    return float(d.get(k, 0.0))
            return None

        pneumonia_score = _get_score(preds, "Pneumonia")
        # supporting findings that commonly indicate or accompany pneumonia
        support_keys = ["Lung Opacity", "Effusion", "Consolidation", "Infiltration", "Atelectasis"]
        supporting = {}
        for sk in support_keys:
            s = _get_score(preds, sk)
            if s is not None:
                supporting[sk] = s

        outobj = {
            "predictions": {"pneumonia": pneumonia_score},
            "supporting_findings": supporting,
            "confidence_scores": preds,  # full model outputs for debugging/audit
            "heatmap_path": heatmap_url,
        }
        out_json = BASE_OUTPUT_DIR / (Path(image_path).stem + ".json")
        _save_json(outobj, out_json)
        return outobj
    except Exception as e:
        return {"predictions": {}, "confidence_scores": {}, "heatmap_path": "", "error": str(e)}
