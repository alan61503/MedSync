import torch
import torchvision
import torchxrayvision as xrv
import skimage.io
from pathlib import Path

img_path = Path(__file__).resolve().parent.parent / "backend" / "uploads" / "testpatient" / "xrays" / "sample_xray.png"
print('Image:', img_path)
img = skimage.io.imread(str(img_path))
# try xrv preprocessing
try:
    img_p = xrv.datasets.normalize(img, 255)
    if img_p.ndim == 3:
        img_p = img_p.mean(2)[None, ...]
    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])
    img_t = transform(img_p)
    tensor = torch.from_numpy(img_t).float()
except Exception as e:
    import numpy as np
    from PIL import Image
    if img.ndim == 3:
        img = img.mean(2).astype('uint8')
    pil = Image.fromarray(img)
    pil = pil.resize((224,224))
    arr = (np.array(pil).astype('float32')/255.0)
    tensor = torch.from_numpy(arr[None,...]).float()

model = xrv.models.DenseNet(weights='densenet121-res224-all')
model.eval()
with torch.no_grad():
    out = model(tensor[None, ...])
    scores = out[0].detach().cpu().numpy()

pathologies = model.pathologies
preds = dict(zip(pathologies, scores.tolist()))
print('\nTop predictions:')
for k,v in sorted(preds.items(), key=lambda x: -x[1])[:6]:
    print(f"{k}: {v:.4f}")

p = preds.get('Pneumonia') if 'Pneumonia' in preds else preds.get('pneumonia')
print('\nPneumonia score:', p)
