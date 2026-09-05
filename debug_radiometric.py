#!/usr/bin/env python3
"""Debug radiometric features"""

from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, laplace
from skimage.filters import sobel

def extract_radiomic_features(arr_norm: np.ndarray) -> dict:
    """Extract clinical-grade bone mineral radiomic features from radiographic image array."""
    # Estimate bone vs background regions
    bone_mask = (arr_norm > np.percentile(arr_norm, 35)).astype(np.float32)
    bone_pixels = arr_norm[bone_mask > 0] if np.sum(bone_mask) > 0 else arr_norm.flatten()
    
    mean_density = float(np.mean(bone_pixels))
    std_density = float(np.std(bone_pixels))
    
    # 1. Cortical bone edge sharpness & gradient
    grad = sobel(arr_norm)
    edge_mean = float(np.mean(grad[bone_mask > 0])) if np.sum(bone_mask) > 0 else float(np.mean(grad))
    cortical_thinning = float(np.clip(1.0 - (edge_mean / 0.028), 0.10, 0.95))
    
    # 2. Trabecular cancellous architecture degradation
    lap = np.abs(laplace(arr_norm))
    trab_energy = float(np.mean(lap[bone_mask > 0])) if np.sum(bone_mask) > 0 else float(np.mean(lap))
    trabecular_loss = float(np.clip(1.0 - (trab_energy / 0.055), 0.10, 0.95))
    
    # 3. Bone mineral attenuation relative to reference
    bmd_attenuation = float(np.clip(1.0 - (mean_density / 0.68) * 0.90, 0.10, 0.95))
    
    print(f"\n  Radiometric Features:")
    print(f"    edge_mean: {edge_mean:.6f} → cortical_thinning: {cortical_thinning:.4f}")
    print(f"    trab_energy: {trab_energy:.6f} → trabecular_loss: {trabecular_loss:.4f}")
    print(f"    mean_density: {mean_density:.6f} → bmd_attenuation: {bmd_attenuation:.4f}")
    
    radiometric_score = float(np.clip(
        0.40 * cortical_thinning + 0.35 * trabecular_loss + 0.20 * bmd_attenuation,
        0.05, 0.98
    ))
    print(f"    radiometric_score: {radiometric_score:.4f}")
    
    return {
        "cortical_thinning": cortical_thinning,
        "trabecular_loss": trabecular_loss,
        "bmd_attenuation": bmd_attenuation,
        "radiometric_score": radiometric_score
    }

# Load test images
print("="*60)
print("RADIOMETRIC FEATURES DEBUG")
print("="*60)

test_osteo = list(Path("dataset/expanded_benchmark/osteoporosis").glob("*"))[0]
test_normal = list(Path("dataset/expanded_benchmark/normal").glob("*"))[0]

print(f"\n[Osteoporosis Image] {test_osteo.name}")
img_osteo = Image.open(str(test_osteo)).convert("L")
img_osteo = img_osteo.resize((224, 224))
gray_osteo = np.array(img_osteo, dtype=np.float32) / 255.0
rad_osteo = extract_radiomic_features(gray_osteo)

print(f"\n[Normal Image] {test_normal.name}")
img_normal = Image.open(str(test_normal)).convert("L")
img_normal = img_normal.resize((224, 224))
gray_normal = np.array(img_normal, dtype=np.float32) / 255.0
rad_normal = extract_radiomic_features(gray_normal)

print("\n" + "="*60)
print("Difference in scores:")
print(f"  Radiometric (Osteo - Normal): {rad_osteo['radiometric_score'] - rad_normal['radiometric_score']:.4f}")
