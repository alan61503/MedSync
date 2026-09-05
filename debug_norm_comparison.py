#!/usr/bin/env python3
"""Debug gray_norm normalization"""

from pathlib import Path
import numpy as np
from PIL import Image

def test_normalization(img_path):
    """Test different normalization methods"""
    pil_img = Image.open(str(img_path)).convert("RGB")
    raw_img = np.array(pil_img)
    
    # Method 1: Simple division (from debug_radiometric.py)
    gray_arr1 = raw_img.mean(2).astype(np.float32) / 255.0
    
    # Method 2: Min-max normalization (from run_inference)
    gray_arr2 = raw_img.mean(2).astype(np.float32) / 255.0
    gray_norm = (gray_arr2 - gray_arr2.min()) / (gray_arr2.max() - gray_arr2.min() + 1e-8)
    
    print(f"\nImage: {Path(img_path).name}")
    print(f"  Simple div (0-1):")
    print(f"    range: [{gray_arr1.min():.4f}, {gray_arr1.max():.4f}]")
    print(f"    mean:  {gray_arr1.mean():.4f}, std: {gray_arr1.std():.4f}")
    
    print(f"  Min-max norm (0-1):")
    print(f"    range: [{gray_norm.min():.4f}, {gray_norm.max():.4f}]")
    print(f"    mean:  {gray_norm.mean():.4f}, std: {gray_norm.std():.4f}")
    
    # Show that they are different
    diff = np.abs(gray_arr1 - gray_norm).mean()
    print(f"  Difference: {diff:.6f}")
    
    return gray_arr1, gray_norm

test_osteo = list(Path("dataset/expanded_benchmark/osteoporosis").glob("*"))[0]
test_normal = list(Path("dataset/expanded_benchmark/normal").glob("*"))[0]

print("="*60)
print("NORMALIZATION COMPARISON")
print("="*60)

simple_osteo, minmax_osteo = test_normalization(test_osteo)
simple_normal, minmax_normal = test_normalization(test_normal)

print("\n" + "="*60)
print("Impact on radiometric features:")
print("="*60)

from scipy.ndimage import laplace
from skimage.filters import sobel

def quick_radio(arr_norm):
    bone_mask = (arr_norm > np.percentile(arr_norm, 35)).astype(np.float32)
    bone_pixels = arr_norm[bone_mask > 0] if np.sum(bone_mask) > 0 else arr_norm.flatten()
    mean_density = float(np.mean(bone_pixels))
    grad = sobel(arr_norm)
    edge_mean = float(np.mean(grad[bone_mask > 0])) if np.sum(bone_mask) > 0 else float(np.mean(grad))
    return edge_mean, mean_density

print(f"\nOsteoporosis image:")
edge1, dens1 = quick_radio(simple_osteo)
edge2, dens2 = quick_radio(minmax_osteo)
print(f"  Simple: edge={edge1:.6f}, density={dens1:.6f}")
print(f"  MinMax: edge={edge2:.6f}, density={dens2:.6f}")

print(f"\nNormal image:")
edge1, dens1 = quick_radio(simple_normal)
edge2, dens2 = quick_radio(minmax_normal)
print(f"  Simple: edge={edge1:.6f}, density={dens1:.6f}")
print(f"  MinMax: edge={edge2:.6f}, density={dens2:.6f}")
