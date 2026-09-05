#!/usr/bin/env python3
"""Debug script to test fine-tuned model integration"""

import sys
from pathlib import Path
from backend.services.xray_service import run_inference, _get_finetuned_osteoporosis_score
import numpy as np
from PIL import Image

print("\n" + "="*60)
print("FINE-TUNED MODEL INTEGRATION DEBUG")
print("="*60)

# Test 1: Test model direct load
print("\n[Test 1] Direct Model Load")
print("-" * 60)
test_osteo = list(Path("dataset/expanded_benchmark/osteoporosis").glob("*"))[0]
test_normal = list(Path("dataset/expanded_benchmark/normal").glob("*"))[0]

img_osteo = Image.open(str(test_osteo)).convert("L")
img_osteo = img_osteo.resize((224, 224))
gray_osteo = np.array(img_osteo, dtype=np.float32) / 255.0

img_normal = Image.open(str(test_normal)).convert("L")
img_normal = img_normal.resize((224, 224))
gray_normal = np.array(img_normal, dtype=np.float32) / 255.0

score_osteo_direct = _get_finetuned_osteoporosis_score(gray_osteo)
score_normal_direct = _get_finetuned_osteoporosis_score(gray_normal)

print(f"Osteoporosis image direct model: {score_osteo_direct:.4f}")
print(f"Normal image direct model:       {score_normal_direct:.4f}")

# Test 2: Test inference pipeline
print("\n[Test 2] Inference Pipeline (Full)")
print("-" * 60)
result_osteo = run_inference(str(test_osteo), save_artifacts=False)
result_normal = run_inference(str(test_normal), save_artifacts=False)

score_osteo_pipeline = result_osteo["osteoporosis"]["score"]
score_normal_pipeline = result_normal["osteoporosis"]["score"]

print(f"Osteoporosis image pipeline: {score_osteo_pipeline:.4f}")
print(f"Normal image pipeline:       {score_normal_pipeline:.4f}")

# Test 3: Diagnostics
print("\n[Test 3] Diagnostics")
print("-" * 60)
print(f"Model weights file: {Path('backend/models/resnet50_finetuned.pt').exists()}")
if Path('backend/models/resnet50_finetuned.pt').exists():
    size_mb = Path('backend/models/resnet50_finetuned.pt').stat().st_size / (1024*1024)
    print(f"  File size: {size_mb:.1f} MB")

print(f"\nOsteoporosis image predictions:")
print(f"  Direct model:  {score_osteo_direct:.4f} → {'✓ CORRECT' if score_osteo_direct > 0.5 else '✗ WRONG'}")
print(f"  Pipeline:      {score_osteo_pipeline:.4f} → {'✓ CORRECT' if score_osteo_pipeline > 0.5 else '✗ WRONG'}")

print(f"\nNormal image predictions:")
print(f"  Direct model:  {score_normal_direct:.4f} → {'✓ CORRECT' if score_normal_direct < 0.5 else '✗ WRONG'}")
print(f"  Pipeline:      {score_normal_pipeline:.4f} → {'✓ CORRECT' if score_normal_pipeline < 0.5 else '✗ WRONG'}")

# Test 4: Run on sample of images
print("\n[Test 4] Sample Accuracy")
print("-" * 60)
osteo_imgs = list(Path("dataset/expanded_benchmark/osteoporosis").glob("*"))[:10]
normal_imgs = list(Path("dataset/expanded_benchmark/normal").glob("*"))[:10]

correct = 0
total = 0

for img_path in osteo_imgs:
    result = run_inference(str(img_path), save_artifacts=False)
    score = result["osteoporosis"]["score"]
    if score >= 0.5:
        correct += 1
    total += 1

for img_path in normal_imgs:
    result = run_inference(str(img_path), save_artifacts=False)
    score = result["osteoporosis"]["score"]
    if score < 0.5:
        correct += 1
    total += 1

accuracy = correct / total if total > 0 else 0
print(f"Sample accuracy (20 images): {accuracy:.1%} ({correct}/{total})")

print("\n" + "="*60)
