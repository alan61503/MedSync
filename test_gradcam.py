#!/usr/bin/env python3
from pathlib import Path
from backend.services.xray_service import run_inference

# Test with a real file path and save artifacts
osteo_path = 'dataset/expanded_benchmark/osteoporosis/benchmark_fractures_01123.png'

print("Testing osteoporosis image with artifacts...")
r1 = run_inference(osteo_path, save_artifacts=True)
print(f"Score: {r1['osteoporosis']['score']}")
print(f"Risk Level: {r1['osteoporosis']['risk_level']}")
print(f"XAI Status: {r1['xai_status']}")
print(f"Heatmap Path: {r1['heatmap_path']}")
print(f"Overlay Path: {r1['overlay_path']}")

# Check if heatmap files actually exist
if r1['heatmap_path'].startswith('/uploads/'):
    local_path = Path('backend') / r1['heatmap_path'].lstrip('/')
    print(f"Looking for heatmap at: {local_path}")
    print(f"Exists: {local_path.exists()}")
