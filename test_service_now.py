#!/usr/bin/env python3
from pathlib import Path
from backend.services.xray_service import run_inference
import json

# Test with a real file path
osteo_path = 'dataset/expanded_benchmark/osteoporosis/benchmark_fractures_01123.png'
normal_path = 'dataset/expanded_benchmark/normal/benchmark_normal_00240.png'

print("Testing osteoporosis image...")
r1 = run_inference(osteo_path, save_artifacts=False)
print(f"Score: {r1['osteoporosis']['score']}")
print(f"Risk Level: {r1['osteoporosis']['risk_level']}")
print(f"XAI Status: {r1['xai_status']}")
print(f"Predictions: {r1['predictions']}")

print("\nTesting normal image...")
r2 = run_inference(normal_path, save_artifacts=False)
print(f"Score: {r2['osteoporosis']['score']}")
print(f"Risk Level: {r2['osteoporosis']['risk_level']}")
print(f"XAI Status: {r2['xai_status']}")

print(f"\nDifference: {r1['osteoporosis']['score'] - r2['osteoporosis']['score']}")
