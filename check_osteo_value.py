#!/usr/bin/env python3
"""Check osteoporosis score"""
from backend.services.xray_service import run_inference

result_osteo = run_inference('dataset/expanded_benchmark/osteoporosis/benchmark_fractures_01123.png', save_artifacts=False)
result_normal = run_inference('dataset/expanded_benchmark/normal/benchmark_normal_00240.png', save_artifacts=False)

print(f"Osteoporosis image:")
print(f"  osteoporosis value: {result_osteo['osteoporosis']}")
print(f"  Type: {type(result_osteo['osteoporosis'])}")

print(f"\nNormal image:")
print(f"  osteoporosis value: {result_normal['osteoporosis']}")

if isinstance(result_osteo['osteoporosis'], dict):
    print("\n  Keys in osteoporosis dict:", result_osteo['osteoporosis'].keys())
    for k, v in result_osteo['osteoporosis'].items():
        print(f"    {k}: {v}")
