#!/usr/bin/env python3
"""Quick inference test"""
from backend.services.xray_service import run_inference

result_osteo = run_inference('dataset/expanded_benchmark/osteoporosis/benchmark_fractures_01123.png', save_artifacts=False)
result_normal = run_inference('dataset/expanded_benchmark/normal/benchmark_normal_00240.png', save_artifacts=False)

print(f"Osteoporosis: {result_osteo['osteoporosis_score']:.4f}")
print(f"Normal:       {result_normal['osteoporosis_score']:.4f}")
print(f"\nDifference:   {result_osteo['osteoporosis_score'] - result_normal['osteoporosis_score']:.4f}")
