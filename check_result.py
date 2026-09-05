#!/usr/bin/env python3
"""Check result structure"""
from backend.services.xray_service import run_inference

result_osteo = run_inference('dataset/expanded_benchmark/osteoporosis/benchmark_fractures_01123.png', save_artifacts=False)
print("Keys in result:", result_osteo.keys())
print("\nResult:")
for k, v in result_osteo.items():
    if isinstance(v, (str, float, int)):
        print(f"  {k}: {v}")
