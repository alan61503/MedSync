#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

from pathlib import Path
from backend.services.inference_service import run_routed_inference
import json

# Test paths that would come through the API
test_cases = [
    "dataset/expanded_benchmark/osteoporosis/benchmark_fractures_01123.png",
]

for test_path in test_cases:
    print(f"\nTesting: {test_path}")
    try:
        result = run_routed_inference(test_path)
        print("Full result keys:", list(result.keys()))
        print("Full result (formatted):")
        print(json.dumps(result, indent=2, default=str)[:500])
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
