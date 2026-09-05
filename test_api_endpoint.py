#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

from pathlib import Path
from backend.services.inference_service import run_routed_inference

# Test paths that would come through the API
test_cases = [
    "dataset/expanded_benchmark/osteoporosis/benchmark_fractures_01123.png",
    "dataset/expanded_benchmark/normal/benchmark_normal_00240.png",
]

for test_path in test_cases:
    print(f"\n{'='*70}")
    print(f"Testing: {test_path}")
    print('='*70)
    try:
        result = run_routed_inference(test_path)
        score = result.get("inference", {}).get("osteoporosis", {}).get("score", "???")
        risk = result.get("inference", {}).get("osteoporosis", {}).get("risk_level", "???")
        xai_status = result.get("inference", {}).get("xai_status", "???")
        print(f"✓ Score: {score}")
        print(f"✓ Risk Level: {risk}")
        print(f"✓ XAI Status: {xai_status}")
        print(f"✓ Source: {result.get('source')}")
        
        # Check for 0.74 (fallback)
        if score == 0.74:
            print("⚠ WARNING: Got fallback score 0.74! There's an exception being caught.")
            if "llm" in result:
                print(f"LLM Result: {result.get('llm')}")
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
