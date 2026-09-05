#!/usr/bin/env python3
"""Test actual inference pipeline"""

from pathlib import Path
from backend.services.xray_service import run_inference

# Test inference
osteo_file = list(Path('dataset/expanded_benchmark/osteoporosis').glob('*'))[0]
normal_file = list(Path('dataset/expanded_benchmark/Osteoporosis Knee X-ray/normal').glob('*'))[0]

print('Testing inference pipeline...\n')

for label, path in [('Osteoporosis', osteo_file), ('Normal', normal_file)]:
    print(f'{label}:')
    result = run_inference(str(path), save_artifacts=False)
    print(f"  Osteoporosis score: {result['osteoporosis_score']:.4f}")
    print(f"  XAI status: {result['xai_status']}")
    print()
