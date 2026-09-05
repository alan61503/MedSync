#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, '.')

from backend.services.xray_service import _resolve_image_path

test_paths = [
    'dataset/expanded_benchmark/osteoporosis/benchmark_fractures_01123.png',
    'dataset/expanded_benchmark/normal/benchmark_normal_00240.png',
    '/dataset/expanded_benchmark/osteoporosis/benchmark_fractures_01123.png',
    'backend/dataset/expanded_benchmark/osteoporosis/benchmark_fractures_01123.png',
]

for path in test_paths:
    resolved = _resolve_image_path(path)
    exists = resolved.exists()
    print(f"{path:60} -> {resolved.name:30} (exists: {exists})")
