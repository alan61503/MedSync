#!/usr/bin/env python3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "dataset" / "expanded_benchmark"

print(f"Script path: {Path(__file__).resolve()}")
print(f"REPO_ROOT: {REPO_ROOT}")
print(f"DATASET_ROOT: {DATASET_ROOT}")
print(f"DATASET_ROOT exists: {DATASET_ROOT.exists()}")

if DATASET_ROOT.exists():
    normal_dir = DATASET_ROOT / "normal"
    osteo_dir = DATASET_ROOT / "osteoporosis"
    print(f"Normal dir exists: {normal_dir.exists()}")
    print(f"Osteo dir exists: {osteo_dir.exists()}")
    
    if normal_dir.exists():
        normal_files = list(normal_dir.glob("*.png"))
        print(f"Normal images: {len(normal_files)}")
    
    if osteo_dir.exists():
        osteo_files = list(osteo_dir.glob("*.png"))
        print(f"Osteoporosis images: {len(osteo_files)}")
