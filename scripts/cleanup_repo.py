"""Cleanup helper for the MedSync workspace.

Usage:
  python scripts/cleanup_repo.py         # dry-run: list candidate paths and sizes
  python scripts/cleanup_repo.py --delete  # actually remove the candidates

The script only deletes paths inside the repository root. It targets common
large or developer-only directories: virtualenvs, node_modules, build caches,
__pycache__, and frontend/.next. It will NOT remove `uploads/` by default.
"""
from pathlib import Path
import shutil
import argparse
import sys
import humanize


REPO_ROOT = Path(__file__).resolve().parent.parent

CANDIDATE_PATTERNS = [
    ".venv",
    "frontend/.venv",
    "backend/.venv",
    "node_modules",
    "frontend/node_modules",
    "frontend/.next",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "dist",
    "build",
    ".idea",
]


def get_dir_size(path: Path) -> int:
    total = 0
    try:
        for p in path.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except Exception:
                    pass
    except Exception:
        pass
    return total


def find_candidates(root: Path):
    found = []
    for pat in CANDIDATE_PATTERNS:
        for p in (root / pat).parent.rglob(Path(pat).name):
            # ensure it's inside repo
            try:
                p = p.resolve()
            except Exception:
                continue
            if root in p.parents or p == root:
                found.append(p)
    # deduplicate
    unique = []
    seen = set()
    for p in found:
        if str(p) not in seen:
            unique.append(p)
            seen.add(str(p))
    return unique


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete", action="store_true", help="Actually delete the candidate paths")
    args = parser.parse_args()

    print("Repo root:", REPO_ROOT)
    candidates = find_candidates(REPO_ROOT)
    if not candidates:
        print("No cleanup candidates found.")
        return

    print("Found candidates:")
    total = 0
    for p in sorted(candidates):
        size = get_dir_size(p) if p.exists() else 0
        total += size
        print(f" - {p}  ({humanize.naturalsize(size)})")
    print(f"Total reclaimable (approx): {humanize.naturalsize(total)}")

    if args.delete:
        print("Deleting candidates...")
        for p in candidates:
            try:
                if p.exists():
                    if p.is_dir():
                        shutil.rmtree(p)
                        print(f"Removed directory {p}")
                    else:
                        p.unlink()
                        print(f"Removed file {p}")
            except Exception as e:
                print(f"Failed to remove {p}: {e}")
        print("Cleanup complete.")
    else:
        print("Run with --delete to remove the above paths.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
