"""Package a clean CS5489 submission code ZIP excluding large datasets, caches, weights, and local tool dirs."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

INCLUDE_ROOT_FILES = (
    "README.md",
    "pyproject.toml",
    "requirements.txt",
)

INCLUDE_DIRS = (
    "src",
    "scripts",
    "tests",
    "configs",
    "docs",
    "reports",
)

EXCLUDE_PARTS = frozenset({"__pycache__", ".pytest_cache", ".DS_Store", "archive"})


def build_code_zip(repo_root: Path, out_zip: Path) -> int:
    repo_root = repo_root.resolve()
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    added = 0
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname in INCLUDE_ROOT_FILES:
            p = repo_root / fname
            if p.exists():
                zf.write(p, arcname=fname)
                added += 1
        for dname in INCLUDE_DIRS:
            dpath = repo_root / dname
            if not dpath.exists():
                continue
            for p in sorted(dpath.rglob("*")):
                if not p.is_file():
                    continue
                rel = p.relative_to(repo_root)
                if any(part in EXCLUDE_PARTS for part in rel.parts):
                    continue
                if p.suffix in {".pyc", ".pyo", ".pt", ".pth", ".npz", ".npy", ".zip"}:
                    continue
                zf.write(p, arcname=str(rel))
                added += 1
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clean CS5489 Code ZIP submission archive")
    parser.add_argument("--group", required=True, help="Canvas group number, e.g. 12")
    parser.add_argument("--out-dir", default="artifacts/submission")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_zip = root / args.out_dir / f"ProjectGroup_{args.group}_Code.zip"
    count = build_code_zip(root, out_zip)
    size_mb = out_zip.stat().st_size / (1024 * 1024)
    print(f"Created {out_zip} ({count} files, {size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
