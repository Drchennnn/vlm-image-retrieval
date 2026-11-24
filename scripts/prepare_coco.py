"""Prepare and validate deterministic 3K/1K/1K COCO val2017 manifests."""

from __future__ import annotations

import json
from pathlib import Path

from vlm_retrieval.data import build_coco_val2017_manifests


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    summary = build_coco_val2017_manifests(
        coco_root=root / "data" / "raw" / "coco",
        manifest_dir=root / "data" / "manifests" / "coco2017",
        seed=5489,
        n_train=3000,
        n_dev=1000,
        n_test=1000,
        n_smoke=100,
        verify_decode=True,
        tracked_ids_path=root / "configs" / "splits" / "coco2017_split_ids.json",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
