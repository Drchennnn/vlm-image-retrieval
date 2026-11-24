"""Dataset manifest construction, loading, and strict split-isolation checks."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image

SPLIT_PROTOCOL_V2 = "coco2017-custom-3k-train-1k-dev-1k-test-v2"


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    relative_path: str
    split: str
    file_sha256: str


@dataclass(frozen=True)
class CaptionRecord:
    caption_id: str
    image_id: str
    text: str
    split: str


@dataclass
class SplitData:
    name: str
    images: List[ImageRecord]
    captions: List[CaptionRecord]
    images_manifest_sha256: str = ""
    captions_manifest_sha256: str = ""

    @property
    def image_ids(self) -> List[str]:
        return [img.image_id for img in self.images]

    @property
    def caption_ids(self) -> List[str]:
        return [cap.caption_id for cap in self.captions]

    @property
    def manifest_sha256(self) -> str:
        combined = f"{self.images_manifest_sha256}:{self.captions_manifest_sha256}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _write_jsonl(path: Path, records: Iterable[object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            payload = asdict(rec) if hasattr(rec, "__dataclass_fields__") else rec
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return sha256_file(path)


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def validate_split_data(split: SplitData) -> None:
    """Validate internal integrity of a single split."""
    img_ids = [r.image_id for r in split.images]
    if len(img_ids) != len(set(img_ids)):
        raise ValueError(f"Duplicate image_id detected in split '{split.name}'")

    cap_ids = [r.caption_id for r in split.captions]
    if len(cap_ids) != len(set(cap_ids)):
        raise ValueError(f"Duplicate caption_id detected in split '{split.name}'")

    img_id_set = set(img_ids)
    for cap in split.captions:
        if cap.image_id not in img_id_set:
            raise ValueError(
                f"Caption '{cap.caption_id}' references missing image_id '{cap.image_id}' in split '{split.name}'"
            )
        if cap.split != split.name:
            raise ValueError(
                f"Caption '{cap.caption_id}' split mismatch: '{cap.split}' != '{split.name}'"
            )
    for img in split.images:
        if img.split != split.name:
            raise ValueError(
                f"Image '{img.image_id}' split mismatch: '{img.split}' != '{split.name}'"
            )


def validate_splits_disjoint(splits: Sequence[SplitData]) -> None:
    """Ensure every split is internally valid and mutually disjoint by image_id and caption_id."""
    seen_images: Dict[str, str] = {}
    seen_captions: Dict[str, str] = {}
    for split in splits:
        validate_split_data(split)
        for img in split.images:
            if img.image_id in seen_images:
                raise ValueError(
                    f"Split leakage: image_id '{img.image_id}' appears in both "
                    f"'{seen_images[img.image_id]}' and '{split.name}'"
                )
            seen_images[img.image_id] = split.name
        for cap in split.captions:
            if cap.caption_id in seen_captions:
                raise ValueError(
                    f"Split leakage: caption_id '{cap.caption_id}' appears in both "
                    f"'{seen_captions[cap.caption_id]}' and '{split.name}'"
                )
            seen_captions[cap.caption_id] = split.name


def build_coco_val2017_manifests(
    coco_root: Path,
    manifest_dir: Path,
    *,
    seed: int = 5489,
    n_train: int = 3000,
    n_dev: int = 1000,
    n_test: int = 1000,
    n_smoke: int = 100,
    verify_decode: bool = True,
    tracked_ids_path: Path | None = None,
) -> dict:
    """Build deterministic, image-ID-isolated train/dev/test + smoke manifests from COCO val2017."""
    coco_root = Path(coco_root)
    manifest_dir = Path(manifest_dir)
    ann_path = coco_root / "annotations" / "captions_val2017.json"
    img_dir = coco_root / "val2017"

    if not ann_path.exists():
        raise FileNotFoundError(f"Missing COCO annotations file: {ann_path}")
    if not img_dir.exists():
        raise FileNotFoundError(f"Missing COCO val2017 image directory: {img_dir}")

    with ann_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    ann_sha256 = sha256_file(ann_path)

    # Map numeric image IDs sorted ascending for determinism
    images_by_num_id: Dict[int, dict] = {}
    for item in raw["images"]:
        num_id = int(item["id"])
        if num_id in images_by_num_id:
            raise ValueError(f"Duplicate COCO image id in annotations: {num_id}")
        images_by_num_id[num_id] = item

    captions_by_num_id: Dict[int, List[dict]] = defaultdict(list)
    seen_ann_ids = set()
    for ann in raw["annotations"]:
        ann_id = int(ann["id"])
        if ann_id in seen_ann_ids:
            raise ValueError(f"Duplicate COCO annotation id: {ann_id}")
        seen_ann_ids.add(ann_id)
        captions_by_num_id[int(ann["image_id"])].append(ann)

    sorted_num_ids = sorted(images_by_num_id.keys())
    total_expected = n_train + n_dev + n_test
    if len(sorted_num_ids) != total_expected:
        raise ValueError(
            f"Expected {total_expected} COCO val2017 images, found {len(sorted_num_ids)}"
        )

    # Verify existence, decodability, and file hashes
    missing_files: List[str] = []
    corrupt_files: List[str] = []
    file_hashes: Dict[int, str] = {}
    hash_to_ids: Dict[str, List[str]] = defaultdict(list)

    for num_id in sorted_num_ids:
        file_name = images_by_num_id[num_id]["file_name"]
        img_path = img_dir / file_name
        if not img_path.exists():
            missing_files.append(str(img_path))
            continue
        if verify_decode:
            try:
                with Image.open(img_path) as im:
                    im.verify()
            except Exception as exc:  # pragma: no cover
                corrupt_files.append(f"{img_path}: {exc}")
                continue
        h = sha256_file(img_path)
        file_hashes[num_id] = h
        hash_to_ids[h].append(f"{num_id:012d}")

    if missing_files or corrupt_files:
        raise RuntimeError(
            f"COCO validation failed: {len(missing_files)} missing, {len(corrupt_files)} corrupt"
        )

    duplicate_content_groups = [ids for ids in hash_to_ids.values() if len(ids) > 1]

    # Deterministic permutation with PCG64(seed)
    rng = np.random.Generator(np.random.PCG64(seed))
    perm = rng.permutation(np.array(sorted_num_ids, dtype=np.int64))

    split_num_ids: Dict[str, List[int]] = {
        "train": sorted(int(x) for x in perm[:n_train]),
        "dev": sorted(int(x) for x in perm[n_train : n_train + n_dev]),
        "test": sorted(int(x) for x in perm[n_train + n_dev : n_train + n_dev + n_test]),
    }
    # Smoke subset is deterministically the first n_smoke images from dev
    smoke_num_ids = split_num_ids["dev"][:n_smoke]

    def _make_split(name: str, num_ids: List[int]) -> SplitData:
        img_recs: List[ImageRecord] = []
        cap_recs: List[CaptionRecord] = []
        for num_id in num_ids:
            img_id_str = f"{num_id:012d}"
            file_name = images_by_num_id[num_id]["file_name"]
            rel_path = f"val2017/{file_name}"
            img_recs.append(
                ImageRecord(
                    image_id=img_id_str,
                    relative_path=rel_path,
                    split=name,
                    file_sha256=file_hashes[num_id],
                )
            )
            caps = sorted(captions_by_num_id[num_id], key=lambda c: int(c["id"]))
            if not caps:
                raise ValueError(f"Image {img_id_str} has zero captions")
            for cap in caps:
                cap_recs.append(
                    CaptionRecord(
                        caption_id=str(int(cap["id"])),
                        image_id=img_id_str,
                        text=str(cap["caption"]).strip(),
                        split=name,
                    )
                )
        return SplitData(name=name, images=img_recs, captions=cap_recs)

    main_splits = {name: _make_split(name, ids) for name, ids in split_num_ids.items()}
    validate_splits_disjoint([main_splits["train"], main_splits["dev"], main_splits["test"]])

    smoke_split = _make_split("smoke_dev100", smoke_num_ids)
    validate_split_data(smoke_split)

    all_splits = {**main_splits, "smoke_dev100": smoke_split}

    manifest_dir.mkdir(parents=True, exist_ok=True)
    split_meta: Dict[str, dict] = {}
    for name, split in all_splits.items():
        img_sha = _write_jsonl(manifest_dir / f"{name}_images.jsonl", split.images)
        cap_sha = _write_jsonl(manifest_dir / f"{name}_captions.jsonl", split.captions)
        split.images_manifest_sha256 = img_sha
        split.captions_manifest_sha256 = cap_sha
        caps_per_img = defaultdict(int)
        for c in split.captions:
            caps_per_img[c.image_id] += 1
        counts_dist: Dict[str, int] = defaultdict(int)
        for cnt in caps_per_img.values():
            counts_dist[str(cnt)] += 1
        split_meta[name] = {
            "num_images": len(split.images),
            "num_captions": len(split.captions),
            "captions_per_image_dist": dict(sorted(counts_dist.items())),
            "images_manifest_sha256": img_sha,
            "captions_manifest_sha256": cap_sha,
            "combined_manifest_sha256": split.manifest_sha256,
        }

    summary = {
        "protocol": SPLIT_PROTOCOL_V2,
        "dataset": "COCO val2017",
        "source_annotations": "annotations/captions_val2017.json",
        "source_annotations_sha256": ann_sha256,
        "seed": seed,
        "rng": "numpy.random.Generator(PCG64)",
        "numpy_version": np.__version__,
        "duplicate_image_content_groups": duplicate_content_groups,
        "num_duplicate_content_groups": len(duplicate_content_groups),
        "splits": split_meta,
    }

    summary_path = manifest_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if tracked_ids_path is not None:
        tracked_ids_path = Path(tracked_ids_path)
        tracked_ids_path.parent.mkdir(parents=True, exist_ok=True)
        tracked_payload = {
            "protocol": SPLIT_PROTOCOL_V2,
            "dataset": "COCO val2017",
            "source_annotations_sha256": ann_sha256,
            "seed": seed,
            "numpy_version": np.__version__,
            "splits": {
                name: {
                    "num_images": len(split.images),
                    "num_captions": len(split.captions),
                    "combined_manifest_sha256": split.manifest_sha256,
                    "image_ids": split.image_ids,
                }
                for name, split in all_splits.items()
            },
        }
        tracked_ids_path.write_text(
            json.dumps(tracked_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return summary


def load_split(manifest_dir: Path, split_name: str) -> SplitData:
    """Load and validate a single split manifest from disk."""
    manifest_dir = Path(manifest_dir)
    img_path = manifest_dir / f"{split_name}_images.jsonl"
    cap_path = manifest_dir / f"{split_name}_captions.jsonl"
    if not img_path.exists() or not cap_path.exists():
        raise FileNotFoundError(f"Missing manifest files for split '{split_name}' in {manifest_dir}")

    images = [ImageRecord(**row) for row in _read_jsonl(img_path)]
    captions = [CaptionRecord(**row) for row in _read_jsonl(cap_path)]
    split = SplitData(
        name=split_name,
        images=images,
        captions=captions,
        images_manifest_sha256=sha256_file(img_path),
        captions_manifest_sha256=sha256_file(cap_path),
    )
    validate_split_data(split)
    return split


def build_relevance_maps(split: SplitData) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Return (t2i_pos_map, i2t_pos_map) mapping query_id -> list of positive candidate IDs."""
    t2i_pos: Dict[str, List[str]] = {}
    i2t_pos: Dict[str, List[str]] = defaultdict(list)
    for cap in split.captions:
        t2i_pos[cap.caption_id] = [cap.image_id]
        i2t_pos[cap.image_id].append(cap.caption_id)
    for img in split.images:
        i2t_pos.setdefault(img.image_id, [])
    return t2i_pos, dict(i2t_pos)
