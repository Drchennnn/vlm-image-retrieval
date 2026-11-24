"""Deterministic embedding cache with provenance key validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

from .model import l2_normalize_checked

KEY_FIELDS = (
    "model_key",
    "model_name",
    "pretrained",
    "text_policy",
    "split_name",
    "manifest_sha256",
    "dtype",
    "dim",
)


@dataclass
class EmbeddingBundle:
    image_ids: List[str]
    image_embeddings: np.ndarray
    caption_ids: List[str]
    caption_embeddings: np.ndarray
    metadata: dict
    cache_key: str


def compute_cache_key(meta: dict) -> str:
    payload = {k: meta[k] for k in KEY_FIELDS if k in meta}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"{meta['model_key']}__{meta['split_name']}__{meta['text_policy']}__{digest}"


def save_embedding_bundle(
    cache_dir: Path,
    *,
    image_ids: Sequence[str],
    image_embeddings: np.ndarray,
    caption_ids: Sequence[str],
    caption_embeddings: np.ndarray,
    metadata: dict,
) -> EmbeddingBundle:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    img_embs = l2_normalize_checked(image_embeddings)
    cap_embs = l2_normalize_checked(caption_embeddings)
    if img_embs.shape[1] != cap_embs.shape[1]:
        raise ValueError(
            f"Image dim ({img_embs.shape[1]}) != caption dim ({cap_embs.shape[1]})"
        )
    if len(image_ids) != img_embs.shape[0]:
        raise ValueError("image_ids length does not match image_embeddings rows")
    if len(caption_ids) != cap_embs.shape[0]:
        raise ValueError("caption_ids length does not match caption_embeddings rows")

    full_meta = dict(metadata)
    full_meta["dtype"] = str(img_embs.dtype)
    full_meta["dim"] = int(img_embs.shape[1])
    full_meta["num_images"] = len(image_ids)
    full_meta["num_captions"] = len(caption_ids)

    key = compute_cache_key(full_meta)
    full_meta["cache_key"] = key

    npz_path = cache_dir / f"{key}.npz"
    meta_path = cache_dir / f"{key}.json"

    np.savez(
        npz_path,
        image_ids=np.array(list(image_ids), dtype=str),
        image_embeddings=img_embs,
        caption_ids=np.array(list(caption_ids), dtype=str),
        caption_embeddings=cap_embs,
    )
    meta_path.write_text(json.dumps(full_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return EmbeddingBundle(
        image_ids=list(image_ids),
        image_embeddings=img_embs,
        caption_ids=list(caption_ids),
        caption_embeddings=cap_embs,
        metadata=full_meta,
        cache_key=key,
    )


def load_embedding_bundle(
    cache_dir: Path,
    cache_key: str,
    *,
    expected_manifest_sha256: str | None = None,
) -> EmbeddingBundle:
    cache_dir = Path(cache_dir)
    npz_path = cache_dir / f"{cache_key}.npz"
    meta_path = cache_dir / f"{cache_key}.json"
    if not npz_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Cache files missing for key '{cache_key}' in {cache_dir}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if expected_manifest_sha256 is not None and meta.get("manifest_sha256") != expected_manifest_sha256:
        raise ValueError(
            f"Cache manifest_sha256 mismatch for '{cache_key}': "
            f"{meta.get('manifest_sha256')} != {expected_manifest_sha256}"
        )

    with np.load(npz_path, allow_pickle=False) as data:
        image_ids = [str(x) for x in data["image_ids"].tolist()]
        caption_ids = [str(x) for x in data["caption_ids"].tolist()]
        image_embs = np.asarray(data["image_embeddings"], dtype=np.float32)
        caption_embs = np.asarray(data["caption_embeddings"], dtype=np.float32)

    for label, arr in (("image_embeddings", image_embs), ("caption_embeddings", caption_embs)):
        if arr.ndim != 2 or not np.all(np.isfinite(arr)):
            raise ValueError(f"Corrupt cached {label} in '{cache_key}'")
        norms = np.linalg.norm(arr, axis=1)
        if np.any(np.abs(norms - 1.0) > 1e-3):
            raise ValueError(f"Cached {label} in '{cache_key}' are not unit-normalized")

    return EmbeddingBundle(
        image_ids=image_ids,
        image_embeddings=image_embs,
        caption_ids=caption_ids,
        caption_embeddings=caption_embs,
        metadata=meta,
        cache_key=cache_key,
    )
