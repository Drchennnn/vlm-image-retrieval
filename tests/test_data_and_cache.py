"""Round 1 synthetic tests for manifest validation, normalization, and embedding cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vlm_retrieval.cache import load_embedding_bundle, save_embedding_bundle
from vlm_retrieval.data import (
    CaptionRecord,
    ImageRecord,
    SplitData,
    build_relevance_maps,
    validate_split_data,
    validate_splits_disjoint,
)
from vlm_retrieval.model import apply_text_policy, l2_normalize_checked


def _sample_split(name: str, img_ids: list[str], cap_prefix: str) -> SplitData:
    images = [
        ImageRecord(image_id=iid, relative_path=f"img/{iid}.jpg", split=name, file_sha256="a" * 64)
        for iid in img_ids
    ]
    captions = [
        CaptionRecord(caption_id=f"{cap_prefix}_{i}_0", image_id=iid, text=f"caption A for {iid}", split=name)
        for i, iid in enumerate(img_ids)
    ] + [
        CaptionRecord(caption_id=f"{cap_prefix}_{i}_1", image_id=iid, text=f"caption B for {iid}", split=name)
        for i, iid in enumerate(img_ids)
    ]
    return SplitData(name=name, images=images, captions=captions)


def test_split_validation_and_leakage_detection() -> None:
    s1 = _sample_split("dev", ["001", "002"], "d")
    s2 = _sample_split("test", ["003", "004"], "t")
    validate_splits_disjoint([s1, s2])

    # Cross-split image_id leakage must fail
    s_leak = _sample_split("test", ["002", "005"], "l")
    with pytest.raises(ValueError, match="Split leakage: image_id"):
        validate_splits_disjoint([s1, s_leak])

    # Duplicate image_id inside a split must fail
    dup_split = SplitData(
        name="dev",
        images=[
            ImageRecord("001", "a.jpg", "dev", "0" * 64),
            ImageRecord("001", "a.jpg", "dev", "0" * 64),
        ],
        captions=[],
    )
    with pytest.raises(ValueError, match="Duplicate image_id"):
        validate_split_data(dup_split)

    # Caption pointing to missing image must fail
    orphan = SplitData(
        name="dev",
        images=[ImageRecord("001", "a.jpg", "dev", "0" * 64)],
        captions=[CaptionRecord("c1", "999", "orphan", "dev")],
    )
    with pytest.raises(ValueError, match="missing image_id"):
        validate_split_data(orphan)


def test_relevance_maps_multi_positive() -> None:
    split = _sample_split("dev", ["001", "002"], "c")
    t2i, i2t = build_relevance_maps(split)
    assert t2i["c_0_0"] == ["001"]
    assert t2i["c_0_1"] == ["001"]
    assert set(i2t["001"]) == {"c_0_0", "c_0_1"}
    assert set(i2t["002"]) == {"c_1_0", "c_1_1"}


def test_text_policy_and_l2_normalize_guards() -> None:
    assert apply_text_policy("a dog", "raw") == "a dog"
    assert apply_text_policy("a dog", "photo_prefix") == "a photo of a dog"
    with pytest.raises(ValueError):
        apply_text_policy("a dog", "invalid_policy")

    vec = l2_normalize_checked(np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32))
    assert np.allclose(np.linalg.norm(vec, axis=1), 1.0, atol=1e-6)

    with pytest.raises(ValueError, match="Zero-norm"):
        l2_normalize_checked(np.array([[0.0, 0.0]], dtype=np.float32))

    with pytest.raises(ValueError, match="Non-finite"):
        l2_normalize_checked(np.array([[np.nan, 1.0]], dtype=np.float32))


def test_cache_roundtrip_and_manifest_invalidation(tmp_path: Path) -> None:
    img_embs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    cap_embs = np.array([[0.6, 0.8], [0.8, 0.6]], dtype=np.float32)
    meta = {
        "model_key": "clip_vit_b32",
        "model_name": "ViT-B-32",
        "pretrained": "openai",
        "text_policy": "raw",
        "split_name": "dev",
        "manifest_sha256": "deadbeef",
    }
    saved = save_embedding_bundle(
        tmp_path,
        image_ids=["001", "002"],
        image_embeddings=img_embs,
        caption_ids=["c1", "c2"],
        caption_embeddings=cap_embs,
        metadata=meta,
    )
    loaded = load_embedding_bundle(tmp_path, saved.cache_key, expected_manifest_sha256="deadbeef")
    assert np.array_equal(saved.image_embeddings, loaded.image_embeddings)
    assert np.array_equal(saved.caption_embeddings, loaded.caption_embeddings)
    assert loaded.image_ids == ["001", "002"]
    assert loaded.caption_ids == ["c1", "c2"]

    with pytest.raises(ValueError, match="manifest_sha256 mismatch"):
        load_embedding_bundle(tmp_path, saved.cache_key, expected_manifest_sha256="otherhash")
