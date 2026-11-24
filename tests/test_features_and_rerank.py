"""Synthetic unit tests for ModalityAligner, CrossModalPCA, Hubness, CSLS, Dual-Softmax, and 5-fold CV."""

from __future__ import annotations

import numpy as np
import pytest

from vlm_retrieval.data import CaptionRecord, ImageRecord, SplitData
from vlm_retrieval.features import (
    CrossModalPCA,
    ModalityAligner,
    analyze_semantic_clusters,
    compute_hubness_stats,
    compute_modality_gap_stats,
)
from vlm_retrieval.metrics import evaluate_retrieval
from vlm_retrieval.rerank import (
    cross_validate_modality_aligner,
    cross_validate_reranker,
    csls_similarity,
    dual_softmax_similarity,
    make_image_grouped_kfold,
)


def _make_synthetic_split(n_images: int = 20, caps_per_img: int = 2, dim: int = 16, seed: int = 7):
    rng = np.random.Generator(np.random.PCG64(seed))
    images = [
        ImageRecord(f"{i:04d}", f"img/{i:04d}.jpg", "dev", "0" * 64)
        for i in range(n_images)
    ]
    captions = []
    img_base = rng.normal(size=(n_images, dim)).astype(np.float32)
    # Inject a systematic modality offset (Modality Gap)
    gap_vec = np.zeros(dim, dtype=np.float32)
    gap_vec[0] = 2.0
    img_embs = img_base + gap_vec[None, :]
    img_embs /= np.linalg.norm(img_embs, axis=1, keepdims=True)

    cap_embs_list = []
    cid = 0
    for i in range(n_images):
        for _ in range(caps_per_img):
            captions.append(CaptionRecord(f"c{cid:04d}", f"{i:04d}", f"caption {cid}", "dev"))
            c_vec = img_base[i] - gap_vec + 0.15 * rng.normal(size=(dim,)).astype(np.float32)
            c_vec /= np.linalg.norm(c_vec)
            cap_embs_list.append(c_vec)
            cid += 1

    split = SplitData(name="dev", images=images, captions=captions)
    return split, img_embs, np.stack(cap_embs_list, axis=0)


def test_modality_aligner_closes_gap_and_prevents_test_leakage() -> None:
    _, v_dev, t_dev = _make_synthetic_split(n_images=30, caps_per_img=2, dim=12, seed=1)
    _, v_test, t_test = _make_synthetic_split(n_images=10, caps_per_img=2, dim=12, seed=2)

    gap_before = compute_modality_gap_stats(v_dev, t_dev)
    assert gap_before["centroid_l2_distance"] > 0.5

    aligner = ModalityAligner(mode="center").fit(v_dev, t_dev)
    v_dev_c, t_dev_c = aligner.transform(v_dev, t_dev)
    gap_after = compute_modality_gap_stats(v_dev_c, t_dev_c)
    assert gap_after["centroid_l2_distance"] < gap_before["centroid_l2_distance"]

    # Anti-leakage: transforming v_test[0] must depend ONLY on fit(dev), not on other test rows
    v_t1, t_t1 = aligner.transform(v_test, t_test)
    v_test_mut = v_test.copy()
    v_test_mut[1:] = -v_test_mut[1:]
    v_t2, _ = aligner.transform(v_test_mut, t_test)
    assert np.allclose(v_t1[0], v_t2[0], atol=1e-6)

    # ZCA whitening modes produce valid unit-norm vectors
    for mode in ("zca_per_modality", "zca_shared"):
        z = ModalityAligner(mode=mode, eps=0.1).fit(v_dev, t_dev)
        vz, tz = z.transform(v_test, t_test)
        assert np.allclose(np.linalg.norm(vz, axis=1), 1.0, atol=1e-5)
        assert np.allclose(np.linalg.norm(tz, axis=1), 1.0, atol=1e-5)


def test_cross_modal_pca_and_clustering() -> None:
    _, v_dev, t_dev = _make_synthetic_split(n_images=25, caps_per_img=2, dim=16, seed=3)
    for mode in ("naive_pca", "uncentered_svd", "modality_centered_pca"):
        pca = CrossModalPCA(n_components=8, seed=5489, mode=mode).fit(v_dev, t_dev)
        vp, tp = pca.transform(v_dev, t_dev)
        assert vp.shape == (25, 8)
        assert tp.shape == (50, 8)
        assert np.allclose(np.linalg.norm(vp, axis=1), 1.0, atol=1e-5)
        assert 0.0 < pca.explained_variance_ratio_sum <= 1.0001

    cl = analyze_semantic_clusters(v_dev, [100.0] * 25, n_clusters=4, seed=5489)
    assert cl["n_clusters"] == 4
    assert sum(c["count"] for c in cl["clusters"]) == 25


def test_hubness_rescue_by_csls_and_dual_softmax() -> None:
    # Construct 6x6 similarity where col 0 is a strong "hub" that beats true diagonals for queries 1..5
    sim = np.eye(6, dtype=np.float32) * 0.70 + 0.10
    # Make column 0 a hub: high similarity (0.78) to EVERY query, while true diagonal is 0.80 for query 0 and 0.75 for 1..5
    sim[:, 0] = 0.78
    sim[0, 0] = 0.85
    for i in range(1, 6):
        sim[i, i] = 0.75

    ids = [f"id{i}" for i in range(6)]
    pos = {x: [x] for x in ids}

    # Naive cosine fails on queries 1..5 (hub col 0 wins!) -> R@1 = 1/6
    base = evaluate_retrieval(sim, ids, ids, pos, ks=(1,), direction="i2t")
    assert base["metrics"]["R@1"] == pytest.approx(100.0 / 6.0)

    # Hubness stats must detect col 0 as max hub (occurrence = 6 at k=1)
    hub = compute_hubness_stats(sim, k=1)
    assert hub["max_occurrence"] == 6
    assert hub["skewness_snk"] > 1.0

    # CSLS penalizes col 0's high mean neighborhood similarity and recovers R@1 = 100%!
    sim_csls = csls_similarity(sim, k=3)
    res_csls = evaluate_retrieval(sim_csls, ids, ids, pos, ks=(1,), direction="i2t")
    assert res_csls["metrics"]["R@1"] == pytest.approx(100.0)

    # Dual-Softmax (both modes) also demotes hub col 0 and recovers R@1 = 100%!
    for mode in ("symmetric_log", "prior_weighted"):
        s_i2t, s_t2i = dual_softmax_similarity(sim, temperature=0.05, mode=mode)
        res_ds = evaluate_retrieval(s_i2t, ids, ids, pos, ks=(1,), direction="i2t")
        assert res_ds["metrics"]["R@1"] == pytest.approx(100.0)
        assert np.all(np.isfinite(s_t2i))

    # Extreme small temperature stability check (no NaN/Inf in symmetric_log)
    s_small, _ = dual_softmax_similarity(sim, temperature=0.001, mode="symmetric_log")
    assert np.all(np.isfinite(s_small))


def test_image_grouped_5fold_cv_disjoint_and_runs_cleanly() -> None:
    split, v, t = _make_synthetic_split(n_images=25, caps_per_img=2, dim=12, seed=9)
    folds = make_image_grouped_kfold(split.image_ids, n_splits=5, seed=5489)
    assert len(folds) == 5

    all_val: list[str] = []
    for tr_ids, val_ids in folds:
        assert set(tr_ids).isdisjoint(set(val_ids))
        assert len(tr_ids) + len(val_ids) == 25
        all_val.extend(val_ids)
    assert sorted(all_val) == sorted(split.image_ids)

    cv_csls = cross_validate_reranker(
        split, v, t, method="csls", param_grid=[{"k": 3}, {"k": 5}], n_splits=5, seed=5489
    )
    assert "best" in cv_csls and len(cv_csls["candidates"]) == 2

    cv_ds = cross_validate_reranker(
        split,
        v,
        t,
        method="dual_softmax",
        param_grid=[{"temperature": 0.05, "mode": "symmetric_log"}, {"temperature": 0.1, "mode": "symmetric_log"}],
        n_splits=5,
        seed=5489,
    )
    assert "best" in cv_ds and len(cv_ds["candidates"]) == 2

    cv_align = cross_validate_modality_aligner(
        split,
        v,
        t,
        param_grid=[{"mode": "center"}, {"mode": "zca_shared", "eps": 0.1}],
        n_splits=5,
        seed=5489,
    )
    assert "best" in cv_align and len(cv_align["candidates"]) == 2
