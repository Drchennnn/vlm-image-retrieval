"""Synthetic unit tests for retrieval ranking, bidirectional Recall@K, Winoground, and paired bootstrap."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from vlm_retrieval.metrics import (
    evaluate_retrieval,
    evaluate_winoground,
    paired_bootstrap_binary,
    paired_bootstrap_retrieval,
    sample_stratified_cases,
)
from vlm_retrieval.retrieval import (
    compute_similarity_matrix,
    rank_similarity_matrix,
    retrieve_topk,
)


def test_exact_vs_chunked_similarity_and_topk() -> None:
    rng = np.random.Generator(np.random.PCG64(42))
    q = rng.normal(size=(23, 16)).astype(np.float32)
    c = rng.normal(size=(17, 16)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    c /= np.linalg.norm(c, axis=1, keepdims=True)

    full = compute_similarity_matrix(q, c, chunk_size=None)
    chunked = compute_similarity_matrix(q, c, chunk_size=5)
    assert np.allclose(full, chunked, atol=1e-6)

    q_ids = [f"q{i:02d}" for i in range(23)]
    c_ids = [f"c{i:02d}" for i in range(17)]
    idx_full, sc_full = rank_similarity_matrix(full, c_ids, top_k=5)
    res_chunked = retrieve_topk(q, c, q_ids, c_ids, top_k=5, chunk_size=7)
    for i, r in enumerate(res_chunked):
        expected_ids = [c_ids[j] for j in idx_full[i].tolist()]
        assert r.ranked_ids == expected_ids
        assert np.allclose(r.ranked_scores, sc_full[i], atol=1e-6)


def test_deterministic_tie_breaking_by_ascending_cand_id() -> None:
    # Identical scores across all candidates -> must sort by ascending cand_id
    sim = np.array([[0.5, 0.5, 0.5], [0.2, 0.8, 0.8]], dtype=np.float32)
    cand_ids = ["img_010", "img_002", "img_001"]
    idx, _ = rank_similarity_matrix(sim, cand_ids, top_k=3)
    assert [cand_ids[j] for j in idx[0].tolist()] == ["img_001", "img_002", "img_010"]
    assert [cand_ids[j] for j in idx[1].tolist()] == ["img_001", "img_002", "img_010"]


def test_retrieval_perfect_known_rank_and_multi_positive_i2t() -> None:
    # 2 images, 4 captions (2 captions per image)
    img_ids = ["I0", "I1"]
    cap_ids = ["C0a", "C0b", "C1a", "C1b"]
    # I0 matches C0b at rank 1 (score 0.9) and C0a at rank 4 (score 0.1) -> multi-positive rank MUST be 1!
    # I1 matches C1a at rank 2 (score 0.6) and C1b at rank 3 (score 0.4) -> multi-positive rank MUST be 2!
    sim_i2t = np.array(
        [
            [0.1, 0.9, 0.5, 0.2],
            [0.8, 0.1, 0.6, 0.4],
        ],
        dtype=np.float32,
    )
    i2t_pos = {"I0": ["C0a", "C0b"], "I1": ["C1a", "C1b"]}
    res = evaluate_retrieval(sim_i2t, img_ids, cap_ids, i2t_pos, ks=(1, 2), direction="i2t")
    assert res["ranks"] == [1, 2]
    assert res["metrics"]["R@1"] == pytest.approx(50.0)
    assert res["metrics"]["R@2"] == pytest.approx(100.0)
    assert res["metrics"]["MedR"] == pytest.approx(1.5)
    assert res["metrics"]["MRR"] == pytest.approx(75.0)
    assert res["positives_per_query_dist"] == {"2": 2}

    # Check axis-swap detection: passing [N_img, N_cap] when expecting [N_cap, N_img] raises
    t2i_pos = {"C0a": ["I0"], "C0b": ["I0"], "C1a": ["I1"], "C1b": ["I1"]}
    with pytest.raises(ValueError, match="does not match"):
        evaluate_retrieval(sim_i2t, cap_ids, img_ids, t2i_pos, direction="t2i")

    # Proper transpose T2I evaluation + invariance under column/row shuffling
    sim_t2i = sim_i2t.T
    res_t2i = evaluate_retrieval(sim_t2i, cap_ids, img_ids, t2i_pos, ks=(1, 2), direction="t2i")
    perm_q = [2, 0, 3, 1]
    perm_c = [1, 0]
    res_shuf = evaluate_retrieval(
        sim_t2i[np.ix_(perm_q, perm_c)],
        [cap_ids[i] for i in perm_q],
        [img_ids[j] for j in perm_c],
        t2i_pos,
        ks=(1, 2),
        direction="t2i",
    )
    assert res_t2i["metrics"] == res_shuf["metrics"]


def test_winoground_strict_inequalities_and_24_permutations() -> None:
    # 1) All pass, text-only pass, image-only pass, tie fails strictly
    cases = [
        (0.9, 0.1, 0.2, 0.8),  # both pass -> group=True
        (0.9, 0.95, 0.1, 0.98),  # text: 0.9>0.1 & 0.98>0.95 (True); image: 0.9>0.95 (False)
        (0.9, 0.1, 0.95, 0.98),  # text: 0.9>0.95 (False); image: 0.9>0.1 & 0.98>0.95 (True)
        (0.8, 0.8, 0.1, 0.9),  # tie s00==s01 -> image False, text True, group False
        (0.8, 0.1, 0.8, 0.9),  # tie s00==s10 -> text False, image True, group False
    ]
    out = evaluate_winoground(cases, tags=[["object"], ["relation"], ["both"], ["object"], ["relation"]])
    assert out["is_partial"] is True
    assert out["per_group"][0]["group_success"] is True
    assert out["per_group"][1]["text_success"] is True and out["per_group"][1]["image_success"] is False
    assert out["per_group"][2]["image_success"] is True and out["per_group"][2]["text_success"] is False
    assert out["per_group"][3]["image_success"] is False and out["per_group"][3]["group_success"] is False
    assert out["per_group"][4]["text_success"] is False and out["per_group"][4]["group_success"] is False

    # 2) Exhaustive 4! = 24 permutations of 4 distinct scores:
    # exact random baselines MUST be text=6/24 (25%), image=6/24 (25%), group=4/24 (16.6667%)
    perms = list(itertools.permutations([1.0, 2.0, 3.0, 4.0]))
    perm_res = evaluate_winoground(perms, full_benchmark_size=24)
    assert perm_res["is_partial"] is False
    assert perm_res["metrics"]["text_score"] == pytest.approx(25.0)
    assert perm_res["metrics"]["image_score"] == pytest.approx(25.0)
    assert perm_res["metrics"]["group_score"] == pytest.approx(100.0 * 4.0 / 24.0)


def test_paired_bootstrap_and_stratified_sampling() -> None:
    ranks_a = [1, 3, 2, 1, 5, 1]
    ranks_b = [1, 1, 1, 1, 2, 1]
    groups = ["img0", "img0", "img1", "img1", "img2", "img2"]

    same = paired_bootstrap_retrieval(ranks_a, ranks_a, groups, k=1, n_resamples=200, seed=5489)
    assert same["diff_b_minus_a"] == 0.0
    assert same["ci_95"] == [0.0, 0.0]

    improved = paired_bootstrap_retrieval(ranks_a, ranks_b, groups, k=1, n_resamples=300, seed=5489)
    assert improved["diff_b_minus_a"] > 0.0
    assert improved["n_groups"] == 3

    wg_boot = paired_bootstrap_binary([1, 0, 0, 1], [1, 1, 0, 1], n_resamples=200, seed=5489)
    assert wg_boot["diff_b_minus_a"] == pytest.approx(25.0)

    sampled = sample_stratified_cases([f"q{i}" for i in range(6)], ranks_a, seed=5489, max_per_stratum=2)
    assert sampled["total_success"] == 3
    assert sampled["total_failure"] == 3
    assert len(sampled["sampled_success_ids"]) == 2
    assert len(sampled["sampled_failure_ids"]) == 2
