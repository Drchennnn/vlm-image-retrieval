"""Pure-function retrieval metrics, Winoground strict scorer, paired bootstrap CI, and case sampling."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .retrieval import rank_similarity_matrix


def evaluate_retrieval(
    sim_matrix: np.ndarray,
    query_ids: Sequence[str],
    cand_ids: Sequence[str],
    pos_map: Mapping[str, Sequence[str]],
    *,
    ks: Sequence[int] = (1, 5, 10),
    direction: str = "t2i",
    top_k_store: int = 10,
) -> dict:
    """Evaluate exact Recall@K, MedR, MeanR, and MRR with multi-positive support and deterministic ties."""
    sim = np.asarray(sim_matrix, dtype=np.float32)
    if sim.shape != (len(query_ids), len(cand_ids)):
        raise ValueError(
            f"Similarity shape {sim.shape} does not match ({len(query_ids)}, {len(cand_ids)})"
        )
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Duplicate query_id detected")
    if len(cand_ids) != len(set(cand_ids)):
        raise ValueError("Duplicate cand_id detected")

    cand_id_to_col = {str(cid): idx for idx, cid in enumerate(cand_ids)}
    indices, scores = rank_similarity_matrix(sim, cand_ids, top_k=None)

    ranks = np.empty(len(query_ids), dtype=np.int64)
    pos_counts: List[int] = []
    per_query_records: List[dict] = []
    store_k = min(int(top_k_store), len(cand_ids))

    for q_idx, qid in enumerate(query_ids):
        qid_str = str(qid)
        if qid_str not in pos_map:
            raise ValueError(f"Missing relevance entry for query_id '{qid_str}'")
        positives = [str(p) for p in pos_map[qid_str]]
        if not positives:
            raise ValueError(f"Query '{qid_str}' has zero positive candidates")
        pos_cols = set()
        for p in positives:
            if p not in cand_id_to_col:
                raise ValueError(f"Positive candidate '{p}' for query '{qid_str}' not in candidate pool")
            pos_cols.add(cand_id_to_col[p])
        pos_counts.append(len(pos_cols))

        row_order = indices[q_idx]
        first_rank = None
        for rank_0, col in enumerate(row_order.tolist()):
            if col in pos_cols:
                first_rank = rank_0 + 1
                break
        if first_rank is None:  # pragma: no cover
            raise RuntimeError(f"No positive found in ranking for query '{qid_str}'")
        ranks[q_idx] = first_rank

        top_cols = row_order[:store_k].tolist()
        top_scores = scores[q_idx, :store_k].tolist()
        per_query_records.append(
            {
                "query_id": qid_str,
                "first_positive_rank": int(first_rank),
                "num_positives": len(pos_cols),
                "positive_ids": positives,
                "top_k_ids": [str(cand_ids[c]) for c in top_cols],
                "top_k_scores": [round(float(s), 6) for s in top_scores],
            }
        )

    metrics: Dict[str, float] = {}
    for k in ks:
        metrics[f"R@{int(k)}"] = float(100.0 * np.mean(ranks <= int(k)))
    metrics["MedR"] = float(np.median(ranks))
    metrics["MeanR"] = float(np.mean(ranks))
    metrics["MRR"] = float(100.0 * np.mean(1.0 / ranks))

    pos_dist = dict(sorted((str(k), int(v)) for k, v in Counter(pos_counts).items()))
    return {
        "direction": direction,
        "num_queries": len(query_ids),
        "num_candidates": len(cand_ids),
        "positives_per_query_dist": pos_dist,
        "metrics": metrics,
        "ranks": ranks.tolist(),
        "per_query": per_query_records,
    }


def evaluate_winoground(
    score_tuples: Sequence[Tuple[float, float, float, float]],
    *,
    group_ids: Sequence[str] | None = None,
    tags: Sequence[Sequence[str]] | None = None,
    full_benchmark_size: int = 400,
) -> dict:
    """Evaluate Winoground text/image/group scores using strict inequalities (ties fail).

    Each tuple must be (s00, s01, s10, s11) where s_ij = s(C_i, I_j):
        text_success  = (s00 > s10) and (s11 > s01)
        image_success = (s00 > s01) and (s11 > s10)
        group_success = text_success and image_success
    """
    n = len(score_tuples)
    if n == 0:
        raise ValueError("score_tuples must be non-empty")
    if group_ids is not None and len(group_ids) != n:
        raise ValueError("group_ids length mismatch")
    if tags is not None and len(tags) != n:
        raise ValueError("tags length mismatch")

    per_group: List[dict] = []
    text_hits = np.zeros(n, dtype=np.int64)
    image_hits = np.zeros(n, dtype=np.int64)
    group_hits = np.zeros(n, dtype=np.int64)
    tag_buckets: Dict[str, List[int]] = defaultdict(list)

    for idx, tup in enumerate(score_tuples):
        if len(tup) != 4:
            raise ValueError(f"Group {idx} must have 4 scores (s00, s01, s10, s11)")
        s00, s01, s10, s11 = (float(x) for x in tup)
        if not np.all(np.isfinite([s00, s01, s10, s11])):
            raise ValueError(f"Non-finite Winoground score in group {idx}")

        t_ok = (s00 > s10) and (s11 > s01)
        i_ok = (s00 > s01) and (s11 > s10)
        g_ok = t_ok and i_ok
        text_hits[idx] = int(t_ok)
        image_hits[idx] = int(i_ok)
        group_hits[idx] = int(g_ok)

        gid = str(group_ids[idx]) if group_ids is not None else str(idx)
        row_tags = [str(t) for t in tags[idx]] if tags is not None else []
        for t in row_tags:
            tag_buckets[t].append(idx)

        per_group.append(
            {
                "group_id": gid,
                "s00": s00,
                "s01": s01,
                "s10": s10,
                "s11": s11,
                "text_success": bool(t_ok),
                "image_success": bool(i_ok),
                "group_success": bool(g_ok),
                "tags": row_tags,
            }
        )

    tag_breakdown: Dict[str, dict] = {}
    for tag_name, idxs in sorted(tag_buckets.items()):
        idx_arr = np.array(idxs, dtype=np.int64)
        tag_breakdown[tag_name] = {
            "count": int(len(idxs)),
            "text_score": float(100.0 * np.mean(text_hits[idx_arr])),
            "image_score": float(100.0 * np.mean(image_hits[idx_arr])),
            "group_score": float(100.0 * np.mean(group_hits[idx_arr])),
        }

    return {
        "num_groups": n,
        "is_partial": bool(n < full_benchmark_size),
        "metrics": {
            "text_score": float(100.0 * np.mean(text_hits)),
            "image_score": float(100.0 * np.mean(image_hits)),
            "group_score": float(100.0 * np.mean(group_hits)),
        },
        "tag_breakdown": tag_breakdown,
        "per_group": per_group,
    }


def paired_bootstrap_retrieval(
    ranks_a: Sequence[int],
    ranks_b: Sequence[int],
    group_keys: Sequence[str],
    *,
    k: int = 1,
    n_resamples: int = 1000,
    seed: int = 5489,
    alpha: float = 0.05,
) -> dict:
    """Image-grouped paired bootstrap for Recall@K difference (B - A) with fixed candidate pool."""
    ra = np.asarray(ranks_a, dtype=np.int64)
    rb = np.asarray(ranks_b, dtype=np.int64)
    if ra.shape != rb.shape or ra.ndim != 1 or len(ra) != len(group_keys):
        raise ValueError("ranks_a, ranks_b, and group_keys must be 1D arrays of equal length")
    if len(ra) == 0:
        raise ValueError("Empty ranks")

    hit_a = (ra <= k).astype(np.float64) * 100.0
    hit_b = (rb <= k).astype(np.float64) * 100.0

    # Group query indices by image_id so all captions of an image are resampled together
    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, gkey in enumerate(group_keys):
        groups[str(gkey)].append(idx)

    ordered_keys = sorted(groups.keys())
    group_sum_a = np.array([hit_a[groups[g]].sum() for g in ordered_keys], dtype=np.float64)
    group_sum_b = np.array([hit_b[groups[g]].sum() for g in ordered_keys], dtype=np.float64)
    group_cnt = np.array([len(groups[g]) for g in ordered_keys], dtype=np.float64)
    n_groups = len(ordered_keys)

    rng = np.random.Generator(np.random.PCG64(seed))
    boot_diffs = np.empty(n_resamples, dtype=np.float64)
    boot_a = np.empty(n_resamples, dtype=np.float64)
    boot_b = np.empty(n_resamples, dtype=np.float64)

    for b in range(n_resamples):
        draw = rng.integers(0, n_groups, size=n_groups)
        total_cnt = group_cnt[draw].sum()
        ma = group_sum_a[draw].sum() / total_cnt
        mb = group_sum_b[draw].sum() / total_cnt
        boot_a[b] = ma
        boot_b[b] = mb
        boot_diffs[b] = mb - ma

    point_a = float(hit_a.mean())
    point_b = float(hit_b.mean())
    point_diff = point_b - point_a
    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    ci_lo, ci_hi = np.percentile(boot_diffs, [lo_q, hi_q])

    # Two-sided sign-based bootstrap p-value around 0
    p_le_0 = float(np.mean(boot_diffs <= 0.0))
    p_ge_0 = float(np.mean(boot_diffs >= 0.0))
    p_two_sided = float(min(1.0, 2.0 * min(p_le_0, p_ge_0)))

    return {
        "k": int(k),
        "n_queries": int(len(ra)),
        "n_groups": int(n_groups),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "metric_a": round(point_a, 4),
        "metric_b": round(point_b, 4),
        "diff_b_minus_a": round(point_diff, 4),
        "ci_95": [round(float(ci_lo), 4), round(float(ci_hi), 4)],
        "ci_a_95": [round(float(np.percentile(boot_a, lo_q)), 4), round(float(np.percentile(boot_a, hi_q)), 4)],
        "ci_b_95": [round(float(np.percentile(boot_b, lo_q)), 4), round(float(np.percentile(boot_b, hi_q)), 4)],
        "p_value_approx": round(p_two_sided, 4),
    }


def paired_bootstrap_binary(
    success_a: Sequence[bool | int],
    success_b: Sequence[bool | int],
    *,
    n_resamples: int = 1000,
    seed: int = 5489,
    alpha: float = 0.05,
) -> dict:
    """Group-level paired bootstrap for binary indicators (e.g., Winoground group score)."""
    sa = np.asarray(success_a, dtype=np.int64)
    sb = np.asarray(success_b, dtype=np.int64)
    if sa.shape != sb.shape or sa.ndim != 1 or len(sa) == 0:
        raise ValueError("success_a and success_b must be non-empty 1D arrays of equal length")
    # Map 1 -> rank 1, 0 -> rank 2 so R@1 equals binary success rate
    ranks_a = np.where(sa > 0, 1, 2)
    ranks_b = np.where(sb > 0, 1, 2)
    group_keys = [str(i) for i in range(len(sa))]
    return paired_bootstrap_retrieval(
        ranks_a,
        ranks_b,
        group_keys,
        k=1,
        n_resamples=n_resamples,
        seed=seed,
        alpha=alpha,
    )


def sample_stratified_cases(
    query_ids: Sequence[str],
    ranks_baseline: Sequence[int],
    *,
    seed: int = 5489,
    max_per_stratum: int = 10,
) -> dict:
    """Deterministically sample up to max_per_stratum R@1 successes and failures."""
    if len(query_ids) != len(ranks_baseline):
        raise ValueError("query_ids and ranks_baseline length mismatch")
    successes = sorted(str(qid) for qid, r in zip(query_ids, ranks_baseline) if int(r) == 1)
    failures = sorted(str(qid) for qid, r in zip(query_ids, ranks_baseline) if int(r) > 1)

    rng = np.random.Generator(np.random.PCG64(seed))

    def _draw(items: List[str]) -> List[str]:
        if len(items) <= max_per_stratum:
            return items
        idx = sorted(int(i) for i in rng.choice(len(items), size=max_per_stratum, replace=False))
        return [items[i] for i in idx]

    return {
        "seed": int(seed),
        "max_per_stratum": int(max_per_stratum),
        "total_success": len(successes),
        "total_failure": len(failures),
        "sampled_success_ids": _draw(successes),
        "sampled_failure_ids": _draw(failures),
    }
