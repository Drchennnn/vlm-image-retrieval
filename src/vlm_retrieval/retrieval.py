"""Exact bidirectional cosine similarity computation and deterministic ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .model import l2_normalize_checked


@dataclass(frozen=True)
class RankedResult:
    query_id: str
    ranked_ids: List[str]
    ranked_scores: List[float]


def _validate_embeddings(embs: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(embs, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty 2D array, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Non-finite values detected in {name}")
    return arr


def compute_similarity_matrix(
    query_embs: np.ndarray,
    cand_embs: np.ndarray,
    *,
    chunk_size: int | None = None,
    ensure_normalized: bool = False,
) -> np.ndarray:
    """Compute exact float32 dot-product (cosine) similarity matrix [Q, C]."""
    q = l2_normalize_checked(query_embs) if ensure_normalized else _validate_embeddings(query_embs, "query_embs")
    c = l2_normalize_checked(cand_embs) if ensure_normalized else _validate_embeddings(cand_embs, "cand_embs")
    if q.shape[1] != c.shape[1]:
        raise ValueError(f"Embedding dimension mismatch: query={q.shape[1]} vs cand={c.shape[1]}")

    if chunk_size is None or chunk_size >= q.shape[0]:
        sim = (q @ c.T).astype(np.float32, copy=False)
    else:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        sim = np.empty((q.shape[0], c.shape[0]), dtype=np.float32)
        for start in range(0, q.shape[0], chunk_size):
            end = min(start + chunk_size, q.shape[0])
            sim[start:end] = (q[start:end] @ c.T).astype(np.float32, copy=False)

    if not np.all(np.isfinite(sim)):
        raise ValueError("Non-finite values detected in similarity matrix")
    return sim


def compute_bidirectional_similarity(
    image_embs: np.ndarray,
    caption_embs: np.ndarray,
    *,
    chunk_size: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (sim_i2t [N_img, N_cap], sim_t2i [N_cap, N_img])."""
    sim_i2t = compute_similarity_matrix(image_embs, caption_embs, chunk_size=chunk_size)
    return sim_i2t, sim_i2t.T.copy()


def _cand_tie_order(cand_ids: Sequence[str]) -> np.ndarray:
    """Return integer rank for each candidate under ascending lexicographic cand_id order."""
    if len(cand_ids) != len(set(cand_ids)):
        raise ValueError("Candidate IDs must be unique")
    sorted_unique = sorted(cand_ids)
    id_to_rank = {cid: idx for idx, cid in enumerate(sorted_unique)}
    return np.array([id_to_rank[cid] for cid in cand_ids], dtype=np.int64)


def rank_similarity_matrix(
    sim_matrix: np.ndarray,
    cand_ids: Sequence[str],
    *,
    top_k: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rank every row by score descending, breaking exact ties by ascending cand_id.

    Returns:
        indices: int64 array [Q, K] of candidate column indices
        scores: float32 array [Q, K] of corresponding similarity scores
    """
    sim = np.asarray(sim_matrix, dtype=np.float32)
    if sim.ndim != 2:
        raise ValueError(f"Expected 2D similarity matrix, got shape {sim.shape}")
    if sim.shape[1] != len(cand_ids):
        raise ValueError(f"Similarity cols ({sim.shape[1]}) != len(cand_ids) ({len(cand_ids)})")
    if not np.all(np.isfinite(sim)):
        raise ValueError("Non-finite values detected in similarity matrix")

    n_q, n_c = sim.shape
    k = n_c if top_k is None else min(int(top_k), n_c)
    if k <= 0:
        raise ValueError("top_k must be positive")

    tie_order = _cand_tie_order(cand_ids)
    out_indices = np.empty((n_q, k), dtype=np.int64)
    out_scores = np.empty((n_q, k), dtype=np.float32)

    for q_idx in range(n_q):
        row = sim[q_idx]
        # lexsort sorts by last key first -> primary key: -row (score desc), secondary: tie_order (cand_id asc)
        order = np.lexsort((tie_order, -row))[:k]
        out_indices[q_idx] = order
        out_scores[q_idx] = row[order]

    return out_indices, out_scores


def retrieve_topk(
    query_embs: np.ndarray,
    cand_embs: np.ndarray,
    query_ids: Sequence[str],
    cand_ids: Sequence[str],
    *,
    top_k: int = 10,
    chunk_size: int = 256,
) -> List[RankedResult]:
    """Chunked exact top-K retrieval guaranteed identical to full-matrix ranking."""
    q = _validate_embeddings(query_embs, "query_embs")
    c = _validate_embeddings(cand_embs, "cand_embs")
    if q.shape[0] != len(query_ids):
        raise ValueError("query_ids length mismatch")
    if c.shape[0] != len(cand_ids):
        raise ValueError("cand_ids length mismatch")
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("query_ids must be unique")

    results: List[RankedResult] = []
    for start in range(0, q.shape[0], chunk_size):
        end = min(start + chunk_size, q.shape[0])
        sim_chunk = compute_similarity_matrix(q[start:end], c)
        idx_chunk, score_chunk = rank_similarity_matrix(sim_chunk, cand_ids, top_k=top_k)
        for local_i, q_idx in enumerate(range(start, end)):
            ranked_ids = [str(cand_ids[j]) for j in idx_chunk[local_i].tolist()]
            ranked_scores = [float(s) for s in score_chunk[local_i].tolist()]
            results.append(
                RankedResult(
                    query_id=str(query_ids[q_idx]),
                    ranked_ids=ranked_ids,
                    ranked_scores=ranked_scores,
                )
            )
    return results
