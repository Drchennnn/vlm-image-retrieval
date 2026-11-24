"""Training-free hubness mitigation (CSLS, Dual-Softmax) and image-grouped K-fold CV."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .data import SplitData, build_relevance_maps
from .features import ModalityAligner
from .metrics import evaluate_retrieval
from .retrieval import compute_similarity_matrix


def csls_similarity(sim_i2t: np.ndarray, *, k: int = 10) -> np.ndarray:
    """Cross-Domain Similarity Local Scaling (Conneau et al., 2018) on [N_img, N_cap].

    CSLS(i, j) = 2 * S[i, j] - r_I(i) - r_T(j),
    where r_I(i) is the mean of top-k similarities in row i, and r_T(j) in column j.
    """
    s = np.asarray(sim_i2t, dtype=np.float32)
    if s.ndim != 2 or s.shape[0] == 0 or s.shape[1] == 0:
        raise ValueError(f"Expected non-empty 2D similarity matrix, got {s.shape}")
    if k <= 0:
        raise ValueError("k must be positive")
    if not np.all(np.isfinite(s)):
        raise ValueError("Non-finite values in similarity matrix")

    n_img, n_cap = s.shape
    k_row = min(int(k), n_cap)
    k_col = min(int(k), n_img)

    if k_row < n_cap:
        row_topk = np.partition(s, n_cap - k_row, axis=1)[:, -k_row:]
    else:
        row_topk = s
    r_i = row_topk.mean(axis=1, keepdims=True)

    if k_col < n_img:
        col_topk = np.partition(s, n_img - k_col, axis=0)[-k_col:, :]
    else:
        col_topk = s
    r_t = col_topk.mean(axis=0, keepdims=True)

    out = (2.0 * s - r_i - r_t).astype(np.float32)
    if not np.all(np.isfinite(out)):
        raise ValueError("Non-finite values after CSLS")
    return out


def _log_softmax(x: np.ndarray, axis: int) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    m = np.max(x64, axis=axis, keepdims=True)
    shifted = x64 - m
    lse = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    return shifted - lse


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    return np.exp(_log_softmax(x, axis=axis))


def dual_softmax_similarity(
    sim_i2t: np.ndarray,
    *,
    temperature: float = 0.05,
    mode: str = "symmetric_log",
) -> Tuple[np.ndarray, np.ndarray]:
    """Dual-Softmax hubness mitigation returning (score_i2t [N, M], score_t2i [M, N]).

    Modes:
        - "symmetric_log": log(softmax_row(S/tau) * softmax_col(S/tau)).
          Numerically stable (no float underflow) and order-equivalent to P_row * P_col.
        - "prior_weighted": I2T uses S * softmax_col(S/tau); T2I uses S.T * softmax_row(S/tau).T.
          Downweights hub candidates using their cross-query softmax prior (Cheng et al., 2021).
    """
    s = np.asarray(sim_i2t, dtype=np.float32)
    if s.ndim != 2 or s.shape[0] == 0 or s.shape[1] == 0:
        raise ValueError(f"Expected non-empty 2D similarity matrix, got {s.shape}")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if not np.all(np.isfinite(s)):
        raise ValueError("Non-finite values in similarity matrix")

    scaled = s.astype(np.float64) / float(temperature)
    if mode == "symmetric_log":
        score = (_log_softmax(scaled, axis=1) + _log_softmax(scaled, axis=0)).astype(np.float32)
        return score, score.T.copy()
    if mode == "prior_weighted":
        # For I2T (rows=images querying captions), downweight hub captions via col-wise softmax
        p_col = _softmax(scaled, axis=0)
        # For T2I (cols=captions querying images), downweight hub images via row-wise softmax
        p_row = _softmax(scaled, axis=1)
        s_i2t = (s.astype(np.float64) * p_col).astype(np.float32)
        s_t2i = (s.astype(np.float64) * p_row).T.astype(np.float32)
        return s_i2t, s_t2i
    raise ValueError(f"Unsupported Dual-Softmax mode '{mode}'")


def make_image_grouped_kfold(
    image_ids: Sequence[str],
    *,
    n_splits: int = 5,
    seed: int = 5489,
) -> List[Tuple[List[str], List[str]]]:
    """Deterministically partition unique image_ids into n_splits mutually disjoint folds."""
    unique_ids = sorted(set(str(x) for x in image_ids))
    if len(unique_ids) != len(image_ids):
        raise ValueError("image_ids must be unique")
    if n_splits < 2 or n_splits > len(unique_ids):
        raise ValueError(f"Invalid n_splits={n_splits} for {len(unique_ids)} images")

    rng = np.random.Generator(np.random.PCG64(seed))
    perm = rng.permutation(len(unique_ids))
    fold_indices = np.array_split(perm, n_splits)

    folds: List[Tuple[List[str], List[str]]] = []
    for val_idx in fold_indices:
        val_set = set(int(i) for i in val_idx.tolist())
        val_ids = sorted(unique_ids[i] for i in val_set)
        train_ids = sorted(unique_ids[i] for i in range(len(unique_ids)) if i not in val_set)
        folds.append((train_ids, val_ids))
    return folds


def _subset_split_arrays(
    split: SplitData,
    img_embs: np.ndarray,
    cap_embs: np.ndarray,
    selected_image_ids: Sequence[str],
) -> Tuple[List[str], np.ndarray, List[str], np.ndarray, Dict[str, List[str]], Dict[str, List[str]]]:
    sel_set = set(selected_image_ids)
    img_rows = [i for i, rec in enumerate(split.images) if rec.image_id in sel_set]
    cap_rows = [i for i, rec in enumerate(split.captions) if rec.image_id in sel_set]

    sub_img_ids = [split.images[i].image_id for i in img_rows]
    sub_cap_ids = [split.captions[i].caption_id for i in cap_rows]
    sub_img_embs = img_embs[img_rows]
    sub_cap_embs = cap_embs[cap_rows]

    t2i_pos: Dict[str, List[str]] = {}
    i2t_pos: Dict[str, List[str]] = {iid: [] for iid in sub_img_ids}
    for i in cap_rows:
        cap = split.captions[i]
        t2i_pos[cap.caption_id] = [cap.image_id]
        i2t_pos[cap.image_id].append(cap.caption_id)

    return sub_img_ids, sub_img_embs, sub_cap_ids, sub_cap_embs, t2i_pos, i2t_pos


def cross_validate_reranker(
    split: SplitData,
    image_embs: np.ndarray,
    caption_embs: np.ndarray,
    *,
    method: str,
    param_grid: Sequence[dict],
    n_splits: int = 5,
    seed: int = 5489,
) -> dict:
    """Run image-grouped K-fold CV on dev split to tune CSLS or Dual-Softmax hyperparameters."""
    if method not in ("csls", "dual_softmax"):
        raise ValueError(f"Unsupported reranker method '{method}'")
    folds = make_image_grouped_kfold(split.image_ids, n_splits=n_splits, seed=seed)

    # Pre-extract fold validation subsets
    val_folds = [
        _subset_split_arrays(split, image_embs, caption_embs, val_ids)
        for _, val_ids in folds
    ]

    results: List[dict] = []
    for params in param_grid:
        fold_t2i: List[float] = []
        fold_i2t: List[float] = []
        fold_mean: List[float] = []
        for sub_img_ids, sub_v, sub_cap_ids, sub_t, t2i_pos, i2t_pos in val_folds:
            base_i2t = compute_similarity_matrix(sub_v, sub_t)
            if method == "csls":
                s_i2t = csls_similarity(base_i2t, k=int(params["k"]))
                s_t2i = s_i2t.T
            else:
                s_i2t, s_t2i = dual_softmax_similarity(
                    base_i2t,
                    temperature=float(params["temperature"]),
                    mode=str(params.get("mode", "symmetric_log")),
                )
            res_t2i = evaluate_retrieval(s_t2i, sub_cap_ids, sub_img_ids, t2i_pos, ks=(1, 5, 10), direction="t2i")
            res_i2t = evaluate_retrieval(s_i2t, sub_img_ids, sub_cap_ids, i2t_pos, ks=(1, 5, 10), direction="i2t")
            r1_t = res_t2i["metrics"]["R@1"]
            r1_i = res_i2t["metrics"]["R@1"]
            fold_t2i.append(r1_t)
            fold_i2t.append(r1_i)
            fold_mean.append(0.5 * (r1_t + r1_i))

        results.append(
            {
                "params": dict(params),
                "mean_r1": round(float(np.mean(fold_mean)), 4),
                "std_r1": round(float(np.std(fold_mean)), 4),
                "t2i_r1_mean": round(float(np.mean(fold_t2i)), 4),
                "i2t_r1_mean": round(float(np.mean(fold_i2t)), 4),
                "fold_mean_r1": [round(x, 4) for x in fold_mean],
            }
        )

    # Deterministic best selection: highest mean_r1, ties broken by grid order
    best_idx = max(range(len(results)), key=lambda i: (results[i]["mean_r1"], -i))
    return {
        "method": method,
        "n_splits": int(n_splits),
        "seed": int(seed),
        "best": results[best_idx],
        "candidates": results,
    }


def cross_validate_modality_aligner(
    split: SplitData,
    image_embs: np.ndarray,
    caption_embs: np.ndarray,
    *,
    param_grid: Sequence[dict],
    n_splits: int = 5,
    seed: int = 5489,
) -> dict:
    """Image-grouped K-fold CV for ModalityAligner: fit ONLY on train folds, score on val fold."""
    folds = make_image_grouped_kfold(split.image_ids, n_splits=n_splits, seed=seed)
    fold_pairs = [
        (
            _subset_split_arrays(split, image_embs, caption_embs, train_ids),
            _subset_split_arrays(split, image_embs, caption_embs, val_ids),
        )
        for train_ids, val_ids in folds
    ]

    results: List[dict] = []
    for params in param_grid:
        mode = str(params["mode"])
        eps = float(params.get("eps", 1e-2))
        fold_t2i: List[float] = []
        fold_i2t: List[float] = []
        fold_mean: List[float] = []
        for (_, tr_v, _, tr_t, _, _), (val_img_ids, val_v, val_cap_ids, val_t, t2i_pos, i2t_pos) in fold_pairs:
            aligner = ModalityAligner(mode=mode, eps=eps).fit(tr_v, tr_t)
            v_aligned, t_aligned = aligner.transform(val_v, val_t)
            s_i2t = compute_similarity_matrix(v_aligned, t_aligned)
            res_t2i = evaluate_retrieval(s_i2t.T, val_cap_ids, val_img_ids, t2i_pos, ks=(1, 5, 10), direction="t2i")
            res_i2t = evaluate_retrieval(s_i2t, val_img_ids, val_cap_ids, i2t_pos, ks=(1, 5, 10), direction="i2t")
            r1_t = res_t2i["metrics"]["R@1"]
            r1_i = res_i2t["metrics"]["R@1"]
            fold_t2i.append(r1_t)
            fold_i2t.append(r1_i)
            fold_mean.append(0.5 * (r1_t + r1_i))

        results.append(
            {
                "params": dict(params),
                "mean_r1": round(float(np.mean(fold_mean)), 4),
                "std_r1": round(float(np.std(fold_mean)), 4),
                "t2i_r1_mean": round(float(np.mean(fold_t2i)), 4),
                "i2t_r1_mean": round(float(np.mean(fold_i2t)), 4),
                "fold_mean_r1": [round(x, 4) for x in fold_mean],
            }
        )

    best_idx = max(range(len(results)), key=lambda i: (results[i]["mean_r1"], -i))
    return {
        "method": "modality_aligner",
        "n_splits": int(n_splits),
        "seed": int(seed),
        "best": results[best_idx],
        "candidates": results,
    }
