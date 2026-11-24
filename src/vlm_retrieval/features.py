"""New feature representations (Modality Gap alignment), PCA compression, clustering, and hubness diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from .model import l2_normalize_checked

VALID_ALIGN_MODES = ("center", "zca_per_modality", "zca_shared")


def _zca_matrix(centered: np.ndarray, eps: float) -> np.ndarray:
    """Compute regularized ZCA whitening matrix (Cov + eps * I)^(-1/2) in float64."""
    x = np.asarray(centered, dtype=np.float64)
    n = x.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 samples to estimate covariance")
    cov = (x.T @ x) / float(n)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    inv_sqrt = 1.0 / np.sqrt(eigvals + float(eps))
    w = (eigvecs * inv_sqrt[None, :]) @ eigvecs.T
    return w.astype(np.float32)


class ModalityAligner:
    """Fit-on-dev cross-modal feature transformation addressing Modality Gap & anisotropy."""

    def __init__(self, mode: str = "center", eps: float = 1e-2) -> None:
        if mode not in VALID_ALIGN_MODES:
            raise ValueError(f"Unsupported mode '{mode}'; expected one of {VALID_ALIGN_MODES}")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.mode = mode
        self.eps = float(eps)
        self.mu_img_: np.ndarray | None = None
        self.mu_cap_: np.ndarray | None = None
        self.w_img_: np.ndarray | None = None
        self.w_cap_: np.ndarray | None = None

    def fit(self, img_embs: np.ndarray, cap_embs: np.ndarray) -> "ModalityAligner":
        v = l2_normalize_checked(img_embs)
        t = l2_normalize_checked(cap_embs)
        if v.shape[1] != t.shape[1]:
            raise ValueError("Image and caption embedding dimensions must match")

        self.mu_img_ = v.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.mu_cap_ = t.mean(axis=0, dtype=np.float64).astype(np.float32)

        vc = v - self.mu_img_[None, :]
        tc = t - self.mu_cap_[None, :]

        if self.mode == "center":
            self.w_img_ = None
            self.w_cap_ = None
        elif self.mode == "zca_per_modality":
            self.w_img_ = _zca_matrix(vc, self.eps)
            self.w_cap_ = _zca_matrix(tc, self.eps)
        elif self.mode == "zca_shared":
            stacked = np.concatenate([vc, tc], axis=0)
            w = _zca_matrix(stacked, self.eps)
            self.w_img_ = w
            self.w_cap_ = w
        return self

    def transform(self, img_embs: np.ndarray, cap_embs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.mu_img_ is None or self.mu_cap_ is None:
            raise RuntimeError("ModalityAligner must be fit() before transform()")
        v = np.asarray(img_embs, dtype=np.float32) - self.mu_img_[None, :]
        t = np.asarray(cap_embs, dtype=np.float32) - self.mu_cap_[None, :]
        if self.w_img_ is not None and self.w_cap_ is not None:
            v = v @ self.w_img_
            t = t @ self.w_cap_
        return l2_normalize_checked(v), l2_normalize_checked(t)


class CrossModalPCA:
    """Fit-on-dev joint PCA dimensionality reduction preserving shared cross-modal space."""

    def __init__(self, n_components: int, seed: int = 5489) -> None:
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        self.n_components = int(n_components)
        self.seed = int(seed)
        self.pca_: PCA | None = None

    def fit(self, img_embs: np.ndarray, cap_embs: np.ndarray) -> "CrossModalPCA":
        v = l2_normalize_checked(img_embs)
        t = l2_normalize_checked(cap_embs)
        if v.shape[1] != t.shape[1]:
            raise ValueError("Image and caption embedding dimensions must match")
        max_dim = min(v.shape[1], v.shape[0] + t.shape[0])
        if self.n_components > max_dim:
            raise ValueError(f"n_components={self.n_components} exceeds max_dim={max_dim}")
        stacked = np.concatenate([v, t], axis=0)
        self.pca_ = PCA(n_components=self.n_components, random_state=self.seed)
        self.pca_.fit(stacked)
        return self

    @property
    def explained_variance_ratio_sum(self) -> float:
        if self.pca_ is None:
            raise RuntimeError("CrossModalPCA must be fit() first")
        return float(np.sum(self.pca_.explained_variance_ratio_))

    def transform(self, img_embs: np.ndarray, cap_embs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.pca_ is None:
            raise RuntimeError("CrossModalPCA must be fit() before transform()")
        v_proj = self.pca_.transform(np.asarray(img_embs, dtype=np.float32)).astype(np.float32)
        t_proj = self.pca_.transform(np.asarray(cap_embs, dtype=np.float32)).astype(np.float32)
        return l2_normalize_checked(v_proj), l2_normalize_checked(t_proj)


def compute_modality_gap_stats(img_embs: np.ndarray, cap_embs: np.ndarray) -> dict:
    """Quantify geometric modality gap between unit-normalized image and caption embeddings."""
    v = l2_normalize_checked(img_embs)
    t = l2_normalize_checked(cap_embs)
    mu_v = v.mean(axis=0, dtype=np.float64)
    mu_t = t.mean(axis=0, dtype=np.float64)
    l2_dist = float(np.linalg.norm(mu_v - mu_t))
    mu_v_norm = float(np.linalg.norm(mu_v))
    mu_t_norm = float(np.linalg.norm(mu_t))
    cos_centroids = (
        float(np.dot(mu_v, mu_t) / (mu_v_norm * mu_t_norm))
        if mu_v_norm > 1e-12 and mu_t_norm > 1e-12
        else 0.0
    )
    # Within-modality average pairwise cosine equals ||mean||^2 (cone concentration / anisotropy)
    return {
        "centroid_l2_distance": round(l2_dist, 6),
        "centroid_cosine": round(cos_centroids, 6),
        "image_mean_norm": round(mu_v_norm, 6),
        "caption_mean_norm": round(mu_t_norm, 6),
        "image_within_avg_cos": round(mu_v_norm**2, 6),
        "caption_within_avg_cos": round(mu_t_norm**2, 6),
    }


def _gini(counts: np.ndarray) -> float:
    x = np.sort(np.asarray(counts, dtype=np.float64))
    total = x.sum()
    if total <= 0.0 or len(x) == 0:
        return 0.0
    n = len(x)
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(idx * x) / (n * total)) - (n + 1.0) / n)


def compute_hubness_stats(sim_matrix: np.ndarray, *, k: int = 10) -> dict:
    """Compute k-occurrence hubness statistics (skewness, Gini, orphans, max hub) for candidates."""
    sim = np.asarray(sim_matrix, dtype=np.float32)
    if sim.ndim != 2:
        raise ValueError("sim_matrix must be 2D")
    n_q, n_c = sim.shape
    k_eff = min(int(k), n_c)
    if k_eff <= 0:
        raise ValueError("k must be positive")

    # Top-k candidate indices per query
    if k_eff < n_c:
        top_idx = np.argpartition(-sim, kth=k_eff - 1, axis=1)[:, :k_eff]
    else:
        top_idx = np.broadcast_to(np.arange(n_c)[None, :], (n_q, n_c))

    counts = np.bincount(top_idx.ravel(), minlength=n_c).astype(np.float64)
    mean_c = float(counts.mean())
    std_c = float(counts.std())
    if std_c > 1e-12:
        skewness = float(np.mean(((counts - mean_c) / std_c) ** 3))
    else:
        skewness = 0.0

    # Robin Hood / orphan metrics
    expected = n_q * k_eff / float(n_c)
    orphan_rate = float(np.mean(counts == 0))
    robin_hood = float(0.5 * np.sum(np.abs(counts - expected)) / (n_q * k_eff))

    return {
        "k": int(k_eff),
        "num_queries": int(n_q),
        "num_candidates": int(n_c),
        "expected_occurrence": round(expected, 4),
        "max_occurrence": int(counts.max()),
        "p95_occurrence": float(np.percentile(counts, 95)),
        "p99_occurrence": float(np.percentile(counts, 99)),
        "orphan_rate": round(orphan_rate, 4),
        "skewness_snk": round(skewness, 4),
        "gini": round(_gini(counts), 4),
        "robin_hood_index": round(robin_hood, 4),
        "occurrences": counts.astype(int).tolist(),
    }


def analyze_semantic_clusters(
    img_embs: np.ndarray,
    per_image_r1: Sequence[float],
    *,
    n_clusters: int = 8,
    seed: int = 5489,
    fit_img_embs: np.ndarray | None = None,
) -> dict:
    """Cluster image embeddings with K-Means and summarize per-cluster retrieval accuracy."""
    v = l2_normalize_checked(img_embs)
    r1 = np.asarray(per_image_r1, dtype=np.float64)
    if len(r1) != v.shape[0]:
        raise ValueError("per_image_r1 length must match img_embs rows")

    fit_v = l2_normalize_checked(fit_img_embs) if fit_img_embs is not None else v
    km = KMeans(n_clusters=int(n_clusters), random_state=int(seed), n_init=10)
    km.fit(fit_v)
    labels = km.predict(v)

    sil = float(silhouette_score(v, labels, metric="cosine")) if len(set(labels.tolist())) > 1 else 0.0
    clusters: List[dict] = []
    for cid in range(int(n_clusters)):
        mask = labels == cid
        cnt = int(mask.sum())
        clusters.append(
            {
                "cluster_id": cid,
                "count": cnt,
                "mean_r1": round(float(r1[mask].mean()), 2) if cnt > 0 else 0.0,
            }
        )

    return {
        "n_clusters": int(n_clusters),
        "seed": int(seed),
        "silhouette_cosine": round(sil, 4),
        "clusters": clusters,
        "labels": labels.tolist(),
    }
