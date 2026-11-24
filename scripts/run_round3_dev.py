"""Round 3 (T4): Full Phase 1 dev-set experiments (1,000 images / 5,005 captions).

Runs strictly on the dev split (zero access to held-out test):
  1. Encode & cache CLIP ViT-B/32, CLIP RN50, and SigLIP ViT-B/16 (raw & photo_prefix).
  2. Baseline bidirectional evaluation (E0, E1, E2) + paired bootstrap for prefix effect.
  3. Data exploration, Modality Gap geometry, Hubness stats, 5-fold CV PCA compression, K-Means clustering.
  4. Image-grouped 5-fold CV for E3 (ModalityAligner) and E4 (CSLS & Dual-Softmax).
  5. Export machine-readable summary, frozen Phase 1 config, and dev diagnostic figures.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from vlm_retrieval.cache import EmbeddingBundle, load_embedding_bundle, save_embedding_bundle
from vlm_retrieval.data import SplitData, build_relevance_maps, load_split
from vlm_retrieval.evaluate import evaluate_bidirectional_scores, write_run_bundle
from vlm_retrieval.features import (
    CrossModalPCA,
    ModalityAligner,
    analyze_semantic_clusters,
    compute_hubness_stats,
    compute_modality_gap_stats,
)
from vlm_retrieval.metrics import evaluate_retrieval, paired_bootstrap_retrieval
from vlm_retrieval.model import VLMWrapper, load_model_specs
from vlm_retrieval.rerank import (
    cross_validate_modality_aligner,
    cross_validate_reranker,
    csls_similarity,
    dual_softmax_similarity,
    make_image_grouped_kfold,
    _subset_split_arrays,
)
from vlm_retrieval.retrieval import compute_similarity_matrix

SEED = 5489


def _find_or_encode_dev(
    root: Path,
    split: SplitData,
    model_key: str,
    spec,
    cache_dir: Path,
) -> Dict[str, EmbeddingBundle]:
    """Encode a model on dev once for images and twice for raw/photo_prefix, reusing cache if present."""
    coco_root = root / "data" / "raw" / "coco"
    bundles: Dict[str, EmbeddingBundle] = {}

    # Check cache hits first
    missing_policies: List[str] = []
    for policy in ("raw", "photo_prefix"):
        prefix = f"{model_key}__{split.name}__{policy}__"
        hit = None
        if cache_dir.exists():
            for mf in sorted(cache_dir.glob(f"{prefix}*.json")):
                meta = json.loads(mf.read_text(encoding="utf-8"))
                if meta.get("manifest_sha256") == split.manifest_sha256:
                    hit = load_embedding_bundle(
                        cache_dir, meta["cache_key"], expected_manifest_sha256=split.manifest_sha256
                    )
                    break
        if hit is not None:
            bundles[policy] = hit
        else:
            missing_policies.append(policy)

    if not missing_policies:
        return bundles

    t0 = time.perf_counter()
    wrapper = VLMWrapper(spec)
    t_load = time.perf_counter() - t0

    image_paths = [coco_root / rec.relative_path for rec in split.images]
    caption_texts = [rec.text for rec in split.captions]

    t1 = time.perf_counter()
    img_embs = wrapper.encode_images(image_paths, batch_size=64)
    t_img = time.perf_counter() - t1

    for policy in missing_policies:
        t2 = time.perf_counter()
        cap_embs, trunc_stats = wrapper.encode_texts(caption_texts, policy=policy, batch_size=128)
        t_txt = time.perf_counter() - t2
        meta = {
            **wrapper.metadata,
            "text_policy": policy,
            "split_name": split.name,
            "manifest_sha256": split.manifest_sha256,
            "truncation": trunc_stats,
            "load_seconds": round(t_load, 3),
            "image_encode_seconds": round(t_img, 3),
            "text_encode_seconds": round(t_txt, 3),
        }
        bundles[policy] = save_embedding_bundle(
            cache_dir,
            image_ids=split.image_ids,
            image_embeddings=img_embs,
            caption_ids=split.caption_ids,
            caption_embeddings=cap_embs,
            metadata=meta,
        )
    return bundles


def _cv_baseline_cosine(split: SplitData, v: np.ndarray, t: np.ndarray, n_splits: int = 5) -> dict:
    folds = make_image_grouped_kfold(split.image_ids, n_splits=n_splits, seed=SEED)
    fold_t2i, fold_i2t, fold_mean = [], [], []
    for _, val_ids in folds:
        sub_img_ids, sub_v, sub_cap_ids, sub_t, t2i_pos, i2t_pos = _subset_split_arrays(split, v, t, val_ids)
        s = compute_similarity_matrix(sub_v, sub_t)
        r_t2i = evaluate_retrieval(s.T, sub_cap_ids, sub_img_ids, t2i_pos, ks=(1, 5, 10), direction="t2i")["metrics"]["R@1"]
        r_i2t = evaluate_retrieval(s, sub_img_ids, sub_cap_ids, i2t_pos, ks=(1, 5, 10), direction="i2t")["metrics"]["R@1"]
        fold_t2i.append(r_t2i)
        fold_i2t.append(r_i2t)
        fold_mean.append(0.5 * (r_t2i + r_i2t))
    return {
        "mean_r1": round(float(np.mean(fold_mean)), 4),
        "std_r1": round(float(np.std(fold_mean)), 4),
        "t2i_r1_mean": round(float(np.mean(fold_t2i)), 4),
        "i2t_r1_mean": round(float(np.mean(fold_i2t)), 4),
    }


def _cv_pca_compression(
    split: SplitData,
    v: np.ndarray,
    t: np.ndarray,
    dims: List[int],
    n_splits: int = 5,
    mode: str = "uncentered_svd",
) -> List[dict]:
    folds = make_image_grouped_kfold(split.image_ids, n_splits=n_splits, seed=SEED)
    fold_pairs = [
        (
            _subset_split_arrays(split, v, t, tr_ids),
            _subset_split_arrays(split, v, t, val_ids),
        )
        for tr_ids, val_ids in folds
    ]
    full_dim = v.shape[1]
    out: List[dict] = []
    for d in dims:
        if d > full_dim:
            continue
        t2i_r1s, i2t_r1s, t2i_r5s, i2t_r5s, ev_ratios = [], [], [], [], []
        for (_, tr_v, _, tr_t, _, _), (val_img_ids, val_v, val_cap_ids, val_t, t2i_pos, i2t_pos) in fold_pairs:
            if d == full_dim:
                vp, tp = val_v, val_t
                ev = 1.0
            else:
                pca = CrossModalPCA(n_components=d, seed=SEED, mode=mode).fit(tr_v, tr_t)
                vp, tp = pca.transform(val_v, val_t)
                ev = pca.explained_variance_ratio_sum
            s = compute_similarity_matrix(vp, tp)
            m_t2i = evaluate_retrieval(s.T, val_cap_ids, val_img_ids, t2i_pos, ks=(1, 5, 10), direction="t2i")["metrics"]
            m_i2t = evaluate_retrieval(s, val_img_ids, val_cap_ids, i2t_pos, ks=(1, 5, 10), direction="i2t")["metrics"]
            t2i_r1s.append(m_t2i["R@1"])
            i2t_r1s.append(m_i2t["R@1"])
            t2i_r5s.append(m_t2i["R@5"])
            i2t_r5s.append(m_i2t["R@5"])
            ev_ratios.append(ev)
        out.append(
            {
                "mode": mode,
                "dim": int(d),
                "compression_ratio": round(full_dim / float(d), 2),
                "explained_variance": round(float(np.mean(ev_ratios)), 4),
                "cv_t2i_r1": round(float(np.mean(t2i_r1s)), 2),
                "cv_i2t_r1": round(float(np.mean(i2t_r1s)), 2),
                "cv_mean_r1": round(float(0.5 * (np.mean(t2i_r1s) + np.mean(i2t_r1s))), 2),
                "cv_t2i_r5": round(float(np.mean(t2i_r5s)), 2),
                "cv_i2t_r5": round(float(np.mean(i2t_r5s)), 2),
            }
        )
    return out


def _plot_dev_figures(
    fig_dir: Path,
    split: SplitData,
    clip_bundle: EmbeddingBundle,
    aligner_center: ModalityAligner,
    hub_base_t2i: dict,
    hub_csls_t2i: dict,
    hub_ds_t2i: dict,
    pca_curves: Dict[str, List[dict]],
    cluster_res: dict,
    cv_csls: dict,
    cv_ds_sym: dict,
) -> List[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []

    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 180,
    })

    # Figure 1: Modality Gap 2D PCA before vs after Mean-Centering (CLIP ViT-B/32 dev)
    v_raw = clip_bundle.image_embeddings
    t_raw = clip_bundle.caption_embeddings
    v_ctr, t_ctr = aligner_center.transform(v_raw, t_raw)

    rng = np.random.Generator(np.random.PCG64(SEED))
    sample_cap_idx = rng.choice(len(t_raw), size=len(v_raw), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for ax, (vv, tt, title) in zip(
        axes,
        [
            (v_raw, t_raw[sample_cap_idx], "Before Alignment (Raw CLIP ViT-B/32)"),
            (v_ctr, t_ctr[sample_cap_idx], "After Mean-Centering Alignment"),
        ],
    ):
        joint = np.concatenate([vv, tt], axis=0)
        proj = PCA(n_components=2, random_state=SEED).fit_transform(joint)
        pv, pt = proj[: len(vv)], proj[len(vv) :]
        gap = compute_modality_gap_stats(vv, tt)["centroid_l2_distance"]
        ax.scatter(pv[:, 0], pv[:, 1], s=10, alpha=0.55, label="Images (n=1,000)", color="#1f77b4")
        ax.scatter(pt[:, 0], pt[:, 1], s=10, alpha=0.45, label="Captions (n=1,000 sample)", color="#ff7f0e")
        ax.set_title(f"{title}\nCentroid L2 Gap = {gap:.3f}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    p1 = fig_dir / "dev_modality_gap_pca.png"
    fig.savefig(p1)
    plt.close(fig)
    saved.append(str(p1))

    # Figure 2: Hubness k-occurrence distribution (T2I, k=10): Baseline vs CSLS vs Dual-Softmax
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for label, hdict, color in [
        (f"Cosine Baseline (skew={hub_base_t2i['skewness_snk']:.2f}, max={hub_base_t2i['max_occurrence']})", hub_base_t2i, "#d62728"),
        (f"CSLS (skew={hub_csls_t2i['skewness_snk']:.2f}, max={hub_csls_t2i['max_occurrence']})", hub_csls_t2i, "#2ca02c"),
        (f"Dual-Softmax (skew={hub_ds_t2i['skewness_snk']:.2f}, max={hub_ds_t2i['max_occurrence']})", hub_ds_t2i, "#1f77b4"),
    ]:
        occ = np.sort(np.array(hdict["occurrences"]))[::-1]
        ax.plot(np.arange(1, len(occ) + 1), occ, label=label, lw=1.8, color=color)
    ax.axhline(hub_base_t2i["expected_occurrence"], ls="--", color="#555555", lw=1.0, label=f"Uniform Expected ({hub_base_t2i['expected_occurrence']:.1f})")
    ax.set_title("T2I Image Candidate Hubness on Dev (Top-10 Occurrence Rank Curve)")
    ax.set_xlabel("Image Candidate Rank (sorted by Top-10 appearance count)")
    ax.set_ylabel("Top-10 Retrieval Appearances across 5,005 Queries")
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    p2 = fig_dir / "dev_hubness_distribution.png"
    fig.savefig(p2)
    plt.close(fig)
    saved.append(str(p2))

    # Figure 3: 5-Fold CV PCA Compression Curve & Reranking CV Hyperparameter Curves
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.9))

    # 3a: DimRed 3-way comparison (Mean R@1 vs dim)
    dims = [r["dim"] for r in pca_curves["uncentered_svd"]]
    axes[0].plot(dims, [r["cv_mean_r1"] for r in pca_curves["uncentered_svd"]], "o-", label="Uncentered SVD", color="#1f77b4")
    axes[0].plot(dims, [r["cv_mean_r1"] for r in pca_curves["modality_centered_pca"]], "s-", label="Modality-Centered PCA", color="#2ca02c")
    axes[0].plot(dims, [r["cv_mean_r1"] for r in pca_curves["naive_pca"]], "^--", label="Naive Stacked PCA", color="#d62728")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(dims)
    axes[0].set_xticklabels([str(d) for d in dims])
    axes[0].set_xlabel("Compressed Dimension (log2)")
    axes[0].set_ylabel("5-Fold CV Mean R@1 (%)")
    axes[0].set_title("DimRed Compression (CLIP ViT-B/32)")
    axes[0].legend(frameon=False, fontsize=8)

    # 3b: CSLS k CV
    ks = [c["params"]["k"] for c in cv_csls["candidates"]]
    means = [c["mean_r1"] for c in cv_csls["candidates"]]
    stds = [c["std_r1"] for c in cv_csls["candidates"]]
    axes[1].errorbar(ks, means, yerr=stds, fmt="o-", capsize=3, color="#2ca02c")
    axes[1].set_xlabel("Neighborhood Size k")
    axes[1].set_ylabel("5-Fold CV Mean R@1 (%)")
    axes[1].set_title("CSLS 5-Fold CV (Dev)")

    # 3c: Dual-Softmax tau CV
    taus = [c["params"]["temperature"] for c in cv_ds_sym["candidates"]]
    means_ds = [c["mean_r1"] for c in cv_ds_sym["candidates"]]
    stds_ds = [c["std_r1"] for c in cv_ds_sym["candidates"]]
    axes[2].errorbar(taus, means_ds, yerr=stds_ds, fmt="s-", capsize=3, color="#9467bd")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Temperature τ (log)")
    axes[2].set_ylabel("5-Fold CV Mean R@1 (%)")
    axes[2].set_title("Dual-Softmax 5-Fold CV (Dev)")

    fig.tight_layout()
    p3 = fig_dir / "dev_pca_and_cv_curves.png"
    fig.savefig(p3)
    plt.close(fig)
    saved.append(str(p3))

    # Figure 4: Data Exploration (Caption Word Length) & K-Means Semantic Cluster Difficulty
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    word_lens = [len(c.text.split()) for c in split.captions]
    axes[0].hist(word_lens, bins=range(4, max(word_lens) + 2), color="#4c78a8", alpha=0.85, edgecolor="white")
    axes[0].axvline(np.mean(word_lens), color="#d62728", ls="--", label=f"Mean = {np.mean(word_lens):.1f} words")
    axes[0].set_title("Dev Caption Length Distribution (n=5,005)")
    axes[0].set_xlabel("Words per Caption")
    axes[0].set_ylabel("Count")
    axes[0].legend(frameon=False)

    cl_sorted = sorted(cluster_res["clusters"], key=lambda c: c["mean_r1"])
    axes[1].barh(
        [f"C{c['cluster_id']} (n={c['count']})" for c in cl_sorted],
        [c["mean_r1"] for c in cl_sorted],
        color="#72b7b2",
    )
    axes[1].set_xlabel("Mean I2T R@1 (%)")
    axes[1].set_title(f"K-Means Image Clusters (K={cluster_res['n_clusters']}, Sil={cluster_res['silhouette_cosine']:.3f})")
    axes[1].set_xlim(0, 100)

    fig.tight_layout()
    p4 = fig_dir / "dev_data_and_clusters.png"
    fig.savefig(p4)
    plt.close(fig)
    saved.append(str(p4))

    return saved


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_dir = root / "data" / "manifests" / "coco2017"
    cache_dir = root / "artifacts" / "cache"
    runs_dir = root / "artifacts" / "runs" / "dev"
    fig_dir = root / "reports" / "figures"

    split = load_split(manifest_dir, "dev")
    specs = load_model_specs(root / "configs" / "models.yaml")
    cap_group_keys = [c.image_id for c in split.captions]
    img_group_keys = [im.image_id for im in split.images]

    summary: dict = {
        "split": "dev",
        "num_images": len(split.images),
        "num_captions": len(split.captions),
        "manifest_sha256": split.manifest_sha256,
        "data_exploration": {},
        "baselines": {},
        "prefix_bootstrap": {},
        "modality_gap": {},
        "cv_results": {},
        "dev_full_methods": {},
        "pca_compression_cv": {},
        "clustering": {},
        "hubness": {},
    }

    # 1. Data exploration on dev captions
    word_lens = [len(c.text.split()) for c in split.captions]
    char_lens = [len(c.text) for c in split.captions]
    summary["data_exploration"] = {
        "num_images": len(split.images),
        "num_captions": len(split.captions),
        "word_len_min": int(min(word_lens)),
        "word_len_mean": round(float(np.mean(word_lens)), 2),
        "word_len_median": float(np.median(word_lens)),
        "word_len_p95": float(np.percentile(word_lens, 95)),
        "word_len_max": int(max(word_lens)),
        "char_len_mean": round(float(np.mean(char_lens)), 2),
    }

    all_bundles: Dict[str, Dict[str, EmbeddingBundle]] = {}
    eval_cache: Dict[str, Dict[str, dict]] = {}

    # 2. Encode + Baseline E0/E1/E2 evaluation across all 3 models x 2 policies
    for model_key, spec in specs.items():
        print(f"[Round 3] Encoding & evaluating {model_key} on dev (1,000 images / 5,005 captions)...")
        bundles = _find_or_encode_dev(root, split, model_key, spec, cache_dir)
        all_bundles[model_key] = bundles
        summary["baselines"][model_key] = {}
        eval_cache[model_key] = {}

        for policy in ("raw", "photo_prefix"):
            b = bundles[policy]
            sim_i2t = compute_similarity_matrix(b.image_embeddings, b.caption_embeddings)
            res = evaluate_bidirectional_scores(split, sim_i2t)
            eval_cache[model_key][policy] = res

            run_id = f"dev__{model_key}__{policy}__baseline"
            write_run_bundle(
                runs_dir,
                run_id=run_id,
                config={"model_key": model_key, "split": "dev", "policy": policy, "method": "baseline_cosine"},
                eval_result=res,
                provenance_extra={"cache_key": b.cache_key, "truncation": b.metadata.get("truncation")},
                log_lines=[f"{run_id}: mean_R@1={res['mean_R@1']:.2f}%"],
                repo_root=root,
            )
            summary["baselines"][model_key][policy] = {
                "mean_R@1": res["mean_R@1"],
                "t2i": res["t2i"]["metrics"],
                "i2t": res["i2t"]["metrics"],
                "truncation": b.metadata.get("truncation"),
                "encode_seconds": {
                    "image": b.metadata.get("image_encode_seconds"),
                    "text": b.metadata.get("text_encode_seconds"),
                },
            }

        # Paired bootstrap for prefix (E1) - raw (E0)
        raw_res = eval_cache[model_key]["raw"]
        pre_res = eval_cache[model_key]["photo_prefix"]
        summary["prefix_bootstrap"][model_key] = {
            "t2i_r1": paired_bootstrap_retrieval(
                raw_res["t2i"]["ranks"], pre_res["t2i"]["ranks"], cap_group_keys, k=1, n_resamples=1000, seed=SEED
            ),
            "i2t_r1": paired_bootstrap_retrieval(
                raw_res["i2t"]["ranks"], pre_res["i2t"]["ranks"], img_group_keys, k=1, n_resamples=1000, seed=SEED
            ),
        }

        # Select dev-preferred policy per model (ties -> raw)
        raw_m = raw_res["mean_R@1"]
        pre_m = pre_res["mean_R@1"]
        best_policy = "photo_prefix" if pre_m > raw_m else "raw"

        # 3. Modality Gap & 5-Fold CV for E3 (ModalityAligner) and E4 (CSLS, Dual-Softmax) on dev-preferred policy
        # Also always run on raw for clip_vit_b32 to keep E0 direct apples-to-apples traceability
        b_sel = bundles[best_policy]
        v_sel, t_sel = b_sel.image_embeddings, b_sel.caption_embeddings
        summary["modality_gap"][model_key] = {
            "policy": best_policy,
            "raw_embeddings": compute_modality_gap_stats(v_sel, t_sel),
        }

        print(f"[Round 3] Running 5-fold image-grouped CV for {model_key} (policy={best_policy})...")
        cv_base = _cv_baseline_cosine(split, v_sel, t_sel, n_splits=5)

        align_grid = [
            {"mode": "center"},
            {"mode": "zca_shared", "eps": 0.01},
            {"mode": "zca_shared", "eps": 0.05},
            {"mode": "zca_shared", "eps": 0.1},
            {"mode": "zca_shared", "eps": 0.25},
            {"mode": "zca_shared", "eps": 0.5},
            {"mode": "zca_per_modality", "eps": 0.05},
            {"mode": "zca_per_modality", "eps": 0.1},
            {"mode": "zca_per_modality", "eps": 0.25},
            {"mode": "zca_per_modality", "eps": 0.5},
        ]
        cv_align = cross_validate_modality_aligner(split, v_sel, t_sel, param_grid=align_grid, n_splits=5, seed=SEED)

        csls_grid = [{"k": k} for k in (5, 10, 20, 30, 50)]
        cv_csls = cross_validate_reranker(split, v_sel, t_sel, method="csls", param_grid=csls_grid, n_splits=5, seed=SEED)

        ds_sym_grid = [{"temperature": tau, "mode": "symmetric_log"} for tau in (0.01, 0.02, 0.05, 0.1, 0.2)]
        cv_ds_sym = cross_validate_reranker(split, v_sel, t_sel, method="dual_softmax", param_grid=ds_sym_grid, n_splits=5, seed=SEED)

        ds_pw_grid = [{"temperature": tau, "mode": "prior_weighted"} for tau in (0.01, 0.02, 0.05, 0.1, 0.2)]
        cv_ds_pw = cross_validate_reranker(split, v_sel, t_sel, method="dual_softmax", param_grid=ds_pw_grid, n_splits=5, seed=SEED)

        summary["cv_results"][model_key] = {
            "selected_policy": best_policy,
            "baseline_5fold": cv_base,
            "modality_aligner_5fold": cv_align,
            "csls_5fold": cv_csls,
            "dual_softmax_symmetric_5fold": cv_ds_sym,
            "dual_softmax_prior_weighted_5fold": cv_ds_pw,
        }

        # Evaluate full-dev gallery with CV-selected best hyperparameters (diagnostic on dev)
        best_align_params = cv_align["best"]["params"]
        aligner_best = ModalityAligner(**best_align_params).fit(v_sel, t_sel)
        v_al, t_al = aligner_best.transform(v_sel, t_sel)
        summary["modality_gap"][model_key]["after_best_aligner"] = compute_modality_gap_stats(v_al, t_al)

        best_k = int(cv_csls["best"]["params"]["k"])
        best_ds_params = (
            cv_ds_sym["best"]["params"]
            if cv_ds_sym["best"]["mean_r1"] >= cv_ds_pw["best"]["mean_r1"]
            else cv_ds_pw["best"]["params"]
        )

        sim_base = compute_similarity_matrix(v_sel, t_sel)
        sim_align = compute_similarity_matrix(v_al, t_al)
        sim_csls = csls_similarity(sim_base, k=best_k)
        sim_ds_i2t, sim_ds_t2i = dual_softmax_similarity(
            sim_base, temperature=float(best_ds_params["temperature"]), mode=str(best_ds_params["mode"])
        )
        # Combined: best aligner + best CSLS & Dual-Softmax
        sim_al_csls = csls_similarity(sim_align, k=best_k)
        sim_al_ds_i2t, sim_al_ds_t2i = dual_softmax_similarity(
            sim_align, temperature=float(best_ds_params["temperature"]), mode=str(best_ds_params["mode"])
        )

        dev_methods = {}
        for mname, s_i2t, s_t2i in [
            ("baseline", sim_base, sim_base.T),
            ("aligner_best", sim_align, sim_align.T),
            ("csls_best", sim_csls, sim_csls.T),
            ("dual_softmax_best", sim_ds_i2t, sim_ds_t2i),
            ("aligner_plus_csls", sim_al_csls, sim_al_csls.T),
            ("aligner_plus_dual_softmax", sim_al_ds_i2t, sim_al_ds_t2i),
        ]:
            ev = evaluate_bidirectional_scores(split, s_i2t, s_t2i)
            dev_methods[mname] = {
                "mean_R@1": ev["mean_R@1"],
                "t2i": ev["t2i"]["metrics"],
                "i2t": ev["i2t"]["metrics"],
            }
        summary["dev_full_methods"][model_key] = {
            "policy": best_policy,
            "best_align_params": best_align_params,
            "best_csls_k": best_k,
            "best_dual_softmax_params": best_ds_params,
            "methods": dev_methods,
        }

        # Hubness diagnostics on dev full gallery (T2I: image candidates)
        hub_base = compute_hubness_stats(sim_base.T, k=10)
        hub_csls = compute_hubness_stats(sim_csls.T, k=10)
        hub_ds = compute_hubness_stats(sim_ds_t2i, k=10)
        summary["hubness"][model_key] = {
            "t2i_k10_baseline": {k: v for k, v in hub_base.items() if k != "occurrences"},
            "t2i_k10_csls": {k: v for k, v in hub_csls.items() if k != "occurrences"},
            "t2i_k10_dual_softmax": {k: v for k, v in hub_ds.items() if k != "occurrences"},
        }

        if model_key == "clip_vit_b32":
            # Save detailed objects for 3-way PCA/SVD compression, K-Means clusters, and figures
            print("[Round 3] Running 5-fold CV DimRed (3 modes) & K-Means cluster diagnostics on CLIP ViT-B/32...")
            pca_curves = {
                mode: _cv_pca_compression(split, v_sel, t_sel, dims=[512, 256, 128, 64, 32], n_splits=5, mode=mode)
                for mode in ("naive_pca", "uncentered_svd", "modality_centered_pca")
            }
            summary["pca_compression_cv"]["clip_vit_b32"] = pca_curves

            base_i2t_ranks = eval_cache["clip_vit_b32"][best_policy]["i2t"]["ranks"]
            per_img_r1 = [100.0 if r == 1 else 0.0 for r in base_i2t_ranks]
            best_sil, best_cl = -2.0, None
            sil_sweep = {}
            for k_cl in (6, 8, 10, 12):
                cl = analyze_semantic_clusters(v_sel, per_img_r1, n_clusters=k_cl, seed=SEED)
                sil_sweep[str(k_cl)] = cl["silhouette_cosine"]
                if cl["silhouette_cosine"] > best_sil:
                    best_sil = cl["silhouette_cosine"]
                    best_cl = cl
            # Extract 2 representative captions per cluster for interpretability
            cap_by_img = {}
            for c in split.captions:
                cap_by_img.setdefault(c.image_id, []).append(c.text)
            for cl_entry in best_cl["clusters"]:
                cid = cl_entry["cluster_id"]
                member_idxs = [i for i, lbl in enumerate(best_cl["labels"]) if lbl == cid][:3]
                cl_entry["sample_captions"] = [cap_by_img[split.images[i].image_id][0] for i in member_idxs]

            summary["clustering"]["clip_vit_b32"] = {
                "silhouette_sweep": sil_sweep,
                "selected_k": best_cl["n_clusters"],
                "silhouette_cosine": best_cl["silhouette_cosine"],
                "clusters": best_cl["clusters"],
            }

            aligner_center = ModalityAligner(mode="center").fit(v_sel, t_sel)
            fig_paths = _plot_dev_figures(
                fig_dir,
                split,
                b_sel,
                aligner_center,
                hub_base,
                hub_csls,
                hub_ds,
                pca_curves,
                best_cl,
                cv_csls,
                cv_ds_sym,
            )
            summary["dev_figures"] = [str(Path(p).relative_to(root)) for p in fig_paths]

    # Save frozen Phase 1 config derived strictly from dev 5-fold CV
    frozen_phase1 = {
        "protocol_version": "v2",
        "frozen_from_split": "dev",
        "dev_manifest_sha256": split.manifest_sha256,
        "seed": SEED,
        "models": {},
    }
    for model_key in specs:
        dm = summary["dev_full_methods"][model_key]
        frozen_phase1["models"][model_key] = {
            "selected_policy": dm["policy"],
            "best_align_params": dm["best_align_params"],
            "best_csls_k": dm["best_csls_k"],
            "best_dual_softmax_params": dm["best_dual_softmax_params"],
        }

    (root / "configs" / "frozen_phase1.json").write_text(
        json.dumps(frozen_phase1, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    out_summary = root / "reports" / "dev_phase1_summary.json"
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Compact console overview
    print("\n=== ROUND 3 DEV SUMMARY ===")
    for mk in specs:
        r_raw = summary["baselines"][mk]["raw"]
        r_pre = summary["baselines"][mk]["photo_prefix"]
        dm = summary["dev_full_methods"][mk]["methods"]
        print(
            f"{mk:15s} | raw={r_raw['mean_R@1']:.2f}% (T2I {r_raw['t2i']['R@1']:.2f} / I2T {r_raw['i2t']['R@1']:.2f}) "
            f"| prefix={r_pre['mean_R@1']:.2f}% | align={dm['aligner_best']['mean_R@1']:.2f}% "
            f"| CSLS={dm['csls_best']['mean_R@1']:.2f}% | DualSM={dm['dual_softmax_best']['mean_R@1']:.2f}% "
            f"| Align+DS={dm['aligner_plus_dual_softmax']['mean_R@1']:.2f}%"
        )
    print(f"Saved summary -> {out_summary}")
    print(f"Saved frozen config -> {root / 'configs' / 'frozen_phase1.json'}")


if __name__ == "__main__":
    main()
