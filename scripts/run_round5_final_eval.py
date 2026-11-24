"""Round 5 (T6): One-shot frozen held-out test (1K COCO) + full 400-group Winoground evaluation.

Strictly enforces anti-leakage protocol:
  - Uses ONLY frozen configurations from configs/frozen_all.json and dev-fitted transforms.
  - Evaluates held-out test once and saves full run bundles (metrics, provenance, per-query ranks).
  - Evaluates all 400 official Winoground groups with strict inequalities (no gallery reranking).
  - Computes 1,000x image/group-grouped paired bootstrap 95% CIs (seed=5489).
  - Samples stratified success/failure cases (seed=5489) and renders 4 publication-ready final figures.
"""

from __future__ import annotations

import io
import json
import resource
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image

from vlm_retrieval.adapt import (
    ResidualImageAdapter,
    encode_images_with_adapter,
    encode_texts_with_model,
    evaluate_compositional_hard_neg_acc,
    generate_hard_negative,
)
from vlm_retrieval.cache import EmbeddingBundle, load_embedding_bundle, save_embedding_bundle
from vlm_retrieval.data import SplitData, load_split, sha256_file, validate_splits_disjoint
from vlm_retrieval.evaluate import (
    collect_runtime_provenance,
    evaluate_bidirectional_scores,
    write_run_bundle,
)
from vlm_retrieval.features import (
    CrossModalPCA,
    ModalityAligner,
    compute_hubness_stats,
    compute_modality_gap_stats,
)
from vlm_retrieval.metrics import (
    evaluate_winoground,
    paired_bootstrap_binary,
    paired_bootstrap_retrieval,
    sample_stratified_cases,
)
from vlm_retrieval.model import VLMWrapper, l2_normalize_checked, load_model_specs
from vlm_retrieval.rerank import csls_similarity, dual_softmax_similarity
from vlm_retrieval.retrieval import compute_similarity_matrix

SEED = 5489


def _find_cached(cache_dir: Path, model_key: str, split_name: str, policy: str, manifest_sha256: str) -> EmbeddingBundle | None:
    if not cache_dir.exists():
        return None
    prefix = f"{model_key}__{split_name}__{policy}__"
    for mf in sorted(cache_dir.glob(f"{prefix}*.json")):
        meta = json.loads(mf.read_text(encoding="utf-8"))
        if meta.get("manifest_sha256") == manifest_sha256:
            return load_embedding_bundle(cache_dir, meta["cache_key"], expected_manifest_sha256=manifest_sha256)
    return None


def _encode_or_load_split(
    root: Path,
    split: SplitData,
    model_key: str,
    spec,
    cache_dir: Path,
) -> Dict[str, EmbeddingBundle]:
    bundles: Dict[str, EmbeddingBundle] = {}
    missing = []
    for policy in ("raw", "photo_prefix"):
        hit = _find_cached(cache_dir, model_key, split.name, policy, split.manifest_sha256)
        if hit is not None:
            bundles[policy] = hit
        else:
            missing.append(policy)
    if not missing:
        return bundles

    wrapper = VLMWrapper(spec)
    paths = [root / "data" / "raw" / "coco" / rec.relative_path for rec in split.images]
    texts = [rec.text for rec in split.captions]
    t0 = time.perf_counter()
    img_embs = wrapper.encode_images(paths, batch_size=64)
    t_img = time.perf_counter() - t0

    for policy in missing:
        t1 = time.perf_counter()
        cap_embs, trunc = wrapper.encode_texts(texts, policy=policy, batch_size=128)
        t_txt = time.perf_counter() - t1
        meta = {
            **wrapper.metadata,
            "text_policy": policy,
            "split_name": split.name,
            "manifest_sha256": split.manifest_sha256,
            "truncation": trunc,
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


def _load_negclip_lite(spec, ckpt_path: Path):
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    wrapper = VLMWrapper(spec)
    state = wrapper.model.state_dict()
    for k, v in payload["text_trainable_state_dict"].items():
        if k not in state:
            raise KeyError(f"Unexpected trainable key in checkpoint: {k}")
        state[k] = v
    wrapper.model.load_state_dict(state, strict=True)
    wrapper.model.eval()

    adapter = None
    if payload.get("img_adapter_state_dict") is not None:
        dim = int(payload["img_adapter_state_dict"]["net.2.bias"].shape[0])
        adapter = ResidualImageAdapter(dim=dim).to(wrapper.device)
        adapter.load_state_dict(payload["img_adapter_state_dict"], strict=True)
        adapter.eval()

    return wrapper, adapter, payload


@torch.inference_mode()
def _encode_pil_images(wrapper: VLMWrapper, pil_images: Sequence[Image.Image], batch_size: int = 64) -> np.ndarray:
    chunks: List[np.ndarray] = []
    for start in range(0, len(pil_images), batch_size):
        tensors = [wrapper.preprocess(im.convert("RGB")) for im in pil_images[start : start + batch_size]]
        batch = torch.stack(tensors, dim=0).to(wrapper.device)
        feats = F.normalize(wrapper.model.encode_image(batch).float(), dim=-1)
        chunks.append(feats.cpu().numpy().astype(np.float32))
    return l2_normalize_checked(np.concatenate(chunks, axis=0))


def _score_winoground_embeddings(
    img0_embs: np.ndarray,
    img1_embs: np.ndarray,
    cap0_embs: np.ndarray,
    cap1_embs: np.ndarray,
) -> List[Tuple[float, float, float, float]]:
    # s_ij = s(C_i, I_j)
    s00 = np.sum(cap0_embs * img0_embs, axis=1)
    s01 = np.sum(cap0_embs * img1_embs, axis=1)
    s10 = np.sum(cap1_embs * img0_embs, axis=1)
    s11 = np.sum(cap1_embs * img1_embs, axis=1)
    return [
        (float(a), float(b), float(c), float(d))
        for a, b, c, d in zip(s00, s01, s10, s11)
    ]


def _plot_final_figures(
    fig_dir: Path,
    test_results: dict,
    bootstrap_deltas: dict,
    hubness_test: dict,
    dimred_test: Dict[str, List[dict]],
    wg_results: dict,
) -> List[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []

    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
    })

    # 1. Final Held-out Test: Model & Method Comparison (T2I, I2T, Mean R@1)
    labels = [
        "CLIP RN50\n(Cosine)",
        "CLIP RN50\n(+Dual-SM)",
        "CLIP ViT-B/32\n(Cosine)",
        "CLIP ViT-B/32\n(+Aligner)",
        "CLIP ViT-B/32\n(+CSLS)",
        "CLIP ViT-B/32\n(+Dual-SM)",
        "NegCLIP-lite\n(Cosine)",
        "NegCLIP-lite\n(+Dual-SM)",
        "SigLIP B/16\n(Cosine)",
        "SigLIP B/16\n(+Dual-SM)",
    ]
    entries = [
        test_results["clip_rn50"]["photo_prefix_baseline"],
        test_results["clip_rn50"]["dual_softmax"],
        test_results["clip_vit_b32"]["photo_prefix_baseline"],
        test_results["clip_vit_b32"]["aligner"],
        test_results["clip_vit_b32"]["csls"],
        test_results["clip_vit_b32"]["dual_softmax"],
        test_results["negclip_lite"]["cosine"],
        test_results["negclip_lite"]["dual_softmax"],
        test_results["siglip_vit_b16"]["photo_prefix_baseline"],
        test_results["siglip_vit_b16"]["dual_softmax"],
    ]
    t2i_vals = [e["t2i"]["R@1"] for e in entries]
    i2t_vals = [e["i2t"]["R@1"] for e in entries]
    mean_vals = [e["mean_R@1"] for e in entries]

    x = np.arange(len(labels))
    w = 0.27
    fig, ax = plt.subplots(figsize=(13.2, 4.8))
    b1 = ax.bar(x - w, t2i_vals, width=w, label="T2I R@1 (5,002 queries → 1K imgs)", color="#4c78a8")
    b2 = ax.bar(x, i2t_vals, width=w, label="I2T R@1 (1K imgs → 5,002 caps)", color="#f58518")
    b3 = ax.bar(x + w, mean_vals, width=w, label="Mean R@1", color="#54a24b")
    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.1f}", ha="center", va="bottom", fontsize=7.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 102)
    ax.set_ylabel("Recall@1 (%)")
    ax.set_title("Held-Out COCO Test (1,000 Images / 5,002 Captions): Frozen Backbones, Reranking & NegCLIP-lite")
    ax.legend(frameon=False, loc="upper left", ncol=3)
    fig.tight_layout()
    p1 = fig_dir / "final_test_model_method_comparison.png"
    fig.savefig(p1)
    plt.close(fig)
    saved.append(str(p1))

    # 2. Held-out Test: Hubness Curve + 3-way DimRed Compression (fit on dev -> applied to test)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.3))
    hb = hubness_test["clip_vit_b32"]
    for label, key, color in [
        (f"Cosine Baseline (skew={hb['baseline']['skewness_snk']:.2f}, max={hb['baseline']['max_occurrence']})", "baseline", "#d62728"),
        (f"CSLS k=5 (skew={hb['csls']['skewness_snk']:.2f}, max={hb['csls']['max_occurrence']})", "csls", "#2ca02c"),
        (f"Dual-Softmax τ=0.02 (skew={hb['dual_softmax']['skewness_snk']:.2f}, max={hb['dual_softmax']['max_occurrence']})", "dual_softmax", "#1f77b4"),
    ]:
        occ = np.sort(np.array(hb[key]["occurrences"]))[::-1]
        axes[0].plot(np.arange(1, len(occ) + 1), occ, label=label, lw=1.8, color=color)
    axes[0].axhline(hb["baseline"]["expected_occurrence"], ls="--", color="#555555", lw=1.0, label=f"Uniform Expected ({hb['baseline']['expected_occurrence']:.1f})")
    axes[0].set_title("Held-Out Test T2I Hubness (CLIP ViT-B/32)")
    axes[0].set_xlabel("Image Candidate Rank")
    axes[0].set_ylabel("Top-10 Appearances across 5,002 Queries")
    axes[0].legend(frameon=False, fontsize=8.2)

    dims = [r["dim"] for r in dimred_test["uncentered_svd"]]
    axes[1].plot(dims, [r["mean_R@1"] for r in dimred_test["uncentered_svd"]], "o-", label="Uncentered SVD (dev-fit)", color="#1f77b4", lw=1.8)
    axes[1].plot(dims, [r["mean_R@1"] for r in dimred_test["modality_centered_pca"]], "s-", label="Modality-Centered PCA (dev-fit)", color="#2ca02c", lw=1.8)
    axes[1].plot(dims, [r["mean_R@1"] for r in dimred_test["naive_pca"]], "^--", label="Naive Stacked PCA (dev-fit)", color="#d62728", lw=1.8)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(dims)
    axes[1].set_xticklabels([str(d) for d in dims])
    axes[1].set_xlabel("Compressed Embedding Dimension (log2)")
    axes[1].set_ylabel("Held-Out Test Mean R@1 (%)")
    axes[1].set_title("Held-Out Test DimRed Compression (CLIP ViT-B/32)")
    axes[1].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    p2 = fig_dir / "final_test_hubness_and_dimred.png"
    fig.savefig(p2)
    plt.close(fig)
    saved.append(str(p2))

    # 3. Full 400-Group Winoground Comparison + Official Tag Breakdown
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    wg_keys = [
        ("clip_rn50__photo_prefix", "CLIP RN50"),
        ("clip_vit_b32__raw", "CLIP ViT-B/32\n(raw)"),
        ("clip_vit_b32__photo_prefix", "CLIP ViT-B/32\n(prefix)"),
        ("negclip_lite__photo_prefix", "NegCLIP-lite\n(Phase 2)"),
        ("siglip_vit_b16__photo_prefix", "SigLIP B/16\n(prefix)"),
    ]
    x_wg = np.arange(len(wg_keys))
    w_wg = 0.25
    t_sc = [wg_results[k]["metrics"]["text_score"] for k, _ in wg_keys]
    i_sc = [wg_results[k]["metrics"]["image_score"] for k, _ in wg_keys]
    g_sc = [wg_results[k]["metrics"]["group_score"] for k, _ in wg_keys]

    b_t = axes[0].bar(x_wg - w_wg, t_sc, width=w_wg, label="Text Score", color="#4c78a8")
    b_i = axes[0].bar(x_wg, i_sc, width=w_wg, label="Image Score", color="#f58518")
    b_g = axes[0].bar(x_wg + w_wg, g_sc, width=w_wg, label="Group Score", color="#54a24b")
    for bars in (b_t, b_i, b_g):
        for bar in bars:
            h = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.1f}", ha="center", va="bottom", fontsize=7.5)
    axes[0].axhline(25.0, ls="--", color="#888888", lw=1.0, label="Random Text/Image (25.0%)")
    axes[0].axhline(16.67, ls=":", color="#444444", lw=1.1, label="Random Group (16.7%)")
    axes[0].set_xticks(x_wg)
    axes[0].set_xticklabels([lbl for _, lbl in wg_keys], fontsize=8.5)
    axes[0].set_ylabel("Winoground Accuracy (%)")
    axes[0].set_ylim(0, max(max(t_sc), max(i_sc), 50) + 10)
    axes[0].set_title("Winoground (400 Groups): Overall Compositional Scores")
    axes[0].legend(frameon=False, fontsize=7.8, loc="upper left", ncol=2)

    # Panel 3b: Group Score by collapsed_tag (Object / Relation / Both)
    tags_order = ["Object", "Relation", "Both"]
    tag_models = [
        ("clip_vit_b32__photo_prefix", "CLIP ViT-B/32", "#4c78a8"),
        ("negclip_lite__photo_prefix", "NegCLIP-lite", "#2ca02c"),
        ("siglip_vit_b16__photo_prefix", "SigLIP B/16", "#9467bd"),
    ]
    xt = np.arange(len(tags_order))
    wt = 0.25
    for idx_m, (mk, mlbl, mcol) in enumerate(tag_models):
        vals = [wg_results[mk]["tag_breakdown"].get(tg, {}).get("group_score", 0.0) for tg in tags_order]
        bars = axes[1].bar(xt + (idx_m - 1) * wt, vals, width=wt, label=mlbl, color=mcol)
        for bar in bars:
            h = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width() / 2, h + 0.6, f"{h:.1f}", ha="center", va="bottom", fontsize=8)
    counts_lbl = [
        f"{tg}\n(n={wg_results['clip_vit_b32__photo_prefix']['tag_breakdown'][tg]['count']})"
        for tg in tags_order
    ]
    axes[1].axhline(16.67, ls=":", color="#444444", lw=1.1, label="Random Group (16.7%)")
    axes[1].set_xticks(xt)
    axes[1].set_xticklabels(counts_lbl)
    axes[1].set_ylabel("Winoground Group Score (%)")
    axes[1].set_title("Winoground Group Score by Official Collapsed Tag")
    axes[1].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    p3 = fig_dir / "final_winoground_comparison.png"
    fig.savefig(p3)
    plt.close(fig)
    saved.append(str(p3))

    # 4. Paired Bootstrap 95% CI Forest Plot (Held-Out Test R@1 Deltas)
    comp_items = [
        ("clip_vit_b32_prefix_vs_raw", "CLIP ViT-B/32: Prefix − Raw"),
        ("clip_vit_b32_aligner_vs_base", "CLIP ViT-B/32: Aligner − Base"),
        ("clip_vit_b32_csls_vs_base", "CLIP ViT-B/32: CSLS − Base"),
        ("clip_vit_b32_ds_vs_base", "CLIP ViT-B/32: Dual-SM − Base"),
        ("negclip_vs_clip_cosine", "NegCLIP-lite − CLIP (Cosine)"),
        ("negclip_ds_vs_clip_ds", "NegCLIP-lite − CLIP (Dual-SM)"),
        ("clip_rn50_ds_vs_base", "CLIP RN50: Dual-SM − Base"),
        ("siglip_ds_vs_base", "SigLIP B/16: Dual-SM − Base"),
        ("siglip_vs_clip_base", "SigLIP B/16 − CLIP ViT-B/32"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), sharey=True)
    y_pos = np.arange(len(comp_items))
    for ax, direction, title, col in [
        (axes[0], "t2i_r1", "T2I R@1 Difference (95% Paired Bootstrap CI)", "#1f77b4"),
        (axes[1], "i2t_r1", "I2T R@1 Difference (95% Paired Bootstrap CI)", "#ff7f0e"),
    ]:
        ax.axvline(0.0, ls="--", color="#555555", lw=1.0)
        for yi, (ckey, _) in enumerate(comp_items):
            d = bootstrap_deltas[ckey][direction]
            pt = d["diff_b_minus_a"]
            lo, hi = d["ci_95"]
            ax.errorbar(pt, yi, xerr=[[pt - lo], [hi - pt]], fmt="o", color=col, capsize=3.5, lw=1.6)
            ax.text(hi + 0.35, yi, f"{pt:+.2f}", va="center", fontsize=8)
        ax.set_xlabel("Δ Recall@1 (percentage points)")
        ax.set_title(title)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels([lbl for _, lbl in comp_items], fontsize=8.8)
    axes[0].invert_yaxis()
    fig.tight_layout()
    p4 = fig_dir / "final_bootstrap_deltas.png"
    fig.savefig(p4)
    plt.close(fig)
    saved.append(str(p4))

    return saved


def main() -> None:
    t_start = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    manifest_dir = root / "data" / "manifests" / "coco2017"
    cache_dir = root / "artifacts" / "cache"
    runs_test_dir = root / "artifacts" / "runs" / "test"
    runs_wg_dir = root / "artifacts" / "runs" / "winoground"
    runs_wg_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = root / "reports" / "figures"

    frozen = json.loads((root / "configs" / "frozen_all.json").read_text(encoding="utf-8"))
    specs = load_model_specs(root / "configs" / "models.yaml")

    dev_split = load_split(manifest_dir, "dev")
    test_split = load_split(manifest_dir, "test")
    validate_splits_disjoint([dev_split, test_split])

    cap_group_keys = [c.image_id for c in test_split.captions]
    img_group_keys = [im.image_id for im in test_split.images]

    print("[Round 5] Part 1: One-shot held-out COCO test evaluation (1,000 images / 5,002 captions)...")
    test_bundles: Dict[str, Dict[str, EmbeddingBundle]] = {}
    dev_bundles: Dict[str, Dict[str, EmbeddingBundle]] = {}
    eval_objs: Dict[str, Dict[str, dict]] = {}
    test_summary: Dict[str, dict] = {}
    hubness_test: Dict[str, dict] = {}
    modality_gap_test: Dict[str, dict] = {}

    for model_key, spec in specs.items():
        print(f"  Encoding & evaluating {model_key} on held-out test...")
        test_bundles[model_key] = _encode_or_load_split(root, test_split, model_key, spec, cache_dir)
        dev_bundles[model_key] = _encode_or_load_split(root, dev_split, model_key, spec, cache_dir)

        fcfg = frozen["models"][model_key]
        sel_pol = fcfg["selected_policy"]
        align_params = fcfg["best_align_params"]
        csls_k = int(fcfg["best_csls_k"])
        ds_params = fcfg["best_dual_softmax_params"]

        eval_objs[model_key] = {}
        test_summary[model_key] = {}

        # Evaluate raw and photo_prefix baselines
        for pol in ("raw", "photo_prefix"):
            tb = test_bundles[model_key][pol]
            sim = compute_similarity_matrix(tb.image_embeddings, tb.caption_embeddings)
            res = evaluate_bidirectional_scores(test_split, sim)
            key_name = f"{pol}_baseline"
            eval_objs[model_key][key_name] = res
            write_run_bundle(
                runs_test_dir,
                run_id=f"test__{model_key}__{pol}__baseline",
                config={"model_key": model_key, "split": "test", "policy": pol, "method": "baseline"},
                eval_result=res,
                provenance_extra={"cache_key": tb.cache_key, "truncation": tb.metadata.get("truncation")},
                log_lines=[f"test {model_key} {pol} baseline mean_R@1={res['mean_R@1']:.2f}%"],
                repo_root=root,
            )
            test_summary[model_key][key_name] = {
                "mean_R@1": res["mean_R@1"],
                "t2i": res["t2i"]["metrics"],
                "i2t": res["i2t"]["metrics"],
                "truncation": tb.metadata.get("truncation"),
            }

        # Apply frozen dev-selected policy for E3 (Aligner fit STRICTLY on dev!) and E4 (CSLS & Dual-Softmax)
        db_sel = dev_bundles[model_key][sel_pol]
        tb_sel = test_bundles[model_key][sel_pol]
        v_test, t_test = tb_sel.image_embeddings, tb_sel.caption_embeddings
        sim_base = compute_similarity_matrix(v_test, t_test)

        # E3: ModalityAligner fit ONLY on dev -> transform test
        aligner = ModalityAligner(**align_params).fit(db_sel.image_embeddings, db_sel.caption_embeddings)
        v_al, t_al = aligner.transform(v_test, t_test)
        sim_al = compute_similarity_matrix(v_al, t_al)
        res_al = evaluate_bidirectional_scores(test_split, sim_al)
        eval_objs[model_key]["aligner"] = res_al
        write_run_bundle(
            runs_test_dir,
            run_id=f"test__{model_key}__{sel_pol}__aligner",
            config={"model_key": model_key, "split": "test", "policy": sel_pol, "method": "aligner", "align_params": align_params, "fit_split": "dev"},
            eval_result=res_al,
            repo_root=root,
        )
        test_summary[model_key]["aligner"] = {
            "align_params": align_params,
            "mean_R@1": res_al["mean_R@1"],
            "t2i": res_al["t2i"]["metrics"],
            "i2t": res_al["i2t"]["metrics"],
        }
        modality_gap_test[model_key] = {
            "before_aligner": compute_modality_gap_stats(v_test, t_test),
            "after_dev_fit_aligner": compute_modality_gap_stats(v_al, t_al),
        }

        # E4a: CSLS
        sim_csls = csls_similarity(sim_base, k=csls_k)
        res_csls = evaluate_bidirectional_scores(test_split, sim_csls)
        eval_objs[model_key]["csls"] = res_csls
        write_run_bundle(
            runs_test_dir,
            run_id=f"test__{model_key}__{sel_pol}__csls",
            config={"model_key": model_key, "split": "test", "policy": sel_pol, "method": "csls", "k": csls_k},
            eval_result=res_csls,
            repo_root=root,
        )
        test_summary[model_key]["csls"] = {
            "k": csls_k,
            "mean_R@1": res_csls["mean_R@1"],
            "t2i": res_csls["t2i"]["metrics"],
            "i2t": res_csls["i2t"]["metrics"],
        }

        # E4b: Dual-Softmax (frozen best) + symmetric_log reference
        s_ds_i2t, s_ds_t2i = dual_softmax_similarity(
            sim_base, temperature=float(ds_params["temperature"]), mode=str(ds_params["mode"])
        )
        res_ds = evaluate_bidirectional_scores(test_split, s_ds_i2t, s_ds_t2i)
        eval_objs[model_key]["dual_softmax"] = res_ds
        write_run_bundle(
            runs_test_dir,
            run_id=f"test__{model_key}__{sel_pol}__dual_softmax",
            config={"model_key": model_key, "split": "test", "policy": sel_pol, "method": "dual_softmax", **ds_params},
            eval_result=res_ds,
            repo_root=root,
        )
        test_summary[model_key]["dual_softmax"] = {
            **ds_params,
            "mean_R@1": res_ds["mean_R@1"],
            "t2i": res_ds["t2i"]["metrics"],
            "i2t": res_ds["i2t"]["metrics"],
        }

        s_sym_i2t, s_sym_t2i = dual_softmax_similarity(sim_base, temperature=0.02, mode="symmetric_log")
        res_sym = evaluate_bidirectional_scores(test_split, s_sym_i2t, s_sym_t2i)
        test_summary[model_key]["dual_softmax_symmetric_log_ref"] = {
            "temperature": 0.02,
            "mode": "symmetric_log",
            "mean_R@1": res_sym["mean_R@1"],
            "t2i": res_sym["t2i"]["metrics"],
            "i2t": res_sym["i2t"]["metrics"],
        }

        hubness_test[model_key] = {
            "baseline": compute_hubness_stats(sim_base.T, k=10),
            "csls": compute_hubness_stats(sim_csls.T, k=10),
            "dual_softmax": compute_hubness_stats(s_ds_t2i, k=10),
        }

    # Held-out test 3-way DimRed compression on CLIP ViT-B/32 (fit STRICTLY on dev, evaluated on test!)
    print("  Evaluating 3-way DimRed compression on held-out test (dev-fitted)...")
    clip_pol = frozen["models"]["clip_vit_b32"]["selected_policy"]
    v_dev_clip = dev_bundles["clip_vit_b32"][clip_pol].image_embeddings
    t_dev_clip = dev_bundles["clip_vit_b32"][clip_pol].caption_embeddings
    v_test_clip = test_bundles["clip_vit_b32"][clip_pol].image_embeddings
    t_test_clip = test_bundles["clip_vit_b32"][clip_pol].caption_embeddings

    dimred_test: Dict[str, List[dict]] = {}
    for mode in ("naive_pca", "uncentered_svd", "modality_centered_pca"):
        rows = []
        for d in (512, 256, 128, 64, 32):
            if d == 512:
                vp, tp, ev_ratio = v_test_clip, t_test_clip, 1.0
            else:
                dr = CrossModalPCA(n_components=d, seed=SEED, mode=mode).fit(v_dev_clip, t_dev_clip)
                vp, tp = dr.transform(v_test_clip, t_test_clip)
                ev_ratio = dr.explained_variance_ratio_sum
            s = compute_similarity_matrix(vp, tp)
            ev = evaluate_bidirectional_scores(test_split, s)
            rows.append(
                {
                    "mode": mode,
                    "dim": d,
                    "compression_ratio": round(512.0 / d, 2),
                    "explained_variance_dev": round(ev_ratio, 4),
                    "t2i_R@1": round(ev["t2i"]["metrics"]["R@1"], 2),
                    "i2t_R@1": round(ev["i2t"]["metrics"]["R@1"], 2),
                    "mean_R@1": round(ev["mean_R@1"], 2),
                    "t2i_R@5": round(ev["t2i"]["metrics"]["R@5"], 2),
                    "i2t_R@5": round(ev["i2t"]["metrics"]["R@5"], 2),
                }
            )
        dimred_test[mode] = rows

    # Phase 2 NegCLIP-lite (E5) one-shot evaluation on held-out test!
    print("  Evaluating frozen Phase 2 NegCLIP-lite on held-out test...")
    p2_cfg = frozen["phase2_negclip_lite"]
    neg_wrapper, neg_adapter, neg_payload = _load_negclip_lite(specs["clip_vit_b32"], root / p2_cfg["checkpoint_path"])
    p2_pol = p2_cfg["selected_config"]["policy"]

    test_texts = [c.text for c in test_split.captions]
    v_test_neg = encode_images_with_adapter(test_bundles["clip_vit_b32"][p2_pol].image_embeddings, neg_adapter, neg_wrapper.device)
    t_test_neg = encode_texts_with_model(neg_wrapper.model, neg_wrapper.tokenizer, test_texts, policy=p2_pol, device=neg_wrapper.device)

    sim_neg = compute_similarity_matrix(v_test_neg, t_test_neg)
    res_neg_cos = evaluate_bidirectional_scores(test_split, sim_neg)
    sim_neg_csls = csls_similarity(sim_neg, k=int(frozen["models"]["clip_vit_b32"]["best_csls_k"]))
    res_neg_csls = evaluate_bidirectional_scores(test_split, sim_neg_csls)
    ds_p = frozen["models"]["clip_vit_b32"]["best_dual_softmax_params"]
    s_neg_ds_i2t, s_neg_ds_t2i = dual_softmax_similarity(
        sim_neg, temperature=float(ds_p["temperature"]), mode=str(ds_p["mode"])
    )
    res_neg_ds = evaluate_bidirectional_scores(test_split, s_neg_ds_i2t, s_neg_ds_t2i)

    eval_objs["negclip_lite"] = {
        "cosine": res_neg_cos,
        "csls": res_neg_csls,
        "dual_softmax": res_neg_ds,
    }
    for mname, res_obj in eval_objs["negclip_lite"].items():
        write_run_bundle(
            runs_test_dir,
            run_id=f"test__negclip_lite__{p2_pol}__{mname}",
            config={"model_key": "negclip_lite", "split": "test", "policy": p2_pol, "method": mname, "phase2_config": p2_cfg["selected_config"]},
            eval_result=res_obj,
            repo_root=root,
        )

    # Held-out test compositional hard-negative accuracy (Frozen CLIP vs NegCLIP-lite)
    test_rng = np.random.Generator(np.random.PCG64(SEED + 200))
    test_neg_pairs = [generate_hard_negative(t, test_rng) for t in test_texts]
    test_neg_texts = [p[0] for p in test_neg_pairs]
    test_img_row = {rec.image_id: i for i, rec in enumerate(test_split.images)}
    test_cap_to_img_row = [test_img_row[c.image_id] for c in test_split.captions]

    base_clip_wrapper = VLMWrapper(specs["clip_vit_b32"])
    t_test_hn_base, _ = base_clip_wrapper.encode_texts(test_neg_texts, policy=p2_pol, batch_size=128)
    t_test_hn_neg = encode_texts_with_model(neg_wrapper.model, neg_wrapper.tokenizer, test_neg_texts, policy=p2_pol, device=neg_wrapper.device)

    test_comp_base = evaluate_compositional_hard_neg_acc(
        v_test_clip, t_test_clip, t_test_hn_base, test_cap_to_img_row
    )
    test_comp_neg = evaluate_compositional_hard_neg_acc(
        v_test_neg, t_test_neg, t_test_hn_neg, test_cap_to_img_row
    )
    del base_clip_wrapper

    test_summary["negclip_lite"] = {
        "selected_config": p2_cfg["selected_config"]["name"],
        "selected_epoch": p2_cfg["selected_epoch"],
        "cosine": {"mean_R@1": res_neg_cos["mean_R@1"], "t2i": res_neg_cos["t2i"]["metrics"], "i2t": res_neg_cos["i2t"]["metrics"]},
        "csls": {"mean_R@1": res_neg_csls["mean_R@1"], "t2i": res_neg_csls["t2i"]["metrics"], "i2t": res_neg_csls["i2t"]["metrics"]},
        "dual_softmax": {"mean_R@1": res_neg_ds["mean_R@1"], "t2i": res_neg_ds["t2i"]["metrics"], "i2t": res_neg_ds["i2t"]["metrics"]},
        "test_hard_neg_comp_acc": {
            "frozen_clip_vit_b32": round(test_comp_base, 2),
            "negclip_lite": round(test_comp_neg, 2),
            "delta": round(test_comp_neg - test_comp_base, 2),
            "operator_counts": dict(Counter(p[1] for p in test_neg_pairs)),
        },
    }

    # Paired bootstrap 95% CI (1,000 image-grouped resamples, seed=5489) on held-out test
    print("  Computing 1,000x image-grouped paired bootstrap 95% CIs on held-out test...")

    def _boot_pair(res_a: dict, res_b: dict) -> dict:
        return {
            "t2i_r1": paired_bootstrap_retrieval(res_a["t2i"]["ranks"], res_b["t2i"]["ranks"], cap_group_keys, k=1, n_resamples=1000, seed=SEED),
            "i2t_r1": paired_bootstrap_retrieval(res_a["i2t"]["ranks"], res_b["i2t"]["ranks"], img_group_keys, k=1, n_resamples=1000, seed=SEED),
        }

    bootstrap_deltas = {
        "clip_vit_b32_prefix_vs_raw": _boot_pair(eval_objs["clip_vit_b32"]["raw_baseline"], eval_objs["clip_vit_b32"]["photo_prefix_baseline"]),
        "clip_rn50_prefix_vs_raw": _boot_pair(eval_objs["clip_rn50"]["raw_baseline"], eval_objs["clip_rn50"]["photo_prefix_baseline"]),
        "siglip_prefix_vs_raw": _boot_pair(eval_objs["siglip_vit_b16"]["raw_baseline"], eval_objs["siglip_vit_b16"]["photo_prefix_baseline"]),
        "clip_vit_b32_aligner_vs_base": _boot_pair(eval_objs["clip_vit_b32"]["photo_prefix_baseline"], eval_objs["clip_vit_b32"]["aligner"]),
        "clip_vit_b32_csls_vs_base": _boot_pair(eval_objs["clip_vit_b32"]["photo_prefix_baseline"], eval_objs["clip_vit_b32"]["csls"]),
        "clip_vit_b32_ds_vs_base": _boot_pair(eval_objs["clip_vit_b32"]["photo_prefix_baseline"], eval_objs["clip_vit_b32"]["dual_softmax"]),
        "clip_rn50_ds_vs_base": _boot_pair(eval_objs["clip_rn50"]["photo_prefix_baseline"], eval_objs["clip_rn50"]["dual_softmax"]),
        "siglip_ds_vs_base": _boot_pair(eval_objs["siglip_vit_b16"]["photo_prefix_baseline"], eval_objs["siglip_vit_b16"]["dual_softmax"]),
        "siglip_vs_clip_base": _boot_pair(eval_objs["clip_vit_b32"]["photo_prefix_baseline"], eval_objs["siglip_vit_b16"]["photo_prefix_baseline"]),
        "negclip_vs_clip_cosine": _boot_pair(eval_objs["clip_vit_b32"]["photo_prefix_baseline"], eval_objs["negclip_lite"]["cosine"]),
        "negclip_ds_vs_clip_ds": _boot_pair(eval_objs["clip_vit_b32"]["dual_softmax"], eval_objs["negclip_lite"]["dual_softmax"]),
    }

    # Stratified success/failure case sampling (seed=5489) on E0 (clip_vit_b32 raw)
    print("  Sampling stratified T2I & I2T success/failure cases (seed=5489)...")
    e0_res = eval_objs["clip_vit_b32"]["raw_baseline"]
    t2i_sample = sample_stratified_cases(test_split.caption_ids, e0_res["t2i"]["ranks"], seed=SEED, max_per_stratum=10)
    i2t_sample = sample_stratified_cases(test_split.image_ids, e0_res["i2t"]["ranks"], seed=SEED, max_per_stratum=10)

    cap_by_id = {c.caption_id: c for c in test_split.captions}
    img_by_id = {im.image_id: im for im in test_split.images}
    method_lookup = {
        "E0_clip_raw": eval_objs["clip_vit_b32"]["raw_baseline"],
        "E1_clip_prefix": eval_objs["clip_vit_b32"]["photo_prefix_baseline"],
        "E4_clip_csls": eval_objs["clip_vit_b32"]["csls"],
        "E4_clip_dual_softmax": eval_objs["clip_vit_b32"]["dual_softmax"],
        "E5_negclip_lite": eval_objs["negclip_lite"]["cosine"],
        "E2_siglip_prefix": eval_objs["siglip_vit_b16"]["photo_prefix_baseline"],
    }

    def _collect_cases(sampled_ids: List[str], direction: str) -> List[dict]:
        per_q_maps = {
            mname: {rec["query_id"]: rec for rec in mobj[direction]["per_query"]}
            for mname, mobj in method_lookup.items()
        }
        cases = []
        for qid in sampled_ids:
            entry: dict = {"query_id": qid}
            if direction == "t2i":
                c = cap_by_id[qid]
                entry["caption_text"] = c.text
                entry["gt_image_id"] = c.image_id
                entry["gt_image_path"] = img_by_id[c.image_id].relative_path
            else:
                im = img_by_id[qid]
                entry["image_path"] = im.relative_path
                entry["gt_captions"] = [c.text for c in test_split.captions if c.image_id == qid]
            entry["methods"] = {}
            for mname, qmap in per_q_maps.items():
                rec = qmap[qid]
                top1_id = rec["top_k_ids"][0]
                m_info = {
                    "rank": rec["first_positive_rank"],
                    "top1_id": top1_id,
                    "top1_score": rec["top_k_scores"][0],
                }
                if direction == "i2t":
                    m_info["top1_caption"] = cap_by_id[top1_id].text
                entry["methods"][mname] = m_info
            cases.append(entry)
        return cases

    # =========================================================================
    # Part 2: Full 400-Group Winoground Evaluation (Evaluation-Only!)
    # =========================================================================
    print("\n[Round 5] Part 2: Full 400-group Winoground evaluation (strict evaluation-only)...")
    parquet_path = Path(
        hf_hub_download("facebook/winoground", filename="data/test-00000-of-00001.parquet", repo_type="dataset")
    )
    wg_parquet_sha256 = sha256_file(parquet_path)
    tbl = pq.read_table(parquet_path)
    n_wg = tbl.num_rows
    assert n_wg == 400, f"Expected 400 Winoground groups, got {n_wg}"

    wg_ids = [str(x) for x in tbl["id"].to_pylist()]
    wg_cap0 = [str(x) for x in tbl["caption_0"].to_pylist()]
    wg_cap1 = [str(x) for x in tbl["caption_1"].to_pylist()]
    wg_collapsed_tags = [[str(x)] for x in tbl["collapsed_tag"].to_pylist()]
    wg_fine_tags = [str(x) for x in tbl["tag"].to_pylist()]

    wg_img0 = [Image.open(io.BytesIO(cell["bytes"])).convert("RGB") for cell in tbl["image_0"].to_pylist()]
    wg_img1 = [Image.open(io.BytesIO(cell["bytes"])).convert("RGB") for cell in tbl["image_1"].to_pylist()]

    wg_results_summary: Dict[str, dict] = {}
    wg_full_objs: Dict[str, dict] = {}

    def _record_wg(run_key: str, tuples: List[Tuple[float, float, float, float]], extra_meta: dict) -> None:
        res = evaluate_winoground(tuples, group_ids=wg_ids, tags=wg_collapsed_tags, full_benchmark_size=400)
        wg_full_objs[run_key] = res
        wg_results_summary[run_key] = {
            **extra_meta,
            "num_groups": res["num_groups"],
            "is_partial": res["is_partial"],
            "metrics": {k: round(v, 2) for k, v in res["metrics"].items()},
            "tag_breakdown": {
                tg: {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()}
                for tg, d in res["tag_breakdown"].items()
            },
        }
        out_file = runs_wg_dir / f"winoground__{run_key}.json"
        out_file.write_text(
            json.dumps(
                {
                    "run_key": run_key,
                    "parquet_sha256": wg_parquet_sha256,
                    **extra_meta,
                    "summary": wg_results_summary[run_key],
                    "per_group": res["per_group"],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        m = res["metrics"]
        print(
            f"  [Winoground] {run_key:34s} | Text={m['text_score']:5.2f}% | Image={m['image_score']:5.2f}% | Group={m['group_score']:5.2f}%"
        )

    # Frozen pretrained models on Winoground
    clip_wg_img0_base = None
    clip_wg_img1_base = None
    for model_key, spec in specs.items():
        w = VLMWrapper(spec)
        im0 = _encode_pil_images(w, wg_img0, batch_size=64)
        im1 = _encode_pil_images(w, wg_img1, batch_size=64)
        if model_key == "clip_vit_b32":
            clip_wg_img0_base, clip_wg_img1_base = im0, im1

        for pol in ("raw", "photo_prefix"):
            c0, trunc0 = w.encode_texts(wg_cap0, policy=pol, batch_size=128)
            c1, trunc1 = w.encode_texts(wg_cap1, policy=pol, batch_size=128)
            tuples = _score_winoground_embeddings(im0, im1, c0, c1)
            _record_wg(
                f"{model_key}__{pol}",
                tuples,
                {
                    "model_key": model_key,
                    "policy": pol,
                    "num_truncated": trunc0["num_truncated"] + trunc1["num_truncated"],
                },
            )
            # Also evaluate dev-fitted ModalityAligner on CLIP ViT-B/32 photo_prefix for diagnostic comparison
            if model_key == "clip_vit_b32" and pol == clip_pol:
                al_params = frozen["models"]["clip_vit_b32"]["best_align_params"]
                al = ModalityAligner(**al_params).fit(v_dev_clip, t_dev_clip)
                im0_a, c0_a = al.transform(im0, c0)
                im1_a, c1_a = al.transform(im1, c1)
                tuples_al = _score_winoground_embeddings(im0_a, im1_a, c0_a, c1_a)
                _record_wg(
                    f"{model_key}__{pol}__aligner_devfit",
                    tuples_al,
                    {"model_key": model_key, "policy": pol, "aligner": al_params},
                )
        del w

    # Phase 2 NegCLIP-lite on Winoground!
    neg_wg_im0 = encode_images_with_adapter(clip_wg_img0_base, neg_adapter, neg_wrapper.device)
    neg_wg_im1 = encode_images_with_adapter(clip_wg_img1_base, neg_adapter, neg_wrapper.device)
    for pol in ("raw", "photo_prefix"):
        c0_n = encode_texts_with_model(neg_wrapper.model, neg_wrapper.tokenizer, wg_cap0, policy=pol, device=neg_wrapper.device)
        c1_n = encode_texts_with_model(neg_wrapper.model, neg_wrapper.tokenizer, wg_cap1, policy=pol, device=neg_wrapper.device)
        tuples_n = _score_winoground_embeddings(neg_wg_im0, neg_wg_im1, c0_n, c1_n)
        _record_wg(
            f"negclip_lite__{pol}",
            tuples_n,
            {
                "model_key": "negclip_lite",
                "policy": pol,
                "phase2_config": p2_cfg["selected_config"]["name"],
                "selected_epoch": p2_cfg["selected_epoch"],
            },
        )

    # Winoground paired bootstrap (1,000 group resamples, seed=5489)
    def _wg_boot(key_a: str, key_b: str) -> dict:
        pa = wg_full_objs[key_a]["per_group"]
        pb = wg_full_objs[key_b]["per_group"]
        return {
            "text": paired_bootstrap_binary([r["text_success"] for r in pa], [r["text_success"] for r in pb], n_resamples=1000, seed=SEED),
            "image": paired_bootstrap_binary([r["image_success"] for r in pa], [r["image_success"] for r in pb], n_resamples=1000, seed=SEED),
            "group": paired_bootstrap_binary([r["group_success"] for r in pa], [r["group_success"] for r in pb], n_resamples=1000, seed=SEED),
        }

    wg_bootstrap = {
        "clip_vit_b32_prefix_vs_raw": _wg_boot("clip_vit_b32__raw", "clip_vit_b32__photo_prefix"),
        "negclip_vs_clip_prefix": _wg_boot("clip_vit_b32__photo_prefix", "negclip_lite__photo_prefix"),
        "negclip_vs_clip_raw": _wg_boot("clip_vit_b32__raw", "negclip_lite__raw"),
        "siglip_vs_clip_prefix": _wg_boot("clip_vit_b32__photo_prefix", "siglip_vit_b16__photo_prefix"),
    }

    # Stratified Winoground case samples based on E0 (clip_vit_b32 raw) group success/failure
    wg_e0_ranks = [1 if r["group_success"] else 2 for r in wg_full_objs["clip_vit_b32__raw"]["per_group"]]
    wg_sample = sample_stratified_cases(wg_ids, wg_e0_ranks, seed=SEED, max_per_stratum=10)

    def _collect_wg_cases(sampled_gids: List[str]) -> List[dict]:
        idx_map = {gid: i for i, gid in enumerate(wg_ids)}
        out = []
        for gid in sampled_gids:
            i = idx_map[gid]
            row = {
                "group_id": gid,
                "caption_0": wg_cap0[i],
                "caption_1": wg_cap1[i],
                "collapsed_tag": wg_collapsed_tags[i][0],
                "fine_tag": wg_fine_tags[i],
                "models": {},
            }
            for mk in ("clip_vit_b32__raw", "clip_vit_b32__photo_prefix", "negclip_lite__photo_prefix", "siglip_vit_b16__photo_prefix"):
                rec = wg_full_objs[mk]["per_group"][i]
                row["models"][mk] = {
                    "text_success": rec["text_success"],
                    "image_success": rec["image_success"],
                    "group_success": rec["group_success"],
                    "scores": [round(rec["s00"], 4), round(rec["s01"], 4), round(rec["s10"], 4), round(rec["s11"], 4)],
                }
            out.append(row)
        return out

    # Render 4 publication-ready final figures
    print("\n[Round 5] Generating 4 publication-ready final figures...")
    final_figs = _plot_final_figures(
        fig_dir,
        test_summary,
        bootstrap_deltas,
        hubness_test,
        dimred_test,
        wg_results_summary,
    )

    # Save complete summaries
    hubness_compact = {
        mk: {
            mname: {k: v for k, v in hd.items() if k != "occurrences"}
            for mname, hd in hmap.items()
        }
        for mk, hmap in hubness_test.items()
    }
    final_payload = {
        "provenance": collect_runtime_provenance(
            repo_root=root,
            extra={
                "test_manifest_sha256": test_split.manifest_sha256,
                "dev_manifest_sha256": dev_split.manifest_sha256,
                "winoground_parquet_sha256": wg_parquet_sha256,
                "total_wall_seconds": round(time.perf_counter() - t_start, 2),
                "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024), 1),
            },
        ),
        "test_split": {
            "name": "test",
            "num_images": len(test_split.images),
            "num_captions": len(test_split.captions),
            "manifest_sha256": test_split.manifest_sha256,
        },
        "test_results": test_summary,
        "test_bootstrap_95ci": bootstrap_deltas,
        "test_modality_gap": modality_gap_test,
        "test_hubness_t2i_k10": hubness_compact,
        "test_dimred_compression_clip_vit_b32": dimred_test,
        "winoground": {
            "num_groups": 400,
            "parquet_sha256": wg_parquet_sha256,
            "random_baseline": {"text_score": 25.0, "image_score": 25.0, "group_score": 16.67},
            "results": wg_results_summary,
            "bootstrap_95ci": wg_bootstrap,
        },
        "final_figures": [str(Path(p).relative_to(root)) for p in final_figs],
    }

    out_final = root / "reports" / "final_test_summary.json"
    out_final.write_text(json.dumps(final_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    stratified_payload = {
        "seed": SEED,
        "coco_test_t2i": {
            "sampling_summary": t2i_sample,
            "success_cases": _collect_cases(t2i_sample["sampled_success_ids"], "t2i"),
            "failure_cases": _collect_cases(t2i_sample["sampled_failure_ids"], "t2i"),
        },
        "coco_test_i2t": {
            "sampling_summary": i2t_sample,
            "success_cases": _collect_cases(i2t_sample["sampled_success_ids"], "i2t"),
            "failure_cases": _collect_cases(i2t_sample["sampled_failure_ids"], "i2t"),
        },
        "winoground_group": {
            "sampling_summary": wg_sample,
            "success_cases": _collect_wg_cases(wg_sample["sampled_success_ids"]),
            "failure_cases": _collect_wg_cases(wg_sample["sampled_failure_ids"]),
        },
    }
    out_cases = root / "reports" / "stratified_case_analysis.json"
    out_cases.write_text(json.dumps(stratified_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n=== ROUND 5 FINAL HELD-OUT TEST SUMMARY ===")
    for mk in ("clip_vit_b32", "clip_rn50", "siglip_vit_b16"):
        r = test_summary[mk]
        print(
            f"{mk:15s} | raw={r['raw_baseline']['mean_R@1']:.2f}% | prefix={r['photo_prefix_baseline']['mean_R@1']:.2f}% "
            f"| align={r['aligner']['mean_R@1']:.2f}% | CSLS={r['csls']['mean_R@1']:.2f}% | DualSM={r['dual_softmax']['mean_R@1']:.2f}%"
        )
    nr = test_summary["negclip_lite"]
    print(
        f"{'negclip_lite':15s} | cos={nr['cosine']['mean_R@1']:.2f}% (T2I {nr['cosine']['t2i']['R@1']:.2f} / I2T {nr['cosine']['i2t']['R@1']:.2f}) "
        f"| CSLS={nr['csls']['mean_R@1']:.2f}% | DualSM={nr['dual_softmax']['mean_R@1']:.2f}% "
        f"| test_comp={nr['test_hard_neg_comp_acc']['frozen_clip_vit_b32']:.2f}% -> {nr['test_hard_neg_comp_acc']['negclip_lite']:.2f}%"
    )
    print(f"Saved -> {out_final} & {out_cases}")


if __name__ == "__main__":
    main()
