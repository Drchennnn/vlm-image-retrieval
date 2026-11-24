"""Generate reports/RESULTS_SUMMARY.md directly from machine-readable experiment JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path


def _fmt(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dev_p1 = json.loads((root / "reports" / "dev_phase1_summary.json").read_text(encoding="utf-8"))
    dev_p2 = json.loads((root / "reports" / "dev_phase2_summary.json").read_text(encoding="utf-8"))
    final = json.loads((root / "reports" / "final_test_summary.json").read_text(encoding="utf-8"))
    cases = json.loads((root / "reports" / "stratified_case_analysis.json").read_text(encoding="utf-8"))
    frozen = json.loads((root / "configs" / "frozen_all.json").read_text(encoding="utf-8"))
    split_ids = json.loads((root / "configs" / "splits" / "coco2017_split_ids.json").read_text(encoding="utf-8"))

    lines: list[str] = []
    add = lines.append

    add("# CS5489 Experimental Results Summary (Programmatically Generated)")
    add("")
    add(
        f"Generated directly from `reports/dev_phase1_summary.json`, `reports/dev_phase2_summary.json`, "
        f"`reports/final_test_summary.json`, and `reports/stratified_case_analysis.json` "
        f"(protocol `{final['provenance']['protocol_version']}`, git commit `{final['provenance']['git_commit'][:7]}`)."
    )
    add("")

    # Table 1: Dataset Splits & Provenance
    add("## Table 1: Dataset Splits & Provenance")
    add("")
    add("| Dataset / Split | Role | Images / Groups | Captions | Manifest / Parquet SHA256 (prefix) |")
    add("|---|---|---:|---:|---|")
    for sname, role in [
        ("train", "Phase 2 NegCLIP-lite adaptation only"),
        ("dev", "Phase 1 5-fold CV tuning, feature fitting, Phase 2 selection"),
        ("test", "One-shot held-out retrieval evaluation (after freeze)"),
        ("smoke_dev100", "Round 1 feasibility smoke subset (first 100 dev images)"),
    ]:
        sp = split_ids["splits"][sname]
        add(
            f"| COCO val2017 (`{sname}`) | {role} | {sp['num_images']:,} | {sp['num_captions']:,} | `{sp['combined_manifest_sha256'][:12]}` |"
        )
    add(
        f"| Winoground (`test`) | Strict evaluation-only compositional diagnostic | "
        f"{final['winoground']['num_groups']} groups (800 imgs) | 800 | `{final['winoground']['parquet_sha256'][:12]}` |"
    )
    add("")

    # Table 2: Dev 5-Fold CV Hyperparameter Selection
    add("## Table 2: Dev Image-Grouped 5-Fold Cross-Validation Summary")
    add("")
    add("| Model | Policy | Baseline 5-Fold Mean R@1 | Best ModalityAligner (5-Fold) | Best CSLS (5-Fold) | Best Dual-Softmax (5-Fold) |")
    add("|---|---|---:|---|---|---|")
    for mk in ("clip_vit_b32", "clip_rn50", "siglip_vit_b16"):
        cv = dev_p1["cv_results"][mk]
        b = cv["baseline_5fold"]
        al = cv["modality_aligner_5fold"]["best"]
        cs = cv["csls_5fold"]["best"]
        ds_sym = cv["dual_softmax_symmetric_5fold"]["best"]
        ds_pw = cv["dual_softmax_prior_weighted_5fold"]["best"]
        ds = ds_sym if ds_sym["mean_r1"] >= ds_pw["mean_r1"] else ds_pw
        add(
            f"| `{mk}` | `{cv['selected_policy']}` | {_fmt(b['mean_r1'])}% ± {_fmt(b['std_r1'])} "
            f"| `{al['params']['mode']}` (ε={al['params'].get('eps', '-')}) → {_fmt(al['mean_r1'])}% "
            f"| `k={cs['params']['k']}` → **{_fmt(cs['mean_r1'])}%** ± {_fmt(cs['std_r1'])} "
            f"| `{ds['params']['mode']}` (τ={ds['params']['temperature']}) → **{_fmt(ds['mean_r1'])}%** ± {_fmt(ds['std_r1'])} |"
        )
    add("")

    # Table 3: Main Held-Out 1K COCO Test Results
    add("## Table 3: Held-Out COCO Test Results (1,000 Images / 5,002 Captions)")
    add("")
    add("| Model & Method | T2I R@1 | T2I R@5 | T2I R@10 | T2I MRR | I2T R@1 | I2T R@5 | I2T R@10 | I2T MRR | **Mean R@1** |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    tr = final["test_results"]
    rows = [
        ("CLIP ViT-B/32 (`raw`, E0)", tr["clip_vit_b32"]["raw_baseline"]),
        ("CLIP ViT-B/32 (`photo_prefix`, E1)", tr["clip_vit_b32"]["photo_prefix_baseline"]),
        ("CLIP ViT-B/32 + ModalityAligner (`zca_shared`, dev-fit, E3)", tr["clip_vit_b32"]["aligner"]),
        ("CLIP ViT-B/32 + CSLS (`k=5`, E4a)", tr["clip_vit_b32"]["csls"]),
        ("CLIP ViT-B/32 + Dual-Softmax (`τ=0.02`, E4b)", tr["clip_vit_b32"]["dual_softmax"]),
        ("CLIP RN50 (`raw`, E2)", tr["clip_rn50"]["raw_baseline"]),
        ("CLIP RN50 (`photo_prefix`, E2)", tr["clip_rn50"]["photo_prefix_baseline"]),
        ("CLIP RN50 + ModalityAligner (`zca_per_modality`, dev-fit)", tr["clip_rn50"]["aligner"]),
        ("CLIP RN50 + CSLS (`k=5`)", tr["clip_rn50"]["csls"]),
        ("CLIP RN50 + Dual-Softmax (`τ=0.02`)", tr["clip_rn50"]["dual_softmax"]),
        ("SigLIP ViT-B/16 (`raw`, E2)", tr["siglip_vit_b16"]["raw_baseline"]),
        ("SigLIP ViT-B/16 (`photo_prefix`, E2)", tr["siglip_vit_b16"]["photo_prefix_baseline"]),
        ("SigLIP ViT-B/16 + ModalityAligner (`zca_per_modality`, dev-fit)", tr["siglip_vit_b16"]["aligner"]),
        ("SigLIP ViT-B/16 + CSLS (`k=10`)", tr["siglip_vit_b16"]["csls"]),
        ("SigLIP ViT-B/16 + Dual-Softmax (`τ=0.02`)", tr["siglip_vit_b16"]["dual_softmax"]),
        ("NegCLIP-lite Cosine (Phase 2 `cfg1` ep3, E5)", tr["negclip_lite"]["cosine"]),
        ("NegCLIP-lite + CSLS (`k=5`)", tr["negclip_lite"]["csls"]),
        ("NegCLIP-lite + Dual-Softmax (`τ=0.02`)", tr["negclip_lite"]["dual_softmax"]),
    ]
    for name, obj in rows:
        t2i, i2t = obj["t2i"], obj["i2t"]
        add(
            f"| {name} | {_fmt(t2i['R@1'])} | {_fmt(t2i['R@5'])} | {_fmt(t2i['R@10'])} | {_fmt(t2i['MRR'])} "
            f"| {_fmt(i2t['R@1'])} | {_fmt(i2t['R@5'])} | {_fmt(i2t['R@10'])} | {_fmt(i2t['MRR'])} | **{_fmt(obj['mean_R@1'])}** |"
        )
    add("")

    # Table 4: Paired Bootstrap 95% CIs on Held-Out Test
    add("## Table 4: 1,000× Image-Grouped Paired Bootstrap 95% CIs (Held-Out Test R@1)")
    add("")
    add("| Comparison (B − A) | T2I ΔR@1 (95% CI) | T2I $p$ | I2T ΔR@1 (95% CI) | I2T $p$ |")
    add("|---|---|---:|---|---:|")
    boot_labels = [
        ("clip_vit_b32_prefix_vs_raw", "CLIP ViT-B/32: Prefix − Raw"),
        ("clip_rn50_prefix_vs_raw", "CLIP RN50: Prefix − Raw"),
        ("siglip_prefix_vs_raw", "SigLIP B/16: Prefix − Raw"),
        ("clip_vit_b32_aligner_vs_base", "CLIP ViT-B/32: Aligner (dev-fit) − Prefix"),
        ("clip_vit_b32_csls_vs_base", "CLIP ViT-B/32: CSLS − Prefix"),
        ("clip_vit_b32_ds_vs_base", "CLIP ViT-B/32: Dual-Softmax − Prefix"),
        ("clip_rn50_ds_vs_base", "CLIP RN50: Dual-Softmax − Prefix"),
        ("siglip_ds_vs_base", "SigLIP B/16: Dual-Softmax − Prefix"),
        ("siglip_vs_clip_base", "SigLIP B/16 − CLIP ViT-B/32 (Prefix)"),
        ("negclip_vs_clip_cosine", "NegCLIP-lite − Frozen CLIP (Cosine)"),
        ("negclip_ds_vs_clip_ds", "NegCLIP-lite − Frozen CLIP (Dual-Softmax)"),
    ]
    for key, label in boot_labels:
        bt = final["test_bootstrap_95ci"][key]["t2i_r1"]
        bi = final["test_bootstrap_95ci"][key]["i2t_r1"]
        add(
            f"| {label} | {bt['diff_b_minus_a']:+.2f} `[{bt['ci_95'][0]:+.2f}, {bt['ci_95'][1]:+.2f}]` | {bt['p_value_approx']:.3f} "
            f"| {bi['diff_b_minus_a']:+.2f} `[{bi['ci_95'][0]:+.2f}, {bi['ci_95'][1]:+.2f}]` | {bi['p_value_approx']:.3f} |"
        )
    add("")

    # Table 5: Modality Gap & Hubness on Held-Out Test
    add("## Table 5: Modality Gap & T2I Hubness Diagnostics (Held-Out Test)")
    add("")
    add("| Model | Centroid L2 Gap (Raw → Aligner) | Baseline Max Hub / Skew / Gini | CSLS Max Hub / Skew / Gini | Dual-Softmax Max Hub / Skew / Gini |")
    add("|---|---|---|---|---|")
    for mk in ("clip_vit_b32", "clip_rn50", "siglip_vit_b16"):
        mg = final["test_modality_gap"][mk]
        hb = final["test_hubness_t2i_k10"][mk]
        add(
            f"| `{mk}` | {mg['before_aligner']['centroid_l2_distance']:.3f} → {mg['after_dev_fit_aligner']['centroid_l2_distance']:.3f} "
            f"| {hb['baseline']['max_occurrence']} / {hb['baseline']['skewness_snk']:.2f} / {hb['baseline']['gini']:.3f} "
            f"| {hb['csls']['max_occurrence']} / {hb['csls']['skewness_snk']:.2f} / {hb['csls']['gini']:.3f} "
            f"| **{hb['dual_softmax']['max_occurrence']}** / **{hb['dual_softmax']['skewness_snk']:.2f}** / **{hb['dual_softmax']['gini']:.3f}** |"
        )
    add("")

    # Table 6: Held-Out Test 3-Way DimRed Compression + K-Means Clusters
    add("## Table 6: Held-Out Test 3-Way Dimensionality Reduction (CLIP ViT-B/32, Fit on Dev)")
    add("")
    add("| Dim (Compression) | Naive Stacked PCA Mean R@1 (T2I / I2T) | Uncentered SVD Mean R@1 (T2I / I2T) | Modality-Centered PCA Mean R@1 (T2I / I2T) |")
    add("|---|---|---|---|")
    dr = final["test_dimred_compression_clip_vit_b32"]
    for i in range(len(dr["uncentered_svd"])):
        n_r = dr["naive_pca"][i]
        u_r = dr["uncentered_svd"][i]
        m_r = dr["modality_centered_pca"][i]
        add(
            f"| {u_r['dim']}d ({u_r['compression_ratio']:.0f}×) "
            f"| {n_r['mean_R@1']:.2f}% ({n_r['t2i_R@1']:.1f} / {n_r['i2t_R@1']:.1f}) "
            f"| {u_r['mean_R@1']:.2f}% ({u_r['t2i_R@1']:.1f} / {u_r['i2t_R@1']:.1f}) "
            f"| {m_r['mean_R@1']:.2f}% ({m_r['t2i_R@1']:.1f} / {m_r['i2t_R@1']:.1f}) |"
        )
    add("")

    add("### K-Means Semantic Clusters on Dev (`CLIP ViT-B/32`, K=8, Silhouette=0.125)")
    add("")
    add("| Cluster | Images | Mean I2T R@1 | Representative Captions |")
    add("|---|---:|---:|---|")
    for cl in sorted(dev_p1["clustering"]["clip_vit_b32"]["clusters"], key=lambda x: x["mean_r1"]):
        ex = "; ".join(cl["sample_captions"][:2])
        add(f"| Cluster {cl['cluster_id']} | {cl['count']} | {cl['mean_r1']:.2f}% | {ex} |")
    add("")

    # Table 7: Phase 2 NegCLIP-lite Dev Ablation & Test Hard-Neg Generalization
    add("## Table 7: Phase 2 NegCLIP-lite Dev Ablation & Held-Out Hard-Negative Generalization")
    add("")
    add("| Configuration | Trainable Params | Best Ep | Dev Cosine Mean R@1 (T2I / I2T) | Dev Dual-SM Mean R@1 | Dev Hard-Neg Acc |")
    add("|---|---:|---:|---|---:|---:|")
    fb = dev_p2["frozen_baseline"]
    add(
        f"| Frozen CLIP ViT-B/32 (`photo_prefix`) | 0 (0.00%) | 0 | {fb['dev_cosine']['mean_R@1']:.2f}% ({fb['dev_cosine']['t2i_R@1']:.2f} / {fb['dev_cosine']['i2t_R@1']:.2f}) | {fb['dev_dual_softmax']['mean_R@1']:.2f}% | {fb['dev_comp_acc']:.2f}% |"
    )
    for cs in dev_p2["configs"]:
        ep = cs["best_epoch_record"]
        sel_tag = " **(Frozen)**" if cs["config"]["name"] == dev_p2["selected"]["config_name"] else ""
        add(
            f"| `{cs['config']['name']}`{sel_tag} | {cs['trainable_params']:,} ({cs['trainable_pct']:.2f}%) | {cs['best_epoch']} "
            f"| {ep['dev_cosine']['mean_R@1']:.2f}% ({ep['dev_cosine']['t2i_R@1']:.2f} / {ep['dev_cosine']['i2t_R@1']:.2f}) "
            f"| {ep['dev_dual_softmax']['mean_R@1']:.2f}% | {ep['dev_comp_acc']:.2f}% |"
        )
    add("")
    thn = tr["negclip_lite"]["test_hard_neg_comp_acc"]
    add(
        f"- **Held-out COCO Test Hard-Negative Pairwise Accuracy** (5,002 captions): "
        f"Frozen CLIP `{thn['frozen_clip_vit_b32']:.2f}%` → NegCLIP-lite **`{thn['negclip_lite']:.2f}%`** (`{thn['delta']:+.2f}%`)."
    )
    add("")

    # Table 8: Full 400-Group Winoground Diagnostic
    add("## Table 8: Full 400-Group Winoground Compositional Diagnostic (Strict Evaluation-Only)")
    add("")
    add("| Model / Policy | Text Score | Image Score | **Group Score** | Object (`n=141`) | Relation (`n=233`) | Both (`n=26`) |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    add("| *Random Baseline (exact 4! permutation)* | *25.00%* | *25.00%* | *16.67%* | *16.67%* | *16.67%* | *16.67%* |")
    wg = final["winoground"]["results"]
    wg_rows = [
        ("CLIP ViT-B/32 (`raw`)", "clip_vit_b32__raw"),
        ("CLIP ViT-B/32 (`photo_prefix`)", "clip_vit_b32__photo_prefix"),
        ("CLIP ViT-B/32 + Aligner (dev-fit)", "clip_vit_b32__photo_prefix__aligner_devfit"),
        ("CLIP RN50 (`raw`)", "clip_rn50__raw"),
        ("CLIP RN50 (`photo_prefix`)", "clip_rn50__photo_prefix"),
        ("NegCLIP-lite (`raw`, Phase 2)", "negclip_lite__raw"),
        ("NegCLIP-lite (`photo_prefix`, Phase 2)", "negclip_lite__photo_prefix"),
        ("SigLIP ViT-B/16 (`raw`)", "siglip_vit_b16__raw"),
        ("SigLIP ViT-B/16 (`photo_prefix`)", "siglip_vit_b16__photo_prefix"),
    ]
    for lbl, k in wg_rows:
        m = wg[k]["metrics"]
        tb = wg[k]["tag_breakdown"]
        add(
            f"| {lbl} | {m['text_score']:.2f}% | {m['image_score']:.2f}% | **{m['group_score']:.2f}%** "
            f"| {tb['Object']['group_score']:.2f}% | {tb['Relation']['group_score']:.2f}% | {tb['Both']['group_score']:.2f}% |"
        )
    add("")

    # Table 9: Sample Stratified Failure Cases (first 5 T2I failures & 5 Winoground failures from seed=5489)
    add("## Table 9: Sample Stratified Failure Cases (Fixed `seed=5489`)")
    add("")
    add("### Held-Out COCO T2I Failures under E0 (`CLIP ViT-B/32 raw`) — Rank Across Methods")
    add("")
    add("| Query ID | Caption | E0 Raw | E1 Prefix | CSLS | Dual-SM | NegCLIP-lite | SigLIP |")
    add("|---|---|---:|---:|---:|---:|---:|---:|")
    for c in cases["coco_test_t2i"]["failure_cases"][:5]:
        ms = c["methods"]
        add(
            f"| `{c['query_id']}` | {c['caption_text']} | {ms['E0_clip_raw']['rank']} | {ms['E1_clip_prefix']['rank']} "
            f"| {ms['E4_clip_csls']['rank']} | {ms['E4_clip_dual_softmax']['rank']} "
            f"| {ms['E5_negclip_lite']['rank']} | {ms['E2_siglip_prefix']['rank']} |"
        )
    add("")

    add("### Winoground Group Failures under E0 (`CLIP ViT-B/32 raw`)")
    add("")
    add("| Group ID | Tag | Caption 0 / Caption 1 | CLIP raw | NegCLIP-lite | SigLIP |")
    add("|---|---|---|---|---|---|")
    for c in cases["winoground_group"]["failure_cases"][:5]:
        ms = c["models"]
        def _flag(k: str) -> str:
            r = ms[k]
            return f"T:{int(r['text_success'])} I:{int(r['image_success'])} G:{int(r['group_success'])}"
        add(
            f"| `{c['group_id']}` | {c['collapsed_tag']} ({c['fine_tag']}) "
            f"| `{c['caption_0']}` vs `{c['caption_1']}` "
            f"| {_flag('clip_vit_b32__raw')} | {_flag('negclip_lite__photo_prefix')} | {_flag('siglip_vit_b16__photo_prefix')} |"
        )
    add("")

    # Figure Index
    add("## Figure Index (`reports/figures/`)")
    add("")
    fig_desc = [
        ("figures/dev_modality_gap_pca.png", "2D PCA visualization of CLIP ViT-B/32 Modality Gap before vs. after Mean-Centering (dev)"),
        ("figures/dev_hubness_distribution.png", "Dev T2I Top-10 hubness rank curve: Baseline vs. CSLS vs. Dual-Softmax"),
        ("figures/dev_pca_and_cv_curves.png", "Dev 5-fold CV 3-way DimRed compression + CSLS k & Dual-Softmax τ tuning curves"),
        ("figures/dev_data_and_clusters.png", "Dev caption length histogram & K-Means (K=8) semantic cluster I2T R@1 difficulty"),
        ("figures/dev_phase2_negclip_lite.png", "Phase 2 NegCLIP-lite 3-config dev retrieval & hard-negative accuracy trajectories"),
        ("figures/final_test_model_method_comparison.png", "Held-out 1K COCO test comparison across all backbones, rerankers, and NegCLIP-lite"),
        ("figures/final_test_hubness_and_dimred.png", "Held-out test hubness mitigation & 3-way dev-fitted DimRed compression curves"),
        ("figures/final_winoground_comparison.png", "Full 400-group Winoground overall scores & official Object/Relation/Both tag breakdown"),
        ("figures/final_bootstrap_deltas.png", "Forest plot of 1,000× paired bootstrap 95% CIs on held-out test R@1 differences"),
    ]
    for path_rel, desc in fig_desc:
        add(f"- [`{path_rel}`]({path_rel}): {desc}")
    add("")

    out_path = root / "reports" / "RESULTS_SUMMARY.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated {out_path} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
