# CS5489 Experimental Results Summary (Programmatically Generated)

Generated directly from `reports/dev_phase1_summary.json`, `reports/dev_phase2_summary.json`, `reports/final_test_summary.json`, and `reports/stratified_case_analysis.json` (protocol `v2`, git commit `c33d02c`).

## Table 1: Dataset Splits & Provenance

| Dataset / Split | Role | Images / Groups | Captions | Manifest / Parquet SHA256 (prefix) |
|---|---|---:|---:|---|
| COCO val2017 (`train`) | Phase 2 NegCLIP-lite adaptation only | 3,000 | 15,007 | `78eafdd86146` |
| COCO val2017 (`dev`) | Phase 1 5-fold CV tuning, feature fitting, Phase 2 selection | 1,000 | 5,005 | `901400596ba9` |
| COCO val2017 (`test`) | One-shot held-out retrieval evaluation (after freeze) | 1,000 | 5,002 | `5b6d8fde92de` |
| COCO val2017 (`smoke_dev100`) | Round 1 feasibility smoke subset (first 100 dev images) | 100 | 501 | `6bbfbcf891b7` |
| Winoground (`test`) | Strict evaluation-only compositional diagnostic | 400 groups (800 imgs) | 800 | `499e49636d65` |

## Table 2: Dev Image-Grouped 5-Fold Cross-Validation Summary

| Model | Policy | Baseline 5-Fold Mean R@1 | Best ModalityAligner (5-Fold) | Best CSLS (5-Fold) | Best Dual-Softmax (5-Fold) |
|---|---|---:|---|---|---|
| `clip_vit_b32` | `photo_prefix` | 80.24% ± 1.41 | `zca_shared` (ε=0.05) → 79.04% | `k=5` → **84.50%** ± 1.33 | `prior_weighted` (τ=0.02) → **85.55%** ± 0.99 |
| `clip_rn50` | `photo_prefix` | 79.13% ± 1.61 | `zca_per_modality` (ε=0.05) → 78.43% | `k=5` → **84.33%** ± 1.75 | `prior_weighted` (τ=0.02) → **85.52%** ± 1.92 |
| `siglip_vit_b16` | `photo_prefix` | 88.24% ± 2.00 | `zca_per_modality` (ε=0.05) → 85.84% | `k=10` → **90.83%** ± 1.90 | `prior_weighted` (τ=0.02) → **91.39%** ± 1.30 |

## Table 3: Held-Out COCO Test Results (1,000 Images / 5,002 Captions)

| Model & Method | T2I R@1 | T2I R@5 | T2I R@10 | T2I MRR | I2T R@1 | I2T R@5 | I2T R@10 | I2T MRR | **Mean R@1** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CLIP ViT-B/32 (`raw`, E0) | 49.70 | 77.05 | 87.05 | 62.19 | 69.90 | 89.90 | 95.20 | 78.73 | **59.80** |
| CLIP ViT-B/32 (`photo_prefix`, E1) | 49.94 | 78.15 | 87.62 | 62.51 | 71.50 | 89.90 | 95.90 | 79.76 | **60.72** |
| CLIP ViT-B/32 + ModalityAligner (`zca_shared`, dev-fit, E3) | 51.56 | 79.11 | 88.28 | 63.92 | 67.30 | 88.90 | 95.00 | 76.75 | **59.43** |
| CLIP ViT-B/32 + CSLS (`k=5`, E4a) | 57.14 | 83.45 | 91.58 | 68.79 | 76.80 | 92.50 | 96.70 | 83.70 | **66.97** |
| CLIP ViT-B/32 + Dual-Softmax (`τ=0.02`, E4b) | 59.42 | 84.41 | 91.76 | 70.69 | 78.30 | 93.50 | 96.00 | 84.78 | **68.86** |
| CLIP RN50 (`raw`, E2) | 47.90 | 76.91 | 87.15 | 60.99 | 71.30 | 91.40 | 95.90 | 79.95 | **59.60** |
| CLIP RN50 (`photo_prefix`, E2) | 48.60 | 77.35 | 87.50 | 61.45 | 71.10 | 91.50 | 95.80 | 79.97 | **59.85** |
| CLIP RN50 + ModalityAligner (`zca_per_modality`, dev-fit) | 49.92 | 78.93 | 88.92 | 62.97 | 69.00 | 89.30 | 94.40 | 78.23 | **59.46** |
| CLIP RN50 + CSLS (`k=5`) | 56.68 | 83.13 | 91.40 | 68.46 | 76.70 | 93.10 | 96.70 | 83.83 | **66.69** |
| CLIP RN50 + Dual-Softmax (`τ=0.02`) | 59.38 | 84.19 | 91.86 | 70.51 | 77.10 | 93.40 | 96.70 | 84.29 | **68.24** |
| SigLIP ViT-B/16 (`raw`, E2) | 66.27 | 88.16 | 93.86 | 75.99 | 82.60 | 96.00 | 98.40 | 88.35 | **74.44** |
| SigLIP ViT-B/16 (`photo_prefix`, E2) | 66.71 | 88.20 | 94.26 | 76.32 | 83.70 | 95.70 | 97.80 | 88.97 | **75.21** |
| SigLIP ViT-B/16 + ModalityAligner (`zca_per_modality`, dev-fit) | 64.57 | 86.87 | 93.44 | 74.48 | 78.60 | 94.50 | 97.70 | 85.30 | **71.59** |
| SigLIP ViT-B/16 + CSLS (`k=10`) | 71.25 | 91.12 | 95.94 | 79.94 | 86.60 | 96.80 | 98.50 | 91.01 | **78.93** |
| SigLIP ViT-B/16 + Dual-Softmax (`τ=0.02`) | 71.47 | 91.26 | 95.88 | 80.20 | 89.40 | 97.10 | 98.60 | 92.66 | **80.44** |
| NegCLIP-lite Cosine (Phase 2 `cfg1` ep3, E5) | 53.46 | 81.51 | 89.60 | 65.76 | 71.20 | 90.40 | 94.90 | 79.81 | **62.33** |
| NegCLIP-lite + CSLS (`k=5`) | 59.74 | 84.83 | 92.82 | 70.94 | 77.00 | 92.80 | 96.20 | 84.10 | **68.37** |
| NegCLIP-lite + Dual-Softmax (`τ=0.02`) | 60.62 | 84.81 | 92.16 | 71.52 | 78.40 | 93.30 | 96.30 | 85.18 | **69.51** |

## Table 4: 1,000× Image-Grouped Paired Bootstrap 95% CIs (Held-Out Test R@1)

| Comparison (B − A) | T2I ΔR@1 (95% CI) | T2I $p$ | I2T ΔR@1 (95% CI) | I2T $p$ |
|---|---|---:|---|---:|
| CLIP ViT-B/32: Prefix − Raw | +0.24 `[-0.50, +1.08]` | 0.614 | +1.60 `[-0.50, +3.60]` | 0.132 |
| CLIP RN50: Prefix − Raw | +0.70 `[-0.04, +1.46]` | 0.058 | -0.20 `[-2.10, +1.70]` | 0.898 |
| SigLIP B/16: Prefix − Raw | +0.44 `[-0.36, +1.14]` | 0.280 | +1.10 `[-0.50, +2.70]` | 0.200 |
| CLIP ViT-B/32: Aligner (dev-fit) − Prefix | +1.62 `[+0.26, +2.94]` | 0.022 | -4.20 `[-6.70, -1.80]` | 0.000 |
| CLIP ViT-B/32: CSLS − Prefix | +7.20 `[+6.04, +8.36]` | 0.000 | +5.30 `[+3.50, +7.30]` | 0.000 |
| CLIP ViT-B/32: Dual-Softmax − Prefix | +9.48 `[+7.88, +11.02]` | 0.000 | +6.80 `[+4.60, +9.30]` | 0.000 |
| CLIP RN50: Dual-Softmax − Prefix | +10.78 `[+9.06, +12.46]` | 0.000 | +6.00 `[+3.60, +8.30]` | 0.000 |
| SigLIP B/16: Dual-Softmax − Prefix | +4.76 `[+3.54, +6.02]` | 0.000 | +5.70 `[+3.80, +7.30]` | 0.000 |
| SigLIP B/16 − CLIP ViT-B/32 (Prefix) | +16.77 `[+14.74, +18.69]` | 0.000 | +12.20 `[+9.60, +15.20]` | 0.000 |
| NegCLIP-lite − Frozen CLIP (Cosine) | +3.52 `[+2.62, +4.42]` | 0.000 | -0.30 `[-2.50, +2.00]` | 0.808 |
| NegCLIP-lite − Frozen CLIP (Dual-Softmax) | +1.20 `[+0.56, +1.82]` | 0.000 | +0.10 `[-1.00, +1.10]` | 0.958 |

## Table 5: Modality Gap & T2I Hubness Diagnostics (Held-Out Test)

| Model | Centroid L2 Gap (Raw → Aligner) | Baseline Max Hub / Skew / Gini | CSLS Max Hub / Skew / Gini | Dual-Softmax Max Hub / Skew / Gini |
|---|---|---|---|---|
| `clip_vit_b32` | 0.811 → 0.043 | 407 / 2.15 / 0.390 | 416 / 2.63 / 0.328 | **110** / **-0.04** / **0.208** |
| `clip_rn50` | 0.806 → 0.046 | 311 / 1.73 / 0.398 | 309 / 1.89 / 0.316 | **110** / **0.02** / **0.192** |
| `siglip_vit_b16` | 1.046 → 0.044 | 302 / 1.80 / 0.326 | 192 / 1.44 / 0.268 | **158** / **0.95** / **0.251** |

## Table 6: Held-Out Test 3-Way Dimensionality Reduction (CLIP ViT-B/32, Fit on Dev)

| Dim (Compression) | Naive Stacked PCA Mean R@1 (T2I / I2T) | Uncentered SVD Mean R@1 (T2I / I2T) | Modality-Centered PCA Mean R@1 (T2I / I2T) |
|---|---|---|---|
| 512d (1×) | 60.72% (49.9 / 71.5) | 60.72% (49.9 / 71.5) | 60.72% (49.9 / 71.5) |
| 256d (2×) | 40.09% (50.5 / 29.7) | 58.48% (48.0 / 69.0) | 57.23% (49.6 / 64.9) |
| 128d (4×) | 34.21% (45.2 / 23.2) | 51.53% (41.9 / 61.2) | 52.64% (45.0 / 60.3) |
| 64d (8×) | 23.55% (35.8 / 11.3) | 34.48% (28.5 / 40.5) | 40.75% (35.6 / 45.9) |
| 32d (16×) | 14.36% (22.0 / 6.7) | 13.72% (11.6 / 15.8) | 24.94% (22.0 / 27.9) |

### K-Means Semantic Clusters on Dev (`CLIP ViT-B/32`, K=8, Silhouette=0.125)

| Cluster | Images | Mean I2T R@1 | Representative Captions |
|---|---:|---:|---|
| Cluster 2 | 63 | 49.21% | A couple of baseball player standing on a field.; A baseball player holding a bat next to a base. |
| Cluster 0 | 139 | 64.75% | A man on a blue raft attempting to catch a ride on a large wave.; A person standing on top of a ski covered slope. |
| Cluster 4 | 128 | 64.84% | Decorated coffee cup and knife sitting on a patterned surface.; a couple of buckets in a white room |
| Cluster 5 | 120 | 70.00% | A small group of sheep standing together next to a building.; The two young girls are petting the two goats. |
| Cluster 3 | 192 | 74.48% | The red, double decker bus is driving past other buses.; Row of buildings on empty street near side walk. |
| Cluster 6 | 157 | 77.07% | Man preforming a trick with a disk toy.; A man sitting in front of a laptop computer in an office. |
| Cluster 7 | 117 | 81.20% | A bowl with rice, broccoli and a purple relish.; many different slices of pizza with a white sauce. |
| Cluster 1 | 84 | 84.52% | A young boy laying in bed with a pacifier in his mouth; A teddy bear sits by a keyboards and microphone. |

## Table 7: Phase 2 NegCLIP-lite Dev Ablation & Held-Out Hard-Negative Generalization

| Configuration | Trainable Params | Best Ep | Dev Cosine Mean R@1 (T2I / I2T) | Dev Dual-SM Mean R@1 | Dev Hard-Neg Acc |
|---|---:|---:|---|---:|---:|
| Frozen CLIP ViT-B/32 (`photo_prefix`) | 0 (0.00%) | 0 | 60.64% (49.47 / 71.80) | 67.90% | 65.19% |
| `cfg1_text_adapter_lr1e5_hn0.5` **(Frozen)** | 6,830,849 (4.51%) | 3 | 62.74% (53.49 / 72.00) | 68.55% | 88.27% |
| `cfg2_text_adapter_lr3e5_hn1.0` | 6,830,849 (4.51%) | 2 | 61.12% (54.45 / 67.80) | 69.70% | 92.45% |
| `cfg3_text_only_lr2e5_hn1.0` | 6,567,936 (4.34%) | 2 | 61.50% (54.21 / 68.80) | 69.43% | 91.57% |

- **Held-out COCO Test Hard-Negative Pairwise Accuracy** (5,002 captions): Frozen CLIP `65.59%` → NegCLIP-lite **`88.02%`** (`+22.43%`).

## Table 8: Full 400-Group Winoground Compositional Diagnostic (Strict Evaluation-Only)

| Model / Policy | Text Score | Image Score | **Group Score** | Object (`n=141`) | Relation (`n=233`) | Both (`n=26`) |
|---|---:|---:|---:|---:|---:|---:|
| *Random Baseline (exact 4! permutation)* | *25.00%* | *25.00%* | *16.67%* | *16.67%* | *16.67%* | *16.67%* |
| CLIP ViT-B/32 (`raw`) | 31.25% | 11.00% | **8.75%** | 7.80% | 6.01% | 38.46% |
| CLIP ViT-B/32 (`photo_prefix`) | 29.50% | 12.00% | **8.50%** | 11.35% | 4.72% | 26.92% |
| CLIP ViT-B/32 + Aligner (dev-fit) | 28.75% | 13.00% | **8.00%** | 5.67% | 6.87% | 30.77% |
| CLIP RN50 (`raw`) | 26.00% | 11.25% | **8.50%** | 7.80% | 6.44% | 30.77% |
| CLIP RN50 (`photo_prefix`) | 26.50% | 11.50% | **7.25%** | 7.80% | 4.72% | 26.92% |
| NegCLIP-lite (`raw`, Phase 2) | 26.75% | 11.25% | **8.00%** | 7.80% | 5.15% | 34.62% |
| NegCLIP-lite (`photo_prefix`, Phase 2) | 28.50% | 13.00% | **9.00%** | 9.93% | 6.01% | 30.77% |
| SigLIP ViT-B/16 (`raw`) | 33.00% | 12.75% | **10.25%** | 9.22% | 7.73% | 38.46% |
| SigLIP ViT-B/16 (`photo_prefix`) | 34.00% | 15.00% | **10.25%** | 11.35% | 7.73% | 26.92% |

## Table 9: Sample Stratified Failure Cases (Fixed `seed=5489`)

### Held-Out COCO T2I Failures under E0 (`CLIP ViT-B/32 raw`) — Rank Across Methods

| Query ID | Caption | E0 Raw | E1 Prefix | CSLS | Dual-SM | NegCLIP-lite | SigLIP |
|---|---|---:|---:|---:|---:|---:|---:|
| `197951` | a table with many plates of food on top | 9 | 9 | 6 | 2 | 7 | 3 |
| `206730` | a close up of a lot of animals with a fence | 17 | 14 | 6 | 4 | 4 | 3 |
| `298668` | An elaborate decoration hangs on the side of a tall building. | 2 | 1 | 2 | 2 | 1 | 1 |
| `438390` | A man in helmet riding an elephant next to trees. | 5 | 7 | 2 | 3 | 4 | 1 |
| `455166` | A toy smiley face in the middle of a doughnut. | 2 | 1 | 4 | 4 | 1 | 1 |

### Winoground Group Failures under E0 (`CLIP ViT-B/32 raw`)

| Group ID | Tag | Caption 0 / Caption 1 | CLIP raw | NegCLIP-lite | SigLIP |
|---|---|---|---|---|---|
| `14` | Object (Determiner-Numeral, Noun Phrase) | `there is a mug in some grass` vs `there is some grass in a mug` | T:0 I:0 G:0 | T:0 I:1 G:0 | T:1 I:1 G:1 |
| `144` | Object (Noun) | `there are more people than stools` vs `there are more stools than people` | T:0 I:0 G:0 | T:0 I:0 G:0 | T:0 I:0 G:0 |
| `19` | Object (Noun) | `there is more dirt than empty space in the jar` vs `there is more empty space than dirt in the jar` | T:0 I:0 G:0 | T:0 I:0 G:0 | T:0 I:0 G:0 |
| `244` | Relation (Preposition) | `the knife on the left and the fork on the right` vs `the knife on the right and the fork on the left` | T:0 I:0 G:0 | T:0 I:0 G:0 | T:0 I:0 G:0 |
| `253` | Relation (Adjective-Color) | `a large pink inflatable flamingo along with some other lilos on the water including one that is mostly white` vs `a large white inflatable flamingo along with some other lilos on the water including one that is mostly pink` | T:0 I:0 G:0 | T:0 I:0 G:0 | T:0 I:0 G:0 |

## Figure Index (`reports/figures/`)

- [`figures/dev_modality_gap_pca.png`](figures/dev_modality_gap_pca.png): 2D PCA visualization of CLIP ViT-B/32 Modality Gap before vs. after Mean-Centering (dev)
- [`figures/dev_hubness_distribution.png`](figures/dev_hubness_distribution.png): Dev T2I Top-10 hubness rank curve: Baseline vs. CSLS vs. Dual-Softmax
- [`figures/dev_pca_and_cv_curves.png`](figures/dev_pca_and_cv_curves.png): Dev 5-fold CV 3-way DimRed compression + CSLS k & Dual-Softmax τ tuning curves
- [`figures/dev_data_and_clusters.png`](figures/dev_data_and_clusters.png): Dev caption length histogram & K-Means (K=8) semantic cluster I2T R@1 difficulty
- [`figures/dev_phase2_negclip_lite.png`](figures/dev_phase2_negclip_lite.png): Phase 2 NegCLIP-lite 3-config dev retrieval & hard-negative accuracy trajectories
- [`figures/final_test_model_method_comparison.png`](figures/final_test_model_method_comparison.png): Held-out 1K COCO test comparison across all backbones, rerankers, and NegCLIP-lite
- [`figures/final_test_hubness_and_dimred.png`](figures/final_test_hubness_and_dimred.png): Held-out test hubness mitigation & 3-way dev-fitted DimRed compression curves
- [`figures/final_winoground_comparison.png`](figures/final_winoground_comparison.png): Full 400-group Winoground overall scores & official Object/Relation/Both tag breakdown
- [`figures/final_bootstrap_deltas.png`](figures/final_bootstrap_deltas.png): Forest plot of 1,000× paired bootstrap 95% CIs on held-out test R@1 differences
