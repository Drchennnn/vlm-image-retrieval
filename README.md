# Natural-Language Image Retrieval with Pretrained Vision-Language Models

CS5489 Machine Learning (2026/27 Semester A) · City University of Hong Kong · Team: **Benyi Chen**, **Muyao Yan**, **Jianheng Qiao**

## Overview

This repository implements a reproducible bidirectional cross-modal retrieval system (**text-to-image** and **image-to-text**) built on pretrained vision-language representation spaces without training a foundation model from scratch. Beyond a frozen CLIP baseline, the study systematically investigates:

1. **Contrastive Pretraining Paradigms & Visual Backbones**: OpenAI `CLIP ViT-B/32` (ViT + Softmax InfoNCE), OpenAI `CLIP RN50` (CNN + Softmax InfoNCE), and `SigLIP ViT-B/16` (ViT + Sigmoid pairwise loss), plus controlled `raw` vs. `a photo of ` prompt evaluation.
2. **Embedding Geometry, Dimensionality Reduction & Clustering**: Quantifying and visualizing the **Modality Gap**, testing dev-fitted **Mean-Centering / ZCA Whitening** (`ModalityAligner`), comparing 3-way dimensionality compression (**Naive Stacked PCA** vs. **Uncentered SVD** vs. **Modality-Centered PCA**), and diagnosing retrieval difficulty across **K-Means** semantic clusters.
3. **Training-Free Hubness Mitigation with 5-Fold CV**: Mitigating high-dimensional **Hubness** via **CSLS** and **Dual-Softmax**, tuned strictly via image-grouped 5-fold cross-validation on the development split.
4. **Phase 2 Compositional Hard-Negative Adaptation (`NegCLIP-lite`)**: Lightweight adaptation (only 4.51% trainable parameters: top-2 text Transformer blocks + `ln_final` + `text_projection` + zero-initialized residual image adapter) trained on an independent 3,000-image `train` split with rule-based compositional hard negatives, followed by strict evaluation-only testing on all **400 official Winoground groups**.

All held-out test metrics, 1,000× paired bootstrap 95% confidence intervals, stratified failure cases, and 9 figures are summarized in [reports/RESULTS_SUMMARY.md](reports/RESULTS_SUMMARY.md).

---

## Key Held-Out Results (COCO 1K Test & 400-Group Winoground)

| Model & Method | COCO Test T2I R@1 | COCO Test I2T R@1 | COCO Test Mean R@1 | Winoground Text / Image / **Group** |
|---|---:|---:|---:|---|
| `CLIP ViT-B/32` (`raw`, E0) | 49.70% | 69.90% | 59.80% | 31.25% / 11.00% / **8.75%** |
| `CLIP ViT-B/32` (`photo_prefix`, E1) | 49.94% | 71.50% | 60.72% | 29.50% / 12.00% / **8.50%** |
| `CLIP ViT-B/32 + ModalityAligner` (dev-fit, E3) | 51.56% | 67.30% | 59.43% | 28.75% / 13.00% / **8.00%** |
| `CLIP ViT-B/32 + CSLS` (`k=5`, E4a) | 57.14% | 76.80% | **66.97%** | *(gallery-level reranking; N/A)* |
| `CLIP ViT-B/32 + Dual-Softmax` (`τ=0.02`, E4b) | 59.42% | 78.30% | **68.86%** | *(gallery-level reranking; N/A)* |
| `CLIP RN50` (`photo_prefix`) → `+Dual-Softmax` | 48.60% → 59.38% | 71.10% → 77.10% | 59.85% → **68.24%** | 26.50% / 11.50% / **7.25%** |
| `SigLIP ViT-B/16` (`photo_prefix`) → `+Dual-Softmax` | 66.71% → 71.47% | 83.70% → 89.40% | 75.21% → **80.44%** | 34.00% / 15.00% / **10.25%** |
| `NegCLIP-lite` Cosine (Phase 2 `cfg1` ep3, E5) | **53.46%** | 71.20% | **62.33%** | 28.50% / 13.00% / **9.00%** |
| `NegCLIP-lite + Dual-Softmax` (E5 + E4b) | **60.62%** | **78.40%** | **69.51%** | *(gallery-level reranking; N/A)* |

*Random Winoground reference: Text 25.00%, Image 25.00%, Group 16.67% (`4/24`). Held-out COCO hard-negative accuracy: Frozen CLIP `65.59%` → `NegCLIP-lite` **`88.02%` (`+22.43%`)**.*

---

## Repository Layout

```text
.
├── README.md                              # Project guide and reproduction instructions
├── pyproject.toml                         # Package & pytest configuration
├── requirements.txt                       # Locked Python 3.12 dependency versions
├── configs/
│   ├── models.yaml                        # Pretrained model specifications (CLIP ViT-B/32, RN50, SigLIP)
│   ├── frozen_phase1.json                 # Dev 5-fold CV frozen Phase 1 hyperparameters
│   ├── frozen_all.json                    # Complete frozen Phase 1 + Phase 2 configuration
│   └── splits/coco2017_split_ids.json     # Deterministic 3K/1K/1K image-ID manifests & SHA256
├── src/vlm_retrieval/
│   ├── data.py                            # Manifest construction, validation, split isolation checks
│   ├── model.py                           # Unified frozen VLM wrapper (open_clip) + truncation stats
│   ├── cache.py                           # Bit-exact embedding cache with manifest-hash validation
│   ├── retrieval.py                       # Exact float32 cosine similarity & deterministic tie-breaking
│   ├── metrics.py                         # Pure-function R@K, Winoground scorer, paired bootstrap CI
│   ├── features.py                        # ModalityAligner, 3-mode CrossModalPCA, Hubness, K-Means
│   ├── rerank.py                          # CSLS, Dual-Softmax, image-grouped 5-fold CV
│   ├── adapt.py                           # Phase 2 NegCLIP-lite hard-negative generator & adaptation
│   ├── evaluate.py                        # Run-bundle provenance & metrics serialization
│   └── cli.py                             # Command-line interface (encode / evaluate / search)
├── scripts/
│   ├── prepare_coco.py                    # Validate COCO val2017 & build 3K/1K/1K JSONL manifests
│   ├── smoke_round1.py                    # 100-image 3-model feasibility smoke test
│   ├── run_round3_dev.py                  # Phase 1 dev evaluation, 5-fold CV, diagnostics & figures
│   ├── run_round4_phase2.py               # Phase 2 NegCLIP-lite 3-config training & dev selection
│   ├── run_round5_final_eval.py           # One-shot held-out 1K test + 400-group Winoground + figures
│   ├── generate_report_tables.py          # Generate reports/RESULTS_SUMMARY.md from JSON outputs
│   └── package_code_zip.py                # Build clean ProjectGroup_<X>_Code.zip submission archive
├── tests/                                 # 17 synthetic unit tests (zero external downloads needed)
├── docs/                                  # REQUIREMENTS.md, PLAN.md, EVALUATION.md, PROGRESS.md
└── reports/
    ├── RESULTS_SUMMARY.md                 # Complete auto-generated tables & figure index
    ├── dev_phase1_summary.json            # Full dev Phase 1 metrics & CV logs
    ├── dev_phase2_summary.json            # Full Phase 2 training logs & ablation metrics
    ├── final_test_summary.json            # Held-out test, bootstrap CIs & Winoground results
    ├── stratified_case_analysis.json      # Seed-5489 stratified T2I, I2T & Winoground cases
    └── figures/                           # 9 publication-ready PNG figures
```

---

## Setup & Dataset Preparation

Large datasets and model weights are excluded from Git and Code ZIP archives.

### 1. Environment Installation (Python 3.12)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps -e .
```

### 2. Download Datasets

**A. Official COCO `val2017` (Public, ~1.0 GB total)**

- Images (`val2017.zip`): `http://images.cocodataset.org/zips/val2017.zip`
- Annotations (`annotations_trainval2017.zip`): `http://images.cocodataset.org/annotations/annotations_trainval2017.zip`

Place and extract into `data/raw/coco/` with the exact directory layout below:

```bash
mkdir -p data/raw/coco
curl -fL -o data/raw/coco/val2017.zip http://images.cocodataset.org/zips/val2017.zip
curl -fL -o data/raw/coco/annotations_trainval2017.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q data/raw/coco/val2017.zip -d data/raw/coco
unzip -q data/raw/coco/annotations_trainval2017.zip "annotations/captions_val2017.json" -d data/raw/coco
```

Expected data paths:
- `data/raw/coco/val2017/*.jpg` (5,000 images)
- `data/raw/coco/annotations/captions_val2017.json`

**B. Official Winoground (Gated HuggingFace Dataset, 400 groups)**

- Dataset page: `https://huggingface.co/datasets/facebook/winoground`
- Accept the dataset terms on HuggingFace and log in locally before running Round 5:

```bash
hf auth login
```

Pretrained model weights (`CLIP ViT-B-32`, `CLIP RN50`, `ViT-B-16-SigLIP`) are downloaded automatically by `open-clip-torch` into the local HuggingFace cache on first run.

---

## Reproduction Commands

### 1. Run Synthetic Unit Tests (No Dataset Download Required)

```bash
pytest -v
```

Covers multi-positive I2T ranking, deterministic tie-breaking, chunked consistency, split-leakage rejection, exact Winoground 24-permutation random baseline (`25% / 25% / 16.67%`), hubness rescue by CSLS & Dual-Softmax, dev-only fitting guards, and Phase 2 hard-negative / loss checks (17 tests).

### 2. Build Deterministic 3K / 1K / 1K COCO Manifests

```bash
python scripts/prepare_coco.py
```

Generates `data/manifests/coco2017/{train,dev,test,smoke_dev100}_{images,captions}.jsonl` using `PCG64(5489)` (protocol `coco2017-custom-3k-train-1k-dev-1k-test-v2`; verified against `configs/splits/coco2017_split_ids.json`).

### 3. Run Full Experimental Pipeline (Phase 1 + Phase 2 + Held-Out Test + Winoground)

```bash
# Optional 100-image feasibility smoke across all 3 backbones
python scripts/smoke_round1.py

# Phase 1: 1K dev encoding, baselines, 5-fold CV tuning, DimRed/K-Means/Hubness diagnostics
python scripts/run_round3_dev.py

# Phase 2: NegCLIP-lite 3-config adaptation on 3K train, dev selection & config freeze
python scripts/run_round4_phase2.py

# Final one-shot held-out 1K test + 400-group Winoground + 1,000x paired bootstrap CI + figures
python scripts/run_round5_final_eval.py

# Regenerate Markdown report tables directly from JSON artifacts
python scripts/generate_report_tables.py
```

### 4. Interactive CLI Usage (Encode, Evaluate, and Bidirectional Search)

```bash
# Encode a split into cache
python -m vlm_retrieval.cli encode --model clip_vit_b32 --split dev --policy photo_prefix

# Evaluate cached split with Dual-Softmax reranking
python -m vlm_retrieval.cli evaluate --model clip_vit_b32 --split dev --policy photo_prefix \
  --rerank dual_softmax --ds-temp 0.02 --ds-mode prior_weighted

# Text-to-Image (T2I) query
python -m vlm_retrieval.cli search --model clip_vit_b32 --split dev --policy photo_prefix \
  --text "a baseball player swinging a bat on a field" --top-k 5

# Image-to-Text (I2T) query by image_id
python -m vlm_retrieval.cli search --model clip_vit_b32 --split dev --policy photo_prefix \
  --image-id 000000000872 --top-k 5
```

### 5. Build Clean Submission Code ZIP

```bash
python scripts/package_code_zip.py --group <YourGroupNumber>
```

---

## Documentation & References

- [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) · [docs/PLAN.md](docs/PLAN.md) · [docs/EVALUATION.md](docs/EVALUATION.md) · [docs/PROGRESS.md](docs/PROGRESS.md) · [reports/RESULTS_SUMMARY.md](reports/RESULTS_SUMMARY.md)
- Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
- Zhai et al. (2023). *Sigmoid Loss for Language Image Pre-Training (SigLIP)*. [arXiv:2303.15343](https://arxiv.org/abs/2303.15343)
- Thrush et al. (2022). *Winoground: Probing Vision and Language Models for Visio-Linguistic Compositionality*. [arXiv:2204.03162](https://arxiv.org/abs/2204.03162)
- Conneau et al. (2018). *Word Translation Without Parallel Data (CSLS)*. [arXiv:1710.04087](https://arxiv.org/abs/1710.04087)
- Cheng et al. (2021). *Improving Video-Text Retrieval by Multi-Stream Corpus Alignment and Dual Softmax Loss*. [arXiv:2109.04290](https://arxiv.org/abs/2109.04290)
- Liang et al. (2022). *Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning*. [arXiv:2203.02053](https://arxiv.org/abs/2203.02053)
- Yuksekgonul et al. (2023). *When and Why Vision-Language Models Behave Like Bags-of-Words, and What to Do About It? (NegCLIP)*. [arXiv:2210.01936](https://arxiv.org/abs/2210.01936)
