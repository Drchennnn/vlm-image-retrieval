# Natural-Language Image Retrieval with Pretrained Vision-Language Models

CS5489 Machine Learning (2026/27 Semester A) · Updated 24 November 2026 · Team: Benyi Chen, Muyao Yan, Jianheng Qiao

## Status

**Round 3 (T4 — Phase 1 dev experiments) completed**: all three pretrained backbones (`CLIP ViT-B/32`, `CLIP RN50`, `SigLIP ViT-B/16`) have been encoded and evaluated on the full 1,000-image / 5,005-caption COCO `dev` split with image-grouped 5-fold CV tuning (`CSLS`, `Dual-Softmax`, `ModalityAligner`), 3-mode PCA/SVD dimensionality reduction, K-Means cluster diagnostics, and hubness analysis (`reports/dev_phase1_summary.json`, `configs/frozen_phase1.json`, and 4 diagnostic figures in `reports/figures/`). Phase 2 compositional hard-negative adaptation (Round 4) and final held-out test / Winoground evaluation (Round 5) are next.

## Overview

Following the CS5489 project specification ([requirements & rubric](docs/REQUIREMENTS.md)), this project builds a reproducible bidirectional image-text retrieval system (text-to-image and image-to-text) on pretrained vision-language representations, explains contrastive pretraining (Softmax InfoNCE vs. Sigmoid pairwise loss), and investigates **modality gap**, **hubness**, and **compositional robustness**. Foundation-model pretraining from scratch is out of scope.

### Experimental Design (v2)

| Dimension | Planned Design |
|---|---|
| Pretrained backbones | OpenAI CLIP `ViT-B/32` (baseline), CLIP `RN50` (CNN vs. ViT), and `SigLIP` (Softmax InfoNCE vs. Sigmoid loss) |
| Prompt control | Raw captions vs. fixed `a photo of ` prefix (CLIP Appendix E.1) |
| Datasets | Flickr30k (Karpathy split) if access is verified on Day 1; otherwise official COCO `val2017` with an explicitly labelled, image-ID-isolated custom split |
| Representation & analysis | PCA/UMAP modality-gap visualization, PCA dimensionality compression, K-Means semantic cluster analysis, hubness distribution, and **mean-centering / whitening** feature alignment (fit on dev only) |
| Training-free reranking | **CSLS** and **Dual-Softmax** hubness mitigation, tuned via **image-grouped 5-fold cross-validation** on dev |
| Compositional diagnostic | **Winoground** (400 groups, strict text/image/group scoring, evaluation-only, conditional on licensed access) |
| Optional Phase-2 sprint | Lightweight compositional hard-negative adaptation (NegCLIP-lite LoRA/Adapter) on an independent train split, strictly isolated from test and Winoground |

No performance gain is claimed prior to experiments. "Zero-shot" means no task-specific weight updates and does not guarantee absence of pretraining overlap.

## Repository Layout

```text
.
├── README.md                  # Project overview (and future Code ZIP instructions)
├── AGENTS.md                  # Engineering and review rules
├── requirements.txt           # Historical candidate list (to be locked in T1)
├── docs/
│   ├── REQUIREMENTS.md        # Assignment prompt, Canvas rubric, and submission rules
│   ├── PLAN.md                # Target package architecture and 7-day sprint plan
│   ├── EVALUATION.md          # Authoritative experimental protocol (v2)
│   ├── PROGRESS.md            # Single source of truth for task status and decisions
│   └── archive/               # Original prompt, Oct-20 review, and submitted Proposal
└── paper/                     # Reference PDFs (CLIP and Winoground)
```

Implementation folders (`src/vlm_retrieval/`, `configs/`, `tests/`, `data/`, `artifacts/`, `reports/`) will be created as implementation milestones start. Dataset download URLs, preprocessing scripts, and runnable CLI commands will be documented here once verified.

## References

- Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) · [Local PDF](paper/Learning%20Transferable%20Visual%20Models%20From%20Natural%20Language%20Supervision.pdf) · [Code](https://github.com/openai/CLIP)
- Thrush et al. (2022). *Winoground: Probing Vision and Language Models for Visio-Linguistic Compositionality*. [arXiv:2204.03162](https://arxiv.org/abs/2204.03162) · [Local PDF](paper/Winoground.pdf) · [Dataset](https://huggingface.co/datasets/facebook/winoground)
- [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/) · [Karpathy splits](https://cs.stanford.edu/people/karpathy/deepimagesent/) · [COCO](https://cocodataset.org/#download)
