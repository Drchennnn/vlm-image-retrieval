# Natural-Language Image Retrieval with Pretrained Vision-Language Models

[中文](README.zh-CN.md) · CS5489 Machine Learning · Reviewed 20 October 2026

## Status and scope

**Planning and independent review completed; implementation and experiments have not started.** The repository contains the assignment, reference papers and planning documents. No dataset acquisition, installed project environment, model execution or experimental metric has been verified.

Following [the assignment](1.md), this project will build a reproducible text-to-image and image-to-text retrieval system using pretrained CLIP, explain its contrastive pretraining objective, and investigate the limits of compositional matching. Foundation-model pretraining is out of scope. Fine-tuning is optional and conditional on a validated baseline, suitable training data and measured compute capacity.

## Proposed study

| Component | Decision / condition |
|---|---|
| Baseline | Frozen OpenAI CLIP ViT-B/32 with its supplied preprocessing and tokenizer |
| Primary data | Flickr30k, conditional on authorized access and validation of published retrieval split IDs |
| Access fallback | COCO val2017 with an explicitly labelled custom development/evaluation split; never described as the Karpathy test split |
| Retrieval | L2-normalized embeddings, exact cosine ranking, cached features and batched scoring |
| Primary metrics | Recall@1, @5 and @10, separately for both retrieval directions |
| Controlled comparison | Raw captions versus the fixed `a photo of ` prefix, motivated by CLIP Appendix E.1 |
| Diagnostic | Winoground text, image and group scores, evaluation-only and conditional on licensed access |
| Optional extension | One bounded adaptation experiment, only after the baseline review gate |

These are project design choices, not measured findings. No performance improvement is promised. “Zero-shot” means no task-specific weight updates; it does not establish absence of overlap with CLIP's pretraining data.

## Method

CLIP maps images and captions into a shared representation space. During pretraining, a symmetric cross-entropy objective promotes matching image-caption pairs over other pairings within a batch. In this project, the initial encoders remain frozen. Normalized image and text embeddings are ranked by their dot product; image-to-text evaluation accepts any annotated caption for the query image. See the [evaluation protocol](docs/EVALUATION.md) for exact definitions, split isolation and Winoground scoring.

## Documentation

- [Project Proposal](PROJECT_PROPOSAL.md): standalone English submission text; [PDF](output/pdf/Project_Proposal.pdf).
- [Independent review](docs/REVIEW.md): assignment traceability, corrections, evidence and unresolved assumptions.
- [Project plan](docs/PROJECT_PLAN.md): intended structure, interfaces, six-week schedule, execution and review gates.
- [Evaluation protocol](docs/EVALUATION.md): data sources, metrics, experiment controls and reproducibility requirements.
- [Progress ledger](docs/PROGRESS.md): authoritative task status and evidence; [checklist](CHECKLIST.md) is a navigation summary.
- [Agent instructions](AGENTS.md): operational rules for future contributors.

The implementation layout in the plan is **not yet created**. `requirements.txt` remains an unpinned historical candidate list, not a validated installation specification. Dependency selection and locking are part of the environment milestone. No application command is advertised before it exists and passes a smoke test.

## References

- Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. [Paper](https://arxiv.org/abs/2103.00020) · [Local PDF](paper/Learning%20Transferable%20Visual%20Models%20From%20Natural%20Language%20Supervision.pdf) · [Official implementation](https://github.com/openai/CLIP).
- Thrush et al. (2022). *Winoground: Probing Vision and Language Models for Visio-Linguistic Compositionality*. [Paper](https://arxiv.org/abs/2204.03162) · [Local PDF](paper/Winoground.pdf) · [Dataset](https://huggingface.co/datasets/facebook/winoground).
- [Flickr30k source and access terms](https://shannon.cs.illinois.edu/DenotationGraph/) · [Published split release](https://cs.stanford.edu/people/karpathy/deepimagesent/) · [COCO](https://cocodataset.org/#download).
