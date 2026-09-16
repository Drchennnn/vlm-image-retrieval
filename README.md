# Natural-Language Image Retrieval with Pretrained Vision-Language Models

[中文 / Chinese](README.zh-CN.md)

## Project Status

This project is in the planning stage.

The goal is to build a reproducible bidirectional image-text retrieval baseline using a pretrained vision-language model:

- text-to-image retrieval;
- image-to-text retrieval.

The first priority is a trustworthy baseline. Fine-tuning will be considered only after the baseline has been implemented and evaluated.

## Method Overview

The planned baseline will use CLIP or a compatible pretrained model. An image encoder and a text encoder map both modalities into a shared embedding space. Normalized embeddings can be compared using cosine similarity or an equivalent dot product.

CLIP-style contrastive pretraining uses matched image-text pairs as positives and other combinations in the same batch as negatives. A symmetric objective increases the similarity of matched pairs and decreases the similarity of mismatched pairs.

These descriptions summarize the referenced papers. They do not imply that this project has trained a model, downloaded data, or reproduced the original training setup.

## Evaluation Plan

The project will evaluate:

- text-to-image retrieval;
- image-to-text retrieval;
- Winoground as an additional compositionality stress test.

The dataset, checkpoint, preprocessing, metrics, indexing method, and computing environment are not finalized. Candidate retrieval metrics include Recall@K, median rank, mean rank, and mean reciprocal rank.

Winoground is intended to remain evaluation-only unless a future experiment explicitly documents another use.

## Current Evidence

- Reference papers are present in the repository root.
- No project implementation has been verified.
- No dataset, model run, experiment result, or metric has been verified.
- `.agents`, `.specify`, `openspec`, and `.omx` are local tooling directories and are not part of the project upload set.
- Candidate dependencies are listed in `requirements.txt`; they have not been installed.
- A virtual environment named `vlm` is planned but has not been created locally.

See [CHECKLIST.md](CHECKLIST.md) for pending work and possible future directions.

## References

- [Learning Transferable Visual Models From Natural Language Supervision](./Learning%20Transferable%20Visual%20Models%20From%20Natural%20Language%20Supervision.pdf) · [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
- [Winoground](./Winoground.pdf) · [arXiv:2204.03162](https://arxiv.org/abs/2204.03162)
