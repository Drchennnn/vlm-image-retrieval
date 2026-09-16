# Natural-Language Image Retrieval with Pretrained Vision-Language Models
[中文 / Chinese](README.zh-CN.md)

## Project Status

This repository is currently in the planning stage.

- **Existing:** Two reference papers are present in the repository root:
  - *Learning Transferable Visual Models From Natural Language Supervision*
  - *Winoground: Probing Vision and Language Models for Visio-Linguistic Compositionality*
- **Planned:** A reproducible image-text retrieval pipeline based on a pretrained vision-language model.
- **To be decided:** The dataset, model checkpoint/version, evaluation metrics, indexing library, preprocessing details, and computing environment.

No project implementation, dataset, experiment result, or execution command is claimed to exist yet.

## Project Goal

The goal of this project is to study natural-language image retrieval with a pretrained vision-language model.

The planned system will support two retrieval directions:

1. **Text-to-image retrieval:** Given a natural-language query, retrieve the most relevant images from an image collection.
2. **Image-to-text retrieval:** Given an image, retrieve the most relevant captions or text descriptions from a text collection.

The project will focus on understanding how pretrained visual and textual representations can be used for cross-modal similarity search, rather than claiming a completed system at this stage.

## Planned Baseline: CLIP

The initial baseline is planned to use CLIP, or a compatible pretrained CLIP implementation. The exact model family, checkpoint, and version are **to be decided**.

CLIP contains:

- an **image encoder** that maps an image to a vector representation;
- a **text encoder** that maps a text description to a vector representation.

The two encoders are trained so that matching images and texts are close to each other in a shared representation space, while non-matching pairs are farther apart. After encoding and normalization, image-text similarity can be estimated with cosine similarity or an equivalent dot product.

For retrieval:

- Text-to-image retrieval embeds the query text and ranks image embeddings by similarity.
- Image-to-text retrieval embeds the query image and ranks text embeddings by similarity.

### Contrastive Pretraining

For a batch of paired examples, each image is associated with one corresponding text. The model computes a similarity matrix between all image embeddings and all text embeddings in the batch. The matching pairs form the positive pairs on the diagonal, while the other combinations act as in-batch negative pairs.

A symmetric contrastive objective is then used to:

- increase the similarity of the correct image-text pairs;
- decrease the similarity of incorrect image-text pairs;
- learn a shared space that supports cross-modal matching.

The original CLIP paper describes this approach as large-scale pretraining on image-text pairs collected from the web. That reported training setup is background from the paper and does not imply that this repository currently contains the same data, model weights, or usage permissions.

## Planned Experimental Workflow

### 1. Data Preparation — Planned

The project will first select an image-text dataset suitable for retrieval experiments.

The following details are **to be decided**:

- dataset or datasets;
- training, validation, and test split policy;
- language and caption format;
- image resolution and preprocessing;
- duplicate and near-duplicate handling;
- whether captions are treated as individual texts or grouped by image;
- data access procedure and license status.

The prepared data should preserve stable identifiers for images and texts so that retrieval predictions can be evaluated reproducibly.

### 2. Embedding and Index Construction — Planned

For the selected pretrained checkpoint, the planned pipeline will:

1. encode all database images;
2. encode all database texts;
3. normalize the resulting vectors;
4. store embeddings together with item identifiers and metadata;
5. build searchable indexes for both modalities.

The choice between exact similarity search and an approximate nearest-neighbor index is **to be decided**. The first implementation may use the simplest reliable option before optimizing search speed.

The final project should record the checkpoint name, preprocessing configuration, embedding dimension, normalization procedure, similarity function, and index settings.

### 3. Retrieval Evaluation — Planned

The system will be evaluated in both directions:

- text-to-image retrieval;
- image-to-text retrieval.

The primary metrics are **to be decided**. Candidate metrics include:

- Recall@K;
- median rank;
- mean rank;
- mean reciprocal rank.

The evaluation protocol should specify candidate-pool construction, whether multiple ground-truth matches are allowed, tie handling, and the values of `K`.

No experimental results are available yet.

## Winoground Evaluation

Winoground is planned as an additional evaluation set for compositional vision-language understanding.

Unlike ordinary retrieval benchmarks, Winoground contains paired images and captions designed to test whether a model is sensitive to compositional structure, including changes in word order and relations. Its examples require matching two images with two captions whose words are the same or closely related but arranged differently.

The project will use Winoground as:

- a probing and stress-test evaluation;
- a way to examine compositional failures that may not be visible in aggregate retrieval metrics;
- a possible source of qualitative error analysis.

Winoground is **not planned as a training set by default**. It should remain evaluation-only unless a future experiment explicitly defines and documents a different use.

The Winoground paper reports text, image, and group scores. Whether these metrics will be reproduced exactly in this project is **to be decided**, together with the evaluation implementation and any required dataset access procedure.

## Optional Future Improvements

Possible follow-up directions include:

- prompt ensembling or prompt-template analysis;
- hard-negative mining for retrieval;
- reranking retrieved candidates with a second model;
- lightweight fine-tuning or adapter-based adaptation;
- comparison of multiple pretrained checkpoints;
- analysis by caption length, relation type, or compositional category;
- improved indexing and retrieval efficiency;
- multilingual or domain-specific retrieval.

These are optional future directions, not commitments for the first implementation.

## Proposed Project Structure

The following structure is a proposal only. These directories and files do not currently exist and must not be interpreted as completed implementation.

```text
.
├── README.md
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── configs/
│   └── retrieval.yaml
├── src/
│   ├── data/
│   ├── models/
│   ├── indexing/
│   ├── retrieval/
│   └── evaluation/
├── scripts/
├── notebooks/
├── tests/
├── results/
│   └── README.md
└── docs/
    ├── methodology.md
    └── reproduction.md
```

The exact layout is **to be decided** after the implementation plan is finalized.

## Reproducibility Information

A reproducible version of the project should document the following:

- Python and framework versions;
- model family, checkpoint name, and model version;
- tokenizer and image preprocessing configuration;
- dataset name, version, split, and access procedure;
- data license and usage constraints, once verified;
- embedding normalization and similarity calculation;
- index type and index parameters;
- random seeds;
- hardware and runtime settings;
- evaluation protocol and metric definitions;
- configuration files or recorded parameter values;
- code revision used for each experiment.

These details are not available yet. No installation instructions or run commands are included because the implementation and dependency choices have not been finalized.

## Next Steps

1. Select the retrieval dataset and document its access and licensing conditions.
2. Select the initial pretrained model checkpoint.
3. Define the data schema, splits, preprocessing, and evaluation metrics.
4. Implement the two retrieval directions.
5. Build and validate the embedding indexes.
6. Add retrieval evaluation and reproducible result recording.
7. Add Winoground as an evaluation-only compositionality test.
8. Document verified commands, dependencies, results, and limitations.

## References

1. Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever.  
   **Learning Transferable Visual Models From Natural Language Supervision.**  
   [Local paper PDF](./Learning%20Transferable%20Visual%20Models%20From%20Natural%20Language%20Supervision.pdf) · [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)

2. Tristan Thrush, Ryan Jiang, Max Bartolo, Amanpreet Singh, Adina Williams, Douwe Kiela, and Candace Ross.  
   **Winoground: Probing Vision and Language Models for Visio-Linguistic Compositionality.**  
   [Local paper PDF](./Winoground.pdf) · [arXiv:2204.03162](https://arxiv.org/abs/2204.03162)
