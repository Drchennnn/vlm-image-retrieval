# Natural-Language Image Retrieval with Pretrained Vision-Language Models

**Project Proposal | CS5489 Machine Learning**  
**Benyi Chen, Muyao Yan, and Jianheng Qiao**  
20 October 2026

## Abstract

This project will develop and evaluate a bidirectional image-text retrieval system using a pretrained Contrastive Language-Image Pretraining (CLIP) model. It will retrieve images from natural-language descriptions and retrieve existing descriptions from images, without training a foundation model from scratch. The study will establish a reproducible frozen-model baseline, compare raw captions with a fixed textual prefix, and examine compositional matching through Winoground. Flickr30k is the preferred retrieval dataset, subject to authorized access and split verification. The contribution will be a carefully validated retrieval pipeline and an evidence-based analysis of its capabilities and limitations. Lightweight adaptation will remain an optional extension rather than a prerequisite for success.

## 1. Background and Motivation

Natural-language image retrieval connects visual content with users' descriptions without requiring a fixed vocabulary of class labels. CLIP provides a suitable foundation by learning image and text representations in a shared embedding space [1]. Its pretrained encoders allow cross-modal similarity to be computed directly, making retrieval possible without task-specific weight updates.

However, successful semantic retrieval does not necessarily establish sensitivity to relationships or word order. Winoground probes this limitation using paired images and captions whose words are rearranged to express different meanings [2]. Evaluating both conventional retrieval and controlled compositional matching will therefore provide a more informative assessment than reporting retrieval accuracy alone. The project will also explain the contrastive learning mechanism behind CLIP and distinguish published findings from results obtained in our own experiments.

## 2. Objectives and Research Questions

The project has three research questions:

1. How effectively does a frozen CLIP model perform text-to-image and image-to-text retrieval under a clearly specified evaluation protocol?
2. How does adding the fixed prefix "a photo of " to a caption affect retrieval performance relative to the unchanged caption?
3. What limitations become visible when the same model is evaluated on compositional image-caption matching?

The prefix comparison is motivated by CLIP's original retrieval experiments, rather than presented as a novel method [1, Appendix E.1]. The intended contribution is a reproducible implementation and controlled empirical analysis. A production service, large-scale approximate indexing, and foundation-model pretraining are outside the scope.

## 3. Data and Access Strategy

Flickr30k will be the preferred retrieval benchmark because it provides multiple human-written descriptions per image [3]. We will use published retrieval split annotations associated with Karpathy and Fei-Fei [4], targeting the established 1,000-image test protocol. Before implementation proceeds to full evaluation, we will verify image identifiers, caption associations, split membership, missing files, and the exact number of usable records. All descriptions belonging to an image will remain in the same split.

Access is a project dependency, not an assumed completed step. Flickr30k images require authorized access and are subject to noncommercial research and educational restrictions. If access cannot be secured during the first project week, we will use officially distributed COCO val2017 data [5]. This fallback will use a documented custom split of 1,000 development images and 4,000 evaluation images, with a fixed seed and saved image-ID manifests. It will not be described as the Karpathy test split, and its scores will not be directly compared with results using that protocol.

Winoground will be reserved for evaluation after the experimental configurations are frozen. Its 400 groups each contain two images and two captions [2]. Access requires acceptance of the dataset's terms. If access remains unavailable, the final report will explicitly identify the missing diagnostic rather than substitute synthetic examples for benchmark results. Dataset images will not be redistributed without permission.

## 4. Methodology

The baseline will use the pretrained OpenAI CLIP ViT-B/32 checkpoint with its corresponding image preprocessing and tokenizer [6]. Both encoders will remain frozen. Each image and caption will be encoded, and the resulting vectors will be L2-normalized. Their dot product will then equal cosine similarity. Exact ranking over the candidate collection will support both retrieval directions. Cached embeddings and batched scoring will limit repeated computation.

To explain contrastive pretraining, we will consider a batch of N matched image-caption pairs. CLIP constructs an N-by-N matrix of temperature-scaled similarities. Cross-entropy is computed once with each image selecting its matching caption and once with each caption selecting its matching image; the two losses are averaged. Matching pairs form the diagonal targets, while other batch pairings provide negative candidates [1]. This explanation will distinguish the original training objective from our frozen-model inference pipeline and discuss the potential for semantically related pairs to become false negatives.

Two configurations will be compared: unchanged captions and captions preceded by the exact string "a photo of ". All other factors, including checkpoint, preprocessing, candidate set, and evaluation code, will be held constant. Both configurations will be reported. Truncation rates will be recorded because adding a prefix can affect the available context length.

## 5. Evaluation and Reliability

The primary metrics will be Recall@1, Recall@5, and Recall@10, reported separately for each direction. For text-to-image retrieval, each caption will query all images in its evaluation split. For image-to-text retrieval, each image will query all captions in that split, and any of its annotated captions will count as a correct result. Recall@K will measure the percentage of queries with at least one correct candidate among the first K results. Candidate counts, query counts, and deterministic tie handling will accompany each result.

Winoground evaluation will report text, image, and group scores [2]. Text success requires both images to prefer their correct captions; image success requires both captions to prefer their correct images; group success requires both conditions. Comparisons are strict, so tied scores fail. The diagnostic will not guide prompt selection, hyperparameter tuning, or early stopping.

Evaluation code will first be tested with small examples whose answers can be calculated manually. Tests will cover multiple correct captions, ranking direction, ties, shuffled identifiers, split overlap, and Winoground inequalities. Full and batched scoring will be checked for agreement. Development data will support implementation and configuration decisions; final evaluation data will be used only after configuration freeze. Any subsequent change motivated by test results will be labelled exploratory.

Paired bootstrap intervals will quantify uncertainty in the difference between the two configurations. Retrieval queries will be resampled by image identifier, keeping associated captions together and the candidate gallery fixed. Winoground will be resampled by group. Error analysis will examine representative successes and failures, including relational mismatches and plausible but unannotated matches. Examples will be selected using a documented sampling rule rather than only showcasing favourable outputs.

## 6. Implementation and Work Plan

The implementation will use a compact Python package separating data validation, model encoding, feature caching, ranking, metrics, and command-line interfaces. Every experiment will record its code revision, dependency versions, dataset and split hashes, model identifier, configuration, device, timing, and machine-readable outputs. Specific group responsibilities will be agreed at execution time; each completed task will require a separate review of its evidence.

A six-week schedule is proposed. Week 1 will establish data access, validate splits, and measure feasibility on a small development subset. Week 2 will implement ranking and verify the evaluators. Week 3 will complete the full development-set baseline and retrieval demonstration. Week 4 will compare the two caption configurations and either complete optional adaptation and development-set selection or defer it before freezing the experiment design. Week 5 will conduct held-out evaluation, compositional diagnostics, and error analysis. Week 6 will complete the report, independent reproduction checks, and submission package.

Optional lightweight adaptation will proceed only if the baseline is reliable, independent training data are available, and measured resources support a bounded experiment. The method, trainable parameters, configuration count, and compute ceiling will be recorded before training. Model selection will use development data only. If these conditions cannot be met, adaptation will be omitted without weakening the core study.

## 7. Expected Outcomes and Limitations

The expected deliverables are a reproducible bidirectional retrieval prototype, verified evaluation tools, a controlled caption-prefix comparison, and a report explaining performance and limitations. Success means that results are correct, traceable, and reproducible; no improvement or target score is assumed in advance.

Important limitations include dataset access, finite evaluation size, incomplete relevance annotations, domain bias, and unknown overlap with CLIP's pretraining corpus. Here, zero-shot means that model weights are not updated on the retrieval task; it does not prove that evaluation images were absent from pretraining. Compute feasibility and all performance claims will be established empirically. Negative or inconclusive findings will be retained and discussed.

## References

[1] Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. *Proceedings of ICML*, 139, 8748-8763. https://arxiv.org/abs/2103.00020

[2] Thrush, T., et al. (2022). Winoground: Probing Vision and Language Models for Visio-Linguistic Compositionality. *Proceedings of CVPR*. https://arxiv.org/abs/2204.03162

[3] Young, P., Lai, A., Hodosh, M., and Hockenmaier, J. (2014). From Image Descriptions to Visual Denotations: New Similarity Metrics for Semantic Inference over Event Descriptions. *Transactions of the ACL*, 2, 67-78. Dataset: https://shannon.cs.illinois.edu/DenotationGraph/

[4] Karpathy, A., and Fei-Fei, L. (2015). Deep Visual-Semantic Alignments for Generating Image Descriptions. *Proceedings of CVPR*. Data and splits: https://cs.stanford.edu/people/karpathy/deepimagesent/

[5] Lin, T.-Y., et al. (2014). Microsoft COCO: Common Objects in Context. *Proceedings of ECCV*. Dataset and terms: https://cocodataset.org/

[6] OpenAI. CLIP: Official Implementation and Pretrained Models. https://github.com/openai/CLIP (accessed 20 October 2026).
