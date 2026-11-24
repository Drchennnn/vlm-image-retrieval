"""Round 4 (T5): Phase 2 NegCLIP-lite Compositional Hard-Negative Lightweight Adaptation.

Strictly obeys anti-leakage rules:
  - Trains ONLY on the 3,000-image COCO train split (1 unique positive caption per image per step).
  - Evaluates and selects hyperparameters/checkpoints ONLY on the 1,000-image dev split.
  - Zero access to held-out test split or Winoground.
  - Bounded budget: 3 configs x 3 epochs, frozen vision backbone (cached image embeddings).
"""

from __future__ import annotations

import json
import math
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from vlm_retrieval.adapt import (
    NegCLIPLiteConfig,
    ResidualImageAdapter,
    configure_trainable_clip_text,
    encode_images_with_adapter,
    encode_texts_with_model,
    evaluate_compositional_hard_neg_acc,
    generate_hard_negative,
    negclip_loss,
)
from vlm_retrieval.cache import load_embedding_bundle
from vlm_retrieval.data import SplitData, load_split, validate_splits_disjoint
from vlm_retrieval.evaluate import evaluate_bidirectional_scores, write_run_bundle
from vlm_retrieval.model import VLMWrapper, apply_text_policy, l2_normalize_checked, load_model_specs
from vlm_retrieval.rerank import csls_similarity, dual_softmax_similarity
from vlm_retrieval.retrieval import compute_similarity_matrix

SEED = 5489


def _load_or_cache_train_images(
    root: Path,
    train_split: SplitData,
    spec,
    cache_path: Path,
) -> Tuple[np.ndarray, float]:
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as d:
            if str(d["manifest_sha256"].item()) == train_split.manifest_sha256:
                return np.asarray(d["image_embeddings"], dtype=np.float32), 0.0

    t0 = time.perf_counter()
    wrapper = VLMWrapper(spec)
    paths = [root / "data" / "raw" / "coco" / rec.relative_path for rec in train_split.images]
    img_embs = wrapper.encode_images(paths, batch_size=64)
    elapsed = time.perf_counter() - t0

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        manifest_sha256=np.array(train_split.manifest_sha256),
        image_ids=np.array(train_split.image_ids, dtype=str),
        image_embeddings=img_embs,
    )
    return img_embs, elapsed


def _load_dev_bundle(cache_dir: Path, model_key: str, policy: str, manifest_sha256: str):
    prefix = f"{model_key}__dev__{policy}__"
    for mf in sorted(cache_dir.glob(f"{prefix}*.json")):
        meta = json.loads(mf.read_text(encoding="utf-8"))
        if meta.get("manifest_sha256") == manifest_sha256:
            return load_embedding_bundle(cache_dir, meta["cache_key"], expected_manifest_sha256=manifest_sha256)
    raise FileNotFoundError(f"Missing dev cache for {model_key}/{policy}")


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _train_single_config(
    spec,
    cfg: NegCLIPLiteConfig,
    train_split: SplitData,
    train_img_embs: np.ndarray,
    dev_split: SplitData,
    dev_img_embs: np.ndarray,
    dev_pos_texts: List[str],
    dev_neg_texts: List[str],
    dev_cap_to_img_row: List[int],
    frozen_csls_k: int,
    frozen_ds_params: dict,
) -> Tuple[dict, dict]:
    """Train one NegCLIP-lite config and return (history_summary, best_epoch_state_dict_payload)."""
    _set_seeds(cfg.seed)
    rng = np.random.Generator(np.random.PCG64(cfg.seed))

    wrapper = VLMWrapper(spec)
    device = wrapper.device
    model = wrapper.model
    tokenizer = wrapper.tokenizer

    trainable_text = configure_trainable_clip_text(model, unfreeze_text_blocks=cfg.unfreeze_text_blocks)
    adapter = ResidualImageAdapter(dim=train_img_embs.shape[1]).to(device) if cfg.use_img_adapter else None

    params = list(trainable_text) + (list(adapter.parameters()) if adapter is not None else [])
    n_trainable = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in model.parameters()) + (
        sum(p.numel() for p in adapter.parameters()) if adapter is not None else 0
    )

    # Group captions by image row index so every mini-batch samples 1 unique caption per image (no false negatives)
    img_id_to_row = {rec.image_id: i for i, rec in enumerate(train_split.images)}
    caps_by_img_row: Dict[int, List[str]] = defaultdict(list)
    for cap in train_split.captions:
        caps_by_img_row[img_id_to_row[cap.image_id]].append(cap.text)

    n_train = len(train_split.images)
    steps_per_epoch = math.ceil(n_train / cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = max(1, int(0.1 * total_steps))

    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    train_img_tensor = torch.from_numpy(train_img_embs).to(device)
    logit_scale = model.logit_scale.exp().clamp(max=100.0).detach()

    op_counts: Counter[str] = Counter()
    epochs_log: List[dict] = []
    best_dev_mean_r1 = -1.0
    best_payload: dict | None = None

    for epoch in range(1, cfg.epochs + 1):
        t_ep = time.perf_counter()
        model.train()
        if adapter is not None:
            adapter.train()

        perm = rng.permutation(n_train)
        ep_losses: List[float] = []
        ep_comp_accs: List[float] = []

        for start in range(0, n_train, cfg.batch_size):
            batch_rows = perm[start : start + cfg.batch_size]
            if len(batch_rows) < 4:
                continue

            pos_batch: List[str] = []
            neg_batch: List[str] = []
            for r in batch_rows.tolist():
                cands = caps_by_img_row[r]
                pos_txt = cands[int(rng.integers(0, len(cands)))]
                neg_txt, op = generate_hard_negative(pos_txt, rng)
                op_counts[op] += 1
                pos_batch.append(apply_text_policy(pos_txt, cfg.policy))
                neg_batch.append(apply_text_policy(neg_txt, cfg.policy))

            img_b = train_img_tensor[torch.from_numpy(batch_rows).to(device)]
            if adapter is not None:
                img_b = adapter(img_b)

            pos_tok = tokenizer(pos_batch).to(device)
            neg_tok = tokenizer(neg_batch).to(device)

            pos_feat = F.normalize(model.encode_text(pos_tok).float(), dim=-1)
            neg_feat = F.normalize(model.encode_text(neg_tok).float(), dim=-1)

            loss, info = negclip_loss(img_b, pos_feat, neg_feat, logit_scale, lambda_hn=cfg.lambda_hn)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            ep_losses.append(info["loss_total"])
            ep_comp_accs.append(info["batch_comp_acc"])

        train_secs = time.perf_counter() - t_ep

        # Evaluate on full 1,000-image dev split
        t_ev = time.perf_counter()
        dev_v = encode_images_with_adapter(dev_img_embs, adapter, device)
        dev_t_pos = encode_texts_with_model(model, tokenizer, dev_pos_texts, policy=cfg.policy, device=device)
        dev_t_neg = encode_texts_with_model(model, tokenizer, dev_neg_texts, policy=cfg.policy, device=device)

        sim_dev = compute_similarity_matrix(dev_v, dev_t_pos)
        res_cos = evaluate_bidirectional_scores(dev_split, sim_dev)
        sim_csls = csls_similarity(sim_dev, k=frozen_csls_k)
        res_csls = evaluate_bidirectional_scores(dev_split, sim_csls)
        s_ds_i2t, s_ds_t2i = dual_softmax_similarity(
            sim_dev,
            temperature=float(frozen_ds_params["temperature"]),
            mode=str(frozen_ds_params["mode"]),
        )
        res_ds = evaluate_bidirectional_scores(dev_split, s_ds_i2t, s_ds_t2i)
        dev_comp_acc = evaluate_compositional_hard_neg_acc(dev_v, dev_t_pos, dev_t_neg, dev_cap_to_img_row)
        eval_secs = time.perf_counter() - t_ev

        ep_record = {
            "epoch": epoch,
            "train_seconds": round(train_secs, 2),
            "eval_seconds": round(eval_secs, 2),
            "train_loss_mean": round(float(np.mean(ep_losses)), 4),
            "train_batch_comp_acc": round(float(100.0 * np.mean(ep_comp_accs)), 2),
            "dev_comp_acc": round(dev_comp_acc, 2),
            "dev_cosine": {
                "mean_R@1": res_cos["mean_R@1"],
                "t2i_R@1": round(res_cos["t2i"]["metrics"]["R@1"], 2),
                "i2t_R@1": round(res_cos["i2t"]["metrics"]["R@1"], 2),
                "t2i_R@5": round(res_cos["t2i"]["metrics"]["R@5"], 2),
                "i2t_R@5": round(res_cos["i2t"]["metrics"]["R@5"], 2),
            },
            "dev_csls": {
                "mean_R@1": res_csls["mean_R@1"],
                "t2i_R@1": round(res_csls["t2i"]["metrics"]["R@1"], 2),
                "i2t_R@1": round(res_csls["i2t"]["metrics"]["R@1"], 2),
            },
            "dev_dual_softmax": {
                "mean_R@1": res_ds["mean_R@1"],
                "t2i_R@1": round(res_ds["t2i"]["metrics"]["R@1"], 2),
                "i2t_R@1": round(res_ds["i2t"]["metrics"]["R@1"], 2),
            },
        }
        epochs_log.append(ep_record)
        print(
            f"  [{cfg.name}] ep={epoch}/{cfg.epochs} | train={train_secs:.1f}s | loss={ep_record['train_loss_mean']:.3f} "
            f"| dev_cos={res_cos['mean_R@1']:.2f}% (T2I {ep_record['dev_cosine']['t2i_R@1']:.2f} / I2T {ep_record['dev_cosine']['i2t_R@1']:.2f}) "
            f"| dev_DS={res_ds['mean_R@1']:.2f}% | dev_comp={dev_comp_acc:.2f}%"
        )

        if res_cos["mean_R@1"] > best_dev_mean_r1:
            best_dev_mean_r1 = res_cos["mean_R@1"]
            # Save only trainable weights (compact checkpoint ~25MB instead of 600MB)
            trainable_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
                if any(
                    k.startswith(prefix)
                    for prefix in (
                        *[f"transformer.resblocks.{len(model.transformer.resblocks) - i}" for i in range(1, cfg.unfreeze_text_blocks + 1)],
                        "ln_final",
                        "text_projection",
                    )
                )
            }
            best_payload = {
                "config": cfg.__dict__,
                "selected_epoch": epoch,
                "dev_metrics": ep_record,
                "text_trainable_state_dict": trainable_state,
                "img_adapter_state_dict": (
                    {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
                    if adapter is not None
                    else None
                ),
            }

    summary = {
        "config": cfg.__dict__,
        "trainable_params": int(n_trainable),
        "total_params": int(n_total),
        "trainable_pct": round(100.0 * n_trainable / n_total, 2),
        "device": str(device),
        "hard_neg_operator_counts": dict(op_counts),
        "epochs": epochs_log,
        "best_epoch": best_payload["selected_epoch"],
        "best_dev_cosine_mean_R@1": best_dev_mean_r1,
        "best_epoch_record": best_payload["dev_metrics"],
    }
    return summary, best_payload


def _plot_phase2_figure(fig_path: Path, baseline_rec: dict, config_summaries: List[dict]) -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 180,
    })
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    colors = ["#1f77b4", "#2ca02c", "#9467bd"]
    markers = ["o", "s", "^"]

    # Panel 1: Dev Bidirectional Mean R@1 across epochs
    axes[0].axhline(
        baseline_rec["dev_cosine"]["mean_R@1"],
        ls="--",
        color="#d62728",
        lw=1.5,
        label=f"Frozen CLIP ViT-B/32 ({baseline_rec['dev_cosine']['mean_R@1']:.2f}%)",
    )
    for idx, cs in enumerate(config_summaries):
        eps = [0] + [e["epoch"] for e in cs["epochs"]]
        vals = [baseline_rec["dev_cosine"]["mean_R@1"]] + [e["dev_cosine"]["mean_R@1"] for e in cs["epochs"]]
        axes[0].plot(eps, vals, marker=markers[idx], color=colors[idx], lw=1.8, label=cs["config"]["name"])
    axes[0].set_xticks([0, 1, 2, 3])
    axes[0].set_xlabel("Epoch (0 = Frozen Pretrained)")
    axes[0].set_ylabel("Dev Cosine Mean R@1 (%)")
    axes[0].set_title("NegCLIP-lite Adaptation: Retrieval on Dev (1K)")
    axes[0].legend(frameon=False, fontsize=8.5)

    # Panel 2: Dev Compositional Hard-Negative Pairwise Accuracy across epochs
    axes[1].axhline(
        baseline_rec["dev_comp_acc"],
        ls="--",
        color="#d62728",
        lw=1.5,
        label=f"Frozen CLIP ViT-B/32 ({baseline_rec['dev_comp_acc']:.2f}%)",
    )
    for idx, cs in enumerate(config_summaries):
        eps = [0] + [e["epoch"] for e in cs["epochs"]]
        vals = [baseline_rec["dev_comp_acc"]] + [e["dev_comp_acc"] for e in cs["epochs"]]
        axes[1].plot(eps, vals, marker=markers[idx], color=colors[idx], lw=1.8, label=cs["config"]["name"])
    axes[1].set_xticks([0, 1, 2, 3])
    axes[1].set_xlabel("Epoch (0 = Frozen Pretrained)")
    axes[1].set_ylabel("Dev Hard-Negative Pairwise Acc (%)")
    axes[1].set_title("Compositional Hard-Negative Sensitivity on Dev")
    axes[1].legend(frameon=False, fontsize=8.5)

    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_dir = root / "data" / "manifests" / "coco2017"
    cache_dir = root / "artifacts" / "cache"
    ckpt_dir = root / "artifacts" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_split = load_split(manifest_dir, "train")
    dev_split = load_split(manifest_dir, "dev")
    validate_splits_disjoint([train_split, dev_split])

    frozen_p1 = json.loads((root / "configs" / "frozen_phase1.json").read_text(encoding="utf-8"))
    clip_p1 = frozen_p1["models"]["clip_vit_b32"]
    policy = clip_p1["selected_policy"]
    frozen_csls_k = int(clip_p1["best_csls_k"])
    frozen_ds_params = clip_p1["best_dual_softmax_params"]

    specs = load_model_specs(root / "configs" / "models.yaml")
    spec = specs["clip_vit_b32"]

    print("[Round 4] Loading/caching frozen CLIP ViT-B/32 image embeddings for 3,000 train images...")
    train_img_embs, train_img_secs = _load_or_cache_train_images(
        root, train_split, spec, cache_dir / "train_img_clip_vit_b32.npz"
    )
    print(f"[Round 4] Train image embeddings ready: shape={train_img_embs.shape} (encode_secs={train_img_secs:.1f}s)")

    dev_bundle = _load_dev_bundle(cache_dir, "clip_vit_b32", policy, dev_split.manifest_sha256)
    dev_img_embs = dev_bundle.image_embeddings
    dev_pos_texts = [c.text for c in dev_split.captions]

    # Deterministic dev compositional hard negatives (diagnostic only, never trained on)
    dev_rng = np.random.Generator(np.random.PCG64(SEED + 100))
    dev_neg_pairs = [generate_hard_negative(t, dev_rng) for t in dev_pos_texts]
    dev_neg_texts = [p[0] for p in dev_neg_pairs]
    dev_neg_ops = dict(Counter(p[1] for p in dev_neg_pairs))
    dev_img_row = {rec.image_id: i for i, rec in enumerate(dev_split.images)}
    dev_cap_to_img_row = [dev_img_row[c.image_id] for c in dev_split.captions]

    # Evaluate frozen baseline compositional hard-negative accuracy on dev
    base_wrapper = VLMWrapper(spec)
    dev_neg_embs_base, _ = base_wrapper.encode_texts(dev_neg_texts, policy=policy, batch_size=128)
    sim_base = compute_similarity_matrix(dev_img_embs, dev_bundle.caption_embeddings)
    res_base_cos = evaluate_bidirectional_scores(dev_split, sim_base)
    res_base_csls = evaluate_bidirectional_scores(dev_split, csls_similarity(sim_base, k=frozen_csls_k))
    s_b_ds_i2t, s_b_ds_t2i = dual_softmax_similarity(
        sim_base, temperature=float(frozen_ds_params["temperature"]), mode=str(frozen_ds_params["mode"])
    )
    res_base_ds = evaluate_bidirectional_scores(dev_split, s_b_ds_i2t, s_b_ds_t2i)
    base_comp_acc = evaluate_compositional_hard_neg_acc(
        dev_img_embs, dev_bundle.caption_embeddings, dev_neg_embs_base, dev_cap_to_img_row
    )
    baseline_rec = {
        "policy": policy,
        "dev_comp_acc": round(base_comp_acc, 2),
        "dev_cosine": {
            "mean_R@1": res_base_cos["mean_R@1"],
            "t2i_R@1": round(res_base_cos["t2i"]["metrics"]["R@1"], 2),
            "i2t_R@1": round(res_base_cos["i2t"]["metrics"]["R@1"], 2),
            "t2i_R@5": round(res_base_cos["t2i"]["metrics"]["R@5"], 2),
            "i2t_R@5": round(res_base_cos["i2t"]["metrics"]["R@5"], 2),
        },
        "dev_csls": {
            "mean_R@1": res_base_csls["mean_R@1"],
            "t2i_R@1": round(res_base_csls["t2i"]["metrics"]["R@1"], 2),
            "i2t_R@1": round(res_base_csls["i2t"]["metrics"]["R@1"], 2),
        },
        "dev_dual_softmax": {
            "mean_R@1": res_base_ds["mean_R@1"],
            "t2i_R@1": round(res_base_ds["t2i"]["metrics"]["R@1"], 2),
            "i2t_R@1": round(res_base_ds["i2t"]["metrics"]["R@1"], 2),
        },
    }
    print(
        f"[Round 4] Frozen Baseline | dev_cos={baseline_rec['dev_cosine']['mean_R@1']:.2f}% "
        f"(T2I {baseline_rec['dev_cosine']['t2i_R@1']:.2f} / I2T {baseline_rec['dev_cosine']['i2t_R@1']:.2f}) "
        f"| dev_DS={baseline_rec['dev_dual_softmax']['mean_R@1']:.2f}% | dev_comp={base_comp_acc:.2f}%"
    )
    del base_wrapper

    configs = [
        NegCLIPLiteConfig(
            name="cfg1_text_adapter_lr1e5_hn0.5",
            lr=1e-5,
            lambda_hn=0.5,
            epochs=3,
            batch_size=64,
            unfreeze_text_blocks=2,
            use_img_adapter=True,
            policy=policy,
            seed=SEED,
        ),
        NegCLIPLiteConfig(
            name="cfg2_text_adapter_lr3e5_hn1.0",
            lr=3e-5,
            lambda_hn=1.0,
            epochs=3,
            batch_size=64,
            unfreeze_text_blocks=2,
            use_img_adapter=True,
            policy=policy,
            seed=SEED,
        ),
        NegCLIPLiteConfig(
            name="cfg3_text_only_lr2e5_hn1.0",
            lr=2e-5,
            lambda_hn=1.0,
            epochs=3,
            batch_size=64,
            unfreeze_text_blocks=2,
            use_img_adapter=False,
            policy=policy,
            seed=SEED,
        ),
    ]

    t_all = time.perf_counter()
    summaries: List[dict] = []
    best_overall_payload: dict | None = None
    best_overall_score = -1.0

    for cfg in configs:
        print(f"\n[Round 4] Training {cfg.name} ...")
        cfg_summary, payload = _train_single_config(
            spec,
            cfg,
            train_split,
            train_img_embs,
            dev_split,
            dev_img_embs,
            dev_pos_texts,
            dev_neg_texts,
            dev_cap_to_img_row,
            frozen_csls_k,
            frozen_ds_params,
        )
        summaries.append(cfg_summary)
        if cfg_summary["best_dev_cosine_mean_R@1"] > best_overall_score:
            best_overall_score = cfg_summary["best_dev_cosine_mean_R@1"]
            best_overall_payload = payload

    total_train_wall = time.perf_counter() - t_all
    ckpt_path = ckpt_dir / "negclip_lite_best.pt"
    torch.save(best_overall_payload, ckpt_path)

    fig_path = root / "reports" / "figures" / "dev_phase2_negclip_lite.png"
    _plot_phase2_figure(fig_path, baseline_rec, summaries)

    phase2_summary = {
        "base_model": "clip_vit_b32",
        "train_split": "train",
        "train_manifest_sha256": train_split.manifest_sha256,
        "dev_split": "dev",
        "dev_manifest_sha256": dev_split.manifest_sha256,
        "train_image_encode_seconds": round(train_img_secs, 2),
        "total_training_wall_seconds": round(total_train_wall, 2),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024), 1),
        "dev_hard_neg_operator_counts": dev_neg_ops,
        "frozen_baseline": baseline_rec,
        "configs": summaries,
        "selected": {
            "config_name": best_overall_payload["config"]["name"],
            "selected_epoch": best_overall_payload["selected_epoch"],
            "checkpoint_path": str(ckpt_path.relative_to(root)),
            "dev_metrics": best_overall_payload["dev_metrics"],
            "delta_dev_cosine_mean_R@1": round(
                best_overall_payload["dev_metrics"]["dev_cosine"]["mean_R@1"]
                - baseline_rec["dev_cosine"]["mean_R@1"],
                2,
            ),
            "delta_dev_comp_acc": round(
                best_overall_payload["dev_metrics"]["dev_comp_acc"] - baseline_rec["dev_comp_acc"],
                2,
            ),
        },
        "figure": str(fig_path.relative_to(root)),
    }

    out_summary = root / "reports" / "dev_phase2_summary.json"
    out_summary.write_text(json.dumps(phase2_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Freeze complete Phase 1 + Phase 2 configuration before one-shot held-out test & Winoground!
    frozen_all = {
        **frozen_p1,
        "phase2_negclip_lite": {
            "status": "enabled",
            "base_model": "clip_vit_b32",
            "selected_config": best_overall_payload["config"],
            "selected_epoch": best_overall_payload["selected_epoch"],
            "checkpoint_path": str(ckpt_path.relative_to(root)),
            "train_manifest_sha256": train_split.manifest_sha256,
            "dev_selected_metrics": best_overall_payload["dev_metrics"],
        },
    }
    frozen_all_path = root / "configs" / "frozen_all.json"
    frozen_all_path.write_text(json.dumps(frozen_all, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n=== ROUND 4 PHASE 2 SUMMARY ===")
    sel = phase2_summary["selected"]
    print(
        f"Selected: {sel['config_name']} (epoch {sel['selected_epoch']}) | "
        f"dev_cos={sel['dev_metrics']['dev_cosine']['mean_R@1']:.2f}% ({sel['delta_dev_cosine_mean_R@1']:+.2f}%) | "
        f"dev_DS={sel['dev_metrics']['dev_dual_softmax']['mean_R@1']:.2f}% | "
        f"dev_comp={sel['dev_metrics']['dev_comp_acc']:.2f}% ({sel['delta_dev_comp_acc']:+.2f}%) | "
        f"total_wall={total_train_wall:.1f}s"
    )
    print(f"Saved -> {out_summary} & {frozen_all_path}")


if __name__ == "__main__":
    main()
