"""Command-line interface for encoding, evaluating, and querying cached VLM embeddings."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

from .cache import EmbeddingBundle, load_embedding_bundle, save_embedding_bundle
from .data import load_split
from .evaluate import evaluate_bidirectional_scores, write_run_bundle
from .model import VLMWrapper, load_model_specs
from .rerank import csls_similarity, dual_softmax_similarity
from .retrieval import compute_similarity_matrix, rank_similarity_matrix


def _find_cached_bundle(
    cache_dir: Path,
    model_key: str,
    split_name: str,
    policy: str,
    manifest_sha256: str,
) -> EmbeddingBundle | None:
    if not cache_dir.exists():
        return None
    prefix = f"{model_key}__{split_name}__{policy}__"
    for meta_file in sorted(cache_dir.glob(f"{prefix}*.json")):
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if meta.get("manifest_sha256") == manifest_sha256:
            return load_embedding_bundle(
                cache_dir,
                meta["cache_key"],
                expected_manifest_sha256=manifest_sha256,
            )
    return None


def cmd_encode(args: argparse.Namespace) -> None:
    root = Path(args.repo_root).resolve()
    specs = load_model_specs(root / args.models_config)
    if args.model not in specs:
        raise ValueError(f"Unknown model '{args.model}'; available: {list(specs)}")
    spec = specs[args.model]
    split = load_split(root / args.manifest_dir, args.split)
    cache_dir = root / args.cache_dir

    if not args.force:
        cached = _find_cached_bundle(cache_dir, spec.key, split.name, args.policy, split.manifest_sha256)
        if cached is not None:
            print(json.dumps({"status": "cache_hit", "cache_key": cached.cache_key}, indent=2))
            return

    t0 = time.perf_counter()
    wrapper = VLMWrapper(spec, device=args.device)
    image_paths = [root / args.coco_root / rec.relative_path for rec in split.images]
    caption_texts = [rec.text for rec in split.captions]

    img_embs = wrapper.encode_images(image_paths, batch_size=args.image_batch_size)
    cap_embs, trunc_stats = wrapper.encode_texts(
        caption_texts, policy=args.policy, batch_size=args.text_batch_size
    )
    elapsed = time.perf_counter() - t0

    meta = {
        **wrapper.metadata,
        "text_policy": args.policy,
        "split_name": split.name,
        "manifest_sha256": split.manifest_sha256,
        "truncation": trunc_stats,
        "encode_seconds": round(elapsed, 3),
    }
    bundle = save_embedding_bundle(
        cache_dir,
        image_ids=split.image_ids,
        image_embeddings=img_embs,
        caption_ids=split.caption_ids,
        caption_embeddings=cap_embs,
        metadata=meta,
    )
    print(
        json.dumps(
            {
                "status": "encoded",
                "cache_key": bundle.cache_key,
                "encode_seconds": round(elapsed, 3),
                "truncation_rate": trunc_stats["truncation_rate"],
            },
            indent=2,
        )
    )


def cmd_evaluate(args: argparse.Namespace) -> None:
    root = Path(args.repo_root).resolve()
    split = load_split(root / args.manifest_dir, args.split)
    bundle = _find_cached_bundle(
        root / args.cache_dir, args.model, split.name, args.policy, split.manifest_sha256
    )
    if bundle is None:
        raise FileNotFoundError(
            f"No matching cache for model={args.model}, split={split.name}, policy={args.policy}. Run 'encode' first."
        )

    t0 = time.perf_counter()
    sim_i2t = compute_similarity_matrix(bundle.image_embeddings, bundle.caption_embeddings)
    sim_t2i = sim_i2t.T

    if args.rerank == "csls":
        sim_i2t = csls_similarity(sim_i2t, k=args.csls_k)
        sim_t2i = sim_i2t.T
    elif args.rerank == "dual_softmax":
        sim_i2t, sim_t2i = dual_softmax_similarity(
            sim_i2t, temperature=args.ds_temp, mode=args.ds_mode
        )

    res = evaluate_bidirectional_scores(split, sim_i2t, sim_t2i)
    elapsed = time.perf_counter() - t0

    run_id = args.run_id or f"{args.model}__{split.name}__{args.policy}__{args.rerank}"
    bundle_dir = write_run_bundle(
        root / args.out_dir,
        run_id=run_id,
        config={
            "model_key": args.model,
            "split": split.name,
            "policy": args.policy,
            "rerank": args.rerank,
            "csls_k": args.csls_k if args.rerank == "csls" else None,
            "ds_temp": args.ds_temp if args.rerank == "dual_softmax" else None,
            "ds_mode": args.ds_mode if args.rerank == "dual_softmax" else None,
            "cache_key": bundle.cache_key,
        },
        eval_result=res,
        provenance_extra={
            "cache_metadata": bundle.metadata,
            "eval_seconds": round(elapsed, 4),
        },
        log_lines=[f"Evaluated {run_id} in {elapsed:.4f}s; mean_R@1={res['mean_R@1']:.2f}%"],
        repo_root=root,
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "bundle_dir": str(bundle_dir),
                "mean_R@1": res["mean_R@1"],
                "t2i": res["t2i"]["metrics"],
                "i2t": res["i2t"]["metrics"],
            },
            indent=2,
        )
    )


def cmd_search(args: argparse.Namespace) -> None:
    root = Path(args.repo_root).resolve()
    split = load_split(root / args.manifest_dir, args.split)
    bundle = _find_cached_bundle(
        root / args.cache_dir, args.model, split.name, args.policy, split.manifest_sha256
    )
    if bundle is None:
        raise FileNotFoundError(
            f"No cached embeddings found for model={args.model}, split={split.name}, policy={args.policy}"
        )

    img_by_id = {r.image_id: r for r in split.images}
    cap_by_id = {r.caption_id: r for r in split.captions}

    if args.text is not None:
        specs = load_model_specs(root / args.models_config)
        wrapper = VLMWrapper(specs[args.model], device=args.device)
        q_emb, _ = wrapper.encode_texts([args.text], policy=args.policy, batch_size=1)
        sim = compute_similarity_matrix(q_emb, bundle.image_embeddings)
        idx, scores = rank_similarity_matrix(sim, bundle.image_ids, top_k=args.top_k)
        hits = []
        for rank_i, (col, sc) in enumerate(zip(idx[0].tolist(), scores[0].tolist()), start=1):
            iid = bundle.image_ids[col]
            rec = img_by_id[iid]
            hits.append(
                {
                    "rank": rank_i,
                    "image_id": iid,
                    "score": round(float(sc), 6),
                    "relative_path": rec.relative_path,
                }
            )
        print(json.dumps({"direction": "t2i", "query_text": args.text, "results": hits}, indent=2))
        return

    if args.image_id is not None:
        if args.image_id not in img_by_id:
            raise ValueError(f"image_id '{args.image_id}' not in split '{split.name}'")
        row_idx = bundle.image_ids.index(args.image_id)
        q_emb = bundle.image_embeddings[row_idx : row_idx + 1]
        sim = compute_similarity_matrix(q_emb, bundle.caption_embeddings)
        idx, scores = rank_similarity_matrix(sim, bundle.caption_ids, top_k=args.top_k)
        hits = []
        for rank_i, (col, sc) in enumerate(zip(idx[0].tolist(), scores[0].tolist()), start=1):
            cid = bundle.caption_ids[col]
            rec = cap_by_id[cid]
            hits.append(
                {
                    "rank": rank_i,
                    "caption_id": cid,
                    "image_id": rec.image_id,
                    "is_positive": rec.image_id == args.image_id,
                    "score": round(float(sc), 6),
                    "text": rec.text,
                }
            )
        print(json.dumps({"direction": "i2t", "query_image_id": args.image_id, "results": hits}, indent=2))
        return

    raise ValueError("Specify either --text (for T2I) or --image-id (for I2T)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vlm-retrieval", description="CS5489 Cross-Modal Retrieval CLI")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--coco-root", default="data/raw/coco")
    parser.add_argument("--manifest-dir", default="data/manifests/coco2017")
    parser.add_argument("--cache-dir", default="artifacts/cache")

    sub = parser.add_subparsers(dest="command", required=True)

    p_enc = sub.add_parser("encode", help="Encode a split and cache normalized embeddings")
    p_enc.add_argument("--model", required=True)
    p_enc.add_argument("--split", required=True)
    p_enc.add_argument("--policy", default="raw", choices=["raw", "photo_prefix"])
    p_enc.add_argument("--device", default="auto")
    p_enc.add_argument("--image-batch-size", type=int, default=64)
    p_enc.add_argument("--text-batch-size", type=int, default=128)
    p_enc.add_argument("--force", action="store_true")
    p_enc.set_defaults(func=cmd_encode)

    p_eval = sub.add_parser("evaluate", help="Evaluate cached embeddings and write a run bundle")
    p_eval.add_argument("--model", required=True)
    p_eval.add_argument("--split", required=True)
    p_eval.add_argument("--policy", default="raw", choices=["raw", "photo_prefix"])
    p_eval.add_argument("--rerank", default="none", choices=["none", "csls", "dual_softmax"])
    p_eval.add_argument("--csls-k", type=int, default=10)
    p_eval.add_argument("--ds-temp", type=float, default=0.05)
    p_eval.add_argument("--ds-mode", default="symmetric_log", choices=["symmetric_log", "prior_weighted"])
    p_eval.add_argument("--out-dir", default="artifacts/runs")
    p_eval.add_argument("--run-id", default=None)
    p_eval.set_defaults(func=cmd_evaluate)

    p_search = sub.add_parser("search", help="Run T2I (--text) or I2T (--image-id) lookup")
    p_search.add_argument("--model", required=True)
    p_search.add_argument("--split", required=True)
    p_search.add_argument("--policy", default="raw", choices=["raw", "photo_prefix"])
    p_search.add_argument("--text", default=None)
    p_search.add_argument("--image-id", default=None)
    p_search.add_argument("--top-k", type=int, default=5)
    p_search.add_argument("--device", default="auto")
    p_search.set_defaults(func=cmd_search)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
