"""Round 1 smoke test: verify all 3 models on 100 dev images, cache round-trip, and resource timings."""

from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import numpy as np

from vlm_retrieval.cache import load_embedding_bundle, save_embedding_bundle
from vlm_retrieval.data import build_relevance_maps, load_split
from vlm_retrieval.model import VLMWrapper, load_model_specs


def _quick_r1(sim: np.ndarray, query_ids: list[str], cand_ids: list[str], pos_map: dict[str, list[str]]) -> float:
    cand_index = {cid: idx for idx, cid in enumerate(cand_ids)}
    hits = 0
    for q_idx, qid in enumerate(query_ids):
        scores = sim[q_idx]
        # Deterministic tie-break: score desc, cand_id asc
        order = sorted(range(len(cand_ids)), key=lambda i: (-float(scores[i]), cand_ids[i]))
        top1_id = cand_ids[order[0]]
        if top1_id in pos_map[qid]:
            hits += 1
    return 100.0 * hits / len(query_ids)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    coco_root = root / "data" / "raw" / "coco"
    manifest_dir = root / "data" / "manifests" / "coco2017"
    cache_dir = root / "artifacts" / "cache" / "smoke"
    out_path = root / "artifacts" / "smoke" / "round1_smoke_report.json"

    split = load_split(manifest_dir, "smoke_dev100")
    t2i_pos, i2t_pos = build_relevance_maps(split)
    image_paths = [coco_root / rec.relative_path for rec in split.images]
    caption_texts = [rec.text for rec in split.captions]

    specs = load_model_specs(root / "configs" / "models.yaml")
    report = {
        "tag": "smoke_dev100",
        "num_images": len(split.images),
        "num_captions": len(split.captions),
        "manifest_sha256": split.manifest_sha256,
        "models": {},
    }

    for key, spec in specs.items():
        t0 = time.perf_counter()
        wrapper = VLMWrapper(spec)
        t_load = time.perf_counter() - t0

        t1 = time.perf_counter()
        img_embs = wrapper.encode_images(image_paths, batch_size=32)
        t_img = time.perf_counter() - t1

        model_entry = {
            "metadata": wrapper.metadata,
            "load_seconds": round(t_load, 3),
            "image_encode_seconds": round(t_img, 3),
            "embedding_dim": int(img_embs.shape[1]),
            "image_norm_max_dev": float(np.max(np.abs(np.linalg.norm(img_embs, axis=1) - 1.0))),
            "policies": {},
        }

        for policy in ("raw", "photo_prefix"):
            t2 = time.perf_counter()
            cap_embs, trunc_stats = wrapper.encode_texts(caption_texts, policy=policy, batch_size=128)
            t_txt = time.perf_counter() - t2

            meta = {
                **wrapper.metadata,
                "text_policy": policy,
                "split_name": split.name,
                "manifest_sha256": split.manifest_sha256,
                "truncation": trunc_stats,
            }
            saved = save_embedding_bundle(
                cache_dir,
                image_ids=split.image_ids,
                image_embeddings=img_embs,
                caption_ids=split.caption_ids,
                caption_embeddings=cap_embs,
                metadata=meta,
            )
            reloaded = load_embedding_bundle(
                cache_dir,
                saved.cache_key,
                expected_manifest_sha256=split.manifest_sha256,
            )
            cache_roundtrip_exact = bool(
                np.array_equal(saved.image_embeddings, reloaded.image_embeddings)
                and np.array_equal(saved.caption_embeddings, reloaded.caption_embeddings)
                and saved.image_ids == reloaded.image_ids
                and saved.caption_ids == reloaded.caption_ids
            )
            if not cache_roundtrip_exact:
                raise RuntimeError(f"Cache round-trip mismatch for {key}/{policy}")

            sim_i2t = reloaded.image_embeddings @ reloaded.caption_embeddings.T
            sim_t2i = sim_i2t.T
            r1_t2i = _quick_r1(sim_t2i, split.caption_ids, split.image_ids, t2i_pos)
            r1_i2t = _quick_r1(sim_i2t, split.image_ids, split.caption_ids, i2t_pos)

            model_entry["policies"][policy] = {
                "text_encode_seconds": round(t_txt, 3),
                "cache_key": saved.cache_key,
                "cache_roundtrip_exact": cache_roundtrip_exact,
                "text_norm_max_dev": float(np.max(np.abs(np.linalg.norm(cap_embs, axis=1) - 1.0))),
                "truncation": trunc_stats,
                "smoke_r1_t2i": round(r1_t2i, 2),
                "smoke_r1_i2t": round(r1_i2t, 2),
            }

        report["models"][key] = model_entry

    # macOS ru_maxrss is in bytes
    report["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024), 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
