"""Synthetic tests for run-bundle serialization and CLI evaluate/search flow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vlm_retrieval.cache import save_embedding_bundle
from vlm_retrieval.cli import main as cli_main
from vlm_retrieval.data import CaptionRecord, ImageRecord, SplitData, _write_jsonl
from vlm_retrieval.evaluate import evaluate_bidirectional_scores, write_run_bundle


def test_run_bundle_and_cli_evaluate_and_i2t_search(tmp_path: Path, capsys) -> None:
    manifest_dir = tmp_path / "manifests"
    cache_dir = tmp_path / "cache"
    runs_dir = tmp_path / "runs"

    images = [
        ImageRecord("0001", "val2017/0001.jpg", "mini", "a" * 64),
        ImageRecord("0002", "val2017/0002.jpg", "mini", "b" * 64),
        ImageRecord("0003", "val2017/0003.jpg", "mini", "c" * 64),
    ]
    captions = [
        CaptionRecord("101", "0001", "a red bus", "mini"),
        CaptionRecord("102", "0001", "bus on road", "mini"),
        CaptionRecord("201", "0002", "a yellow dog", "mini"),
        CaptionRecord("301", "0003", "snowy mountain", "mini"),
    ]
    img_sha = _write_jsonl(manifest_dir / "mini_images.jsonl", images)
    cap_sha = _write_jsonl(manifest_dir / "mini_captions.jsonl", captions)
    split = SplitData("mini", images, captions, img_sha, cap_sha)

    img_embs = np.eye(3, dtype=np.float32)
    cap_embs = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.95, 0.3122499, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    save_embedding_bundle(
        cache_dir,
        image_ids=split.image_ids,
        image_embeddings=img_embs,
        caption_ids=split.caption_ids,
        caption_embeddings=cap_embs,
        metadata={
            "model_key": "clip_vit_b32",
            "model_name": "ViT-B-32",
            "pretrained": "openai",
            "text_policy": "raw",
            "split_name": "mini",
            "manifest_sha256": split.manifest_sha256,
        },
    )

    # 1) Direct write_run_bundle check
    res = evaluate_bidirectional_scores(split, img_embs @ cap_embs.T)
    assert res["mean_R@1"] == 100.0
    bdir = write_run_bundle(runs_dir, run_id="direct_check", config={"test": True}, eval_result=res)
    for fname in ("config.json", "provenance.json", "metrics.json", "ranks.jsonl", "run.log"):
        assert (bdir / fname).exists()

    # 2) CLI evaluate check
    cli_main(
        [
            "--repo-root", str(tmp_path),
            "--manifest-dir", "manifests",
            "--cache-dir", "cache",
            "evaluate",
            "--model", "clip_vit_b32",
            "--split", "mini",
            "--policy", "raw",
            "--rerank", "csls",
            "--csls-k", "2",
            "--out-dir", "runs",
            "--run-id", "cli_csls_test",
        ]
    )
    eval_out = json.loads(capsys.readouterr().out)
    assert eval_out["mean_R@1"] == 100.0
    assert (runs_dir / "cli_csls_test" / "metrics.json").exists()

    # 3) CLI I2T search check
    cli_main(
        [
            "--repo-root", str(tmp_path),
            "--manifest-dir", "manifests",
            "--cache-dir", "cache",
            "search",
            "--model", "clip_vit_b32",
            "--split", "mini",
            "--policy", "raw",
            "--image-id", "0001",
            "--top-k", "2",
        ]
    )
    search_out = json.loads(capsys.readouterr().out)
    assert search_out["direction"] == "i2t"
    assert search_out["results"][0]["caption_id"] == "101"
    assert search_out["results"][0]["is_positive"] is True
