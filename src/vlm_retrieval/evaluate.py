"""Run-bundle provenance writer and end-to-end bidirectional retrieval evaluation helpers."""

from __future__ import annotations

import hashlib
import json
import platform
import resource
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from .data import SplitData, build_relevance_maps
from .metrics import evaluate_retrieval

PROTOCOL_VERSION = "v2"


def collect_git_provenance(repo_root: Path | None = None) -> dict:
    cwd = str(repo_root) if repo_root is not None else None
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL, text=True
        ).strip()
        diff_bytes = subprocess.check_output(
            ["git", "diff", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
        )
        status_out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=cwd, stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(status_out)
        dirty_hash = hashlib.sha256(diff_bytes).hexdigest() if diff_bytes else None
    except Exception:
        return {"git_commit": None, "git_dirty": None, "git_dirty_diff_sha256": None}
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "git_dirty_diff_sha256": dirty_hash,
    }


def collect_runtime_provenance(repo_root: Path | None = None, extra: dict | None = None) -> dict:
    import open_clip
    import sklearn
    import torch

    payload: Dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            "torch": torch.__version__,
            "open_clip": open_clip.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024), 1),
        **collect_git_provenance(repo_root),
    }
    if extra:
        payload.update(extra)
    return payload


def evaluate_bidirectional_scores(
    split: SplitData,
    sim_i2t: np.ndarray,
    sim_t2i: np.ndarray | None = None,
    *,
    ks: Sequence[int] = (1, 5, 10),
    top_k_store: int = 10,
) -> dict:
    """Evaluate both T2I and I2T directions given [N_img, N_cap] (and optional [N_cap, N_img]) matrices."""
    t2i_pos, i2t_pos = build_relevance_maps(split)
    mat_t2i = sim_i2t.T if sim_t2i is None else sim_t2i
    t2i_res = evaluate_retrieval(
        mat_t2i,
        split.caption_ids,
        split.image_ids,
        t2i_pos,
        ks=ks,
        direction="t2i",
        top_k_store=top_k_store,
    )
    i2t_res = evaluate_retrieval(
        sim_i2t,
        split.image_ids,
        split.caption_ids,
        i2t_pos,
        ks=ks,
        direction="i2t",
        top_k_store=top_k_store,
    )
    mean_r1 = 0.5 * (t2i_res["metrics"]["R@1"] + i2t_res["metrics"]["R@1"])
    return {
        "split_name": split.name,
        "manifest_sha256": split.manifest_sha256,
        "mean_R@1": round(float(mean_r1), 4),
        "t2i": t2i_res,
        "i2t": i2t_res,
    }


def write_run_bundle(
    out_dir: Path,
    *,
    run_id: str,
    config: dict,
    eval_result: dict,
    provenance_extra: dict | None = None,
    log_lines: Sequence[str] = (),
    repo_root: Path | None = None,
) -> Path:
    """Write standardized run bundle: config.json, provenance.json, metrics.json, ranks.jsonl, run.log."""
    bundle_dir = Path(out_dir) / run_id
    bundle_dir.mkdir(parents=True, exist_ok=True)

    (bundle_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    prov = collect_runtime_provenance(
        repo_root=repo_root,
        extra={"run_id": run_id, **(provenance_extra or {})},
    )
    (bundle_dir / "provenance.json").write_text(
        json.dumps(prov, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    metrics_summary = {
        "run_id": run_id,
        "split_name": eval_result["split_name"],
        "manifest_sha256": eval_result["manifest_sha256"],
        "mean_R@1": eval_result["mean_R@1"],
        "t2i": {
            "num_queries": eval_result["t2i"]["num_queries"],
            "num_candidates": eval_result["t2i"]["num_candidates"],
            "positives_per_query_dist": eval_result["t2i"]["positives_per_query_dist"],
            "metrics": eval_result["t2i"]["metrics"],
        },
        "i2t": {
            "num_queries": eval_result["i2t"]["num_queries"],
            "num_candidates": eval_result["i2t"]["num_candidates"],
            "positives_per_query_dist": eval_result["i2t"]["positives_per_query_dist"],
            "metrics": eval_result["i2t"]["metrics"],
        },
    }
    (bundle_dir / "metrics.json").write_text(
        json.dumps(metrics_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    with (bundle_dir / "ranks.jsonl").open("w", encoding="utf-8") as f:
        for direction in ("t2i", "i2t"):
            for rec in eval_result[direction]["per_query"]:
                f.write(json.dumps({"direction": direction, **rec}, ensure_ascii=False) + "\n")

    log_text = "\n".join(log_lines) + ("\n" if log_lines else "")
    (bundle_dir / "run.log").write_text(log_text, encoding="utf-8")
    return bundle_dir
