"""Unified frozen vision-language model wrapper (OpenAI CLIP & SigLIP via open_clip)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

VALID_TEXT_POLICIES = ("raw", "photo_prefix")
PHOTO_PREFIX = "a photo of "


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_name: str
    pretrained: str
    backbone: str
    loss_type: str
    context_length: int
    quick_gelu: bool = False
    description: str = ""


def load_model_specs(config_path: Path) -> Dict[str, ModelSpec]:
    with Path(config_path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    specs: Dict[str, ModelSpec] = {}
    for key, cfg in raw["models"].items():
        specs[key] = ModelSpec(key=key, **cfg)
    return specs


def apply_text_policy(text: str, policy: str) -> str:
    if policy == "raw":
        return text
    if policy == "photo_prefix":
        return f"{PHOTO_PREFIX}{text}"
    raise ValueError(f"Unsupported text policy '{policy}'; expected one of {VALID_TEXT_POLICIES}")


def resolve_device(device: str | None = None) -> torch.device:
    if device is not None and device != "auto":
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def l2_normalize_checked(embs: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize embeddings as float32 and reject NaN, Inf, or zero-norm vectors."""
    arr = np.asarray(embs, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D embedding array, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Non-finite values (NaN/Inf) detected in embeddings")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    if np.any(norms <= eps):
        raise ValueError("Zero-norm vector detected in embeddings")
    normalized = arr / norms
    if not np.all(np.isfinite(normalized)):
        raise ValueError("Non-finite values after L2 normalization")
    return normalized


class VLMWrapper:
    """Frozen pretrained vision-language encoder with deterministic preprocessing and truncation stats."""

    def __init__(self, spec: ModelSpec, device: str | None = None) -> None:
        self.spec = spec
        self.device = resolve_device(device)
        create_kwargs = {}
        if spec.quick_gelu:
            create_kwargs["force_quick_gelu"] = True
        model, _, preprocess = open_clip.create_model_and_transforms(
            spec.model_name,
            pretrained=spec.pretrained,
            **create_kwargs,
        )
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.model = model.to(self.device)
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(spec.model_name)
        self.context_length = int(getattr(model, "context_length", spec.context_length))

    @property
    def metadata(self) -> dict:
        return {
            "model_key": self.spec.key,
            "model_name": self.spec.model_name,
            "pretrained": self.spec.pretrained,
            "backbone": self.spec.backbone,
            "loss_type": self.spec.loss_type,
            "context_length": self.context_length,
            "open_clip_version": open_clip.__version__,
            "torch_version": torch.__version__,
            "device": str(self.device),
        }

    def count_token_lengths(self, texts: Sequence[str]) -> List[int]:
        """Compute full un-truncated token lengths (including special tokens) for diagnostics."""
        lengths: List[int] = []
        # Case 1: open_clip HFTokenizer (e.g. SigLIP)
        if hasattr(self.tokenizer, "tokenizer"):
            hf_tok = self.tokenizer.tokenizer
            for t in texts:
                ids = hf_tok.encode(t, add_special_tokens=True, truncation=False)
                lengths.append(len(ids))
            return lengths
        # Case 2: open_clip SimpleTokenizer (CLIP BPE)
        if hasattr(self.tokenizer, "encode"):
            for t in texts:
                # +2 for SOT and EOT tokens
                lengths.append(len(self.tokenizer.encode(t)) + 2)
            return lengths
        # Fallback: inspect non-zero tokens from tokenized tensor
        toks = self.tokenizer(list(texts))
        return [int((row != 0).sum().item()) for row in toks]

    @torch.inference_mode()
    def encode_images(
        self,
        image_paths: Sequence[Path],
        *,
        batch_size: int = 64,
    ) -> np.ndarray:
        if not image_paths:
            raise ValueError("image_paths must be non-empty")
        chunks: List[np.ndarray] = []
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            tensors = []
            for p in batch_paths:
                with Image.open(p) as im:
                    tensors.append(self.preprocess(im.convert("RGB")))
            batch_tensor = torch.stack(tensors, dim=0).to(self.device)
            feats = self.model.encode_image(batch_tensor)
            feats = F.normalize(feats.float(), dim=-1)
            chunks.append(feats.cpu().numpy().astype(np.float32))
        embs = np.concatenate(chunks, axis=0)
        return l2_normalize_checked(embs)

    @torch.inference_mode()
    def encode_texts(
        self,
        texts: Sequence[str],
        *,
        policy: str = "raw",
        batch_size: int = 128,
    ) -> Tuple[np.ndarray, dict]:
        if not texts:
            raise ValueError("texts must be non-empty")
        formatted = [apply_text_policy(t, policy) for t in texts]
        raw_lengths = self.count_token_lengths(formatted)
        truncated_indices = [i for i, n in enumerate(raw_lengths) if n > self.context_length]

        chunks: List[np.ndarray] = []
        for start in range(0, len(formatted), batch_size):
            batch_texts = formatted[start : start + batch_size]
            tokens = self.tokenizer(batch_texts).to(self.device)
            feats = self.model.encode_text(tokens)
            feats = F.normalize(feats.float(), dim=-1)
            chunks.append(feats.cpu().numpy().astype(np.float32))
        embs = np.concatenate(chunks, axis=0)
        embs = l2_normalize_checked(embs)

        trunc_stats = {
            "policy": policy,
            "context_length": self.context_length,
            "num_texts": len(formatted),
            "num_truncated": len(truncated_indices),
            "truncation_rate": len(truncated_indices) / len(formatted),
            "max_token_length": max(raw_lengths),
            "mean_token_length": float(np.mean(raw_lengths)),
            "p95_token_length": float(np.percentile(raw_lengths, 95)),
            "truncated_indices": truncated_indices[:50],
        }
        return embs, trunc_stats
