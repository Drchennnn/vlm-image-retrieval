"""Phase 2 (NegCLIP-lite): Compositional hard-negative generation and lightweight text/adapter fine-tuning."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import apply_text_policy, l2_normalize_checked

# Spatial and relational connectors frequently appearing in COCO captions
RELATIONAL_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(next to|in front of|on top of)\b", flags=re.IGNORECASE),
    re.compile(r"\b(behind|beside|under|underneath|above|below|near|holding|riding|wearing|chasing|watching|eating)\b", flags=re.IGNORECASE),
    re.compile(r"\b(on|in|with)\b", flags=re.IGNORECASE),
)

NP_PATTERN = re.compile(
    r"\b((?:a|an|the|two|three|some|several)\s+(?:[a-z]+\s+){0,2}[a-z]+)\b",
    flags=re.IGNORECASE,
)

ANTONYM_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("red", "blue"),
    ("blue", "yellow"),
    ("green", "red"),
    ("white", "black"),
    ("black", "white"),
    ("yellow", "purple"),
    ("brown", "gray"),
    ("orange", "blue"),
    ("left", "right"),
    ("right", "left"),
    ("up", "down"),
    ("down", "up"),
    ("standing", "sitting"),
    ("sitting", "standing"),
    ("man", "woman"),
    ("woman", "man"),
    ("boy", "girl"),
    ("girl", "boy"),
    ("big", "small"),
    ("large", "small"),
    ("small", "large"),
    ("young", "old"),
    ("old", "young"),
    ("inside", "outside"),
    ("outside", "inside"),
)

STOP_WORDS = frozenset(
    {
        "a", "an", "the", "of", "in", "on", "at", "to", "for", "with", "by", "from",
        "is", "are", "was", "were", "be", "being", "been", "and", "or", "that", "this",
        "it", "its", "as", "into", "over", "through", "up", "down", "out", "off",
    }
)


def _preserve_case(template: str, replacement: str) -> str:
    if template and template[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement.lower()


def _swap_relational_clauses(text: str) -> str | None:
    """Swap noun phrases around a spatial/relational connector to flip compositional roles."""
    cleaned = text.strip()
    punct = ""
    if cleaned.endswith((".", "!", "?")):
        punct = cleaned[-1]
        cleaned = cleaned[:-1].strip()

    for pat in RELATIONAL_PATTERNS:
        m = pat.search(cleaned)
        if not m:
            continue
        left = cleaned[: m.start()].strip()
        rel = m.group(0)
        right = cleaned[m.end() :].strip()
        # Require non-trivial noun-like phrases on both sides
        if len(left.split()) < 2 or len(right.split()) < 2:
            continue
        if left.lower() == right.lower():
            continue
        # Preserve leading capitalization cleanly
        new_left = _preserve_case(left, right)
        new_right = left[0].lower() + left[1:] if len(left) > 1 else left.lower()
        swapped = f"{new_left} {rel} {new_right}{punct}"
        if swapped.lower() != text.strip().lower():
            return swapped
    return None


def _swap_two_noun_phrases(text: str, rng: np.random.Generator) -> str | None:
    matches = list(NP_PATTERN.finditer(text))
    if len(matches) < 2:
        return None
    # Pick two non-overlapping distinct phrases
    pairs = []
    for i in range(len(matches)):
        for j in range(i + 1, len(matches)):
            a, b = matches[i], matches[j]
            if a.end() <= b.start() and a.group(0).lower() != b.group(0).lower():
                pairs.append((a, b))
    if not pairs:
        return None
    a, b = pairs[int(rng.integers(0, len(pairs)))]
    rep_a = _preserve_case(a.group(0), b.group(0))
    rep_b = _preserve_case(b.group(0), a.group(0))
    swapped = text[: a.start()] + rep_a + text[a.end() : b.start()] + rep_b + text[b.end() :]
    if swapped.lower() != text.lower():
        return swapped
    return None


def _swap_attribute_antonym(text: str, rng: np.random.Generator) -> str | None:
    candidates = []
    for src, dst in ANTONYM_PAIRS:
        m = re.search(rf"\b{src}\b", text, flags=re.IGNORECASE)
        if m:
            candidates.append((m, dst))
    if not candidates:
        return None
    m, dst = candidates[int(rng.integers(0, len(candidates)))]
    rep = _preserve_case(m.group(0), dst)
    mutated = text[: m.start()] + rep + text[m.end() :]
    if mutated.lower() != text.lower():
        return mutated
    return None


def _shuffle_content_words(text: str, rng: np.random.Generator) -> str:
    tokens = re.findall(r"[A-Za-z]+|[^A-Za-z]+", text)
    word_positions = [
        i for i, tok in enumerate(tokens) if tok.isalpha() and tok.lower() not in STOP_WORDS
    ]
    if len(word_positions) < 2:
        word_positions = [i for i, tok in enumerate(tokens) if tok.isalpha()]
    if len(word_positions) < 2:
        return f"not {text}"

    orig_words = [tokens[i] for i in word_positions]
    if len(set(w.lower() for w in orig_words)) < 2:
        return f"not {text}"

    for _ in range(10):
        perm = rng.permutation(len(word_positions))
        candidate = list(tokens)
        for dst_k, src_k in enumerate(perm):
            pos = word_positions[dst_k]
            candidate[pos] = _preserve_case(tokens[pos], orig_words[int(src_k)])
        out = "".join(candidate)
        if out.lower() != text.lower():
            return out

    # Guaranteed deterministic swap of first two distinct content words
    i0, i1 = word_positions[0], word_positions[-1]
    candidate = list(tokens)
    candidate[i0], candidate[i1] = _preserve_case(tokens[i0], tokens[i1]), _preserve_case(tokens[i1], tokens[i0])
    return "".join(candidate)


def generate_hard_negative(text: str, rng: np.random.Generator) -> Tuple[str, str]:
    """Generate a compositional hard negative caption and return (hard_neg_text, operator_name).

    Always returns a string strictly different from the input caption.
    """
    # Prioritize bag-of-words-preserving compositional swaps (relational & NP swaps)
    op_choice = float(rng.random())
    if op_choice < 0.55:
        res = _swap_relational_clauses(text)
        if res is not None:
            return res, "relational_swap"
        res = _swap_two_noun_phrases(text, rng)
        if res is not None:
            return res, "np_swap"
    elif op_choice < 0.85:
        res = _swap_two_noun_phrases(text, rng)
        if res is not None:
            return res, "np_swap"
        res = _swap_relational_clauses(text)
        if res is not None:
            return res, "relational_swap"
    else:
        res = _swap_attribute_antonym(text, rng)
        if res is not None:
            return res, "attribute_swap"

    # Fallbacks
    res = _swap_attribute_antonym(text, rng)
    if res is not None:
        return res, "attribute_swap"
    return _shuffle_content_words(text, rng), "content_shuffle"


class ResidualImageAdapter(nn.Module):
    """Lightweight residual adapter on cached image embeddings, initialized to identity."""

    def __init__(self, dim: int, hidden_dim: int = 256, init_scale: float = 0.05) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.alpha = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x + self.alpha * self.net(x)
        return F.normalize(out, dim=-1)


def negclip_loss(
    img_feats: torch.Tensor,
    pos_txt_feats: torch.Tensor,
    neg_txt_feats: torch.Tensor,
    logit_scale: torch.Tensor,
    *,
    lambda_hn: float = 0.5,
) -> Tuple[torch.Tensor, dict]:
    """Symmetric InfoNCE with in-batch compositional hard negatives + pairwise margin loss.

    Args:
        img_feats: L2-normalized [B, D]
        pos_txt_feats: L2-normalized [B, D]
        neg_txt_feats: L2-normalized [B, D]
        logit_scale: scalar positive scale (e.g., exp(model.logit_scale).clamp(max=100))
    """
    bsz = img_feats.shape[0]
    labels = torch.arange(bsz, device=img_feats.device)

    all_txt = torch.cat([pos_txt_feats, neg_txt_feats], dim=0)  # [2B, D]
    logits_i2t = logit_scale * (img_feats @ all_txt.T)  # [B, 2B]
    logits_t2i = logit_scale * (pos_txt_feats @ img_feats.T)  # [B, B]

    loss_i2t = F.cross_entropy(logits_i2t, labels)
    loss_t2i = F.cross_entropy(logits_t2i, labels)

    # Direct pairwise compositional contrast: s(I_i, C+_i) > s(I_i, C-_i)
    pos_sim = torch.sum(img_feats * pos_txt_feats, dim=-1)
    neg_sim = torch.sum(img_feats * neg_txt_feats, dim=-1)
    loss_comp = F.softplus(logit_scale * (neg_sim - pos_sim)).mean()

    total = 0.5 * (loss_i2t + loss_t2i) + float(lambda_hn) * loss_comp
    comp_acc = float((pos_sim > neg_sim).float().mean().item())
    return total, {
        "loss_total": float(total.item()),
        "loss_i2t": float(loss_i2t.item()),
        "loss_t2i": float(loss_t2i.item()),
        "loss_comp": float(loss_comp.item()),
        "batch_comp_acc": comp_acc,
    }


def configure_trainable_clip_text(
    clip_model: nn.Module,
    *,
    unfreeze_text_blocks: int = 2,
) -> List[nn.Parameter]:
    """Freeze entire CLIP model, then unfreeze only top-K text transformer blocks + ln_final + text_projection."""
    for p in clip_model.parameters():
        p.requires_grad_(False)

    trainable: List[nn.Parameter] = []
    if hasattr(clip_model, "transformer") and hasattr(clip_model.transformer, "resblocks"):
        blocks = clip_model.transformer.resblocks
        for blk in blocks[-unfreeze_text_blocks:]:
            for p in blk.parameters():
                p.requires_grad_(True)
                trainable.append(p)
    if hasattr(clip_model, "ln_final"):
        for p in clip_model.ln_final.parameters():
            p.requires_grad_(True)
            trainable.append(p)
    if hasattr(clip_model, "text_projection") and isinstance(clip_model.text_projection, nn.Parameter):
        clip_model.text_projection.requires_grad_(True)
        trainable.append(clip_model.text_projection)

    if not trainable:
        raise RuntimeError("No trainable text parameters found on CLIP model")
    return trainable


@dataclass
class NegCLIPLiteConfig:
    name: str
    lr: float = 2e-5
    weight_decay: float = 0.01
    lambda_hn: float = 0.5
    epochs: int = 3
    batch_size: int = 64
    unfreeze_text_blocks: int = 2
    use_img_adapter: bool = True
    policy: str = "photo_prefix"
    seed: int = 5489


@torch.inference_mode()
def encode_texts_with_model(
    clip_model: nn.Module,
    tokenizer,
    texts: Sequence[str],
    *,
    policy: str,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    clip_model.eval()
    formatted = [apply_text_policy(t, policy) for t in texts]
    chunks: List[np.ndarray] = []
    for start in range(0, len(formatted), batch_size):
        toks = tokenizer(formatted[start : start + batch_size]).to(device)
        feats = clip_model.encode_text(toks)
        feats = F.normalize(feats.float(), dim=-1)
        chunks.append(feats.cpu().numpy().astype(np.float32))
    return l2_normalize_checked(np.concatenate(chunks, axis=0))


@torch.inference_mode()
def encode_images_with_adapter(
    base_img_embs: np.ndarray,
    adapter: ResidualImageAdapter | None,
    device: torch.device,
) -> np.ndarray:
    if adapter is None:
        return l2_normalize_checked(base_img_embs)
    adapter.eval()
    x = torch.from_numpy(np.asarray(base_img_embs, dtype=np.float32)).to(device)
    out = adapter(x).cpu().numpy().astype(np.float32)
    return l2_normalize_checked(out)


def evaluate_compositional_hard_neg_acc(
    img_embs: np.ndarray,
    pos_cap_embs: np.ndarray,
    neg_cap_embs: np.ndarray,
    cap_to_img_row: Sequence[int],
) -> float:
    """Fraction of captions where s(I, C+) > s(I, C-) strictly."""
    idx = np.asarray(cap_to_img_row, dtype=np.int64)
    v = img_embs[idx]
    s_pos = np.sum(v * pos_cap_embs, axis=1)
    s_neg = np.sum(v * neg_cap_embs, axis=1)
    return float(100.0 * np.mean(s_pos > s_neg))
