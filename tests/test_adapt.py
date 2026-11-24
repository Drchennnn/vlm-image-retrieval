"""Synthetic unit tests for Phase 2 NegCLIP-lite hard negatives, residual adapter, and loss."""

from __future__ import annotations

import numpy as np
import torch

from vlm_retrieval.adapt import (
    ResidualImageAdapter,
    evaluate_compositional_hard_neg_acc,
    generate_hard_negative,
    negclip_loss,
)


def test_hard_negative_generator_always_mutates_and_covers_relational_swaps() -> None:
    rng = np.random.Generator(np.random.PCG64(5489))
    samples = [
        "A man sitting on a wooden bench.",
        "A small brown dog chasing a red ball on the grass.",
        "The cat next to a silver laptop on a desk.",
        "Two boys wearing blue shirts standing behind a white fence.",
        "A plate of pizza.",
        "Bus.",
    ]
    ops = set()
    for text in samples:
        for _ in range(5):
            neg, op = generate_hard_negative(text, rng)
            assert isinstance(neg, str) and len(neg.strip()) > 0
            assert neg.strip().lower() != text.strip().lower()
            ops.add(op)
    assert "relational_swap" in ops or "np_swap" in ops


def test_residual_adapter_identity_init_and_negclip_loss_gradient() -> None:
    torch.manual_seed(5489)
    dim = 16
    adapter = ResidualImageAdapter(dim=dim, hidden_dim=8)
    x = torch.randn(4, dim)
    x = x / x.norm(dim=-1, keepdim=True)
    y = adapter(x)
    # Zero-initialized last layer => starts as exact identity on unit sphere
    assert torch.allclose(x, y, atol=1e-6)

    pos = x.clone()
    neg = -x.clone()
    scale = torch.tensor(10.0)
    loss, info = negclip_loss(y, pos, neg, scale, lambda_hn=1.0)
    assert torch.isfinite(loss)
    assert info["batch_comp_acc"] == 1.0
    loss.backward()
    # Gradient flows through adapter weights
    assert adapter.net[-1].weight.grad is not None
    assert torch.all(torch.isfinite(adapter.net[-1].weight.grad))


def test_compositional_acc_strict_inequality() -> None:
    img = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    pos = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.8660254]], dtype=np.float32)
    neg = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    # idx 0: 1.0 > 0.0 (pass); idx 1: 1.0 == 1.0 (tie -> fail); idx 2: 0.866 > 0.0 (pass)
    acc = evaluate_compositional_hard_neg_acc(img, pos, neg, [0, 1, 1])
    assert abs(acc - (200.0 / 3.0)) < 1e-4
