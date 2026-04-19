"""Tests for PR D: QuantizedMHA drop-in replacement for nn.MultiheadAttention.

Pins the semantics from ``docs/plans/2026-04-19-sqformat-experiment-plan.md``
→ "PR D · QuantizedMHA + optional flag on MNISTTransformer":

  * ``QuantizedMHA`` matches ``nn.MultiheadAttention`` in FP32 (atol 1e-5).
  * ``MNISTTransformer(use_quantizable_mha=True)`` wires the replacement
    into every encoder layer's ``self_attn``.
  * ``MNISTTransformer(use_quantizable_mha=False)`` (default) preserves
    the legacy model byte-identically.
"""
from __future__ import annotations

import torch


def test_quantizable_mha_matches_nn_mha_in_fp32():
    from examples.model_quantizable import QuantizedMHA

    torch.manual_seed(0)
    embed_dim, num_heads, batch, seq_len = 32, 4, 5, 9

    ref = torch.nn.MultiheadAttention(
        embed_dim, num_heads, bias=True, batch_first=True
    )
    ours = QuantizedMHA(embed_dim, num_heads, bias=True)
    ours.load_from_nn(ref)

    x = torch.randn(batch, seq_len, embed_dim)
    with torch.no_grad():
        y_ref, _ = ref(x, x, x, need_weights=False)
        y_ours = ours(x)
    torch.testing.assert_close(y_ours, y_ref, atol=1e-5, rtol=1e-5)


def test_mnist_transformer_accepts_quantizable_mha_flag():
    from examples.model import MNISTTransformer
    from examples.model_quantizable import QuantizedMHA

    torch.manual_seed(0)
    m = MNISTTransformer(use_quantizable_mha=True)
    # Every encoder layer's self_attn must be a QuantizedMHA now.
    for layer in m.encoder.layers:
        assert isinstance(layer.self_attn, QuantizedMHA)
    x = torch.randn(2, 1, 28, 28)
    y = m(x)
    assert y.shape == (2, 10)


def test_mnist_transformer_default_unchanged():
    """Default flag=False must keep nn.MultiheadAttention so existing
    checkpoints / results stay compatible."""
    from examples.model import MNISTTransformer

    m = MNISTTransformer()
    for layer in m.encoder.layers:
        assert isinstance(layer.self_attn, torch.nn.MultiheadAttention)
