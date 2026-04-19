"""Hand-written self-attention module for end-to-end quantisation.

PyTorch's ``nn.MultiheadAttention`` calls
``F.multi_head_attention_forward``, which reads ``out_proj.weight`` /
``in_proj_weight`` as raw parameter tensors and bypasses
``Module.forward`` on the internal Linears.  This means a
:class:`experiments.fourbit.accuracy.QuantLinear` wrapper cannot
intercept its matmuls — a fatal blind spot for the SQ-Format experiment
where *every* W/A product must be quantised.

:class:`QuantizedMHA` reimplements the attention forward pass with four
explicit ``nn.Linear`` submodules (``q_proj``, ``k_proj``, ``v_proj``,
``out_proj``), so swapping each for a ``QuantLinear`` works uniformly.

It is deliberately a minimal reference: only self-attention, no
dropout, no key-padding mask, no causal mask.  Sufficient for the
MNIST transformer in :mod:`examples.model`.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class QuantizedMHA(nn.Module):
    """Drop-in self-attention that routes every matmul through submodule forwards."""

    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Attributes some nn.TransformerEncoderLayer implementations probe on
        # ``self_attn`` before deciding which dispatch path to take.  Setting
        # them avoids AttributeError when we drop ``QuantizedMHA`` in as a
        # replacement for ``nn.MultiheadAttention``.
        self.batch_first = True
        self._qkv_same_embed_dim = True

    def load_from_nn(self, ref: nn.MultiheadAttention) -> None:
        """Copy Q/K/V/out weights from an ``nn.MultiheadAttention``.

        ``ref.in_proj_weight`` has shape ``(3*embed_dim, embed_dim)``
        with Q / K / V rows stacked in that order.
        """
        W = ref.in_proj_weight.detach()
        b = ref.in_proj_bias.detach() if ref.in_proj_bias is not None else None
        e = self.embed_dim
        self.q_proj.weight.data.copy_(W[0 * e:1 * e])
        self.k_proj.weight.data.copy_(W[1 * e:2 * e])
        self.v_proj.weight.data.copy_(W[2 * e:3 * e])
        if b is not None:
            self.q_proj.bias.data.copy_(b[0 * e:1 * e])
            self.k_proj.bias.data.copy_(b[1 * e:2 * e])
            self.v_proj.bias.data.copy_(b[2 * e:3 * e])
        self.out_proj.weight.data.copy_(ref.out_proj.weight.detach())
        if ref.out_proj.bias is not None:
            self.out_proj.bias.data.copy_(ref.out_proj.bias.detach())

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        **kwargs,
    ):
        """Self-attention forward.

        Accepts the ``(query, key, value, **kwargs)`` signature of
        ``nn.MultiheadAttention`` for drop-in compatibility with
        ``nn.TransformerEncoderLayer._sa_block``.  Only self-attention
        is supported (``query is key is value``); masks and weight
        returns are not.

        Returns ``(out, None)`` when called with the MHA-style kwargs
        (``need_weights`` present) so the encoder layer's ``[0]`` index
        works; otherwise returns the plain tensor (for direct callers
        like the unit test).
        """
        x = query
        mha_style = (key is not None) or (value is not None) or ("need_weights" in kwargs)
        B, T, E = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = scores.softmax(dim=-1)
        ctx = attn @ v
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, E)
        out = self.out_proj(ctx)
        return (out, None) if mha_style else out
