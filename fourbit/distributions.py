"""Synthetic distributions used by the Part-1 experiments.

Three distribution libraries are exposed:

  * ``COMMON_DISTRIBUTIONS``
        Seven "ideal / near-ideal" distributions (Gaussian, Laplace, Student-t,
        Bimodal, channel outliers, spiky outliers, Log-Normal) – the same set
        the rest of the repo uses, re-imported from ``distributions.generators``
        so behaviour stays aligned.

  * ``LINEAR_WEIGHT_ACTIVATION``
        Pairs of (weight, activation) tensors mimicking a typical Transformer
        linear layer: weights ~ N(0, 0.02²) with a few per-channel outliers,
        activations ~ GELU(N(0,1)) with light positive skew.

  * ``SMOOTH_FRIENDLY``
        Pairs engineered to be maximally responsive to SmoothQuant:
        activations with a small set of high-magnitude input channels, weights
        that *lack* those systematic outliers.  SmoothQuant's s_k = |X_k|^α /
        |W_k|^(1-α) is largest on those channels, absorbing the activation
        outliers into the weights.

Each generator returns ``(x, metadata)`` for single-tensor distributions, or
``(X, W, metadata)`` for paired distributions.  Shapes:

    X : (batch, in_features)   e.g. (128, 256)
    W : (out_features, in_features)  e.g. (128, 256)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np

from distributions.generators import (
    gaussian, laplace, student_t_dist, bimodal,
    channel_outliers, spiky_outliers, log_normal,
)


# ── Dataclass wrappers ───────────────────────────────────────────────────────

@dataclass
class DistSpec:
    name: str
    fn: Callable
    tags: List[str] = field(default_factory=list)

    def generate(self, n: int, seed: int):
        return self.fn(n=n, seed=seed)


@dataclass
class LinearSpec:
    """Paired (X, W) specification.

    ``fn(batch, in_features, out_features, seed) -> (X, W, metadata)``.
    """
    name: str
    fn: Callable
    tags: List[str] = field(default_factory=list)

    def generate(self, batch: int, in_features: int, out_features: int, seed: int):
        return self.fn(
            batch=batch, in_features=in_features,
            out_features=out_features, seed=seed,
        )


# ── Common distributions (Part 1.1) ──────────────────────────────────────────

COMMON_DISTRIBUTIONS: list[DistSpec] = [
    DistSpec("Gaussian(σ=1)",
             lambda n, seed: gaussian(n, sigma=1.0, seed=seed),
             tags=["gaussian"]),
    DistSpec("Laplace(b=1)",
             lambda n, seed: laplace(n, b=1.0, seed=seed),
             tags=["heavy-tail"]),
    DistSpec("Student-t(ν=3)",
             lambda n, seed: student_t_dist(n, nu=3.0, seed=seed),
             tags=["heavy-tail"]),
    DistSpec("Student-t(ν=10)",
             lambda n, seed: student_t_dist(n, nu=10.0, seed=seed),
             tags=["heavy-tail"]),
    DistSpec("Bimodal(μ=±3)",
             lambda n, seed: bimodal(n, mu1=-3.0, mu2=3.0, sigma=0.5, seed=seed),
             tags=["multimodal"]),
    DistSpec("ChannelOutlier(σ=30)",
             lambda n, seed: channel_outliers(n, outlier_ratio=0.01,
                                              outlier_sigma=30.0, seed=seed),
             tags=["outlier", "channel"]),
    DistSpec("ChannelOutlier(σ=100)",
             lambda n, seed: channel_outliers(n, outlier_ratio=0.01,
                                              outlier_sigma=100.0, seed=seed),
             tags=["outlier", "channel"]),
    DistSpec("Spiky(50×)",
             lambda n, seed: spiky_outliers(n, spike_ratio=0.001,
                                            spike_multiplier=50.0, seed=seed),
             tags=["outlier", "spiky"]),
    DistSpec("LogNormal(σ=1)",
             lambda n, seed: log_normal(n, mu=0.0, sigma=1.0, seed=seed),
             tags=["lognormal"]),
]


# ── Paired (W, X) distributions for linear simulation ───────────────────────

def _weight_transformer(batch: int, in_features: int, out_features: int, seed: int):
    """Typical transformer weight: N(0, 0.02²) scaled, no outliers.

    Activations: N(0,1) (pre-normalised), mild heavy-tail via Student-t(10).
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, 0.02, size=(out_features, in_features)).astype(np.float32)
    # Mild heavy-tail activations
    X = rng.standard_t(df=10.0, size=(batch, in_features)).astype(np.float32)
    meta = {"type": "transformer_typical"}
    return X, W, meta


def _weight_moe(batch: int, in_features: int, out_features: int, seed: int):
    """Wider weight distribution (FFN up-projection), Laplace tails.

    Activations: Laplace-like (post-GELU style)."""
    rng = np.random.default_rng(seed)
    W = rng.laplace(0.0, 0.03, size=(out_features, in_features)).astype(np.float32)
    X = rng.standard_t(df=5.0, size=(batch, in_features)).astype(np.float32)
    meta = {"type": "ffn_up_projection"}
    return X, W, meta


def _weight_attention(batch: int, in_features: int, out_features: int, seed: int):
    """Attention QKV projection: tight weight distribution, bimodal activations.

    Activations in attention can be bimodal (pre-softmax).
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, 0.015, size=(out_features, in_features)).astype(np.float32)
    half = batch // 2
    X1 = rng.normal(-2.0, 0.6, size=(half, in_features)).astype(np.float32)
    X2 = rng.normal(+2.0, 0.6, size=(batch - half, in_features)).astype(np.float32)
    X = np.vstack([X1, X2])
    rng.shuffle(X, axis=0)
    meta = {"type": "attention_qkv"}
    return X, W, meta


LINEAR_WEIGHT_ACTIVATION: list[LinearSpec] = [
    LinearSpec("TransformerTypical", _weight_transformer, tags=["typical"]),
    LinearSpec("FFN-UpProjection",   _weight_moe,         tags=["ffn"]),
    LinearSpec("AttentionQKV",       _weight_attention,   tags=["attention"]),
]


# ── SmoothQuant-friendly paired distributions (Part 1.3) ─────────────────────

def _smooth_friendly_mild(batch: int, in_features: int, out_features: int, seed: int):
    """5% of input channels have 20× larger activation magnitude.

    Weights have a matching 'quiet' profile – their input-channel absmax is
    nearly constant across channels.  SmoothQuant moves the outlier mass
    from X to W, flattening the *activation* crest factor and slightly
    enlarging the *weight* crest factor, which 4-bit per-channel-scaled
    formats tolerate much better.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(batch, in_features)).astype(np.float32)
    W = rng.normal(0.0, 0.02, size=(out_features, in_features)).astype(np.float32)

    outlier_ratio = 0.05
    n_outlier = max(1, int(outlier_ratio * in_features))
    outlier_idx = rng.choice(in_features, size=n_outlier, replace=False)
    X[:, outlier_idx] *= 20.0                    # 20× larger activation channels

    meta = {
        "type": "smooth_friendly_mild",
        "outlier_ratio": outlier_ratio,
        "outlier_mult": 20.0,
    }
    return X, W, meta


def _smooth_friendly_severe(batch: int, in_features: int, out_features: int, seed: int):
    """Same shape as mild but with 100× activation outliers in 2% of channels.

    This is roughly the LLM.int8() regime – many 4-bit formats collapse under
    direct quantization, whereas SmoothQuant / Hadamard recover substantial
    QSNR.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(batch, in_features)).astype(np.float32)
    W = rng.normal(0.0, 0.02, size=(out_features, in_features)).astype(np.float32)

    outlier_ratio = 0.02
    n_outlier = max(1, int(outlier_ratio * in_features))
    outlier_idx = rng.choice(in_features, size=n_outlier, replace=False)
    X[:, outlier_idx] *= 100.0

    meta = {
        "type": "smooth_friendly_severe",
        "outlier_ratio": outlier_ratio,
        "outlier_mult": 100.0,
    }
    return X, W, meta


def _smooth_friendly_balanced(batch: int, in_features: int, out_features: int, seed: int):
    """Outliers on both sides – SmoothQuant alpha=0.5 is not optimal, HAD usually wins.

    Tests the contrast: smooth works when outliers are one-sided; HAD
    handles the two-sided case better because it redistributes energy
    across all channels.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(batch, in_features)).astype(np.float32)
    W = rng.normal(0.0, 0.02, size=(out_features, in_features)).astype(np.float32)

    n_act = max(1, int(0.02 * in_features))
    n_w   = max(1, int(0.02 * out_features))
    act_idx = rng.choice(in_features,  size=n_act, replace=False)
    w_idx   = rng.choice(out_features, size=n_w,   replace=False)

    X[:, act_idx] *= 30.0
    W[w_idx, :]   *= 15.0

    meta = {
        "type": "smooth_friendly_balanced",
        "act_outlier_ratio": 0.02,
        "w_outlier_ratio":   0.02,
    }
    return X, W, meta


SMOOTH_FRIENDLY: list[LinearSpec] = [
    LinearSpec("SmoothFriendly-Mild",     _smooth_friendly_mild,     tags=["smooth", "mild"]),
    LinearSpec("SmoothFriendly-Severe",   _smooth_friendly_severe,   tags=["smooth", "severe"]),
    LinearSpec("SmoothFriendly-Balanced", _smooth_friendly_balanced, tags=["smooth", "balanced"]),
]
