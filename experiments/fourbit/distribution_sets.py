"""Curated distribution sets for the 4-bit study (Part 1).

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

from distributions.generators import (
    gaussian, laplace, student_t_dist, bimodal,
    channel_outliers, spiky_outliers, log_normal,
)
from distributions.linear_pairs import (
    weight_transformer, weight_moe, weight_attention,
    smooth_friendly_mild, smooth_friendly_severe, smooth_friendly_balanced,
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

LINEAR_WEIGHT_ACTIVATION: list[LinearSpec] = [
    LinearSpec("TransformerTypical", weight_transformer, tags=["typical"]),
    LinearSpec("FFN-UpProjection",   weight_moe,         tags=["ffn"]),
    LinearSpec("AttentionQKV",       weight_attention,   tags=["attention"]),
]


# ── SmoothQuant-friendly paired distributions (Part 1.3) ─────────────────────

SMOOTH_FRIENDLY: list[LinearSpec] = [
    LinearSpec("SmoothFriendly-Mild",     smooth_friendly_mild,     tags=["smooth", "mild"]),
    LinearSpec("SmoothFriendly-Severe",   smooth_friendly_severe,   tags=["smooth", "severe"]),
    LinearSpec("SmoothFriendly-Balanced", smooth_friendly_balanced, tags=["smooth", "balanced"]),
]
