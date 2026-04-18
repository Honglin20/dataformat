"""Paired (X, W) generators simulating common linear-layer regimes."""
from __future__ import annotations

import numpy as np


def weight_transformer(batch: int, in_features: int, out_features: int, seed: int):
    """Typical transformer weight: N(0, 0.02²) scaled, no outliers.

    Activations: N(0,1) (pre-normalised), mild heavy-tail via Student-t(10).
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, 0.02, size=(out_features, in_features)).astype(np.float32)
    # Mild heavy-tail activations
    X = rng.standard_t(df=10.0, size=(batch, in_features)).astype(np.float32)
    meta = {"type": "transformer_typical"}
    return X, W, meta


def weight_moe(batch: int, in_features: int, out_features: int, seed: int):
    """Wider weight distribution (FFN up-projection), Laplace tails.

    Activations: Laplace-like (post-GELU style)."""
    rng = np.random.default_rng(seed)
    W = rng.laplace(0.0, 0.03, size=(out_features, in_features)).astype(np.float32)
    X = rng.standard_t(df=5.0, size=(batch, in_features)).astype(np.float32)
    meta = {"type": "ffn_up_projection"}
    return X, W, meta


def weight_attention(batch: int, in_features: int, out_features: int, seed: int):
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


def smooth_friendly_mild(batch: int, in_features: int, out_features: int, seed: int):
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


def smooth_friendly_severe(batch: int, in_features: int, out_features: int, seed: int):
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


def smooth_friendly_balanced(batch: int, in_features: int, out_features: int, seed: int):
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
