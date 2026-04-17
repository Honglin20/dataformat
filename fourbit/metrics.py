"""Quantization-quality metrics used by the 4-bit study.

Centralised here so that every experiment – Part 1 and Part 2 – reports the
same numbers.

QSNR  = 10 · log10( Var(x) / MSE(x, x_hat) )

Crest factor of a tensor is the peak-to-RMS ratio:

    crest(x) = max(|x|) / sqrt(mean(x^2))

Crest factor is >= 1 for any non-zero tensor.  A Gaussian has crest ≈ 3.7
(for 4096 samples), a Laplace crest ≈ 7, and heavy-outlier tensors can reach
crest >= 50.  It is the single number that best predicts how many bits are
wasted on outlier headroom under per-tensor/per-channel absmax quantization.
"""
from __future__ import annotations

import numpy as np


def qsnr_db(x: np.ndarray, x_hat: np.ndarray) -> float:
    """Quantization SNR in dB.  Returns +inf when MSE == 0."""
    x   = np.asarray(x,   dtype=np.float64).ravel()
    xh  = np.asarray(x_hat, dtype=np.float64).ravel()
    mse = float(np.mean((x - xh) ** 2))
    var = float(np.var(x))
    if mse <= 0.0:
        return float("inf")
    if var <= 0.0:
        return 0.0
    return 10.0 * np.log10(var / mse)


def fp16_quantize(x: np.ndarray) -> np.ndarray:
    """Round-trip ``x`` through float16.

    Used as the "infinite-precision" baseline against every 4-bit format:
    FP16 QSNR is an upper bound on what any 4-bit scheme can achieve, and
    also represents the industry-standard inference datatype that W4A4
    replaces.  Both casts are exact (saturating) per IEEE-754 round-to-
    nearest-even.
    """
    x = np.asarray(x, dtype=np.float32)
    return x.astype(np.float16).astype(np.float32)


def fp16_qsnr_db(x: np.ndarray) -> float:
    """QSNR (dB) of FP16-rounded ``x`` vs FP32 ``x``.  Baseline precision."""
    return qsnr_db(x, fp16_quantize(x))


def mse(x: np.ndarray, x_hat: np.ndarray) -> float:
    x  = np.asarray(x,  dtype=np.float64)
    xh = np.asarray(x_hat, dtype=np.float64)
    return float(np.mean((x - xh) ** 2))


def crest_factor(x: np.ndarray) -> float:
    """Peak-to-RMS ratio ``max(|x|) / rms(x)``.  Returns 0.0 for an all-zero tensor."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return 0.0
    peak = float(np.max(np.abs(x)))
    rms  = float(np.sqrt(np.mean(x * x)))
    if rms <= 0.0:
        return 0.0
    return peak / rms


def tensor_summary(x: np.ndarray) -> dict:
    """Compact stat bundle for a tensor: std, max, crest, kurtosis."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return {"std": 0.0, "max_abs": 0.0, "crest": 0.0, "kurtosis": 0.0, "n": 0}
    std = float(np.std(x))
    peak = float(np.max(np.abs(x)))
    mean = float(np.mean(x))
    dev = x - mean
    var = float(np.mean(dev * dev))
    kurt = float(np.mean(dev ** 4) / (var ** 2)) - 3.0 if var > 0 else 0.0
    crest = peak / std if std > 0 else 0.0
    return {
        "std":      std,
        "max_abs":  peak,
        "crest":    crest,    # peak / std (typical machine-learning convention)
        "crest_rms": crest_factor(x),  # peak / rms (includes mean)
        "kurtosis": kurt,
        "n":        int(x.size),
    }
