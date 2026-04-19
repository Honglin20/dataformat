"""5-metric quantization quality evaluation suite.

Metrics:
  1. MSE      — Mean Squared Error (primary distortion measure)
  2. SNR      — Signal-to-Noise Ratio in dB
  3. KL       — KL divergence (distribution shape preservation)
  4. Max-AE   — Maximum Absolute Error (worst-case single-element error)
  5. EffBits  — Effective bit-width via rate-distortion theory
"""

from typing import Callable, Dict

import numpy as np
from scipy.stats import entropy as _scipy_entropy


# ── 1. MSE ───────────────────────────────────────────────────────────────────

def mse(x_orig: np.ndarray, x_recon: np.ndarray) -> float:
    """Mean Squared Error between original and reconstructed tensor."""
    x_orig = x_orig.astype(np.float64)
    x_recon = x_recon.astype(np.float64)
    return float(np.mean((x_orig - x_recon) ** 2))


# ── 2. SNR ───────────────────────────────────────────────────────────────────

def snr_db(x_orig: np.ndarray, x_recon: np.ndarray) -> float:
    """Signal-to-Noise Ratio in decibels.

    SNR = 10 · log₁₀( Var(x) / MSE(x, x̂) )

    Higher is better. Returns -inf if MSE == 0 (perfect reconstruction).
    """
    x_orig = x_orig.astype(np.float64)
    error_mse = mse(x_orig, x_recon)
    signal_var = float(np.var(x_orig))
    if error_mse == 0.0:
        return float("inf")
    if signal_var == 0.0:
        return 0.0
    return float(10.0 * np.log10(signal_var / error_mse))


# ── 3. KL Divergence ─────────────────────────────────────────────────────────

def kl_divergence(
    x_orig: np.ndarray,
    x_recon: np.ndarray,
    n_bins: int = 256,
    eps: float = 1e-10,
) -> float:
    """Kullback-Leibler divergence KL(P_orig ‖ P_recon).

    Histograms are computed over the union of both tensors' range.
    Smaller is better (0 = identical distributions).

    Parameters
    ----------
    n_bins : int
        Number of histogram bins (higher → more accurate but noisier for small N).
    eps : float
        Small constant to avoid log(0).
    """
    x_orig = x_orig.astype(np.float64)
    x_recon = x_recon.astype(np.float64)

    all_vals = np.concatenate([x_orig, x_recon])
    lo, hi = np.min(all_vals), np.max(all_vals)

    if hi == lo:
        return 0.0

    bins = np.linspace(lo, hi, n_bins + 1)
    p, _ = np.histogram(x_orig, bins=bins, density=False)
    q, _ = np.histogram(x_recon, bins=bins, density=False)

    p = p.astype(np.float64) + eps
    q = q.astype(np.float64) + eps
    p /= p.sum()
    q /= q.sum()

    return float(_scipy_entropy(p, q))


# ── 4. Maximum Absolute Error ─────────────────────────────────────────────────

def max_absolute_error(x_orig: np.ndarray, x_recon: np.ndarray) -> float:
    """Maximum absolute error — critical for outlier-heavy tensors."""
    return float(np.max(np.abs(x_orig.astype(np.float64) - x_recon.astype(np.float64))))


# ── 5. Effective Bit-Width ────────────────────────────────────────────────────

def effective_bits(x_orig: np.ndarray, x_recon: np.ndarray) -> float:
    """Information-theoretically equivalent bit-width via rate-distortion theory.

    For a Gaussian source with variance σ², quantized to MSE distortion D,
    the rate-distortion bound gives:
        R(D) = ½ log₂(σ²/D)   [bits per sample]

    We use this as a proxy "effective bits" regardless of the actual source
    distribution, interpreting it as: how many bits of a Gaussian code would
    achieve this MSE level?

    EffBits = ½ log₂( Var(x) / MSE )

    - EffBits ≈ nominal bit-width → format is working close to theoretical limit.
    - EffBits << nominal bit-width → quantization is wasting precision.
    - EffBits < 0 → quantization is worse than no quantization.
    """
    x_orig = x_orig.astype(np.float64)
    error_mse = mse(x_orig, x_recon)
    signal_var = float(np.var(x_orig))
    if error_mse == 0.0:
        return float("inf")
    if signal_var == 0.0 or signal_var < error_mse:
        return 0.0
    return float(0.5 * np.log2(signal_var / error_mse))


# ── Combined evaluation ───────────────────────────────────────────────────────

def evaluate_all(x_orig: np.ndarray, x_recon: np.ndarray) -> dict:
    """Compute all 5 metrics. Returns dict with float values."""
    return {
        "mse":       mse(x_orig, x_recon),
        "snr_db":    snr_db(x_orig, x_recon),
        "kl_div":    kl_divergence(x_orig, x_recon),
        "max_ae":    max_absolute_error(x_orig, x_recon),
        "eff_bits":  effective_bits(x_orig, x_recon),
    }


# ── Aliases and FP16 baseline helpers ──────────────────────────────────────────

qsnr_db = snr_db  # alias used by fourbit code paths

def fp16_quantize(x: np.ndarray) -> np.ndarray:
    """Round-trip ``x`` through float16. FP16 is the upper-bound baseline
    for 4-bit quantization QSNR comparisons."""
    return np.asarray(x, dtype=np.float32).astype(np.float16).astype(np.float32)

def fp16_qsnr_db(x: np.ndarray) -> float:
    """QSNR (dB) of FP16-rounded ``x`` vs FP32 ``x``."""
    return snr_db(x, fp16_quantize(x))

def crest_factor(x: np.ndarray) -> float:
    """Peak-to-RMS ratio ``max(|x|) / rms(x)``. Returns 0 for all-zero tensor."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return 0.0
    peak = float(np.max(np.abs(x)))
    rms  = float(np.sqrt(np.mean(x * x)))
    return peak / rms if rms > 0 else 0.0

def kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis of ``x`` (``0`` for a Gaussian)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return 0.0
    mean = float(np.mean(x))
    dev  = x - mean
    var  = float(np.mean(dev * dev))
    if var <= 0.0:
        return 0.0
    return float(np.mean(dev ** 4) / (var ** 2)) - 3.0


def tensor_summary(x: np.ndarray) -> dict:
    """Compact stat bundle: std, max_abs, crest (peak/std), crest_rms
    (peak/rms including mean), kurtosis, n."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return {"std": 0.0, "max_abs": 0.0, "crest": 0.0,
                "crest_rms": 0.0, "kurtosis": 0.0, "n": 0}
    std  = float(np.std(x))
    peak = float(np.max(np.abs(x)))
    mean = float(np.mean(x))
    dev  = x - mean
    var  = float(np.mean(dev * dev))
    kurt = float(np.mean(dev ** 4) / (var ** 2)) - 3.0 if var > 0 else 0.0
    return {
        "std":       std,
        "max_abs":   peak,
        "crest":     peak / std if std > 0 else 0.0,
        "crest_rms": crest_factor(x),
        "kurtosis":  kurt,
        "n":         int(x.size),
    }


# ── Registries ────────────────────────────────────────────────────────────────
#
# ``METRIC_REGISTRY`` holds pairwise (reference, quantized) metric functions;
# ``TENSOR_STAT_REGISTRY`` holds single-tensor descriptive statistics.  Both
# are extendable at runtime via :func:`register_metric`, which lets
# experiments plug in custom columns without modifying this module.

METRIC_REGISTRY: Dict[str, Callable] = {
    "qsnr_db":      qsnr_db,
    "snr_db":       snr_db,
    "mse":          mse,
    "fp16_qsnr_db": lambda ref, _q: fp16_qsnr_db(ref),   # single-tensor shim
}

TENSOR_STAT_REGISTRY: Dict[str, Callable[[np.ndarray], float]] = {
    "crest":    crest_factor,
    "kurtosis": kurtosis,
}


def register_metric(name: str, fn: Callable, kind: str = "pair") -> None:
    """Register a metric under ``name``.

    ``kind="pair"`` adds to :data:`METRIC_REGISTRY` (signature
    ``fn(ref, quant) -> float``). ``kind="tensor_stat"`` adds to
    :data:`TENSOR_STAT_REGISTRY` (signature ``fn(tensor) -> float``).
    Any other ``kind`` raises :class:`ValueError`.
    """
    if kind == "pair":
        METRIC_REGISTRY[name] = fn
    elif kind == "tensor_stat":
        TENSOR_STAT_REGISTRY[name] = fn
    else:
        raise ValueError(f"kind must be 'pair' or 'tensor_stat', got {kind!r}")
