"""5-metric quantization quality evaluation suite.

Metrics:
  1. MSE      — Mean Squared Error (primary distortion measure)
  2. SNR      — Signal-to-Noise Ratio in dB
  3. KL       — KL divergence (distribution shape preservation)
  4. Max-AE   — Maximum Absolute Error (worst-case single-element error)
  5. EffBits  — Effective bit-width via rate-distortion theory
"""

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
