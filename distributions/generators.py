"""7-type synthetic distribution generators for outlier robustness testing.

All generators return flat np.ndarray of shape (n_samples,) with dtype float32.
Each generator also returns a metadata dict describing the distribution parameters.
"""

import numpy as np
from scipy.stats import t as student_t, lognorm


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# ────────────────────────────────────────────────────────────────────────────
# 1. Standard Gaussian
# ────────────────────────────────────────────────────────────────────────────

def gaussian(n: int = 4096, sigma: float = 1.0, seed: int = 42) -> tuple:
    """N(0, σ²) — ideal quantization baseline."""
    rng = _rng(seed)
    x = rng.normal(0.0, sigma, size=n).astype(np.float32)
    meta = {"type": "gaussian", "sigma": sigma}
    return x, meta


# ────────────────────────────────────────────────────────────────────────────
# 2. Laplace
# ────────────────────────────────────────────────────────────────────────────

def laplace(n: int = 4096, b: float = 1.0, seed: int = 42) -> tuple:
    """Laplace(0, b) — models FFN weight distributions (heavier tails than Gaussian)."""
    rng = _rng(seed)
    x = rng.laplace(0.0, b, size=n).astype(np.float32)
    meta = {"type": "laplace", "b": b}
    return x, meta


# ────────────────────────────────────────────────────────────────────────────
# 3. Student-t
# ────────────────────────────────────────────────────────────────────────────

def student_t_dist(n: int = 4096, nu: float = 5.0, seed: int = 42) -> tuple:
    """Student-t(ν) — models typical Transformer activation long tails.

    Smaller ν → heavier tails (ν=3: very heavy, ν=10: near-Gaussian).
    """
    rng = _rng(seed)
    # scipy.stats.t.rvs uses numpy internally
    np.random.seed(seed)
    x = student_t.rvs(df=nu, size=n).astype(np.float32)
    meta = {"type": "student_t", "nu": nu}
    return x, meta


# ────────────────────────────────────────────────────────────────────────────
# 4. Bimodal
# ────────────────────────────────────────────────────────────────────────────

def bimodal(
    n: int = 4096,
    mu1: float = -3.0,
    mu2: float = 3.0,
    sigma: float = 0.5,
    seed: int = 42,
) -> tuple:
    """Mixture of two Gaussians — models Attention Softmax output or layernorm'd activations."""
    rng = _rng(seed)
    half = n // 2
    x1 = rng.normal(mu1, sigma, size=half).astype(np.float32)
    x2 = rng.normal(mu2, sigma, size=n - half).astype(np.float32)
    x = np.concatenate([x1, x2])
    rng.shuffle(x)
    meta = {"type": "bimodal", "mu1": mu1, "mu2": mu2, "sigma": sigma}
    return x, meta


# ────────────────────────────────────────────────────────────────────────────
# 5. Mixture Gaussian with systematic channel outliers
# ────────────────────────────────────────────────────────────────────────────

def channel_outliers(
    n: int = 4096,
    outlier_ratio: float = 0.01,
    outlier_sigma: float = 50.0,
    seed: int = 42,
) -> tuple:
    """LLM systematic channel outliers: fixed channels always have large magnitude.

    Models the key observation from LLM.int8(): outliers are NOT random —
    they consistently appear in the same channels across all tokens.

    Parameters
    ----------
    n : int
        Total tensor size.
    outlier_ratio : float
        Fraction of channels (elements) that are salient/outlier channels.
    outlier_sigma : float
        Standard deviation of outlier channels (e.g., 30, 50, 100).
    """
    rng = _rng(seed)
    x = rng.normal(0.0, 1.0, size=n).astype(np.float32)

    # Deterministically select outlier indices (always the same channels)
    n_outliers = max(1, int(np.ceil(outlier_ratio * n)))
    # Use evenly-spaced indices to simulate fixed-channel outliers
    outlier_idx = np.linspace(0, n - 1, n_outliers, dtype=int)
    x[outlier_idx] = rng.normal(0.0, outlier_sigma, size=n_outliers).astype(np.float32)

    meta = {
        "type": "channel_outliers",
        "outlier_ratio": outlier_ratio,
        "outlier_sigma": outlier_sigma,
        "n_outlier_channels": n_outliers,
    }
    return x, meta


# ────────────────────────────────────────────────────────────────────────────
# 6. Spiky outliers (random point injection)
# ────────────────────────────────────────────────────────────────────────────

def spiky_outliers(
    n: int = 4096,
    spike_ratio: float = 0.001,
    spike_multiplier: float = 50.0,
    seed: int = 42,
) -> tuple:
    """Random spiky outliers injected into N(0,1) background.

    Models the AWQ observation: a tiny fraction of weights have extremely
    large magnitudes, causing catastrophic quantization error.

    Parameters
    ----------
    spike_ratio : float
        Fraction of elements that are spikes (default 0.1%).
    spike_multiplier : float
        Multiplier relative to the background std (10×, 50×, 100×).
    """
    rng = _rng(seed)
    x = rng.normal(0.0, 1.0, size=n).astype(np.float32)

    n_spikes = max(1, int(np.ceil(spike_ratio * n)))
    spike_idx = rng.choice(n, size=n_spikes, replace=False)
    signs = rng.choice([-1.0, 1.0], size=n_spikes).astype(np.float32)
    x[spike_idx] = signs * spike_multiplier

    meta = {
        "type": "spiky_outliers",
        "spike_ratio": spike_ratio,
        "spike_multiplier": spike_multiplier,
        "n_spikes": n_spikes,
    }
    return x, meta


# ────────────────────────────────────────────────────────────────────────────
# 7. Log-Normal
# ────────────────────────────────────────────────────────────────────────────

def log_normal(
    n: int = 4096,
    mu: float = 0.0,
    sigma: float = 1.0,
    seed: int = 42,
) -> tuple:
    """Log-Normal distribution — models activation distributions after SiLU/GELU.

    Signed version: random sign applied to positive log-normal draws.
    """
    rng = _rng(seed)
    x_pos = rng.lognormal(mean=mu, sigma=sigma, size=n).astype(np.float32)
    signs = rng.choice([-1.0, 1.0], size=n).astype(np.float32)
    x = signs * x_pos
    meta = {"type": "log_normal", "mu": mu, "sigma": sigma}
    return x, meta


# ────────────────────────────────────────────────────────────────────────────
# Master sweep: generate all distributions for a full experiment
# ────────────────────────────────────────────────────────────────────────────

def generate_all_distributions(n: int = 4096, seed: int = 42) -> list:
    """Return list of (name, tensor, metadata) tuples for all 7 distribution types."""
    dists = []

    # 1. Gaussian variants
    dists.append(("Gaussian(σ=1)", *gaussian(n, sigma=1.0, seed=seed)))

    # 2. Laplace variants
    for b in [0.5, 1.0, 2.0]:
        dists.append((f"Laplace(b={b})", *laplace(n, b=b, seed=seed)))

    # 3. Student-t variants
    for nu in [3, 5, 10]:
        dists.append((f"Student-t(ν={nu})", *student_t_dist(n, nu=nu, seed=seed)))

    # 4. Bimodal
    dists.append(("Bimodal(μ=±3)", *bimodal(n, mu1=-3.0, mu2=3.0, sigma=0.5, seed=seed)))

    # 5. Channel outliers (systematic)
    for sig in [30.0, 50.0, 100.0]:
        dists.append((
            f"ChannelOutlier(σ={int(sig)})",
            *channel_outliers(n, outlier_ratio=0.01, outlier_sigma=sig, seed=seed)
        ))

    # 6. Spiky outliers
    for mult in [10.0, 50.0, 100.0]:
        dists.append((
            f"Spiky({int(mult)}×)",
            *spiky_outliers(n, spike_ratio=0.001, spike_multiplier=mult, seed=seed)
        ))

    # 7. Log-Normal variants
    for sig in [1.0, 2.0]:
        dists.append((f"LogNormal(σ={sig})", *log_normal(n, mu=0.0, sigma=sig, seed=seed)))

    return dists
