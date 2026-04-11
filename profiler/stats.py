"""Online statistics for per-layer tensor distribution analysis.

Three classes:
  WelfordStats      — incremental mean, std, abs_max across batches
  RunningHistogram  — fixed 256-bin histogram, range set by first batch
  QuantStats        — incremental MSE, SNR, EffBits, max_ae for quant error
"""
from __future__ import annotations
import numpy as np


class WelfordStats:
    """Welford's online algorithm for mean and standard deviation.

    Supports batch updates (parallel Welford combination).
    Tracks absolute maximum value (not quantization error).
    """

    def __init__(self):
        self._n: int = 0
        self._mean: float = 0.0
        self._M2: float = 0.0
        self._abs_max: float = 0.0

    def update(self, x: np.ndarray) -> None:
        x = x.astype(np.float64).ravel()
        n_b = len(x)
        if n_b == 0:
            return
        mean_b = float(np.mean(x))
        M2_b = float(np.sum((x - mean_b) ** 2))
        n_new = self._n + n_b
        delta = mean_b - self._mean
        self._M2 = self._M2 + M2_b + delta ** 2 * self._n * n_b / n_new
        self._mean = (self._mean * self._n + mean_b * n_b) / n_new
        self._n = n_new
        self._abs_max = max(self._abs_max, float(np.max(np.abs(x))))

    def finalize(self) -> dict:
        if self._n == 0:
            raise RuntimeError("No data recorded — call update() before finalize().")
        return {
            "mean": self._mean,
            "std": float(np.sqrt(self._M2 / self._n)) if self._n > 1 else 0.0,
            "abs_max": self._abs_max,
            "n_elements": self._n,
        }


class RunningHistogram:
    """Fixed 256-bin histogram accumulated across batches.

    Range is determined by the first call to update(). Subsequent batches
    use the same bin edges; out-of-range values are counted as outliers
    but clamped into the edge bins for the histogram counts.
    """

    N_BINS: int = 256

    def __init__(self):
        self._counts = None
        self._edges = None
        self._n_total: int = 0
        self._n_outliers: int = 0

    def update(self, x: np.ndarray) -> None:
        x = x.astype(np.float64).ravel()
        if len(x) == 0:
            return
        if self._edges is None:
            lo, hi = float(np.min(x)), float(np.max(x))
            if lo == hi:
                hi = lo + 1e-10
            self._edges = np.linspace(lo, hi, self.N_BINS + 1)
            self._counts = np.zeros(self.N_BINS, dtype=np.int64)
        lo, hi = self._edges[0], self._edges[-1]
        self._n_outliers += int(np.sum((x < lo) | (x > hi)))
        self._n_total += len(x)
        clipped = np.clip(x, lo, hi)
        counts, _ = np.histogram(clipped, bins=self._edges)
        self._counts += counts

    def finalize(self) -> dict:
        if self._counts is None:
            return {"outlier_ratio": 0.0, "hist_counts": [], "hist_edges": []}
        return {
            "outlier_ratio": self._n_outliers / max(self._n_total, 1),
            "hist_counts": self._counts.tolist(),
            "hist_edges": self._edges.tolist(),
        }


class QuantStats:
    """Incremental quantization error statistics."""

    def __init__(self):
        self._sum_sq_err: float = 0.0
        self._sum_sq_orig: float = 0.0
        self._sum_orig: float = 0.0
        self._n: int = 0
        self._max_ae: float = 0.0

    def update(self, x_orig: np.ndarray, x_quant: np.ndarray) -> None:
        x_orig = x_orig.astype(np.float64).ravel()
        x_quant = x_quant.astype(np.float64).ravel()
        if x_orig.shape != x_quant.shape:
            raise ValueError(
                f"x_orig and x_quant must have the same number of elements after ravel, "
                f"got {x_orig.shape} vs {x_quant.shape}"
            )
        err = x_orig - x_quant
        self._sum_sq_err += float(np.sum(err ** 2))
        self._sum_sq_orig += float(np.sum(x_orig ** 2))
        self._sum_orig += float(np.sum(x_orig))
        self._n += len(x_orig)
        self._max_ae = max(self._max_ae, float(np.max(np.abs(err))))

    def finalize(self) -> dict:
        if self._n == 0:
            return {"mse": float("nan"), "snr_db": float("nan"), "eff_bits": float("nan"), "max_ae": float("nan")}
        mse = self._sum_sq_err / self._n
        mean = self._sum_orig / self._n
        var = max(self._sum_sq_orig / self._n - mean ** 2, 0.0)
        if mse == 0.0:
            snr_db = float("inf")
            eff_bits = float("inf")
        elif var <= 0.0:
            snr_db = 0.0
            eff_bits = 0.0
        else:
            snr_db = float(10.0 * np.log10(var / mse))
            eff_bits = float(0.5 * np.log2(var / mse))
        return {"mse": mse, "snr_db": snr_db, "eff_bits": eff_bits, "max_ae": self._max_ae}
