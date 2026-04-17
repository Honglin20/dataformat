"""Online statistics for per-layer tensor distribution analysis.

Three classes:
  WelfordStats      — incremental mean, std, skewness, excess kurtosis, min, max
  RunningHistogram  — fixed 256-bin histogram, range set by first batch
  QuantStats        — incremental MSE, SNR, EffBits, MaxAE, MARE, saturation rate

Moment combination uses the parallel Pebay (2008) / Chan et al. (1979) formulas
for M2, M3, M4 so that all statistics are computed in a single online pass with
no stored samples.
"""
from __future__ import annotations
import numpy as np


class WelfordStats:
    """Parallel Welford algorithm for mean, variance, skewness, excess kurtosis.

    Stores unnormalized central moments M2 = Σ(xᵢ-μ)², M3 = Σ(xᵢ-μ)³,
    M4 = Σ(xᵢ-μ)⁴ and combines them across batches using exact parallel
    formulas (Pebay 2008, equations 3.1–3.2).

    Skewness      = M3 / (n * σ³)
    ExcessKurtosis = M4 / (n * σ⁴) - 3     (Gaussian → 0, Laplace → 3)
    """

    def __init__(self):
        self._n: int = 0
        self._mean: float = 0.0
        self._M2: float = 0.0   # Σ(xᵢ - μ)²
        self._M3: float = 0.0   # Σ(xᵢ - μ)³
        self._M4: float = 0.0   # Σ(xᵢ - μ)⁴
        self._abs_max: float = 0.0
        self._min: float = float("inf")
        self._max: float = float("-inf")

    def update(self, x: np.ndarray) -> None:
        x = x.astype(np.float64).ravel()
        n_b = len(x)
        if n_b == 0:
            return

        mean_b = float(np.mean(x))
        dev_b = x - mean_b
        M2_b = float(np.sum(dev_b ** 2))
        M3_b = float(np.sum(dev_b ** 3))
        M4_b = float(np.sum(dev_b ** 4))

        n_a = self._n
        n_new = n_a + n_b
        delta = mean_b - self._mean
        delta2 = delta * delta

        # Pebay (2008) parallel update — must update in M4→M3→M2 order so each
        # formula reads only the *old* values of the lower moments.
        self._M4 = (
            self._M4 + M4_b
            + delta2 ** 2 * n_a * n_b * (n_a ** 2 - n_a * n_b + n_b ** 2) / n_new ** 3
            + 6.0 * delta2 * (n_a ** 2 * M2_b + n_b ** 2 * self._M2) / n_new ** 2
            + 4.0 * delta * (n_a * M3_b - n_b * self._M3) / n_new
        )
        self._M3 = (
            self._M3 + M3_b
            + delta ** 3 * n_a * n_b * (n_a - n_b) / n_new ** 2
            + 3.0 * delta * (n_a * M2_b - n_b * self._M2) / n_new
        )
        self._M2 = self._M2 + M2_b + delta2 * n_a * n_b / n_new
        self._mean = (self._mean * n_a + mean_b * n_b) / n_new
        self._n = n_new
        self._abs_max = max(self._abs_max, float(np.max(np.abs(x))))
        self._min = min(self._min, float(np.min(x)))
        self._max = max(self._max, float(np.max(x)))

    def finalize(self) -> dict:
        if self._n == 0:
            raise RuntimeError("No data recorded — call update() before finalize().")
        variance = self._M2 / self._n if self._n > 0 else 0.0
        std = float(np.sqrt(max(variance, 0.0)))
        skewness = 0.0
        kurtosis = 0.0  # excess kurtosis; Gaussian = 0
        if variance > 1e-30 and self._n >= 4:
            skewness = float(self._M3 / self._n / std ** 3)
            kurtosis = float(self._M4 / self._n / variance ** 2) - 3.0
        return {
            "mean": self._mean,
            "std": std,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "abs_max": self._abs_max,
            "min": self._min,
            "max": self._max,
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
    """Incremental quantization error statistics.

    Metrics:
      mse             — mean squared error
      snr_db          — 10 log₁₀(signal_var / mse)  (can be negative)
      eff_bits        — 0.5 log₂(signal_var / mse)
      max_ae          — maximum absolute error across all updates
      mare            — mean absolute relative error: mean(|err| / (|x| + 1e-8))
      saturation_rate — fraction of elements in provided saturation_mask
    """

    def __init__(self):
        self._sum_sq_err: float = 0.0
        self._sum_sq_orig: float = 0.0
        self._sum_orig: float = 0.0
        self._sum_abs_rel_err: float = 0.0
        self._n: int = 0
        self._n_saturated: int = 0
        self._max_ae: float = 0.0

    def update(
        self,
        x_orig: np.ndarray,
        x_quant: np.ndarray,
        saturation_mask: np.ndarray | None = None,
    ) -> None:
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
        self._sum_abs_rel_err += float(np.sum(np.abs(err) / (np.abs(x_orig) + 1e-8)))
        self._n += len(x_orig)
        self._max_ae = max(self._max_ae, float(np.max(np.abs(err))))
        if saturation_mask is not None:
            self._n_saturated += int(np.sum(saturation_mask.ravel()))

    def finalize(self) -> dict:
        nan = float("nan")
        if self._n == 0:
            return {
                "mse": nan, "snr_db": nan, "eff_bits": nan,
                "max_ae": nan, "mare": nan, "saturation_rate": nan,
            }
        mse = self._sum_sq_err / self._n
        mare = self._sum_abs_rel_err / self._n
        saturation_rate = self._n_saturated / self._n
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
        return {
            "mse": mse,
            "snr_db": snr_db,
            "eff_bits": eff_bits,
            "max_ae": self._max_ae,
            "mare": mare,
            "saturation_rate": saturation_rate,
        }
