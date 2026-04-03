"""NF4: NormalFloat4 — information-theoretically optimal 4-bit format for
normally distributed weights (from QLoRA, Dettmers et al. 2023).

16 quantization levels are placed such that each quantile of N(0,1) is equally
likely to be selected → minimizes expected quantization error for Gaussian data.

Quantization is per-tensor with a single absmax scale factor.
"""

import numpy as np
from config import NF4_LEVELS


class NF4Format:
    """Per-tensor NormalFloat4 quantization.

    Dequantized values ∈ NF4_LEVELS scaled by absmax of the input tensor.
    """
    name = "NF4"
    bits = 4

    def __init__(self):
        self._levels = NF4_LEVELS.copy()  # shape (16,)

    def quantize(self, x: np.ndarray, bits: int = 4) -> np.ndarray:
        x = x.astype(np.float32)
        absmax = np.max(np.abs(x))
        if absmax == 0:
            return np.zeros_like(x)

        # Normalize to [-1, 1]
        x_norm = x / absmax

        # Nearest-neighbor into NF4 levels
        dists = np.abs(x_norm[..., None] - self._levels)  # (..., 16)
        idx = np.argmin(dists, axis=-1)                    # (...)
        q_norm = self._levels[idx]

        self._absmax = absmax
        return q_norm * absmax  # dequantized in original scale

    def dequantize(self, q: np.ndarray) -> np.ndarray:
        return q.astype(np.float32)

    def encoding_overhead(self) -> dict:
        # 4 bits data + 1 FP32 scale per tensor (negligible metadata per element)
        return {
            "data_bits_per_element": 4,
            "metadata_bits_per_element": 0,   # per-tensor scale amortized to ~0
            "bandwidth_amplification": 1.0,
        }
