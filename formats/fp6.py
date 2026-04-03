"""FP6 E3M2: 6-bit floating-point format (1 sign + 3 exponent + 2 mantissa).

Exponent bias = 3.  Subnormals supported (exp=000).
Max representable positive value: (1 + 3/4) × 2^(7-3) = 1.75 × 16 = 28.0
Min positive normal: 1.0 × 2^(1-3) = 0.25
Min positive subnormal: 0.25 × 2^(-3) × (1/4) ≈ 0.0078125... actually:
  subnormal step = 2^(1-bias) / 4 = 2^(-2) / 4 = 0.0625

References: TC-FPx (Kim et al., 2023) — GPU Tensor Core support for FP6.
"""

import numpy as np


# Pre-compute all positive FP6 E3M2 levels for fast lookup
def _build_fp6_levels() -> np.ndarray:
    levels = []
    bias = 3
    for exp_code in range(8):   # 3-bit exponent: 0..7
        for mant_code in range(4):  # 2-bit mantissa: 0..3
            if exp_code == 0:
                # Subnormal: 0.mant × 2^(1-bias) = 0.mant × 2^(-2)
                val = (mant_code / 4.0) * (2 ** (1 - bias))
            else:
                # Normal: 1.mant × 2^(exp-bias)
                val = (1.0 + mant_code / 4.0) * (2 ** (exp_code - bias))
            levels.append(val)
    return np.array(levels, dtype=np.float32)


_FP6_POS_LEVELS = _build_fp6_levels()
_FP6_MAX = _FP6_POS_LEVELS[-1]  # 28.0


class FP6Format:
    """Per-tensor FP6 E3M2 quantization with absmax scaling.

    Symmetric: negative levels are mirror of positive.
    Scale maps max(|x|) → FP6_MAX.
    """
    name = "FP6"
    bits = 6

    def __init__(self):
        # Full symmetric level set (including 0)
        neg_levels = -_FP6_POS_LEVELS[1:][::-1]
        self._all_levels = np.concatenate([neg_levels, _FP6_POS_LEVELS])

    def quantize(self, x: np.ndarray, bits: int = 6) -> np.ndarray:
        x = x.astype(np.float32)
        absmax = np.max(np.abs(x))
        if absmax == 0:
            return np.zeros_like(x)

        scale = absmax / _FP6_MAX
        x_scaled = x / scale

        # Clip then nearest-neighbor
        x_clipped = np.clip(x_scaled, self._all_levels[0], self._all_levels[-1])
        dists = np.abs(x_clipped[..., None] - self._all_levels)  # (..., 63)
        idx = np.argmin(dists, axis=-1)
        q = self._all_levels[idx]

        self._scale = scale
        return q * scale

    def dequantize(self, q: np.ndarray) -> np.ndarray:
        return q.astype(np.float32)

    def encoding_overhead(self) -> dict:
        # 6-bit data, no block metadata → 6 bits/element
        # Hardware note: 6-bit requires bit-packing (3 elements per 2 bytes)
        return {
            "data_bits_per_element": 6,
            "metadata_bits_per_element": 0,
            "bandwidth_amplification": 1.0,
            "packing_overhead": "3 elements per 18 bits (requires bit-shift at decode)",
        }
