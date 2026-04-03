"""NVFP4: NVIDIA Blackwell E2M1 4-bit floating-point format.

Structure: [sign(1)] [exp(2)] [mantissa(1)]
Exponent bias = 1 (so exp codes 00=0, 01=1, 10=2, 11=3 → biased exp = -1,0,1,2)
Special: 0000 = 0.0, no infinities/NaN (finite range only).

Representable positive values (from NVIDIA Blackwell spec):
  0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
"""

import numpy as np

# All 16 representable values of E2M1 (NVFP4)
# Sign bit controls negation. Positive half:
_NVFP4_POS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
_NVFP4_ALL = np.concatenate([-_NVFP4_POS[::-1], _NVFP4_POS])  # symmetric around 0


class NVFP4Format:
    """Per-tensor NVFP4 quantization with a single FP32 scaling factor."""
    name = "NVFP4"
    bits = 4

    def __init__(self):
        self._levels = _NVFP4_POS  # positive levels only; sign handled separately

    def quantize(self, x: np.ndarray, bits: int = 4) -> np.ndarray:
        x = x.astype(np.float32)
        # Per-tensor scale: map max(|x|) → max representable (6.0)
        max_val = np.max(np.abs(x))
        scale = max_val / 6.0 if max_val > 0 else 1.0
        x_scaled = x / scale

        sign = np.sign(x_scaled)
        sign[sign == 0] = 1.0
        x_abs = np.abs(x_scaled)

        # Nearest-neighbor quantization into positive levels
        dists = np.abs(x_abs[..., None] - self._levels)  # (..., 8)
        idx = np.argmin(dists, axis=-1)                   # (...,)
        q = sign * self._levels[idx]

        # Pack metadata alongside quantized values
        self._scale = scale
        return q * scale   # return in original scale (dequant included for simplicity)

    def dequantize(self, q: np.ndarray) -> np.ndarray:
        return q.astype(np.float32)

    def encoding_overhead(self) -> dict:
        # 4 bits data, 32 bits scale per tensor (amortized → ~0 per element at large N)
        return {"data_bits_per_element": 4, "metadata_bits_per_element": 0}
