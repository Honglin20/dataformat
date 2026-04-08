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
    """Per-tensor NVFP4 quantization with a single POT scaling factor.

    Implementation note — two-level scaling in real hardware (NVIDIA Blackwell):
      Level 1 (modelled here): per-tensor POT scale mapped to the nearest
        power-of-two ≤ max(|x|)/6.0 — hardware-friendly (right-shift in HW).
      Level 2 (NOT modelled): in the actual Blackwell FP4 tensor-core spec,
        a per-group (16 elements) E8M0 scale is also stored, plus an outer
        per-tensor FP32 "master scale" for flexibility.
        The FP32 outer scale requires an FP32 multiplier in the decode path —
        hardware-UNFRIENDLY compared to pure-POT INT formats.

    Hardware cost implication: the FP32 outer scale adds one FP32 multiply
    per 16-element group decode, costing ~0.08× extra area vs INT4 array.
    This is captured in the area/energy model (plot_ppa_bubble.py).
    """
    name = "NVFP4"
    bits = 4

    def __init__(self):
        self._levels = _NVFP4_POS  # positive levels only; sign handled separately

    def quantize(self, x: np.ndarray, bits: int = 4) -> np.ndarray:
        x = x.astype(np.float32)
        # Per-tensor scale: map max(|x|) → max representable (6.0)
        max_val = np.max(np.abs(x))
        # OCP-aligned POT scale: 2^(floor(log2(max_val)) - floor(log2(6.0)))
        #                      = 2^(floor(log2(max_val)) - 2)
        #
        # Why NOT floor(log2(max_val/6.0)):
        #   log2(6.0) = 2.585 (non-integer). The naive formula gives
        #   floor(log2(max_val)) - 3 in ~58.5% of cases (scale 2× too small),
        #   causing max_val/scale ∈ [8,12) instead of [4,8) → saturates to 6.0.
        if max_val > 0:
            log2_max = int(np.floor(np.log2(float(max_val) + 1e-38)))
            scale = float(2.0 ** (log2_max - 2))   # floor(log2(6.0)) = 2
        else:
            scale = 1.0
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
        # 4 bits data; POT per-tensor scale (amortized to ~0 per element at large N).
        # Real Blackwell: 8-bit E8M0 per 16-element group + FP32 outer scale.
        # Group-level scale metadata ≈ 8/16 = 0.5 bits/element overhead.
        return {
            "data_bits_per_element": 4,
            "metadata_bits_per_element": 0.5,  # E8M0 per 16 elements (real spec)
            "bandwidth_amplification": 4.5 / 4,
            "hw_note": "FP32 outer scale requires FP32 multiplier in decode path",
        }
