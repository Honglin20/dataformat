"""MXFP4 and MXFP8: OCP Microscaling Floating-Point formats.

OCP MX spec (Open Compute Project):
  - Block size = 32 elements share one 8-bit E8M0 scale factor (biased exponent only).
  - MXFP4: E2M1 element format (same encoding as NVFP4 E2M1).
  - MXFP8: E4M3 or E5M2 element format.

The shared scale is stored as an 8-bit exponent (E8M0, bias=127):
  scale_value = 2^(scale_byte - 127)

This yields:  x ≈ scale_value × element_fp_value
"""

import numpy as np
from config import MX_BLOCK_SIZE

# MXFP4 E2M1 positive levels (same as NVFP4)
_E2M1_POS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)

# MXFP8 E4M3 positive levels (143 finite values, we use clamp+round approach)
_FP8_E4M3_MAX = 448.0
_FP8_E5M2_MAX = 57344.0


def _fp8_e4m3_quantize_scalar(v: float) -> float:
    """Simulate FP8 E4M3 (4 exp bits, 3 mantissa bits, bias=7, no inf)."""
    if v == 0.0:
        return 0.0
    sign = -1.0 if v < 0 else 1.0
    v_abs = abs(v)
    v_abs = min(v_abs, _FP8_E4M3_MAX)
    # Find biased exponent
    exp = int(np.floor(np.log2(v_abs + 1e-38)))
    exp_biased = exp + 7  # bias=7
    exp_biased = max(0, min(15, exp_biased))
    # Compute mantissa (3 bits → 8 levels per exponent)
    if exp_biased == 0:
        # Subnormal: value = (m/8) × 2^(1-7) = m × 2^(-9). Clamp m to [0,7].
        mant_int = min(7, round(v_abs / 2**(-6) * 8))
        mant = mant_int / 8 * 2**(-6)
    else:
        step = 2 ** (exp_biased - 7) / 8
        mant_int = round((v_abs - 2 ** (exp_biased - 7)) / step)
        mant_int = max(0, min(7, mant_int))
        mant = 2 ** (exp_biased - 7) + mant_int * step
    return sign * mant


_fp8_e4m3_vec = np.vectorize(_fp8_e4m3_quantize_scalar)


class MXFPFormat:
    """Block-scaled microscaling FP format (MXFP4 or MXFP8).

    Parameters
    ----------
    element_bits : int
        4 for MXFP4 (E2M1), 8 for MXFP8 (E4M3).
    block_size : int
        Number of elements sharing one scale (default 32, OCP standard).
    """

    def __init__(self, element_bits: int = 4, block_size: int = MX_BLOCK_SIZE):
        assert element_bits in (4, 8), "element_bits must be 4 or 8"
        self.element_bits = element_bits
        self.block_size = block_size
        self.name = f"MXFP{element_bits}"
        self.bits = element_bits

    def _quantize_block(self, block: np.ndarray) -> np.ndarray:
        """Quantize one block of elements with shared E8M0 scale."""
        max_abs = np.max(np.abs(block))
        if max_abs == 0:
            return np.zeros_like(block)

        # Compute E8M0 scale per OCP MX spec:
        #   scale = 2^(floor(log2(max_abs)) - floor(log2(max_elem_val)))
        #
        # This is NOT floor(log2(max_abs / max_elem_val)).
        # Since log2(6.0)=2.585 and log2(448)=8.807 are non-integer,
        # the naive formula gives a scale 2× too small in ~58.5% (FP4) and
        # ~80.7% (FP8) of blocks respectively, causing severe saturation.
        if self.element_bits == 4:
            max_elem_val = 6.0
        else:  # 8-bit E4M3
            max_elem_val = _FP8_E4M3_MAX

        log2_max      = int(np.floor(np.log2(max_abs + 1e-38)))
        log2_elem_max = int(np.floor(np.log2(max_elem_val)))
        scale = 2.0 ** (log2_max - log2_elem_max)

        # Scale and quantize elements
        x_scaled = block / scale

        if self.element_bits == 4:
            sign = np.sign(x_scaled)
            sign[sign == 0] = 1.0
            x_abs = np.clip(np.abs(x_scaled), 0, 6.0)
            dists = np.abs(x_abs[..., None] - _E2M1_POS)
            idx = np.argmin(dists, axis=-1)
            q = sign * _E2M1_POS[idx]
        else:
            q = _fp8_e4m3_vec(x_scaled)

        return q * scale  # dequantized value

    def _quantize_1d(self, flat: np.ndarray) -> np.ndarray:
        """Quantize a 1-D float32 array block-by-block."""
        n = len(flat)
        out = np.zeros(n, dtype=np.float32)
        pad = (-n) % self.block_size
        padded = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
        for i in range(0, len(padded), self.block_size):
            block = padded[i:i + self.block_size]
            out_block = self._quantize_block(block)
            end = min(i + self.block_size, n)
            out[i:end] = out_block[:end - i]
        return out

    def quantize(self, x: np.ndarray, bits: int = None) -> np.ndarray:
        x = x.astype(np.float32)
        n = x.size
        if x.ndim == 2:
            # Apply block quantization independently per row (output channel).
            # Flattening across rows would cause cross-channel contamination.
            out = np.stack([self._quantize_1d(row) for row in x])
        else:
            out = self._quantize_1d(x.ravel()).reshape(x.shape)
        # Store bandwidth overhead info
        n_blocks = int(np.ceil(n / self.block_size))
        self._metadata_bits = n_blocks * 8  # 8-bit E8M0 per block
        self._n_elements = n
        return out

    def dequantize(self, q: np.ndarray) -> np.ndarray:
        return q.astype(np.float32)

    def encoding_overhead(self) -> dict:
        # Scale metadata: 8 bits per block of 32 elements = 0.25 bits/element
        meta = 8 / self.block_size
        return {
            "data_bits_per_element": self.element_bits,
            "metadata_bits_per_element": meta,
            "bandwidth_amplification": (self.element_bits + meta) / self.element_bits,
        }
