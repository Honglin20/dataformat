"""MXINT4 and MXINT8: OCP Microscaling Integer formats.

Block of 32 elements share one FP32 (or scaled-integer) scale factor.
Element values are standard two's-complement integers.
Scale stored as E8M0 (8-bit power-of-two) per OCP spec.
"""

import numpy as np
from config import MX_BLOCK_SIZE


class MXINTFormat:
    """Block-scaled integer quantization (MXINT4 or MXINT8).

    Parameters
    ----------
    element_bits : int
        4 for MXINT4, 8 for MXINT8.
    block_size : int
        Elements sharing one scale factor (OCP default = 32).
    """

    def __init__(self, element_bits: int = 4, block_size: int = MX_BLOCK_SIZE):
        assert element_bits in (4, 8), "element_bits must be 4 or 8"
        self.element_bits = element_bits
        self.block_size = block_size
        self.name = f"MXINT{element_bits}"
        self.bits = element_bits

        # Symmetric signed integer range [-2^(b-1)+1, 2^(b-1)-1]
        self._q_max = 2 ** (element_bits - 1) - 1
        self._q_min = -(2 ** (element_bits - 1) - 1)  # symmetric (no -128 for INT8)

    def _quantize_block(self, block: np.ndarray) -> np.ndarray:
        max_abs = np.max(np.abs(block))
        if max_abs == 0:
            return np.zeros_like(block)

        # OCP MX E8M0 scale: 2^(floor(log2(max_abs)) - (bits-2))
        # Equivalently: floor(log2(max_abs)) - floor(log2(q_max)), since
        #   floor(log2(2^(bits-1)-1)) = bits-2  for bits ∈ {4,8}.
        #
        # NOTE: this does NOT guarantee no clipping for INT4.  When max_abs ∈
        #   (q_max·scale, 2·q_max·scale) = (1.75·2^k, 2^(k+1)) the block max
        #   exceeds q_max·scale and gets clipped (~12.5% of the per-octave range).
        #   For INT8 the clip window is only ~1.6% (127/128 ≈ 0.992 coverage).
        #   Both INT-PerChannel and MXINT use the same formula, so comparisons
        #   between them are fair despite the systematic underflow.
        log2_max = np.floor(np.log2(max_abs + 1e-38))
        scale = 2.0 ** (log2_max - (self.element_bits - 2))

        # Quantize to integers
        x_scaled = block / scale
        q_int = np.round(x_scaled).astype(np.int32)
        q_int = np.clip(q_int, self._q_min, self._q_max)

        # Dequantize
        return q_int.astype(np.float32) * scale

    def quantize(self, x: np.ndarray, bits: int = None) -> np.ndarray:
        x = x.astype(np.float32)
        flat = x.ravel()
        n = len(flat)
        out = np.zeros_like(flat)

        pad = (-n) % self.block_size
        padded = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])

        for i in range(0, len(padded), self.block_size):
            block = padded[i:i + self.block_size]
            out_block = self._quantize_block(block)
            start = i
            end = min(i + self.block_size, n)
            out[start:end] = out_block[:end - start]

        n_blocks = int(np.ceil(n / self.block_size))
        self._metadata_bits = n_blocks * 8
        self._n_elements = n

        return out.reshape(x.shape)

    def dequantize(self, q: np.ndarray) -> np.ndarray:
        return q.astype(np.float32)

    def encoding_overhead(self) -> dict:
        meta = 8 / self.block_size  # 8-bit E8M0 per 32 elements = 0.25 bits/element
        return {
            "data_bits_per_element": self.element_bits,
            "metadata_bits_per_element": meta,
            "bandwidth_amplification": (self.element_bits + meta) / self.element_bits,
        }
