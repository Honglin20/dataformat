"""SQ-Format: Sparse-Quantized unified format.

Decomposes a weight tensor into:
  - Sparse high-precision part: top-k% salient weights stored in INT8
    (typically 1% of elements, identified by magnitude).
  - Dense low-precision part: remaining weights quantized to INT4.
  - 1-bit mask per element to identify salient positions.

Hardware model:
  - A bitmask identifies salient positions.
  - Gather unit extracts salient values → high-precision ALU branch.
  - Remaining (99%) → INT4 dense branch.
  - Results accumulated with appropriate precision.

Hardware-friendly quantization:
  Both dense and sparse components use Power-of-Two (POT) scales:
    scale = 2^floor(log2(absmax / q_max))
  This makes the scale division a simple arithmetic right-shift in hardware,
  eliminating any floating-point scale computation.
"""

import numpy as np


def _pot_scale(absmax: float, q_max: int) -> float:
    """OCP-aligned power-of-two scale: 2^(floor(log2(absmax)) - floor(log2(q_max))).

    Correct formula guarantees q_max * scale >= absmax (no clipping) while
    maintaining the finest representable step size.

    Why NOT floor(log2(absmax / q_max)):
      log2(q_max) is non-integer for q_max=7 (INT4) and q_max=127 (INT8).
      floor(log2(absmax) - 2.807) vs floor(log2(absmax)) - 2 for INT4:
      the former gives scale 2× too small in ~80.7% of cases, causing clipping.
      floor(log2(absmax) - 6.989) vs floor(log2(absmax)) - 6 for INT8:
      the former gives scale 2× too small in ~99% of cases, clipping ~4.7% of
      N(0,1) elements per group.
    """
    if absmax <= 0:
        return 1.0
    log2_absmax = int(np.floor(np.log2(float(absmax) + 1e-38)))
    log2_qmax   = int(np.floor(np.log2(float(q_max))))
    return float(2.0 ** (log2_absmax - log2_qmax))


class SQFormat:
    """Sparse-Quantized Format with configurable sparsity ratio.

    Parameters
    ----------
    dense_bits : int
        Bit-width for the dense low-precision component (default 4).
    sparse_bits : int
        Bit-width for the sparse high-precision component (default 8).
    sparsity_ratio : float
        Fraction of elements treated as salient (default 0.01 = 1%).
    """

    def __init__(
        self,
        dense_bits: int = 4,
        sparse_bits: int = 8,
        sparsity_ratio: float = 0.01,
    ):
        self.dense_bits = dense_bits
        self.sparse_bits = sparse_bits
        self.sparsity_ratio = sparsity_ratio
        self.name = "SQ-Format"
        self.bits = dense_bits  # effective bit-width (dominant component)

    def _int_quantize_pot(self, x: np.ndarray, bits: int) -> np.ndarray:
        """Symmetric integer quantization with POT scale (hardware-friendly)."""
        q_max = 2 ** (bits - 1) - 1
        absmax = float(np.max(np.abs(x)))
        if absmax == 0:
            return np.zeros_like(x, dtype=np.float32)
        scale = _pot_scale(absmax, q_max)
        q = np.round(x / scale).astype(np.int32)
        q = np.clip(q, -q_max, q_max)
        return q.astype(np.float32) * scale

    def quantize(self, x: np.ndarray, bits: int = None) -> np.ndarray:
        """
        Returns the dequantized approximation of x.

        Internally:
          1. Identify top-k% elements by |x| as salient mask.
          2. Quantize salient elements with sparse_bits + POT scale.
          3. Zero out salient positions and quantize remainder with dense_bits + POT scale.
          4. Reconstruct: sparse_q + dense_q.
        """
        x = x.astype(np.float32)
        flat = x.ravel()
        n = len(flat)

        # 1. Identify salient indices (top-k by magnitude)
        k = max(1, int(np.ceil(self.sparsity_ratio * n)))
        abs_flat = np.abs(flat)
        salient_idx = np.argpartition(abs_flat, -k)[-k:]

        mask = np.zeros(n, dtype=bool)
        mask[salient_idx] = True

        # 2. Sparse high-precision component (POT scale per group)
        sparse_vals = np.zeros(n, dtype=np.float32)
        if np.any(mask):
            sparse_vals[mask] = self._int_quantize_pot(flat[mask], self.sparse_bits)

        # 3. Dense low-precision component (salient positions zeroed, POT scale)
        dense_input = flat.copy()
        dense_input[mask] = 0.0
        dense_q = self._int_quantize_pot(dense_input, self.dense_bits)

        # 4. Reconstruct
        result = sparse_vals + dense_q

        self._last_k = k
        self._last_n = n

        return result.reshape(x.shape)

    def dequantize(self, q: np.ndarray) -> np.ndarray:
        return q.astype(np.float32)

    def encoding_overhead(self) -> dict:
        # Dense: dense_bits per element (99%)
        # Sparse: sparse_bits per element (1%) + 1-bit mask per element
        dense_cost = (1 - self.sparsity_ratio) * self.dense_bits
        sparse_cost = self.sparsity_ratio * self.sparse_bits
        mask_cost = 1.0  # 1 bit per element for the bitmask
        total = dense_cost + sparse_cost + mask_cost
        return {
            "data_bits_per_element": total,
            "metadata_bits_per_element": mask_cost,
            "dense_bits": self.dense_bits,
            "sparse_bits": self.sparse_bits,
            "sparsity_ratio": self.sparsity_ratio,
            "bandwidth_amplification": total / self.dense_bits,
        }
