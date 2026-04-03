"""SQ-Format: Sparse-Quantized unified format.

Decomposes a weight tensor into:
  - Sparse high-precision part: top-k% salient weights stored in INT8/FP16
    (typically 1% of elements, identified by magnitude).
  - Dense low-precision part: remaining weights quantized to INT4.

This mirrors the AWQ insight that 1% of weights contribute disproportionately
to model quality. SQ-Format is a hardware-level format that natively supports
this decomposition via Gather/Scatter units.

Hardware model:
  - A bitmask identifies salient positions.
  - Gather unit extracts salient values → high-precision ALU branch.
  - Remaining (99%) → INT4 dense branch.
  - Results accumulated with appropriate precision.
"""

import numpy as np


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
        self.name = f"HAD+SQ" if sparsity_ratio == 0.01 else f"SQ(d{dense_bits}s{sparse_bits})"
        self.bits = dense_bits  # effective bit-width (dominant component)

    def _int_quantize(self, x: np.ndarray, bits: int) -> np.ndarray:
        """Symmetric uniform integer quantization."""
        q_max = 2 ** (bits - 1) - 1
        absmax = np.max(np.abs(x))
        if absmax == 0:
            return np.zeros_like(x)
        scale = absmax / q_max
        q = np.round(x / scale).astype(np.int32)
        q = np.clip(q, -q_max, q_max)
        return q.astype(np.float32) * scale

    def quantize(self, x: np.ndarray, bits: int = None) -> np.ndarray:
        """
        Returns the dequantized approximation of x.

        Internally:
          1. Identify top-k% elements by |x| as salient mask.
          2. Quantize salient elements with sparse_bits precision.
          3. Zero out salient positions and quantize remainder with dense_bits.
          4. Reconstruct: sparse_q + dense_q.
        """
        x = x.astype(np.float32)
        flat = x.ravel()
        n = len(flat)

        # 1. Identify salient indices
        k = max(1, int(np.ceil(self.sparsity_ratio * n)))
        abs_flat = np.abs(flat)
        salient_idx = np.argpartition(abs_flat, -k)[-k:]  # top-k indices

        # Salient mask
        mask = np.zeros(n, dtype=bool)
        mask[salient_idx] = True

        # 2. Sparse high-precision component
        sparse_vals = np.zeros(n, dtype=np.float32)
        sparse_vals[mask] = self._int_quantize(flat[mask], self.sparse_bits)

        # 3. Dense low-precision component (salient zeroed out)
        dense_input = flat.copy()
        dense_input[mask] = 0.0
        dense_q = self._int_quantize(dense_input, self.dense_bits)

        # 4. Reconstruct
        result = sparse_vals + dense_q

        # Store metadata for hardware modeling
        self._mask = mask
        self._k = k
        self._n = n

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

        # Gather/Scatter unit: additional hardware logic (not bits, modeled in PyRTL)
        return {
            "data_bits_per_element": total,
            "metadata_bits_per_element": mask_cost,
            "dense_bits": self.dense_bits,
            "sparse_bits": self.sparse_bits,
            "sparsity_ratio": self.sparsity_ratio,
            "bandwidth_amplification": total / self.dense_bits,
        }
