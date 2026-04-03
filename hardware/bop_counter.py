"""Bit-Operations (BOPs) counter for theoretical compute cost analysis.

BOPs = sum over all operations of (bit_width_a × bit_width_b)
     for multiply-accumulate operations.

This metric is used in the NLP/quantization literature as a hardware-
agnostic proxy for compute cost that captures the benefit of reduced
precision more accurately than FLOPs.

For a matrix multiply A[M×K] × B[K×N]:
  BOPs = M × K × N × bits_a × bits_b

For transform pre/post-processing:
  HAD: N × log2(N) additions @ bits_width → BOPs = N × log2(N) × bits²
  (additions use bits² BOPs as a rough approximation of integer adder area)
"""

import math
import numpy as np


class BopCounter:
    """Accumulates BOPs across forward-pass operations."""

    def __init__(self):
        self._bops = 0.0
        self._log = []

    def reset(self):
        self._bops = 0.0
        self._log = []

    @property
    def total_bops(self) -> float:
        return self._bops

    # ── Core operations ──────────────────────────────────────────────────────

    def matmul(
        self,
        M: int,
        K: int,
        N: int,
        bits_a: int,
        bits_b: int,
        label: str = "matmul",
    ) -> float:
        """Register BOPs for matrix multiply A[M×K] @ B[K×N]."""
        bops = M * K * N * bits_a * bits_b
        self._bops += bops
        self._log.append({"op": label, "bops": bops, "bits_a": bits_a, "bits_b": bits_b})
        return bops

    def hadamard(self, N: int, bits: int, label: str = "FWHT") -> float:
        """Register BOPs for N-point FWHT.

        N × log2(N) butterfly add/sub operations, each on bits-wide integers.
        BOPs = N × log2(N) × bits² (each b-bit add ≈ b² bit-ops by convention).
        """
        n_ops = N * math.log2(max(N, 2))
        bops = n_ops * bits * bits
        self._bops += bops
        self._log.append({"op": label, "bops": bops, "bits": bits, "n": N})
        return bops

    def smoothquant_scale(self, C: int, bits: int = 32, label: str = "SQ_scale") -> float:
        """Register BOPs for per-channel scale multiply (SmoothQuant).

        C multiplications, treated as 32-bit FP multiply (scale is FP32).
        """
        bops = C * bits * bits
        self._bops += bops
        self._log.append({"op": label, "bops": bops, "bits": bits, "C": C})
        return bops

    def random_rotation(self, N: int, bits: int = 8, label: str = "RandRot") -> float:
        """Register BOPs for dense N×N orthogonal matrix-vector multiply."""
        bops = N * N * bits * bits   # N² MACs
        self._bops += bops
        self._log.append({"op": label, "bops": bops, "bits": bits, "n": N})
        return bops

    def turbo_sign_flip(self, N: int, label: str = "TurboSign") -> float:
        """Register BOPs for ±1 diagonal scaling (essentially XOR ops)."""
        bops = N * 1   # 1-bit XOR per element
        self._bops += bops
        self._log.append({"op": label, "bops": bops, "n": N})
        return bops

    def mx_scale_apply(
        self, N: int, element_bits: int, block_size: int = 32, label: str = "MX_scale"
    ) -> float:
        """Register BOPs for MX block scale application.

        Each block: 1 scale read (8-bit) + block_size shift operations.
        Shift: 1-bit × element_bits BOPs.
        """
        n_blocks = math.ceil(N / block_size)
        bops = n_blocks * (8 * 8 + block_size * 1 * element_bits)
        self._bops += bops
        self._log.append({"op": label, "bops": bops, "n": N, "element_bits": element_bits})
        return bops

    def sq_gather_scatter(
        self, N: int, dense_bits: int, sparse_bits: int, k: int, label: str = "SQ_GS"
    ) -> float:
        """Register BOPs for SQ-Format Gather/Scatter.

        Gather: k mux operations (each 1-bit mask × sparse_bits data).
        Scatter: N mux operations (1-bit × sparse_bits).
        """
        bops = k * sparse_bits + N * 1
        self._bops += bops
        self._log.append({"op": label, "bops": bops, "n": N, "k": k})
        return bops

    # ── Format-specific forward pass ─────────────────────────────────────────

    def linear_layer_bops(
        self,
        format_name: str,
        M: int,
        K: int,
        N: int,
        weight_bits: int,
        activation_bits: int,
    ) -> dict:
        """Register all BOPs for one linear layer (Y = X @ W + b).

        Includes:
          1. Pre-processing transform (if any).
          2. Matrix multiply.
          3. Post-processing (dequant/rescale).
        Returns itemized breakdown.
        """
        self.reset()
        fmt = format_name.upper()
        breakdown = {}

        # Pre-processing
        if "HAD" in fmt:
            breakdown["transform_bops"] = self.hadamard(K, activation_bits, "FWHT_pre")
        elif "SMOOTHQUANT" in fmt:
            breakdown["transform_bops"] = self.smoothquant_scale(K, 32, "SQ_scale")
        elif "RANDROT" in fmt:
            breakdown["transform_bops"] = self.random_rotation(K, activation_bits, "RandRot")
        else:
            breakdown["transform_bops"] = 0.0

        # Matrix multiply
        breakdown["matmul_bops"] = self.matmul(M, K, N, activation_bits, weight_bits)

        # Format-specific overhead
        if "MX" in fmt:
            breakdown["scale_bops"] = self.mx_scale_apply(K * N, weight_bits, 32)
        elif "SQ" in fmt:
            k_sparse = max(1, int(0.01 * K * N))
            breakdown["gather_bops"] = self.sq_gather_scatter(
                K * N, 4, 8, k_sparse
            )

        # Post-processing inverse transform
        if "HAD" in fmt:
            breakdown["inv_transform_bops"] = self.hadamard(N, activation_bits, "FWHT_post")

        breakdown["total_bops"] = self.total_bops
        breakdown["format"] = format_name
        return breakdown


def compare_formats_bops(
    formats: list,
    M: int = 16,
    K: int = 256,
    N: int = 256,
) -> list:
    """Compare BOPs across multiple formats for the same layer dimensions."""
    counter = BopCounter()
    results = []

    bit_map = {
        "FP32": (32, 32), "BF16": (16, 16), "INT8": (8, 8), "INT4": (4, 4),
        "MXFP8": (8, 8), "MXFP4": (4, 4), "MXINT8": (8, 8), "MXINT4": (4, 4),
        "NF4": (4, 4), "NVFP4": (4, 4), "FP6": (6, 6),
        "HAD+INT8(C)": (8, 8), "HAD+INT8(T)": (8, 8),
        "HAD+INT4(C)": (4, 4), "HAD+INT4(T)": (4, 4),
        "SMOOTHQUANT+INT8": (8, 8), "SMOOTHQUANT+INT4": (4, 4),
        "RANDROT+INT4": (4, 4), "RANDROT+INT8": (8, 8),
        "SQ-FORMAT": (4, 8), "HAD+SQ": (4, 8),
    }

    for fmt in formats:
        key = fmt.upper().replace("+", "").replace("-", "").replace(" ", "")
        a_bits, w_bits = 8, 8  # defaults
        for k, v in bit_map.items():
            if k.replace("+", "").replace("-", "") in key:
                a_bits, w_bits = v
                break

        bd = counter.linear_layer_bops(fmt, M, K, N, w_bits, a_bits)
        results.append({
            "format": fmt,
            "activation_bits": a_bits,
            "weight_bits": w_bits,
            "total_bops": bd["total_bops"],
            "matmul_bops": bd.get("matmul_bops", 0),
            "transform_bops": bd.get("transform_bops", 0),
            "overhead_bops": bd["total_bops"] - bd.get("matmul_bops", 0) - bd.get("transform_bops", 0),
        })

    return results
