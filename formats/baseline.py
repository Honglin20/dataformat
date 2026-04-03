"""FP32 and BF16 baseline formats (lossless / near-lossless reference)."""

import numpy as np
import struct


class FP32Format:
    name = "FP32"
    bits = 32

    def quantize(self, x: np.ndarray, bits: int = 32) -> np.ndarray:
        return x.astype(np.float32)

    def dequantize(self, q: np.ndarray) -> np.ndarray:
        return q.astype(np.float32)

    def encoding_overhead(self) -> dict:
        return {"data_bits_per_element": 32, "metadata_bits_per_element": 0}


class BF16Format:
    """BFloat16: 1 sign + 8 exponent + 7 mantissa bits.
    Simulated by rounding FP32 to BF16 precision via truncation of mantissa bits.
    """
    name = "BF16"
    bits = 16

    def quantize(self, x: np.ndarray, bits: int = 16) -> np.ndarray:
        x32 = x.astype(np.float32)
        # Truncate lower 16 mantissa bits → BF16 precision
        x_bytes = x32.view(np.uint32)
        bf16_bits = (x_bytes & 0xFFFF0000).view(np.float32)
        return bf16_bits

    def dequantize(self, q: np.ndarray) -> np.ndarray:
        return q.astype(np.float32)

    def encoding_overhead(self) -> dict:
        return {"data_bits_per_element": 16, "metadata_bits_per_element": 0}
