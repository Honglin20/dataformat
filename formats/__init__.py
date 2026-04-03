"""Format registry: instantiates all quantization formats and composed pipelines."""

import numpy as np
from scipy.stats import norm as _norm_dist


# ── Plain formats ──────────────────────────────────────────────────────────────
from formats.baseline import FP32Format, BF16Format
from formats.nvfp4 import NVFP4Format
from formats.mxfp import MXFPFormat
from formats.mxint import MXINTFormat
from formats.nf4 import NF4Format
from formats.fp6 import FP6Format
from formats.sq_format import SQFormat

# ── Transforms ────────────────────────────────────────────────────────────────
from formats.transforms.hadamard import HADTransform
from formats.transforms.random_rotation import RandomRotationTransform, TurboQuantTransform
from formats.transforms.smoothquant import SmoothQuantINTQuantizer


# ── Internal helpers ──────────────────────────────────────────────────────────

class _INTQuantizer:
    """Symmetric per-tensor or per-channel INT quantization."""

    def __init__(self, bits: int, per_channel: bool = False):
        self.bits = bits
        self.per_channel = per_channel
        self._q_max = 2 ** (bits - 1) - 1

    def quantize(self, x: np.ndarray, bits: int = None) -> np.ndarray:
        x = x.astype(np.float32)
        if self.per_channel:
            absmax = np.max(np.abs(x), axis=-1, keepdims=True)
        else:
            absmax = np.max(np.abs(x))
        absmax = np.maximum(absmax, 1e-8)
        scale = absmax / self._q_max
        q = np.round(x / scale).astype(np.int32)
        q = np.clip(q, -self._q_max, self._q_max)
        return q.astype(np.float32) * scale

    def dequantize(self, q: np.ndarray) -> np.ndarray:
        return q.astype(np.float32)

    def encoding_overhead(self) -> dict:
        return {
            "data_bits_per_element": self.bits,
            "metadata_bits_per_element": 0,
            "bandwidth_amplification": 1.0,
        }


class _LUTQuantizer:
    """Non-uniform LUT quantization: levels at uniform quantiles of N(0,1)."""

    def __init__(self, bits: int):
        self.bits = bits
        n_levels = 2 ** bits
        ps = np.linspace(1 / (2 * n_levels), 1 - 1 / (2 * n_levels), n_levels)
        self._levels = _norm_dist.ppf(ps).astype(np.float32)

    def quantize(self, x: np.ndarray, bits: int = None) -> np.ndarray:
        x = x.astype(np.float32)
        absmax = np.max(np.abs(x))
        if absmax == 0:
            return np.zeros_like(x)
        x_norm = x / absmax
        dists = np.abs(x_norm[..., None] - self._levels)
        idx = np.argmin(dists, axis=-1)
        return self._levels[idx] * absmax

    def dequantize(self, q: np.ndarray) -> np.ndarray:
        return q.astype(np.float32)

    def encoding_overhead(self) -> dict:
        return {
            "data_bits_per_element": self.bits,
            "metadata_bits_per_element": 0,
            "bandwidth_amplification": 1.0,
        }


class ComposedFormat:
    """Transform → Quantize → InverseTransform composition."""

    def __init__(self, name: str, transform, quantizer, bits: int):
        self.name = name
        self.bits = bits
        self._transform = transform
        self._quantizer = quantizer

    def quantize(self, x: np.ndarray, bits: int = None) -> np.ndarray:
        x_t = self._transform.forward(x)
        q_t = self._quantizer.quantize(x_t)
        return self._transform.inverse(q_t)

    def dequantize(self, q: np.ndarray) -> np.ndarray:
        return q.astype(np.float32)

    def encoding_overhead(self) -> dict:
        return self._quantizer.encoding_overhead()


def build_all_formats(dim: int = 256, seed: int = 42) -> dict:
    """Build and return the full format registry as name→format_object dict.

    Parameters
    ----------
    dim : int
        Dimension used for rotation-based transforms. Must be power of 2 for HAD.
    seed : int
        Random seed for stochastic transforms (RandRot, TurboQuant).
    """
    had = HADTransform(normalize=True)
    rand_rot = RandomRotationTransform(dim=dim, seed=seed)
    turbo = TurboQuantTransform(dim=dim, seed=seed)

    formats = {
        # Baselines
        "FP32":  FP32Format(),
        "BF16":  BF16Format(),

        # Hardware-native formats
        "NVFP4":   NVFP4Format(),
        "MXFP4":   MXFPFormat(element_bits=4),
        "MXFP8":   MXFPFormat(element_bits=8),
        "MXINT4":  MXINTFormat(element_bits=4),
        "MXINT8":  MXINTFormat(element_bits=8),
        "NF4":     NF4Format(),
        "FP6":     FP6Format(),

        # SQ-Format (pure, no pre-transform)
        "SQ-Format": SQFormat(dense_bits=4, sparse_bits=8, sparsity_ratio=0.01),

        # SmoothQuant + INT
        "SmoothQuant+INT4": SmoothQuantINTQuantizer(bits=4),
        "SmoothQuant+INT8": SmoothQuantINTQuantizer(bits=8),

        # HAD + INT (per-channel after transform)
        "HAD+INT4": ComposedFormat("HAD+INT4", had, _INTQuantizer(4, per_channel=True), 4),
        "HAD+INT8": ComposedFormat("HAD+INT8", had, _INTQuantizer(8, per_channel=True), 8),

        # HAD + LUT
        "HAD+LUT4": ComposedFormat("HAD+LUT4", had, _LUTQuantizer(4), 4),

        # HAD + SQ-Format
        "HAD+SQ": ComposedFormat(
            "HAD+SQ", had, SQFormat(dense_bits=4, sparse_bits=8, sparsity_ratio=0.01), 4
        ),

        # Random orthogonal rotation + INT
        "RandRot+INT4": ComposedFormat("RandRot+INT4", rand_rot, _INTQuantizer(4), 4),
        "RandRot+INT8": ComposedFormat("RandRot+INT8", rand_rot, _INTQuantizer(8), 8),

        # TurboQuant (random ±1 diagonal) + INT
        "TurboQuant+INT4": ComposedFormat("TurboQuant+INT4", turbo, _INTQuantizer(4), 4),
        "TurboQuant+INT8": ComposedFormat("TurboQuant+INT8", turbo, _INTQuantizer(8), 8),

        # Plain INT baselines (no transform, for comparison)
        "INT4": _INTQuantizer(4),
        "INT8": _INTQuantizer(8),
    }

    for name, fmt in formats.items():
        if not hasattr(fmt, "name"):
            fmt.name = name

    return formats
