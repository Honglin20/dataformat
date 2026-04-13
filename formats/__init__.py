"""Format registry: instantiates all quantization formats and composed pipelines.

Focus format set (for comparison figures):
  Baselines:         FP32, INT4, INT8
  Hardware-native:   MXINT4, MXINT8, NVFP4, NF4
  Transform-based:   HAD+INT4(C), HAD+INT4(T), HAD+INT8(C), HAD+INT8(T), HAD+SQ
  Sparse-quantized:  SQ-Format
  Upper-bound ref:   RandRot+INT4, RandRot+INT8

All INT-family quantizers use Power-of-Two (POT) scales:
  scale = 2^floor(log2(absmax / q_max))
This makes the scale multiply/divide a hardware right-shift operation.

HADTransform uses normalize=False (hardware model):
  - Forward HAD outputs are integer-valued (no irrational √N division).
  - The quantizer scale automatically absorbs the √N amplification.
  - Inverse divides by N (exact right-shift by log₂N bits).
"""

import numpy as np
from scipy.stats import norm as _norm_dist

# ── Plain formats ──────────────────────────────────────────────────────────────
from formats.baseline import FP32Format, BF16Format, FP16Format
from formats.nvfp4 import NVFP4Format
from formats.mxfp import MXFPFormat
from formats.mxint import MXINTFormat
from formats.nf4 import NF4Format
from formats.fp6 import FP6Format
from formats.sq_format import SQFormat, SQFormatActivations, SQFormatFP

# ── Transforms ────────────────────────────────────────────────────────────────
from formats.transforms.hadamard import HADTransform
from formats.transforms.random_rotation import RandomRotationTransform
from formats.transforms.smoothquant import SmoothQuantINTQuantizer


# ── Hardware-friendly INT quantizer (POT scale) ───────────────────────────────

def _pot_scale(absmax: float, q_max: int) -> float:
    """OCP-aligned power-of-two scale: 2^(floor(log2(absmax)) - floor(log2(q_max))).

    This guarantees the maximum element is covered (no clipping) while
    maintaining as fine a step size as possible.

    Hardware: result is always a power of 2 → scale multiply/divide is an
    arithmetic right-shift, exactly as in E8M0 hardware.

    Why NOT floor(log2(absmax / q_max)):
      log2(q_max) = log2(2^(bits-1)-1) ≈ bits-1-ε  (e.g. log2(127)≈6.99)
      floor(log2(absmax) - 6.99) = floor(log2(absmax)) - 7  for most absmax,
      giving a scale 2× too small and clipping ~5% of N(0,1) elements in INT8.
      The OCP formula floor(log2(absmax)) - floor(log2(q_max)) avoids this
      by treating floor(log2(q_max)) as the exact integer exponent of q_max.
    """
    if absmax <= 0:
        return 1.0
    log2_absmax = int(np.floor(np.log2(float(absmax) + 1e-38)))
    log2_qmax   = int(np.floor(np.log2(float(q_max))))
    return float(2.0 ** (log2_absmax - log2_qmax))


class _POTINTQuantizer:
    """Symmetric per-tensor or per-channel INT quantization with POT scale.

    POT scale = 2^ceil(log2(absmax / q_max))  — smallest power of 2 that covers absmax.
    Division by scale is an arithmetic right-shift in hardware — no FP divider.
    """

    def __init__(self, bits: int, per_channel: bool = False):
        self.bits = bits
        self.per_channel = per_channel
        self._q_max = 2 ** (bits - 1) - 1

    def quantize(self, x: np.ndarray, bits: int = None) -> np.ndarray:
        x = x.astype(np.float32)
        if self.per_channel and x.ndim > 1:
            absmax = np.max(np.abs(x), axis=-1, keepdims=True)
        else:
            absmax = np.max(np.abs(x))

        # Compute POT scale element-wise (or scalar)
        if np.ndim(absmax) == 0:
            scale = _pot_scale(float(absmax), self._q_max)
        else:
            absmax_flat = absmax.ravel()
            scales_flat = np.array([_pot_scale(float(v), self._q_max) for v in absmax_flat],
                                   dtype=np.float32)
            scale = scales_flat.reshape(absmax.shape)

        scale = np.maximum(scale, 1e-38)
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
        Dimension used for rotation-based transforms. Must be power of 2.
    seed : int
        Random seed for stochastic transforms (RandRot).
    """
    # HAD transform (normalize=False: hardware-friendly, fold scale into quantizer)
    had = HADTransform(normalize=False)
    rand_rot = RandomRotationTransform(dim=dim, seed=seed)

    formats = {
        # ── High-precision baselines ──────────────────────────────────────────
        "FP32":  FP32Format(),
        "BF16":  BF16Format(),
        "FP16":  FP16Format(),

        # ── Plain INT (POT scale, hardware-friendly) ──────────────────────────
        "INT4":    _POTINTQuantizer(4, per_channel=False),
        "INT8":    _POTINTQuantizer(8, per_channel=False),
        "INT4(T)": _POTINTQuantizer(4, per_channel=False),
        "INT4(C)": _POTINTQuantizer(4, per_channel=True),
        "INT8(T)": _POTINTQuantizer(8, per_channel=False),
        "INT8(C)": _POTINTQuantizer(8, per_channel=True),

        # ── Hardware-native formats ───────────────────────────────────────────
        "MXINT4":  MXINTFormat(element_bits=4),
        "MXINT8":  MXINTFormat(element_bits=8),
        "MXFP4":   MXFPFormat(element_bits=4),    # secondary reference
        "MXFP8":   MXFPFormat(element_bits=8),    # secondary reference
        "NVFP4":   NVFP4Format(),
        "NF4":     NF4Format(),
        "FP6":     FP6Format(),                   # secondary reference

        # ── SQ-Format-INT / SQ-Format-FP (exp1 canonical configs) ───────────
        # Fixed configuration: high_bits=8, low_bits=4, bank_size=128, sparsity=0.5
        # Appears in both 4-bit and 8-bit exp1 sections (unchanged per user spec).
        "SQ-Format-INT": SQFormat(bank_size=128, sparsity=0.5, high_bits=8, low_bits=4),
        "SQ-Format-FP":  SQFormatFP(bank_size=128, sparsity=0.5, low_bits=4),

        # ── SQ-Format (sparse-quantized, POT scales) ──────────────────────────
        # Algorithm 1 (weight quantization), bank-based, element-level importance.
        # sparsity_ratio=0.01 → 1% of elements per bank are high-precision (s=0.99).
        # This is kept for backward compatibility; see SQ-Format(s=0.5) for the
        # paper's canonical 2:4-equivalent configuration (s=0.5, b=128).
        "SQ-Format":       SQFormat(dense_bits=4, sparse_bits=8, sparsity_ratio=0.01),
        "SQ-Format(8b)":   SQFormat(dense_bits=8, sparse_bits=8, sparsity_ratio=0.01),
        # Paper-faithful configuration: s=0.5 (50% high-prec, 50% low-prec, b=128)
        "SQ-Format(s=0.5)": SQFormat(bank_size=128, sparsity=0.5, high_bits=8, low_bits=4),
        # Algorithm 2 (activation quantization): per-channel importance Ij=|Āj·ΣW'|
        "SQ-Format-A":     SQFormatActivations(bank_size=128, sparsity=0.5,
                                               high_bits=8, low_bits=4),

        # ── HAD + INT per-channel (C) — PRIMARY focus format ─────────────────
        # Per-channel quantization after HAD: each output channel gets its own
        # POT scale. Best quality. For 1D inputs same as per-tensor.
        "HAD+INT4(C)": ComposedFormat(
            "HAD+INT4(C)", had, _POTINTQuantizer(4, per_channel=True), 4
        ),
        "HAD+INT8(C)": ComposedFormat(
            "HAD+INT8(C)", had, _POTINTQuantizer(8, per_channel=True), 8
        ),

        # ── HAD + INT per-tensor (T) — ablation: single global scale ─────────
        # Per-tensor: one POT scale for the entire HAD-domain tensor.
        # For 1D inputs identical to (C). For 2D activations (T) shows
        # the cost of not having per-channel scales after HAD.
        "HAD+INT4(T)": ComposedFormat(
            "HAD+INT4(T)", had, _POTINTQuantizer(4, per_channel=False), 4
        ),
        "HAD+INT8(T)": ComposedFormat(
            "HAD+INT8(T)", had, _POTINTQuantizer(8, per_channel=False), 8
        ),

        # ── HAD + SQ-Format (global redistribution + sparse high-prec) ───────
        "HAD+SQ": ComposedFormat(
            "HAD+SQ", had, SQFormat(dense_bits=4, sparse_bits=8, sparsity_ratio=0.01), 4
        ),

        # ── HAD + LUT (non-uniform, secondary) ───────────────────────────────
        "HAD+LUT4": ComposedFormat("HAD+LUT4", had, _LUTQuantizer(4), 4),

        # ── Random orthogonal rotation + INT (upper-bound reference) ─────────
        # RandRot uses a dense N×N ROM matrix. Quality upper bound for rotation
        # methods. Compared to HAD: approximately equal for large N because
        # HAD spreads energy perfectly uniformly (all outputs same magnitude for
        # single-channel-outlier inputs), while RandRot output has small variance
        # (Normal distribution) with slightly larger max → slightly coarser scale.
        "RandRot+INT4": ComposedFormat(
            "RandRot+INT4", rand_rot, _POTINTQuantizer(4, per_channel=False), 4
        ),
        "RandRot+INT8": ComposedFormat(
            "RandRot+INT8", rand_rot, _POTINTQuantizer(8, per_channel=False), 8
        ),

        # ── SmoothQuant (secondary, for reference) ───────────────────────────
        "SmoothQuant+INT4": SmoothQuantINTQuantizer(bits=4),
        "SmoothQuant+INT8": SmoothQuantINTQuantizer(bits=8),
    }

    # Backward-compatibility aliases
    formats["HAD+INT4"] = formats["HAD+INT4(C)"]
    formats["HAD+INT8"] = formats["HAD+INT8(C)"]

    for name, fmt in formats.items():
        if not hasattr(fmt, "name"):
            fmt.name = name

    return formats


# ── Focus format sets for comparison figures ──────────────────────────────────

#: Core 4-bit comparison: the main research question
FOCUS_4BIT = [
    "INT4",
    "MXINT4",
    "NVFP4",
    "NF4",
    "HAD+INT4(C)",
    "HAD+INT4(T)",
    "SQ-Format",
    "HAD+SQ",
    "RandRot+INT4",
]

#: Core 8-bit comparison
FOCUS_8BIT = [
    "INT8",
    "MXINT8",
    "HAD+INT8(C)",
    "HAD+INT8(T)",
    "SQ-Format(8b)",
    "RandRot+INT8",
]

#: All focus formats combined (with FP32 baseline)
FOCUS_ALL = ["FP32"] + FOCUS_4BIT + FOCUS_8BIT
