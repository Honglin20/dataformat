"""4-bit format implementations used by the study.

All formats share a uniform interface:

    class Format(Protocol):
        name: str
        bits: int
        def quantize(self, x: np.ndarray) -> np.ndarray:
            '''Return dequantized tensor (same shape as x).'''

Per-channel formats (``INT4``, ``FP4``, ``NF4``) pick up their scale/look-up
parameters along the LAST axis of a 2-D tensor (i.e. per output channel of a
weight matrix of shape (out, in), or per feature dim of an activation of
shape (batch, dim)).  For 1-D input the per-channel and per-tensor behaviours
coincide.

NVFP4 keeps the NVIDIA-Blackwell-style per-tensor POT scale (level-1 scale
only; the level-2 per-16-element E8M0 scale is *not* modelled here to keep
the per-channel comparison clean).

MXINT4 and MXFP4 wrap the existing OCP-MX block-scaled implementations
(block size 32).  All formats return float32 arrays.

Extra 4-bit variants added to compare against NF4 fairly:

    * ``INT4_FP``     – symmetric INT4 with per-channel **FP** absmax scale
      (no power-of-two constraint).  Puts INT4 on the same scaling footing
      as NF4 and helps isolate the contribution of the NF4 level set itself.
    * ``APoT4``       – Additive Power-of-Two 4-bit (Li et al. 2020): the 16
      levels are sums / differences of two powers of two, giving a denser
      code near zero than INT4 while remaining multiplier-free on hardware.
    * ``LOG4``        – Pure logarithmic 4-bit, levels = ±2^k ∪ {0}; decode
      is a single shift (hardware-cheapest of all four-bit schemes).
    * ``NF4_FP8``     – Hardware-realistic NF4: per-channel absmax stored in
      **FP8 (E4M3)** instead of FP32, modelling the real QLoRA double-
      quantisation idea (still a 16-entry LUT + one scale multiply).

These variants keep the same per-channel footprint as the originals so the
comparison in Part 2 remains apples-to-apples.
"""
from __future__ import annotations

import numpy as np

# Reuse existing, well-tested element-level encoders.
from formats.mxint import MXINTFormat
from formats.mxfp import MXFPFormat, _E2M1_POS
from formats.nvfp4 import NVFP4Format
# Canonical OCP-floor POT helper; aliased to the legacy private name used below.
from formats._pot import pot_scale_vec as _pot_scale_for_qmax
# Shared 4-bit helpers + the NF4/APoT/LOG/INT4-FP variants now live in
# ``formats.int_variants`` so they can be consumed by any experiment; they are
# re-exported here unchanged for backwards compatibility.
from formats.int_variants import (
    _per_channel_absmax,
    _fp8_e4m3,
    _build_apot4_levels,
    INT4FPScalePerChannel,
    APoT4PerChannel,
    Log4PerChannel,
    NF4FP8PerChannel,
)
# SQ-Format variants are registered so ``experiments/sqformat/config.py``
# can declaratively build the 16-cell study matrix via FormatSpec.kwargs.
from formats.sq_format import SQFormat, SQFormatActivations, SQFormatFP
from config import NF4_LEVELS


# ── INT4 per-channel ─────────────────────────────────────────────────────────

class INT4PerChannel:
    """Symmetric INT4 with a power-of-two per-channel scale."""

    name = "INT4"
    bits = 4

    def __init__(self):
        self._q_max = 7  # 2^(4-1) - 1

    def quantize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        absmax = _per_channel_absmax(x)
        scale  = _pot_scale_for_qmax(absmax, self._q_max)
        scale  = np.maximum(scale, np.finfo(np.float32).tiny)
        q = np.clip(np.round(x / scale), -self._q_max, self._q_max)
        return (q * scale).astype(np.float32)


# ── FP4 per-channel (E2M1 levels, POT scale, per-channel) ────────────────────

class FP4PerChannel:
    """Pure 4-bit floating-point (E2M1) with a POT per-channel scale.

    Element code set is identical to NVFP4's (0, 0.5, 1, 1.5, 2, 3, 4, 6) with
    a sign bit, but unlike NVFP4 the scale is computed per *output channel*
    (last axis) rather than per whole tensor.  No block-level scale — this is
    the "pure" FP4 counterpart to INT4.
    """

    name = "FP4"
    bits = 4
    _levels = _E2M1_POS   # (0, 0.5, 1, 1.5, 2, 3, 4, 6)
    _max_level = 6.0

    def quantize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        absmax = _per_channel_absmax(x)
        scale  = _pot_scale_for_qmax(absmax, self._max_level)
        scale  = np.maximum(scale, np.finfo(np.float32).tiny)
        x_scaled = x / scale

        sign = np.where(x_scaled < 0, -1.0, 1.0).astype(np.float32)
        x_abs = np.clip(np.abs(x_scaled), 0.0, self._max_level)
        # Nearest neighbour into the 8 positive levels.
        dists = np.abs(x_abs[..., None] - self._levels)
        idx = np.argmin(dists, axis=-1)
        q_abs = self._levels[idx]
        return (sign * q_abs * scale).astype(np.float32)


# ── NF4 per-channel ──────────────────────────────────────────────────────────

class NF4PerChannel:
    """QLoRA NormalFloat4 with a per-channel absmax scale.

    Unlike the original per-tensor implementation in ``formats/nf4.py`` this
    class normalises each channel (last axis) by its own absmax before
    snapping to the 16 NF4 levels.  The scale is *not* constrained to a power
    of two because NF4's look-up-table values are already irrational; a PoT
    scale would only add a constant dB offset without changing the shape of
    the quantizer.
    """

    name = "NF4"
    bits = 4
    _levels = NF4_LEVELS.astype(np.float32)

    def quantize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        absmax = _per_channel_absmax(x)
        absmax = np.maximum(absmax, np.finfo(np.float32).tiny)
        x_norm = x / absmax
        dists  = np.abs(x_norm[..., None] - self._levels)
        idx    = np.argmin(dists, axis=-1)
        q_norm = self._levels[idx]
        return (q_norm * absmax).astype(np.float32)


# ── INT4_FP / APoT4 / LOG4 / NF4_FP8 now live in formats/int_variants.py ─────
# (re-imported at the top of this module for backwards compatibility).


# ── NVFP4 — reuse existing per-tensor implementation ─────────────────────────

def make_nvfp4():
    fmt = NVFP4Format()
    fmt.name = "NVFP4"
    return fmt


# ── MXINT4 / MXFP4 — reuse existing block-scaled implementations ─────────────

def make_mxint4(block_size: int = 32):
    fmt = MXINTFormat(element_bits=4, block_size=block_size)
    fmt.name = "MXINT4"
    return fmt


def make_mxfp4(block_size: int = 32):
    fmt = MXFPFormat(element_bits=4, block_size=block_size)
    fmt.name = "MXFP4"
    return fmt


# ── Factory registry — used by config.py / registry.py ───────────────────────

FORMAT_FACTORIES = {
    "int4_per_channel":    INT4PerChannel,
    "int4_fp_per_channel": INT4FPScalePerChannel,
    "fp4_per_channel":     FP4PerChannel,
    "nf4_per_channel":     NF4PerChannel,
    "nf4_fp8_per_channel": NF4FP8PerChannel,
    "apot4_per_channel":   APoT4PerChannel,
    "log4_per_channel":    Log4PerChannel,
    "nvfp4":               make_nvfp4,
    "mxint4":              make_mxint4,
    "mxfp4":               make_mxfp4,
    # SQ-Format family (Algorithm 1 + Algorithm 2 + legacy FP8/INT hybrid).
    # kwargs wire up ``base``/``high_bits``/``low_bits`` per cell.
    "sqformat_alg1":       SQFormat,
    "sqformat_alg2":       SQFormatActivations,
    "sqformat_fp_hybrid":  SQFormatFP,
}


def make_format(factory_name: str, **kwargs):
    """Instantiate a format by its factory key. Unknown keys raise KeyError."""
    if factory_name not in FORMAT_FACTORIES:
        raise KeyError(
            f"Unknown format factory '{factory_name}'. "
            f"Known: {sorted(FORMAT_FACTORIES)}"
        )
    factory = FORMAT_FACTORIES[factory_name]
    return factory(**kwargs)
