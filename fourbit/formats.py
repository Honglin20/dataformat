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
"""
from __future__ import annotations

import numpy as np

# Reuse existing, well-tested element-level encoders.
from formats.mxint import MXINTFormat
from formats.mxfp import MXFPFormat, _E2M1_POS
from formats.nvfp4 import NVFP4Format
from config import NF4_LEVELS


# ── Helpers ──────────────────────────────────────────────────────────────────

def _pot_scale_for_qmax(absmax: np.ndarray, q_max: float) -> np.ndarray:
    """Power-of-two scale: 2^(floor(log2(absmax)) - floor(log2(q_max))).

    Vectorised and safe for zero / subnormal ``absmax``.  Matches the OCP
    MX-spec formula used elsewhere in the repo so comparisons are apples-to-
    apples across formats.
    """
    absmax = np.asarray(absmax, dtype=np.float32)
    log2_q = int(np.floor(np.log2(float(q_max))))
    safe   = np.maximum(absmax, np.finfo(np.float32).tiny)
    log2_a = np.floor(np.log2(safe))
    scale  = 2.0 ** (log2_a - log2_q)
    # For absmax == 0 use unit scale (result is zero anyway).
    return np.where(absmax > 0, scale, 1.0).astype(np.float32)


def _per_channel_absmax(x: np.ndarray) -> np.ndarray:
    """Per-channel absmax along the last axis. Returns array with a trailing 1 axis."""
    if x.ndim <= 1:
        return np.asarray(np.max(np.abs(x)), dtype=np.float32)
    return np.max(np.abs(x), axis=-1, keepdims=True).astype(np.float32)


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
    "int4_per_channel": INT4PerChannel,
    "fp4_per_channel":  FP4PerChannel,
    "nf4_per_channel":  NF4PerChannel,
    "nvfp4":            make_nvfp4,
    "mxint4":           make_mxint4,
    "mxfp4":            make_mxfp4,
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
