"""Extra 4-bit INT / LUT variants used for apples-to-apples NF4 comparisons.

These formats were originally defined inside ``fourbit/formats.py``; moving
them here makes them available to any experiment (not just the ``fourbit/``
study) while keeping the same per-channel footprint as the originals.

    * ``INT4FPScalePerChannel``  (``name='INT4_FP'``) – symmetric INT4 with a
      per-channel **FP** absmax scale (no power-of-two constraint).  Puts
      INT4 on the same scaling footing as NF4 and helps isolate the
      contribution of the NF4 level set itself.
    * ``APoT4PerChannel``        (``name='APoT4'``)   – Additive Power-of-Two
      4-bit (Li et al. 2020): the 16 levels are sums / differences of two
      powers of two, giving a denser code near zero than INT4 while
      remaining multiplier-free on hardware.
    * ``Log4PerChannel``         (``name='LOG4'``)    – Pure logarithmic
      4-bit, levels = ±2^k ∪ {0}; decode is a single shift
      (hardware-cheapest of all four-bit schemes).
    * ``NF4FP8PerChannel``       (``name='NF4_FP8'``) – Hardware-realistic
      NF4: per-channel absmax stored in **FP8 (E4M3)** instead of FP32,
      modelling the real QLoRA double-quantisation idea (still a 16-entry
      LUT + one scale multiply).

Shared helpers (``_per_channel_absmax``, the FP8-E4M3 vectorised wrapper,
``_build_apot4_levels``) also live here and are re-exported by
``fourbit/formats.py`` for backwards compatibility.
"""
from __future__ import annotations

import numpy as np

from formats.mxfp import _fp8_e4m3_quantize_scalar
from config import NF4_LEVELS


# ── Shared helpers ───────────────────────────────────────────────────────────

def _per_channel_absmax(x: np.ndarray) -> np.ndarray:
    """Per-channel absmax along the last axis. Returns array with a trailing 1 axis."""
    if x.ndim <= 1:
        return np.asarray(np.max(np.abs(x)), dtype=np.float32)
    return np.max(np.abs(x), axis=-1, keepdims=True).astype(np.float32)


_fp8_e4m3 = np.vectorize(_fp8_e4m3_quantize_scalar)


# ── INT4 with per-channel FP scale (no POT) ──────────────────────────────────

class INT4FPScalePerChannel:
    """Symmetric INT4 with an unrestricted per-channel FP scale.

    Unlike :class:`fourbit.formats.INT4PerChannel` the scale is a full
    floating-point ``absmax / q_max`` (no power-of-two rounding).  Hardware
    cost: one FP multiply per element at decode time — identical to NF4's
    cost.  This isolates the *level-set* contribution of NF4 from the *scale
    precision* contribution, since both formats now share the same FP scale
    footprint but differ only in where their 16 representable levels sit.
    """

    name = "INT4_FP"
    bits = 4

    def __init__(self):
        self._q_max = 7

    def quantize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        absmax = _per_channel_absmax(x)
        absmax = np.maximum(absmax, np.finfo(np.float32).tiny)
        scale = absmax / float(self._q_max)
        q = np.clip(np.round(x / scale), -self._q_max, self._q_max)
        return (q * scale).astype(np.float32)


# ── APoT4: Additive Powers of Two 4-bit ──────────────────────────────────────

def _build_apot4_levels() -> np.ndarray:
    """16-level APoT set: {0, ±(2^a + 2^b)} with a,b ∈ {-3,-2,-1,0}.

    APoT (Li et al., 2020) keeps the hardware-friendliness of PoT (two
    shift-and-add ops per dequant) while densifying the level set near zero.
    We pick the canonical APoT-4 configuration whose positive values are
    sorted into the 8 finest-grained magnitudes: {0, 0.125, 0.25, 0.375,
    0.5, 0.625, 0.75, 1.0}.  (Equivalent to INT3 + a 0.625 code.)
    """
    # Positive half: 0 + 7 additive PoT values normalised to max=1.0
    pos = np.array([0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0],
                   dtype=np.float32)
    return pos


class APoT4PerChannel:
    """Additive Power-of-Two 4-bit, per-channel FP absmax scale.

    Decode path per element: one LUT lookup (8 positive codes) + one FP
    multiply with the per-channel scale.  Multiplier-free *if* the scale is
    itself a PoT; we expose both the general FP version (here) and the POT
    version via :class:`APoT4PoTPerChannel` below.
    """

    name = "APoT4"
    bits = 4
    _levels_pos = _build_apot4_levels()
    _max_level = 1.0

    def quantize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        absmax = _per_channel_absmax(x)
        absmax = np.maximum(absmax, np.finfo(np.float32).tiny)
        scale = absmax / self._max_level

        x_scaled = x / scale
        sign = np.where(x_scaled < 0, -1.0, 1.0).astype(np.float32)
        x_abs = np.clip(np.abs(x_scaled), 0.0, self._max_level)
        dists = np.abs(x_abs[..., None] - self._levels_pos)
        idx = np.argmin(dists, axis=-1)
        q_abs = self._levels_pos[idx]
        return (sign * q_abs * scale).astype(np.float32)


# ── LOG4: pure logarithmic 4-bit ─────────────────────────────────────────────

class Log4PerChannel:
    """Pure logarithmic 4-bit with per-channel FP absmax scale.

    Positive codes = {0, 2^-6, 2^-5, 2^-4, 2^-3, 2^-2, 2^-1, 2^0}, i.e. eight
    levels including 0.  Sign bit completes the 4-bit code.  Decode is a
    single arithmetic shift (multiplier-free).  This is the *cheapest*
    possible 4-bit format in hardware but its coarse resolution makes it a
    weak baseline — included to show the other end of the
    quality/HW-cost trade-off.
    """

    name = "LOG4"
    bits = 4
    _levels_pos = np.array(
        [0.0, 2 ** -6, 2 ** -5, 2 ** -4, 2 ** -3, 2 ** -2, 2 ** -1, 1.0],
        dtype=np.float32,
    )
    _max_level = 1.0

    def quantize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        absmax = _per_channel_absmax(x)
        absmax = np.maximum(absmax, np.finfo(np.float32).tiny)
        scale = absmax / self._max_level

        x_scaled = x / scale
        sign = np.where(x_scaled < 0, -1.0, 1.0).astype(np.float32)
        x_abs = np.clip(np.abs(x_scaled), 0.0, self._max_level)

        # Log-space nearest-neighbour: compare against the positive code set.
        dists = np.abs(x_abs[..., None] - self._levels_pos)
        idx = np.argmin(dists, axis=-1)
        q_abs = self._levels_pos[idx]
        return (sign * q_abs * scale).astype(np.float32)


# ── NF4 with FP8 (E4M3) scale — hardware-realistic QLoRA-style ──────────────

class NF4FP8PerChannel:
    """NF4 with an FP8-E4M3 per-channel absmax (QLoRA double-quantisation).

    QLoRA's "NF4" stores the FP32 per-block absmax by quantising it a
    second time with an FP8 scale (double quantisation).  This class
    models that hardware cost explicitly: compute per-channel absmax,
    cast it into FP8-E4M3, then use that quantised absmax as the scale.

    Result: 4 bits/element + ~8/channel overhead, same LUT + one FP8
    multiply in decode — the realistic hardware implementation of NF4.
    """

    name = "NF4_FP8"
    bits = 4
    _levels = NF4_LEVELS.astype(np.float32)

    def quantize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        absmax = _per_channel_absmax(x)
        absmax = np.maximum(absmax, np.finfo(np.float32).tiny)
        # Double-quantise the scale into FP8-E4M3.
        absmax_fp8 = _fp8_e4m3(absmax).astype(np.float32)
        absmax_fp8 = np.maximum(absmax_fp8, np.finfo(np.float32).tiny)
        x_norm = x / absmax_fp8
        dists = np.abs(x_norm[..., None] - self._levels)
        idx = np.argmin(dists, axis=-1)
        q_norm = self._levels[idx]
        return (q_norm * absmax_fp8).astype(np.float32)
