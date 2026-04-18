"""Canonical Power-of-Two scale helpers.

Two rounding strategies are in use across the codebase:

  * pot_scale / pot_scale_vec  — OCP floor formula
        scale = 2^(floor(log2(absmax)) - floor(log2(q_max)))
    Trades some clipping for a finer step size; used by the main
    INT / FP4/6/8 formats (see formats/__init__.py, fourbit/formats.py).

  * pot_scale_ceil / pot_scale_ceil_vec  — ceil-of-ratio formula
        scale = 2^ceil(log2(absmax / q_max))
    Guarantees no clipping (q_max * scale >= absmax); used by SQFormat.

Both share the same ``absmax <= 0 -> 1.0`` sentinel.
"""
from __future__ import annotations

import numpy as np


# ── OCP floor variant (majority of formats) ──────────────────────────────────

def pot_scale(absmax: float, q_max: int) -> float:
    """Scalar OCP-floor POT scale. Returns 1.0 for absmax <= 0."""
    if absmax <= 0:
        return 1.0
    log2_absmax = int(np.floor(np.log2(float(absmax) + 1e-38)))
    log2_qmax = int(np.floor(np.log2(float(q_max))))
    return float(2.0 ** (log2_absmax - log2_qmax))


def pot_scale_vec(absmax: np.ndarray, q_max: float) -> np.ndarray:
    """Vectorized OCP-floor POT scale. Safe for zero / subnormal absmax.

    Matches the semantics of ``fourbit/formats.py::_pot_scale_for_qmax``:
    float32 dtype, uses ``np.finfo(np.float32).tiny`` as the zero-guard,
    and returns 1.0 for non-positive inputs.
    """
    absmax = np.asarray(absmax, dtype=np.float32)
    log2_q = int(np.floor(np.log2(float(q_max))))
    safe = np.maximum(absmax, np.finfo(np.float32).tiny)
    log2_a = np.floor(np.log2(safe))
    scale = 2.0 ** (log2_a - log2_q)
    return np.where(absmax > 0, scale, 1.0).astype(np.float32)


# ── Ceil-of-ratio variant (SQFormat) ─────────────────────────────────────────

def pot_scale_ceil(absmax: float, q_max: int) -> float:
    """Smallest power-of-two scale s such that q_max * s >= absmax (no clipping).

    Computes s = 2^ceil(log2(absmax / q_max)).  Using ceil (not the
    floor-of-each-operand-separately approach) guarantees the no-clipping
    property.  Counter-example for the floor approach: absmax=15, q_max=7
    gives floor(log2(15))-floor(log2(7)) = 3-2 = 1 → s=2, but 7×2=14 < 15,
    so the value clips.  The ceil approach gives ceil(log2(15/7))=ceil(1.1)=2
    → s=4, and 7×4=28 >= 15.

    Scale is a power of two → division is a hardware arithmetic right-shift.
    """
    if absmax <= 0:
        return 1.0
    log2_ratio = np.log2(float(absmax) / float(q_max))
    return float(2.0 ** int(np.ceil(log2_ratio)))


def pot_scale_ceil_vec(absmax: np.ndarray, q_max: int) -> np.ndarray:
    """Vectorized pot_scale_ceil: 2^ceil(log2(absmax / q_max)) per element."""
    absmax = np.asarray(absmax, dtype=np.float32)
    result = np.ones_like(absmax)
    valid = absmax > 0
    if not np.any(valid):
        return result
    safe_am = np.where(valid, absmax, float(q_max))   # avoid log2(0)
    log2_ratio = np.log2(safe_am / float(q_max))
    result = np.where(valid, (2.0 ** np.ceil(log2_ratio)).astype(np.float32), 1.0)
    return result.astype(np.float32)
