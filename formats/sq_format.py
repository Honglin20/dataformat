"""SQ-Format: Sparse-Quantized unified format.

Implements two algorithms from the SQ-format paper:

    SQ-format(X) = ([Xquant], [Squant], [m], hhigh, hlow, b, s)

Where:
  [Xquant], [Squant] : quantized matrix and scaling matrix
  [m]                : precision mask (True = high-precision element)
  hhigh, hlow        : high/low precision bit-widths
  b                  : bank size
  s                  : sparsity — fraction of elements using LOW precision
                       top (1-s) per bank → hhigh; remaining s → hlow

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Algorithm 1 — Weight Quantization
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Input : W ∈ R^{K×N}, calibration D, sparsity s, bank size b, hhigh, hlow
  1. W', H  ← Smooth(W, D)
  2. I      ← (W')² / (diag(H⁻¹))²           ← element-level importance
  3. for each bank of b consecutive K-elements per output column:
       mw  ← top (1-s) elements of Iw         ← precision mask
       (w'_high, s'_high) ← Quant(w' ⊙  mw, hhigh)   ← per-column POT scale
       (w'_low,  s'_low)  ← Quant(w' ⊙ ~mw, hlow)    ← per-column POT scale
  4. Output: (Whigh, Wlow, Shigh, Slow)

Hardware bank definition (2D)
──────────────────────────────
Banks are groups of b consecutive K-elements within a SINGLE output column,
not b rows × all-N columns.  With b=4 and s=0.5 this produces exactly 2:4
structured sparsity per output neuron — the pattern directly supported by
NVIDIA Tensor Core sparse pipelines.  A single (scale_high, scale_low) pair
per bank (per output column) is required for dequantization; accessible via
SQFormat._last_bank_scales after each quantize() call.

Sentinel mask (paper §3.2)
───────────────────────────
High-precision positions are flagged within the low-precision integer stream
using the sentinel value v_sentinel = -(2^(hlow-1)), the two's-complement
minimum that symmetric quantization never occupies:
  INT2: normal range {-1, 0, 1}; v_sentinel = -2
  INT4: normal range {-7, …, 7}; v_sentinel = -8
This eliminates the need for a separate 1-bit boolean mask array and is why
encoding_overhead() reports mask_cost = 0.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Algorithm 2 — Activation-Aware Weight Quantization (Static Calibration)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Input : W ∈ R^{K×N}, calibration D, sparsity s, bank size b, hhigh, hlow
  1. W', Ā  ← Smooth(W, D)
  2. for each bank of b input channels:
       Ij   ← |Āj · Σᵢ W'_{j,i}|, ∀j ∈ bank   ← per-channel importance
       mw   ← top (1-s) channels of Ij
       (w', s') ← Quant(w, hhigh, hlow) per channel
  3. Reorder rows of W' by mask m
  4. Output: (Wquant, Squant, m)

NOTE: This is Activation-Aware Weight Quantization (calibration-time only).
Runtime activation quantization is performed separately at inference time
via SQFormatActivations.quantize_runtime_activations(), which uses per-token
(per-row) scales so that each token's quantization is independent of other
tokens in the batch.
"""

import numpy as np

# SQFormat uses the ceil-of-ratio POT scale (no-clipping guarantee); canonical
# implementation lives in ``formats/_pot.py``.  Aliased to the legacy private
# names so the rest of this module (and its tests) need no further changes.
from formats._pot import (
    pot_scale_ceil as _pot_scale,
    pot_scale_ceil_vec as _pot_scale_vec,
)
from formats.mxfp import _E2M1_POS, _fp8_e4m3_vec


# ── Element encoders ──────────────────────────────────────────────────────────
#
# Element-level encoder registry keyed by (base, bits). Each entry is a tuple
# of (encode_fn, q_max).  The encode functions accept the raw value tensor and
# a scale (either scalar or broadcastable array) and return dequantized values
# on the same grid (level_set * scale).  Banking / mask / sentinel logic in
# SQFormat and SQFormatActivations remains unchanged; only the per-element
# encoding swaps.


def _int_encode(x: np.ndarray, scale, q_max: int) -> np.ndarray:
    q = np.clip(np.round(x / np.maximum(scale, 1e-38)), -q_max, q_max)
    return (q * scale).astype(np.float32)


def _fp_e2m1_encode(x: np.ndarray, scale) -> np.ndarray:
    x_scaled = x / np.maximum(scale, 1e-38)
    sign = np.where(x_scaled < 0, -1.0, 1.0).astype(np.float32)
    x_abs = np.clip(np.abs(x_scaled), 0.0, float(_E2M1_POS[-1]))
    dists = np.abs(x_abs[..., None] - _E2M1_POS)
    idx = np.argmin(dists, axis=-1)
    return (sign * _E2M1_POS[idx] * scale).astype(np.float32)


def _fp_e4m3_encode(x: np.ndarray, scale) -> np.ndarray:
    return (_fp8_e4m3_vec(x / np.maximum(scale, 1e-38)) * scale).astype(np.float32)


_ELEMENT_ENCODERS = {
    ("int", 8): (lambda x, scale: _int_encode(x, scale, 127), 127),
    ("int", 4): (lambda x, scale: _int_encode(x, scale,   7),   7),
    ("int", 2): (lambda x, scale: _int_encode(x, scale,   1),   1),
    ("fp",  8): (_fp_e4m3_encode, 448.0),
    ("fp",  4): (_fp_e2m1_encode,   6.0),
}


def _resolve_encoder(base: str, bits: int):
    """Look up ``(encode_fn, q_max)`` for a given (base, bits) cell.

    ``_ELEMENT_ENCODERS`` is authoritative.  For ``base="int"`` we additionally
    fall back to the symmetric-integer formula ``q_max = 2**(bits-1) - 1`` so
    that non-standard widths (e.g. INT6/INT12/INT16 used by ablation tests and
    encoding-overhead tests) continue to work.  ``base="fp"`` is strict: only
    cells explicitly registered are accepted.
    """
    key = (base, bits)
    if key in _ELEMENT_ENCODERS:
        return _ELEMENT_ENCODERS[key]
    if base == "int" and bits >= 2:
        q_max = 2 ** (bits - 1) - 1
        return (lambda x, scale, _qm=q_max: _int_encode(x, scale, _qm), q_max)
    raise ValueError(f"Unsupported SQ-Format cell: base={base!r}, bits={bits}")


# ── Low-level helpers ──────────────────────────────────────────────────────────


def _int_quantize_pot(x: np.ndarray, bits: int) -> np.ndarray:
    """Symmetric integer quantization with per-tensor POT scale."""
    q_max = 2 ** (bits - 1) - 1
    absmax = float(np.max(np.abs(x)))
    if absmax == 0:
        return np.zeros_like(x, dtype=np.float32)
    scale = _pot_scale(absmax, q_max)
    q = np.clip(np.round(x / scale).astype(np.int32), -q_max, q_max)
    return q.astype(np.float32) * scale


def _int_quantize_pot_with_scale(x: np.ndarray, bits: int) -> tuple:
    """Symmetric integer quantization; returns (dequantized, scale)."""
    q_max = 2 ** (bits - 1) - 1
    absmax = float(np.max(np.abs(x)))
    if absmax == 0:
        return np.zeros_like(x, dtype=np.float32), 1.0
    scale = _pot_scale(absmax, q_max)
    q = np.clip(np.round(x / scale).astype(np.int32), -q_max, q_max)
    return q.astype(np.float32) * scale, scale


def _smooth(W: np.ndarray, A_mean: np.ndarray, alpha: float = 0.5):
    """SmoothQuant channel-wise scaling.

    s_j = max(|Ā_j|)^α / max(|W_j|)^(1-α)
    W' = W / s,  Ā' = Ā * s

    Parameters
    ----------
    W      : (K, N) weight matrix
    A_mean : (K,) mean calibration activation per input channel
    alpha  : migration strength (0.5 default)

    Returns
    -------
    W_smooth  : (K, N)
    A_smooth  : (K,)
    sq_scales : (K,) per-channel scales
    """
    A_mean = A_mean.ravel().astype(np.float32)
    W      = W.astype(np.float32)

    x_max = np.maximum(np.abs(A_mean), 1e-8)
    w_max = np.maximum(np.max(np.abs(W), axis=1), 1e-8)

    sq_scales = (x_max ** alpha) / (w_max ** (1.0 - alpha))
    W_smooth  = W / sq_scales[:, np.newaxis]
    A_smooth  = A_mean * sq_scales

    return W_smooth, A_smooth, sq_scales


# ── Algorithm 1: Weight Quantization ──────────────────────────────────────────

class SQFormat:
    """SQ-Format weight quantization — Algorithm 1.

    For 2D W (K×N), banks are groups of b consecutive K-elements within each
    single output column (N:M-style).  With b=4, sparsity=0.5, this produces
    the 2:4 pattern directly accelerated by Tensor Core sparse pipelines.
    Per-column POT scales are computed for both the high- and low-precision
    sub-streams of each bank.

    High-precision positions are encoded using the sentinel value
    v_sentinel = -(2^(hlow-1)) in the low-precision stream (the normally
    unused two's-complement minimum for symmetric quantization), so no
    separate boolean mask array is needed.  encoding_overhead() reports
    mask_cost = 0 to reflect this.

    For 1D inputs, banks are contiguous segments of b elements.

    After each quantize() call, self._last_bank_scales is populated with
    per-bank dicts: {"scale_high", "scale_low", "n_high", "n_low"}.

    Parameters
    ----------
    bank_size     : int   — elements per bank along K (b). Default 128.
    sparsity      : float — fraction using LOW precision (s). Default 0.5.
    high_bits     : int   — hhigh. Default 8.
    low_bits      : int   — hlow.  Default 4.

    Backward-compatible aliases:
    dense_bits    : int   — alias for low_bits.
    sparse_bits   : int   — alias for high_bits.
    sparsity_ratio: float — fraction in HIGH precision; sparsity = 1 - ratio.
    """

    def __init__(
        self,
        bank_size: int = 128,
        sparsity: float = None,
        high_bits: int = None,
        low_bits: int = None,
        dense_bits: int = None,
        sparse_bits: int = None,
        sparsity_ratio: float = None,
        base: str = "int",
    ):
        self.low_bits  = low_bits  if low_bits  is not None else (dense_bits  if dense_bits  is not None else 4)
        self.high_bits = high_bits if high_bits is not None else (sparse_bits if sparse_bits is not None else 8)

        if sparsity is not None:
            self.sparsity = float(sparsity)
        elif sparsity_ratio is not None:
            self.sparsity = 1.0 - float(sparsity_ratio)
        else:
            self.sparsity = 0.5

        self.bank_size = bank_size

        # Backward-compatible aliases
        self.dense_bits     = self.low_bits
        self.sparse_bits    = self.high_bits
        self.sparsity_ratio = 1.0 - self.sparsity

        self.name = "SQ-Format"
        self.bits = self.low_bits

        # ── Element-encoder selection (base ∈ {"int", "fp"}) ────────────────
        # Banking / mask / sentinel logic below stays identical across bases;
        # only the per-element encoding swaps.  Default base="int" preserves
        # byte-identical output for the existing golden CSVs.
        self.base = base
        self._enc_high, self._qmax_high = _resolve_encoder(base, self.high_bits)
        self._enc_low,  self._qmax_low  = _resolve_encoder(base, self.low_bits)

        # Sentinel value: the unused two's-complement minimum for hlow-bit symmetric int.
        # Marks high-precision positions in the low-precision integer stream.
        # e.g., INT4 (hlow=4): v_sentinel = -8  (normal range: -7..7)
        #        INT2 (hlow=2): v_sentinel = -2  (normal range: -1..1)
        self.v_sentinel: int = -(2 ** (self.low_bits - 1))

        self._last_bank_scales: list = []
        # Set by quantize() when A_mean is supplied (Algorithm 1, line 1).
        # Callers must apply X = X * self._sq_scales to activations at inference
        # time so that X·W' = (X·s)·(W/s) = X·W is preserved.
        self._sq_scales: np.ndarray = None

    def _bank_mask_1d(self, importance: np.ndarray, bsz: int) -> np.ndarray:
        """Bool mask: top (1-s) elements of a 1D importance vector."""
        k_high = min(max(1, int(np.round((1.0 - self.sparsity) * bsz))), bsz)
        if k_high >= bsz:
            return np.ones(bsz, dtype=bool)
        mask = np.zeros(bsz, dtype=bool)
        mask[np.argpartition(importance, -k_high)[-k_high:]] = True
        return mask

    def quantize(
        self,
        W: np.ndarray,
        H_inv_diag: np.ndarray = None,
        A_mean: np.ndarray = None,
        bits: int = None,
    ) -> np.ndarray:
        """Algorithm 1: bank-based mixed-precision weight quantization.

        Step 1  (optional): smooth W with calibration activations A_mean.
                            Required by Algorithm 1, line 1; pass A_mean to
                            enable.  Output is in the smoothed W' space.
        Step 2: compute element-level importance I on W'.
        Step 3: for each bank, select top (1-s) elements for hhigh and
                quantize the rest to hlow with per-column POT scales.

        For 2D W (K×N), banks are b consecutive K-elements per output column
        (N:M style, not b-rows × all-N).  For 1D W, banks are contiguous
        segments of b elements.

        Sentinel marks: high-precision positions in the low-prec stream are
        flagged with v_sentinel = -(2^(hlow-1)) in the integer representation.
        The dequantized output is correct regardless; no separate mask stored.

        Parameters
        ----------
        W          : (K, N) weight matrix or 1-D vector.
        H_inv_diag : (K,) diagonal of H⁻¹ for Hessian importance (optional).
        A_mean     : (K,) mean calibration activation; triggers _smooth() call
                     implementing Algorithm 1 line 1 (optional).
        bits       : ignored (harness compatibility).

        Returns
        -------
        np.ndarray — dequantized W, same shape as input.
        """
        W = W.astype(np.float32)
        self._last_bank_scales = []

        # ── Step 1: Smooth (Algorithm 1, line 1) — only when A_mean provided ─
        if A_mean is not None and W.ndim == 2:
            W_base, _, self._sq_scales = _smooth(W, np.asarray(A_mean, dtype=np.float32))
            # self._sq_scales[j] is the per-channel scale applied to W.
            # Caller must multiply activations X by self._sq_scales at inference:
            #   X_smooth = X * self._sq_scales  (activation side of SmoothQuant)
            # so that X_smooth @ W_base = X @ W is preserved.
        else:
            self._sq_scales = None
            W_base = W

        # ── Step 2: Element-level importance (Algorithm 1, line 2) ───────────
        if H_inv_diag is not None and W.ndim == 2:
            h_inv = np.asarray(H_inv_diag, dtype=np.float32).ravel()  # (K,)
            importance = (W_base ** 2) / (h_inv[:, np.newaxis] ** 2 + 1e-38)
        else:
            importance = W_base ** 2

        if W.ndim == 2:
            return self._quantize_2d(W_base, importance)
        else:
            return self._quantize_1d(W_base, importance)

    def _quantize_2d(self, W: np.ndarray, importance: np.ndarray) -> np.ndarray:
        """2D path: N:M-style banks of bank_size elements along K per output column.

        Each bank is a contiguous slice of bank_size K-rows within one output
        column n.  Per-column POT scales are computed independently for the
        high-precision and low-precision sub-streams of every bank.
        """
        K, N = W.shape

        # Pad K to a multiple of bank_size so reshape is clean
        K_pad = int(np.ceil(K / self.bank_size)) * self.bank_size
        pad   = K_pad - K
        n_kb  = K_pad // self.bank_size

        if pad > 0:
            W_work   = np.vstack([W,          np.zeros((pad, N), dtype=np.float32)])
            imp_work = np.vstack([importance,  np.zeros((pad, N), dtype=np.float32)])
        else:
            W_work, imp_work = W, importance

        # (n_kb, bank_size, N): axis-0 = K-bank index, axis-1 = element in bank
        W_blk   = W_work.reshape(n_kb, self.bank_size, N)
        imp_blk = imp_work.reshape(n_kb, self.bank_size, N)

        k_high      = min(max(1, int(np.round((1.0 - self.sparsity) * self.bank_size))), self.bank_size)
        thresh_rank = self.bank_size - k_high   # elements ranked >= this → high-prec

        result_blk = np.zeros((n_kb, self.bank_size, N), dtype=np.float32)

        for kb in range(n_kb):
            W_b   = W_blk[kb]    # (bank_size, N)
            imp_b = imp_blk[kb]  # (bank_size, N)

            # Per-column top-k mask via rank
            # rank_b[i,n] = rank of element i within column n (0 = lowest importance)
            rank_b = np.argsort(np.argsort(imp_b, axis=0), axis=0)   # (bank_size, N)
            mask_b = rank_b >= thresh_rank                             # True = high-prec

            # ── High-precision sub-stream ─────────────────────────────────
            absmax_h = np.where(mask_b, np.abs(W_b), 0.0).max(axis=0)   # (N,)
            scale_h  = _pot_scale_vec(absmax_h, self._qmax_high)         # (N,)
            dq_h = np.where(
                mask_b,
                self._enc_high(W_b, scale_h[np.newaxis, :]),
                0.0,
            )

            # ── Low-precision sub-stream ──────────────────────────────────
            # In hardware the integer stream stores v_sentinel at high-prec positions;
            # in simulation we simply zero-out those positions before dequantizing.
            absmax_l = np.where(~mask_b, np.abs(W_b), 0.0).max(axis=0)  # (N,)
            scale_l  = _pot_scale_vec(absmax_l, self._qmax_low)          # (N,)
            dq_l = np.where(
                ~mask_b,
                self._enc_low(W_b, scale_l[np.newaxis, :]),
                0.0,
            )

            result_blk[kb] = dq_h + dq_l   # non-overlapping by construction

            self._last_bank_scales.append({
                "scale_high": float(scale_h.mean()),
                "scale_low":  float(scale_l.mean()),
                "n_high":     int(mask_b.sum()),
                "n_low":      int((~mask_b).sum()),
            })

        return result_blk.reshape(K_pad, N)[:K]

    @staticmethod
    def _encode_subset(vals: np.ndarray, enc, qmax) -> tuple:
        """Quantize a flat subset via (enc, qmax) with per-group POT scale.

        Returns (dequantized, scale) where scale == 1.0 for empty/zero input.
        """
        absmax = float(np.max(np.abs(vals))) if vals.size else 0.0
        if absmax == 0:
            return np.zeros_like(vals, dtype=np.float32), 1.0
        scale = _pot_scale(absmax, qmax)
        return enc(vals, scale), scale

    def _quantize_1d(self, W: np.ndarray, importance: np.ndarray) -> np.ndarray:
        """1D path: contiguous element banks of bank_size elements."""
        flat = W.ravel()
        imp  = importance.ravel()
        n    = len(flat)
        result_flat = np.zeros(n, dtype=np.float32)

        for start in range(0, n, self.bank_size):
            end      = min(start + self.bank_size, n)
            bsz      = end - start
            bank_W   = flat[start:end]
            bank_imp = imp[start:end]

            mask = self._bank_mask_1d(bank_imp, bsz)
            bank_result = np.zeros(bsz, dtype=np.float32)

            s_h, s_l = 1.0, 1.0
            if np.any(mask):
                bank_result[mask],  s_h = self._encode_subset(bank_W[mask],  self._enc_high, self._qmax_high)
            if np.any(~mask):
                bank_result[~mask], s_l = self._encode_subset(bank_W[~mask], self._enc_low,  self._qmax_low)

            self._last_bank_scales.append({
                "scale_high": s_h,
                "scale_low":  s_l,
                "n_high":     int(np.sum(mask)),
                "n_low":      int(np.sum(~mask)),
            })
            result_flat[start:end] = bank_result

        return result_flat.reshape(W.shape)

    def dequantize(self, q: np.ndarray) -> np.ndarray:
        return q.astype(np.float32)

    def encoding_overhead(self) -> dict:
        high_frac = 1.0 - self.sparsity
        low_frac  = self.sparsity
        high_cost = high_frac * self.high_bits
        low_cost  = low_frac  * self.low_bits
        # Sentinel mask: high-prec positions are flagged by v_sentinel in the
        # low-prec integer stream.  No separate boolean mask array is stored.
        mask_cost = 0.0
        total = high_cost + low_cost + mask_cost
        return {
            "data_bits_per_element":     total,
            "metadata_bits_per_element": mask_cost,
            "high_bits":                 self.high_bits,
            "low_bits":                  self.low_bits,
            "sparsity":                  self.sparsity,
            "v_sentinel":                self.v_sentinel,
            # backward-compat keys
            "dense_bits":                self.low_bits,
            "sparse_bits":               self.high_bits,
            "sparsity_ratio":            self.sparsity_ratio,
            "bandwidth_amplification":   total / self.low_bits,
        }


# ── Algorithm 2: Activation-Aware Weight Quantization ─────────────────────────

class SQFormatActivations:
    """SQ-Format activation-aware weight quantizer — Algorithm 2 (Static Calibration).

    CALIBRATION-TIME: quantize_weights() determines the precision mask from
    joint weight-activation importance and returns a reordered quantized weight
    matrix together with the metadata needed at inference time.

    INFERENCE-TIME: quantize_runtime_activations() applies the calibration mask
    to a live activation batch.  Crucially, the quantization scale for each
    token is computed from THAT token's own values only — scales are per-token
    (per-row), not per-column-across-the-batch.  This prevents the batch size
    from influencing the quantization of any individual token.

    Importance metric (Algorithm 2, line 4 — paper formula):
        Ij = |Āj · Σᵢ W'_{j,i}|,  ∀j ∈ bank

    This is the signed row sum multiplied by the activation, then take absolute
    value.  It faithfully implements the paper's formula.  The previous
    abs-row-sum variant (|Āj| · Σᵢ|W'_{j,i}|) was mathematically distinct and
    has been reverted.

    Parameters
    ----------
    bank_size : int   — input channels per bank (b). Default 128.
    sparsity  : float — fraction of channels using LOW precision. Default 0.5.
    high_bits : int   — hhigh. Default 8.
    low_bits  : int   — hlow.  Default 4.
    alpha     : float — SmoothQuant migration strength. Default 0.5.
    """

    def __init__(
        self,
        bank_size: int = 128,
        sparsity: float = 0.5,
        high_bits: int = 8,
        low_bits: int = 4,
        alpha: float = 0.5,
    ):
        self.bank_size = bank_size
        self.sparsity  = sparsity
        self.high_bits = high_bits
        self.low_bits  = low_bits
        self.alpha     = alpha
        self.name      = "SQ-Format-A"
        self.bits      = low_bits
        self.dense_bits    = low_bits
        self.sparse_bits   = high_bits
        self.sparsity_ratio = 1.0 - sparsity

    def _bank_mask(self, importance: np.ndarray, bsz: int) -> np.ndarray:
        k_high = min(max(1, int(np.round((1.0 - self.sparsity) * bsz))), bsz)
        if k_high >= bsz:
            return np.ones(bsz, dtype=bool)
        mask = np.zeros(bsz, dtype=bool)
        mask[np.argpartition(importance, -k_high)[-k_high:]] = True
        return mask

    def quantize_weights(
        self,
        W: np.ndarray,
        A_mean: np.ndarray,
    ) -> tuple:
        """Algorithm 2: activation-guided mixed-precision weight quantization.

        Parameters
        ----------
        W      : (K, N) weight matrix.
        A_mean : (K,) mean calibration activation per input channel.

        Returns
        -------
        W_quant      : (K, N) quantized W', rows reordered high-prec first.
        scales       : (K,) per-channel POT quantization scales (reordered).
        mask         : (K,) bool high-prec flags (reordered).
        reorder_idx  : (K,) permutation; pass to quantize_runtime_activations().
        sq_scales    : (K,) SmoothQuant per-channel scales in ORIGINAL channel
                       order (NOT reordered).  Pass directly to
                       quantize_runtime_activations() — it applies them as
                       X_smooth = X * sq_scales BEFORE column reordering.
        """
        W      = np.asarray(W,      dtype=np.float32)
        A_mean = np.asarray(A_mean, dtype=np.float32).ravel()
        K, N   = W.shape

        # Step 1: W', Ā ← Smooth(W, D)
        W_smooth, A_smooth, _sq_scales = _smooth(W, A_mean, self.alpha)

        # Step 2: Per-channel importance — paper formula (Algorithm 2, line 4)
        # Ij = |Āj · Σᵢ W'_{j,i}|   (signed row sum, then absolute value)
        W_row_sum  = np.sum(W_smooth, axis=1)       # (K,) signed Σᵢ W'_{j,i}
        importance = np.abs(A_smooth * W_row_sum)   # (K,) |Āj · Σᵢ W'_{j,i}|

        # Step 3: Generate precision mask per bank
        mask = np.zeros(K, dtype=bool)
        for start in range(0, K, self.bank_size):
            end = min(start + self.bank_size, K)
            mask[start:end] = self._bank_mask(importance[start:end], end - start)

        # Step 4: Per-channel mixed-precision quantization
        W_quant = np.zeros_like(W_smooth)
        scales  = np.zeros(K, dtype=np.float32)

        for j in range(K):
            bits_j = self.high_bits if mask[j] else self.low_bits
            q_max  = 2 ** (bits_j - 1) - 1
            row    = W_smooth[j]
            absmax = float(np.max(np.abs(row)))
            if absmax == 0:
                W_quant[j] = 0.0
                scales[j]  = 1.0
            else:
                s        = _pot_scale(absmax, q_max)
                scales[j] = s
                q        = np.clip(np.round(row / s).astype(np.int32), -q_max, q_max)
                W_quant[j] = q.astype(np.float32) * s

        # Step 5: Reorder rows — high-prec channels first within each bank
        reorder_idx = np.arange(K)
        for start in range(0, K, self.bank_size):
            end          = min(start + self.bank_size, K)
            local_mask   = mask[start:end]
            local_order  = np.concatenate([np.where(local_mask)[0], np.where(~local_mask)[0]])
            reorder_idx[start:end] = start + local_order

        # sq_scales is returned in ORIGINAL channel order (not reordered).
        # quantize_runtime_activations applies it as X_smooth = X * sq_scales
        # BEFORE column-reordering, so original order is required.
        return W_quant[reorder_idx], scales[reorder_idx], mask[reorder_idx], reorder_idx, _sq_scales

    def quantize_runtime_activations(
        self,
        X_runtime: np.ndarray,
        mask: np.ndarray,
        sq_scales: np.ndarray,
        reorder_idx: np.ndarray,
    ) -> tuple:
        """Quantize runtime activations using calibration-derived mask (per-token scale).

        Each token's quantization scale is computed from THAT token's own
        activation values across its high- or low-precision channels.  Tokens
        are therefore quantized independently of each other and the result is
        invariant to batch composition or batch size.

        Parameters
        ----------
        X_runtime   : (B, K) or (K,) runtime activation matrix.
        mask        : (K,) bool — high-prec flags in REORDERED space, as
                      returned by quantize_weights().
        sq_scales   : (K,) SmoothQuant per-channel scales from _smooth().
                      Activation side: X_smooth = X * sq_scales.
        reorder_idx : (K,) column permutation from quantize_weights().

        Returns
        -------
        X_high      : (B, K) hhigh-bit quantized activations (reordered cols).
        X_low       : (B, K) hlow-bit  quantized activations (reordered cols).
        X_reordered : (B, K) smoothed + reordered (pre-split, for debugging).
        """
        X = np.asarray(X_runtime, dtype=np.float32)
        squeeze = X.ndim == 1
        if squeeze:
            X = X[np.newaxis, :]  # (1, K)

        # Step 1: Activation-side SmoothQuant scaling
        X_smooth    = X * sq_scales[np.newaxis, :]   # (B, K)

        # Step 2: Reorder columns to align with reordered weight rows
        X_reordered = X_smooth[:, reorder_idx]       # (B, K)

        high_cols = np.where(mask)[0]
        low_cols  = np.where(~mask)[0]

        X_high = np.zeros_like(X_reordered)
        X_low  = np.zeros_like(X_reordered)

        q_max_h = 2 ** (self.high_bits - 1) - 1
        q_max_l = 2 ** (self.low_bits  - 1) - 1

        # Step 3: Per-token (per-row) quantization
        # scale[b] = f(X_reordered[b, <high or low cols>]) only — batch-independent
        if len(high_cols) > 0:
            X_h      = X_reordered[:, high_cols]                              # (B, n_high)
            absmax_h = np.max(np.abs(X_h), axis=1, keepdims=True)             # (B, 1)
            scale_h  = _pot_scale_vec(absmax_h.ravel(), q_max_h).reshape(-1, 1)  # (B, 1)
            scale_h  = np.maximum(scale_h, 1e-38)
            q_h = np.clip(np.round(X_h / scale_h).astype(np.int32), -q_max_h, q_max_h)
            X_high[:, high_cols] = q_h.astype(np.float32) * scale_h

        if len(low_cols) > 0:
            X_l      = X_reordered[:, low_cols]                               # (B, n_low)
            absmax_l = np.max(np.abs(X_l), axis=1, keepdims=True)             # (B, 1)
            scale_l  = _pot_scale_vec(absmax_l.ravel(), q_max_l).reshape(-1, 1)  # (B, 1)
            scale_l  = np.maximum(scale_l, 1e-38)
            q_l = np.clip(np.round(X_l / scale_l).astype(np.int32), -q_max_l, q_max_l)
            X_low[:, low_cols] = q_l.astype(np.float32) * scale_l

        if squeeze:
            return X_high[0], X_low[0], X_reordered[0]
        return X_high, X_low, X_reordered

    def quantize(self, x: np.ndarray, bits: int = None) -> np.ndarray:
        """Simplified single-tensor interface for distribution-testing harness.

        Without calibration data, uses magnitude-based importance:
          2D input (K, N): per-channel (row) importance → channel-level selection.
          1D input       : per-element importance → element-level selection.
        """
        x = np.asarray(x, dtype=np.float32)

        if x.ndim == 2:
            K, N   = x.shape
            result = np.zeros_like(x)
            for start in range(0, K, self.bank_size):
                end  = min(start + self.bank_size, K)
                bank = x[start:end]   # (bsz, N)
                ch_imp = np.max(np.abs(bank), axis=1)   # (bsz,)
                mask   = self._bank_mask(ch_imp, end - start)
                bank_r = np.zeros_like(bank)
                for j in range(end - start):
                    bank_r[j] = _int_quantize_pot(bank[j], self.high_bits if mask[j] else self.low_bits)
                result[start:end] = bank_r
            return result

        flat = x.ravel()
        n    = len(flat)
        result_flat = np.zeros(n, dtype=np.float32)
        for start in range(0, n, self.bank_size):
            end  = min(start + self.bank_size, n)
            bank = flat[start:end]
            mask = self._bank_mask(bank ** 2, end - start)
            bank_r = np.zeros(end - start, dtype=np.float32)
            if np.any(mask):
                bank_r[mask]  = _int_quantize_pot(bank[mask],  self.high_bits)
            if np.any(~mask):
                bank_r[~mask] = _int_quantize_pot(bank[~mask], self.low_bits)
            result_flat[start:end] = bank_r
        return result_flat.reshape(x.shape)

    def dequantize(self, q: np.ndarray) -> np.ndarray:
        return q.astype(np.float32)

    def encoding_overhead(self) -> dict:
        high_frac = 1.0 - self.sparsity
        low_frac  = self.sparsity
        total = high_frac * self.high_bits + low_frac * self.low_bits + 1.0  # +1 mask/channel
        return {
            "data_bits_per_element":     total,
            "metadata_bits_per_element": 1.0,
            "high_bits":                 self.high_bits,
            "low_bits":                  self.low_bits,
            "sparsity":                  self.sparsity,
            "bandwidth_amplification":   total / self.low_bits,
        }


# ── SQ-Format-FP: FP8 E4M3 high-precision + INT low-precision ─────────────────

class SQFormatFP:
    """SQ-Format variant using FP8 E4M3 for high-precision elements.

    Same bank-based importance selection as SQFormat (Algorithm 1), but the
    top (1-s) fraction of elements per bank are quantized to FP8 E4M3 with
    an E8M0 shared scale, rather than INT8.  The remaining s fraction uses
    INT low-precision with a POT scale.

    This gives better dynamic range for the high-precision stream (FP8 covers
    6× more orders of magnitude than INT8 for the same bit-width), at the cost
    of the element-level non-uniformity that FP quantization introduces.

    Parameters
    ----------
    bank_size : int   — elements per bank (b). Default 128.
    sparsity  : float — fraction using low precision (s). Default 0.5.
    low_bits  : int   — low-precision INT bit-width. Default 4.
    """

    _FP8_E4M3_MAX = 448.0
    _LOG2_FP8_MAX = 8  # floor(log2(448)) = 8

    def __init__(self, bank_size: int = 128, sparsity: float = 0.5, low_bits: int = 4):
        self.bank_size = bank_size
        self.sparsity  = sparsity
        self.low_bits  = low_bits
        self.name      = "SQ-Format-FP"
        self.bits      = low_bits
        self._q_max_l  = 2 ** (low_bits - 1) - 1
        self._last_bank_scales: list = []

    def _fp8_e8m0_scale(self, absmax: float) -> float:
        """E8M0 scale for FP8 E4M3: 2^(floor(log2(absmax)) - 8)."""
        if absmax <= 0:
            return 1.0
        log2_abs = int(np.floor(np.log2(float(absmax) + 1e-38)))
        return float(2.0 ** (log2_abs - self._LOG2_FP8_MAX))

    def _quantize_fp8_group(self, vals: np.ndarray) -> np.ndarray:
        """Quantize a group of values to FP8 E4M3 with a shared E8M0 scale."""
        from formats.mxfp import _fp8_e4m3_vec
        absmax = float(np.max(np.abs(vals)))
        if absmax == 0:
            return np.zeros_like(vals)
        scale = self._fp8_e8m0_scale(absmax)
        return _fp8_e4m3_vec(vals / scale) * scale

    def _bank_mask_1d(self, importance: np.ndarray, bsz: int) -> np.ndarray:
        k_high = min(max(1, int(np.round((1.0 - self.sparsity) * bsz))), bsz)
        if k_high >= bsz:
            return np.ones(bsz, dtype=bool)
        mask = np.zeros(bsz, dtype=bool)
        mask[np.argpartition(importance, -k_high)[-k_high:]] = True
        return mask

    def quantize(self, W: np.ndarray, bits: int = None) -> np.ndarray:
        W = W.astype(np.float32)
        importance = W ** 2
        self._last_bank_scales = []

        if W.ndim == 2:
            return self._quantize_2d(W, importance)
        else:
            return self._quantize_1d(W.ravel(), importance.ravel()).reshape(W.shape)

    def _quantize_2d(self, W: np.ndarray, importance: np.ndarray) -> np.ndarray:
        """2D path: bank_size elements per output column, FP8 high / INT low."""
        K, N = W.shape
        K_pad = int(np.ceil(K / self.bank_size)) * self.bank_size
        pad   = K_pad - K
        n_kb  = K_pad // self.bank_size

        if pad > 0:
            W_work   = np.vstack([W,          np.zeros((pad, N), dtype=np.float32)])
            imp_work = np.vstack([importance,  np.zeros((pad, N), dtype=np.float32)])
        else:
            W_work, imp_work = W, importance

        W_blk   = W_work.reshape(n_kb, self.bank_size, N)
        imp_blk = imp_work.reshape(n_kb, self.bank_size, N)

        q_max_l     = self._q_max_l
        k_high      = min(max(1, int(np.round((1.0 - self.sparsity) * self.bank_size))), self.bank_size)
        thresh_rank = self.bank_size - k_high

        result_blk = np.zeros((n_kb, self.bank_size, N), dtype=np.float32)

        for kb in range(n_kb):
            W_b   = W_blk[kb]    # (bank_size, N)
            imp_b = imp_blk[kb]

            rank_b = np.argsort(np.argsort(imp_b, axis=0), axis=0)
            mask_b = rank_b >= thresh_rank  # True = high-prec (FP8)

            dq_h = np.zeros_like(W_b)
            dq_l = np.zeros_like(W_b)

            # High-precision: FP8 E4M3 with per-column E8M0 scale
            for n in range(N):
                col_vals = W_b[:, n][mask_b[:, n]]
                if len(col_vals) > 0:
                    dq_h[mask_b[:, n], n] = self._quantize_fp8_group(col_vals)

            # Low-precision: INT with per-column POT scale
            for n in range(N):
                col_vals = W_b[:, n][~mask_b[:, n]]
                if len(col_vals) > 0:
                    absmax = float(np.max(np.abs(col_vals)))
                    scale  = _pot_scale(absmax, q_max_l)
                    q      = np.clip(np.round(col_vals / scale).astype(np.int32), -q_max_l, q_max_l)
                    dq_l[~mask_b[:, n], n] = q.astype(np.float32) * scale

            result_blk[kb] = dq_h + dq_l
            self._last_bank_scales.append({
                "n_high": int(mask_b.sum()),
                "n_low":  int((~mask_b).sum()),
            })

        return result_blk.reshape(K_pad, N)[:K]

    def _quantize_1d(self, flat: np.ndarray, importance: np.ndarray) -> np.ndarray:
        """1D path: contiguous element banks."""
        n = len(flat)
        result = np.zeros(n, dtype=np.float32)

        for start in range(0, n, self.bank_size):
            end      = min(start + self.bank_size, n)
            bsz      = end - start
            bank_W   = flat[start:end]
            bank_imp = importance[start:end]

            mask = self._bank_mask_1d(bank_imp, bsz)
            bank_r = np.zeros(bsz, dtype=np.float32)

            if np.any(mask):
                bank_r[mask] = self._quantize_fp8_group(bank_W[mask])
            if np.any(~mask):
                vals   = bank_W[~mask]
                absmax = float(np.max(np.abs(vals)))
                scale  = _pot_scale(absmax, self._q_max_l)
                q      = np.clip(np.round(vals / scale).astype(np.int32), -self._q_max_l, self._q_max_l)
                bank_r[~mask] = q.astype(np.float32) * scale

            self._last_bank_scales.append({
                "n_high": int(np.sum(mask)),
                "n_low":  int(np.sum(~mask)),
            })
            result[start:end] = bank_r

        return result

    def dequantize(self, q: np.ndarray) -> np.ndarray:
        return q.astype(np.float32)

    def encoding_overhead(self) -> dict:
        high_frac = 1.0 - self.sparsity
        low_frac  = self.sparsity
        # FP8 high-prec = 8 bits; low-prec = low_bits INT
        total = high_frac * 8.0 + low_frac * self.low_bits
        return {
            "data_bits_per_element":     total,
            "metadata_bits_per_element": 0.0,  # sentinel in low-prec stream
            "high_bits":                 8,
            "low_bits":                  self.low_bits,
            "sparsity":                  self.sparsity,
            "bandwidth_amplification":   total / self.low_bits,
        }
