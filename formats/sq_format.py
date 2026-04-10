"""SQ-Format: Sparse-Quantized unified format.

Implements two algorithms from the SQ-format paper:

    SQ-format(X) = ([Xquant], [Squant], [m], hhigh, hlow, b, s)

Where:
  [Xquant], [Squant] : quantized matrix and scaling matrix
  [m]                : precision mask (True = high-precision channel/element)
  hhigh, hlow        : high/low precision bit-widths
  b                  : bank size
  s                  : sparsity — fraction of elements/channels using LOW precision
                       top (1-s) per bank → hhigh; remaining s → hlow

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Algorithm 1 — Weight Quantization
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Input : W ∈ R^{K×N}, calibration D, sparsity s, bank size b, hhigh, hlow
  1. W', H  ← Smooth(W, D)
  2. I      ← (W')² / (diag(H⁻¹))²           ← element-level importance
  3. for each bank w of W':
       mw  ← top (1-s) elements of Iw         ← precision mask
       (w'_high, s'_high) ← Quant(w' ⊙  mw, hhigh)
       (w'_low,  s'_low)  ← Quant(w' ⊙ ~mw, hlow)
  4. Output: (Whigh, Wlow, Shigh, Slow)

Hardware: low-prec path = full dense matmul (dominant); high-prec path =
  sparse gather+compute (small compute, overlaps with low-prec latency).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Algorithm 2 — Activation Quantization (Static)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Input : W ∈ R^{K×N}, calibration D, sparsity s, bank size b, hhigh, hlow
  1. W', Ā  ← Smooth(W, D)                    ← SmoothQuant on W and Ā
  2. for each bank w of W', ā of Ā:
       Ij   ← |Āj · Σᵢ W'_{j,i}|, ∀j ∈ bank  ← per-channel importance
       mw   ← top (1-s) channels of Ij          ← precision mask
       (w', s') ← Quant(w, hhigh, hlow)         ← mixed-precision per channel
  3. Reorder rows of W' by mask m               ← high-prec channels first/bank
  4. Output: (Wquant, Squant, m)

W and A share the same mask. Channels within each bank are ranked by |Āj·AW^T_j|.
"""

import numpy as np


# ── Low-level helpers ──────────────────────────────────────────────────────────

def _pot_scale(absmax: float, q_max: int) -> float:
    """OCP-aligned power-of-two scale: 2^(floor(log2(absmax)) - floor(log2(q_max))).

    Guarantees q_max * scale >= absmax (no clipping) with finest step size.
    Both dense and sparse components use POT scales → scale division is a
    hardware arithmetic right-shift (no FP divider needed).
    """
    if absmax <= 0:
        return 1.0
    log2_absmax = int(np.floor(np.log2(float(absmax) + 1e-38)))
    log2_qmax   = int(np.floor(np.log2(float(q_max))))
    return float(2.0 ** (log2_absmax - log2_qmax))


def _int_quantize_pot(x: np.ndarray, bits: int) -> np.ndarray:
    """Symmetric integer quantization with POT scale (hardware-friendly)."""
    q_max = 2 ** (bits - 1) - 1
    absmax = float(np.max(np.abs(x)))
    if absmax == 0:
        return np.zeros_like(x, dtype=np.float32)
    scale = _pot_scale(absmax, q_max)
    q = np.round(x / scale).astype(np.int32)
    q = np.clip(q, -q_max, q_max)
    return q.astype(np.float32) * scale


def _smooth(W: np.ndarray, A_mean: np.ndarray, alpha: float = 0.5):
    """SmoothQuant channel-wise scaling.

    Computes per-input-channel scale s_j = max(|Ā_j|)^α / max(|W_j|)^(1-α).
    Returns W' = W / s (weight absorbed offline) and Ā' = Ā * s (activation scaled).
    W and A share the same scale vector, so they share the same precision mask.

    Parameters
    ----------
    W      : (K, N) weight matrix
    A_mean : (K,) mean calibration activation per input channel
    alpha  : migration strength (0.5 default)

    Returns
    -------
    W_smooth  : (K, N) smoothed weight W'
    A_smooth  : (K,) smoothed mean activation Ā'
    sq_scales : (K,) per-channel SmoothQuant scales
    """
    A_mean = A_mean.ravel().astype(np.float32)   # (K,)
    W = W.astype(np.float32)                      # (K, N)

    x_max = np.abs(A_mean)                        # per-input-channel activation max
    w_max = np.max(np.abs(W), axis=1)             # per-input-channel weight max

    x_max = np.maximum(x_max, 1e-8)
    w_max = np.maximum(w_max, 1e-8)

    sq_scales = (x_max ** alpha) / (w_max ** (1.0 - alpha))  # (K,)
    W_smooth  = W / sq_scales[:, np.newaxis]                  # W' = W / s
    A_smooth  = A_mean * sq_scales                            # Ā' = Ā * s

    return W_smooth, A_smooth, sq_scales


# ── Algorithm 1: Weight Quantization ──────────────────────────────────────────

class SQFormat:
    """SQ-Format weight quantization — Algorithm 1.

    Applies bank-based mixed-precision quantization driven by element-level
    importance.  Within each bank of b elements:
      • top (1-s) elements by importance → hhigh (high-precision)
      • remaining s elements             → hlow  (low-precision)

    Importance metric (Algorithm 1, line 3):
      With Hessian H:    I_ij = (W'_ij)² / (diag(H⁻¹)_i)²
      Without Hessian:   I_ij = (W_ij)²   (magnitude-based fallback)

    Hardware model
    ──────────────
    Low-prec  path: full dense matmul with hlow-bit weights (dominant compute).
    High-prec path: gather top-(1-s) elements, compute with hhigh-bit weights.
    Both paths run in parallel; high-prec latency is hidden behind low-prec.
    Fixed bank size b avoids load imbalance and distributed-accumulator issues.

    Parameters
    ----------
    bank_size     : int   — elements per bank (b). Default 128.
    sparsity      : float — fraction using LOW precision (s). Default 0.5.
                            Top (1-s) per bank → hhigh; rest → hlow.
    high_bits     : int   — hhigh bit-width. Default 8.
    low_bits      : int   — hlow  bit-width. Default 4.

    Backward-compatible aliases (old parameter names still accepted):
    dense_bits    : int   — alias for low_bits.
    sparse_bits   : int   — alias for high_bits.
    sparsity_ratio: float — OLD convention: fraction in HIGH precision = (1-s).
                            If provided, sparsity = 1 - sparsity_ratio.
    """

    def __init__(
        self,
        bank_size: int = 128,
        sparsity: float = None,
        high_bits: int = None,
        low_bits: int = None,
        # Backward-compatible parameter names
        dense_bits: int = None,
        sparse_bits: int = None,
        sparsity_ratio: float = None,
    ):
        # Resolve low-precision bit-width (hlow)
        if low_bits is not None:
            self.low_bits = low_bits
        elif dense_bits is not None:
            self.low_bits = dense_bits
        else:
            self.low_bits = 4

        # Resolve high-precision bit-width (hhigh)
        if high_bits is not None:
            self.high_bits = high_bits
        elif sparse_bits is not None:
            self.high_bits = sparse_bits
        else:
            self.high_bits = 8

        # Resolve sparsity s (fraction going to LOW precision)
        if sparsity is not None:
            self.sparsity = float(sparsity)
        elif sparsity_ratio is not None:
            # Old convention: sparsity_ratio = 1 - s (fraction in HIGH precision)
            self.sparsity = 1.0 - float(sparsity_ratio)
        else:
            self.sparsity = 0.5  # default: 2:4-sparse-equivalent

        self.bank_size = bank_size

        # Backward-compatible attribute aliases
        self.dense_bits    = self.low_bits
        self.sparse_bits   = self.high_bits
        self.sparsity_ratio = 1.0 - self.sparsity   # fraction in HIGH precision

        self.name = "SQ-Format"
        self.bits = self.low_bits   # dominant (low-prec) effective bit-width

    def _bank_mask(self, importance: np.ndarray, bank_size_actual: int) -> np.ndarray:
        """Return bool mask selecting top (1-s) elements by importance within a bank."""
        k_high = max(1, int(np.round((1.0 - self.sparsity) * bank_size_actual)))
        k_high = min(k_high, bank_size_actual)
        if k_high >= bank_size_actual:
            return np.ones(bank_size_actual, dtype=bool)
        high_idx = np.argpartition(importance, -k_high)[-k_high:]
        mask = np.zeros(bank_size_actual, dtype=bool)
        mask[high_idx] = True
        return mask

    def quantize(
        self,
        W: np.ndarray,
        H_inv_diag: np.ndarray = None,
        bits: int = None,
    ) -> np.ndarray:
        """Algorithm 1: bank-based mixed-precision weight quantization.

        Parameters
        ----------
        W          : weight matrix (K×N) or flat 1-D vector.
        H_inv_diag : (K,) diagonal of H⁻¹ for Hessian-based importance.
                     If None, falls back to magnitude-based importance (W²).
        bits       : ignored (interface compatibility with other quantizers).

        Returns
        -------
        np.ndarray — dequantized approximation of W, same shape as input.
        """
        W = W.astype(np.float32)
        original_shape = W.shape
        flat = W.ravel()
        n = len(flat)

        # ── Compute element-level importance (Algorithm 1, line 3) ──────────
        # I = (W')² / (diag(H⁻¹))²   with Hessian
        # I = W²                       fallback (magnitude)
        if H_inv_diag is not None and W.ndim == 2:
            h_inv = np.asarray(H_inv_diag, dtype=np.float32).ravel()   # (K,)
            importance = (W ** 2) / (h_inv[:, np.newaxis] ** 2 + 1e-38)
            importance_flat = importance.ravel()
        else:
            importance_flat = flat ** 2

        # ── Bank-based quantization (Algorithm 1, lines 4-8) ────────────────
        result_flat = np.zeros(n, dtype=np.float32)

        for bank_start in range(0, n, self.bank_size):
            bank_end = min(bank_start + self.bank_size, n)
            bsz = bank_end - bank_start
            bank_W   = flat[bank_start:bank_end]
            bank_imp = importance_flat[bank_start:bank_end]

            # Generate precision mask mw: top (1-s) elements → high precision
            mask = self._bank_mask(bank_imp, bsz)

            # Quant(w' ⊙  mask, hhigh) — Algorithm 1, line 6
            bank_result = np.zeros(bsz, dtype=np.float32)
            if np.any(mask):
                bank_result[mask] = _int_quantize_pot(bank_W[mask], self.high_bits)

            # Quant(w' ⊙ ~mask, hlow) — Algorithm 1, line 7
            if np.any(~mask):
                bank_result[~mask] = _int_quantize_pot(bank_W[~mask], self.low_bits)

            result_flat[bank_start:bank_end] = bank_result

        return result_flat.reshape(original_shape)

    def dequantize(self, q: np.ndarray) -> np.ndarray:
        return q.astype(np.float32)

    def encoding_overhead(self) -> dict:
        high_frac = 1.0 - self.sparsity   # fraction in high precision
        low_frac  = self.sparsity          # fraction in low  precision
        high_cost = high_frac * self.high_bits
        low_cost  = low_frac  * self.low_bits
        mask_cost = 1.0   # 1 bit per element for the precision mask
        total = high_cost + low_cost + mask_cost
        return {
            "data_bits_per_element":     total,
            "metadata_bits_per_element": mask_cost,
            "high_bits":                 self.high_bits,
            "low_bits":                  self.low_bits,
            "sparsity":                  self.sparsity,
            # backward-compat keys
            "dense_bits":                self.low_bits,
            "sparse_bits":               self.high_bits,
            "sparsity_ratio":            self.sparsity_ratio,
            "bandwidth_amplification":   total / self.low_bits,
        }


# ── Algorithm 2: Activation Quantization ──────────────────────────────────────

class SQFormatActivations:
    """SQ-Format activation quantization — Algorithm 2 (Static).

    Determines per-channel quantization precision using the combined
    weight-activation importance metric:

        Ij = |Āj · Σᵢ W'_{j,i}|,  ∀j ∈ bank

    where Āj is the SmoothQuant-scaled mean calibration activation for
    input channel j, and Σᵢ W'_{j,i} is the row sum of the smoothed weight.
    This captures the output contribution of channel j: an outlier activation
    on a large-weight channel amplifies errors more than on a small-weight one.

    Within each bank of b input channels:
      • top (1-s) channels by Ij → hhigh (high-precision per-channel scale)
      • remaining s channels     → hlow  (low-precision per-channel scale)

    W and A share the same precision mask (same channel selection).

    After quantization, rows of W' are reordered so high-precision channels
    come first within each bank, enabling hardware-efficient sequential access
    without a scatter pattern at inference time.

    Parameters
    ----------
    bank_size : int   — input channels per bank (b). Default 128.
    sparsity  : float — fraction of channels using LOW precision (s). Default 0.5.
    high_bits : int   — hhigh bit-width. Default 8.
    low_bits  : int   — hlow  bit-width. Default 4.
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
        # Backward-compatible aliases
        self.dense_bits    = low_bits
        self.sparse_bits   = high_bits
        self.sparsity_ratio = 1.0 - sparsity

    def _bank_mask(self, importance: np.ndarray, bank_size_actual: int) -> np.ndarray:
        """Return bool mask selecting top (1-s) channels by importance within a bank."""
        k_high = max(1, int(np.round((1.0 - self.sparsity) * bank_size_actual)))
        k_high = min(k_high, bank_size_actual)
        if k_high >= bank_size_actual:
            return np.ones(bank_size_actual, dtype=bool)
        high_idx = np.argpartition(importance, -k_high)[-k_high:]
        mask = np.zeros(bank_size_actual, dtype=bool)
        mask[high_idx] = True
        return mask

    def quantize_weights(
        self,
        W: np.ndarray,
        A_mean: np.ndarray,
    ) -> tuple:
        """Algorithm 2: activation-guided mixed-precision weight quantization.

        Parameters
        ----------
        W      : (K, N) weight matrix  (K input channels, N output channels).
        A_mean : (K,) mean calibration activation per input channel (Ā).

        Returns
        -------
        W_quant      : (K, N) — quantized W', rows reordered (high-prec first/bank).
        scales       : (K,)   — per-channel POT quantization scales (reordered).
        mask         : (K,) bool — True = high-precision channel (reordered).
        reorder_idx  : (K,)   — row permutation applied; invert to restore order.
        """
        W      = np.asarray(W,      dtype=np.float32)
        A_mean = np.asarray(A_mean, dtype=np.float32).ravel()
        K, N   = W.shape

        # ── Step 1: W', Ā ← Smooth(W, D) ───────────────────────────────────
        W_smooth, A_smooth, _sq_scales = _smooth(W, A_mean, self.alpha)

        # ── Step 2: Per-channel importance within each bank ──────────────────
        # Ij = |Āj · Σᵢ W'_{j,i}|  (Algorithm 2, line 4)
        W_row_sum  = np.sum(W_smooth, axis=1)              # (K,) Σᵢ W'_{j,i}
        importance = np.abs(A_smooth * W_row_sum)          # (K,) per-channel

        # ── Step 3: Generate precision mask per bank (Algorithm 2, line 5) ──
        mask = np.zeros(K, dtype=bool)
        for bank_start in range(0, K, self.bank_size):
            bank_end = min(bank_start + self.bank_size, K)
            bsz = bank_end - bank_start
            mask[bank_start:bank_end] = self._bank_mask(
                importance[bank_start:bank_end], bsz
            )

        # ── Step 4: Mixed-precision quantization of W' (Algorithm 2, line 6) ─
        # (w', s') ← Quant(w, hhigh, hlow) with per-channel POT scale
        W_quant = np.zeros_like(W_smooth)
        scales  = np.zeros(K, dtype=np.float32)

        for j in range(K):
            bits  = self.high_bits if mask[j] else self.low_bits
            q_max = 2 ** (bits - 1) - 1
            row   = W_smooth[j]
            absmax = float(np.max(np.abs(row)))
            if absmax == 0:
                W_quant[j] = 0.0
                scales[j]  = 1.0
            else:
                scale      = _pot_scale(absmax, q_max)
                scales[j]  = scale
                q          = np.round(row / scale).astype(np.int32)
                q          = np.clip(q, -q_max, q_max)
                W_quant[j] = q.astype(np.float32) * scale

        # ── Step 5: Reorder rows by mask (Algorithm 2, line 8) ──────────────
        # High-precision channels first within each bank for hardware efficiency
        reorder_idx = np.arange(K)
        for bank_start in range(0, K, self.bank_size):
            bank_end    = min(bank_start + self.bank_size, K)
            local_mask  = mask[bank_start:bank_end]
            local_high  = np.where( local_mask)[0]
            local_low   = np.where(~local_mask)[0]
            local_order = np.concatenate([local_high, local_low])
            reorder_idx[bank_start:bank_end] = bank_start + local_order

        W_quant_reordered = W_quant[reorder_idx]
        scales_reordered  = scales[reorder_idx]
        mask_reordered    = mask[reorder_idx]

        return W_quant_reordered, scales_reordered, mask_reordered, reorder_idx

    def quantize(self, x: np.ndarray, bits: int = None) -> np.ndarray:
        """Simplified single-tensor interface for distribution-testing harness.

        Without calibration activation data, falls back to magnitude-based
        per-bank channel importance.  This allows SQFormatActivations to be
        evaluated in the same benchmark harness as other quantizers.
        """
        x = np.asarray(x, dtype=np.float32)
        original_shape = x.shape
        flat = x.ravel()
        n    = len(flat)

        result_flat = np.zeros(n, dtype=np.float32)
        for bank_start in range(0, n, self.bank_size):
            bank_end = min(bank_start + self.bank_size, n)
            bsz  = bank_end - bank_start
            bank = flat[bank_start:bank_end]

            # Magnitude-based importance fallback
            mask = self._bank_mask(bank ** 2, bsz)

            bank_result = np.zeros(bsz, dtype=np.float32)
            if np.any(mask):
                bank_result[mask]  = _int_quantize_pot(bank[mask],  self.high_bits)
            if np.any(~mask):
                bank_result[~mask] = _int_quantize_pot(bank[~mask], self.low_bits)
            result_flat[bank_start:bank_end] = bank_result

        return result_flat.reshape(original_shape)

    def dequantize(self, q: np.ndarray) -> np.ndarray:
        return q.astype(np.float32)

    def encoding_overhead(self) -> dict:
        high_frac = 1.0 - self.sparsity
        low_frac  = self.sparsity
        high_cost = high_frac * self.high_bits
        low_cost  = low_frac  * self.low_bits
        mask_cost = 1.0   # 1 bit per channel
        total = high_cost + low_cost + mask_cost
        return {
            "data_bits_per_element":     total,
            "metadata_bits_per_element": mask_cost,
            "high_bits":                 self.high_bits,
            "low_bits":                  self.low_bits,
            "sparsity":                  self.sparsity,
            "bandwidth_amplification":   total / self.low_bits,
        }
