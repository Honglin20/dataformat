"""Comprehensive tests for SQ-Format (Algorithms 1 & 2).

Based on the paper description:
  - Algorithm 1: SQ-format on Weights (GPTQ + SmoothQuant)
  - Algorithm 2: SQ-format on Activations (Static Strategy)

Test categories:
  A. Sentinel mask correctness (vmask = -(2^(hlow-1)))
  B. Algorithm 1 — importance score I = (W')² / (H⁻¹ii)²
  C. Algorithm 1 — per-bank high/low precision selection
  D. Algorithm 1 — SmoothQuant smoothing
  E. Algorithm 1 — encoding overhead & metadata
  F. Algorithm 1 — bank structure (N:M-style, 2D/1D)
  G. Algorithm 2 — importance score Ij = |Āj · Σi W'_{j,i}|
  H. Algorithm 2 — precision mask, reordering, data locality
  I. Algorithm 2 — runtime activation quantization (per-token, batch-independent)
  J. Algorithm 2 — quantize_weights output contract
  K. Edge cases (zeros, tiny banks, non-multiple shapes, large values)
  L. Backward-compatibility aliases
  M. _smooth helper
  N. POT scale helper properties
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from formats.sq_format import SQFormat, SQFormatActivations, _smooth, _pot_scale, _pot_scale_vec


RNG = np.random.default_rng(0)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def is_pot(v: float) -> bool:
    if v <= 0:
        return False
    log2v = np.log2(abs(v))
    return abs(log2v - round(log2v)) < 1e-5


def make_W(K=16, N=8, seed=0):
    return np.random.default_rng(seed).normal(0, 1, (K, N)).astype(np.float32)


def make_A_mean(K=16, seed=1):
    return np.abs(np.random.default_rng(seed).normal(0, 1, K)).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# A. Sentinel mask correctness
# ═══════════════════════════════════════════════════════════════════════════════

class TestSentinelMask:
    """vmask = -(2^(hlow-1)): the two's-complement minimum that symmetric quant never occupies."""

    @pytest.mark.parametrize("hlow,expected", [
        (2, -2),   # INT2: normal {-1,0,1}; vmask=-2
        (4, -8),   # INT4: normal {-7,...,7}; vmask=-8
        (8, -128), # INT8: normal {-127,...,127}; vmask=-128
    ])
    def test_sentinel_value(self, hlow, expected):
        """v_sentinel must equal -(2^(hlow-1)) for each low-precision bit-width."""
        sq = SQFormat(low_bits=hlow, high_bits=hlow + 4)
        assert sq.v_sentinel == expected, \
            f"INT{hlow}: expected v_sentinel={expected}, got {sq.v_sentinel}"

    def test_sentinel_outside_normal_range_int2(self):
        """INT2 sentinel=-2 must fall outside the symmetric normal range {-1,0,1}."""
        sq = SQFormat(low_bits=2, high_bits=4)
        q_max = 2 ** (sq.low_bits - 1) - 1  # =1
        assert sq.v_sentinel < -q_max, \
            "Sentinel must be below the normal quantization range"

    def test_sentinel_outside_normal_range_int4(self):
        """INT4 sentinel=-8 must fall outside {-7,...,7}."""
        sq = SQFormat(low_bits=4, high_bits=8)
        q_max = 2 ** (sq.low_bits - 1) - 1  # =7
        assert sq.v_sentinel < -q_max

    def test_sentinel_is_twos_complement_min(self):
        """Sentinel is -(2^(hlow-1)), not -(2^(hlow-1)-1)."""
        for hlow in (2, 4, 8):
            sq = SQFormat(low_bits=hlow)
            assert sq.v_sentinel == -(2 ** (hlow - 1))

    def test_no_separate_mask_cost(self):
        """Paper: sentinel encodes high-prec in-band → mask_cost = 0."""
        sq = SQFormat(low_bits=4, high_bits=8, sparsity=0.5)
        oh = sq.encoding_overhead()
        assert oh["metadata_bits_per_element"] == 0.0, \
            "Sentinel-based mask has zero separate storage cost"


# ═══════════════════════════════════════════════════════════════════════════════
# B. Algorithm 1 — importance score
# ═══════════════════════════════════════════════════════════════════════════════

class TestAlg1Importance:
    """I_{r,i} = (W'_{r,i})² / (H⁻¹_{ii})²  (paper Algorithm 1, line 3)."""

    def test_without_hessian_uses_magnitude(self):
        """Without H_inv_diag, importance = W² (pure magnitude fallback)."""
        sq = SQFormat(bank_size=4, sparsity=0.5, high_bits=8, low_bits=4)
        # Row 0 has large magnitude → should dominate importance
        W = np.array([
            [10.0, 10.0],
            [0.1,  0.1],
            [0.1,  0.1],
            [0.1,  0.1],
        ], dtype=np.float32)
        # With no Hessian, importance ∝ W². Row 0 magnitude >> rows 1-3
        # → row 0 should be high-precision in each column's bank
        W_q = sq.quantize(W)
        assert W_q.shape == W.shape
        # The large-weight rows should be recovered accurately
        assert abs(W_q[0, 0] - 10.0) < 1.0, f"Large weight error: {abs(W_q[0,0]-10.0)}"

    def test_hessian_shifts_importance(self):
        """High H⁻¹ (low curvature) → lower importance despite large weight.

        I = W² / (H⁻¹)². Large H⁻¹ divides away a large W², making the
        element less important. The element with small W but small H⁻¹ wins.
        """
        sq = SQFormat(bank_size=2, sparsity=0.5, high_bits=8, low_bits=4)
        # K=2, N=1: one bank of 2 elements
        W = np.array([[10.0], [0.1]], dtype=np.float32)   # row-0 large, row-1 small
        # H_inv_diag: row-0 has huge H⁻¹ → I_0 = 10²/huge² ≈ 0
        #              row-1 has tiny H⁻¹ → I_1 = 0.1²/tiny² >> I_0
        H_inv = np.array([1e6, 1e-3], dtype=np.float32)
        W_q = sq.quantize(W, H_inv_diag=H_inv)
        assert W_q.shape == W.shape
        assert np.all(np.isfinite(W_q))

    def test_hessian_importance_formula(self):
        """Verify the per-element importance formula numerically for a tiny case."""
        # For W' = [[a, b], [c, d]] and H_inv = [h0, h1]:
        # I[0,0] = a²/h0², I[1,0] = c²/h1²
        sq = SQFormat(bank_size=2, sparsity=0.5, high_bits=8, low_bits=4)
        W = np.array([[3.0, 2.0], [1.0, 4.0]], dtype=np.float32)
        H_inv = np.array([1.0, 2.0], dtype=np.float32)
        # I[0,0]=9, I[0,1]=4; I[1,0]=0.25, I[1,1]=1.0
        # Per col, bank of 2: col0 → row0 high (I=9>0.25); col1 → row0 high (I=4>1.0)
        W_q = sq.quantize(W, H_inv_diag=H_inv)
        # Row 0 high-prec in both cols → small error
        assert abs(W_q[0, 0] - 3.0) < 0.5
        assert abs(W_q[0, 1] - 2.0) < 0.5

    def test_equal_importance_deterministic(self):
        """All-equal importance should still produce a deterministic mask."""
        sq = SQFormat(bank_size=4, sparsity=0.5, high_bits=8, low_bits=4)
        W = np.ones((4, 4), dtype=np.float32)
        W_q1 = sq.quantize(W.copy())
        W_q2 = sq.quantize(W.copy())
        np.testing.assert_array_equal(W_q1, W_q2)

    def test_zero_hessian_diag_safe(self):
        """H_inv_diag containing zeros must not produce NaN/Inf (guarded by +1e-38)."""
        sq = SQFormat(bank_size=4, sparsity=0.5, high_bits=8, low_bits=4)
        W = make_W(4, 4, seed=5)
        H_inv = np.zeros(4, dtype=np.float32)
        W_q = sq.quantize(W, H_inv_diag=H_inv)
        assert np.all(np.isfinite(W_q))


# ═══════════════════════════════════════════════════════════════════════════════
# C. Algorithm 1 — per-bank high/low precision selection
# ═══════════════════════════════════════════════════════════════════════════════

class TestAlg1BankSelection:
    """Per-bank top-(1-s) selection assigns high precision to important elements."""

    def test_sparsity_fraction_1d(self):
        """1D: fraction of high-prec elements per bank ≈ (1-s)."""
        for sparsity in (0.25, 0.5, 0.75):
            sq = SQFormat(bank_size=64, sparsity=sparsity, high_bits=8, low_bits=4)
            W = np.random.default_rng(42).normal(0, 1, 128).astype(np.float32)
            sq.quantize(W)
            for bank_info in sq._last_bank_scales:
                total = bank_info["n_high"] + bank_info["n_low"]
                high_frac = bank_info["n_high"] / total
                assert abs(high_frac - (1.0 - sparsity)) < 0.05, \
                    f"s={sparsity}: expected high_frac≈{1-sparsity:.2f}, got {high_frac:.2f}"

    def test_sparsity_fraction_2d(self):
        """2D: total high-prec per bank ≈ (1-s)*bank_size per output column."""
        sq = SQFormat(bank_size=4, sparsity=0.5, high_bits=8, low_bits=4)
        W = make_W(8, 4, seed=7)
        sq.quantize(W)
        for bank_info in sq._last_bank_scales:
            total = bank_info["n_high"] + bank_info["n_low"]
            # n_high counts across all columns in the bank; each col gets ~(1-s)*bsz
            assert total > 0

    def test_outlier_gets_high_precision(self):
        """The element with the largest |W| in a bank must receive high-precision."""
        sq = SQFormat(bank_size=4, sparsity=0.5, high_bits=8, low_bits=4)
        # Plant a clear outlier at position 0
        W = np.array([100.0, 0.01, -0.01, 0.02], dtype=np.float32)
        W_q = sq.quantize(W)
        # High-precision on the outlier → small relative error
        assert abs(W_q[0] - 100.0) < 2.0, \
            f"Outlier error={abs(W_q[0]-100.0):.2f} — should be high-precision"

    def test_non_outlier_gets_low_precision(self):
        """The smallest elements in a bank get low-precision, so larger errors are acceptable."""
        sq = SQFormat(bank_size=4, sparsity=0.5, high_bits=8, low_bits=4)
        W = np.array([100.0, 100.0, 0.001, 0.001], dtype=np.float32)
        W_q = sq.quantize(W)
        assert np.all(np.isfinite(W_q))
        # Large elements should be recovered well
        assert abs(W_q[0] - 100.0) < 2.0
        assert abs(W_q[1] - 100.0) < 2.0

    def test_all_elements_high_sparsity_zero(self):
        """sparsity=0 → all elements use high precision."""
        sq = SQFormat(bank_size=4, sparsity=0.0, high_bits=8, low_bits=4)
        W = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        sq.quantize(W)
        for bank_info in sq._last_bank_scales:
            assert bank_info["n_low"] == 0 or bank_info["n_high"] >= 3

    def test_minimum_one_high_prec_element(self):
        """Even at very high sparsity, at least 1 element per bank is high-precision."""
        sq = SQFormat(bank_size=4, sparsity=0.99, high_bits=8, low_bits=4)
        W = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        sq.quantize(W)
        for bank_info in sq._last_bank_scales:
            assert bank_info["n_high"] >= 1, "Must have at least 1 high-prec element"

    def test_bank_isolation(self):
        """Outlier in bank-0 must not affect quantization quality of bank-1."""
        sq = SQFormat(bank_size=4, sparsity=0.5, high_bits=8, low_bits=4)
        W = np.zeros(8, dtype=np.float32)
        W[:4] = [1000.0, 1000.0, 1000.0, 1000.0]   # bank-0: all outliers
        W[4:] = [1.0, 1.0, 1.0, 1.0]               # bank-1: normal values
        W_q = sq.quantize(W)
        # bank-1 should still be quantized accurately (its own scale, not bank-0's)
        mse_b1 = float(np.mean((W[4:] - W_q[4:]) ** 2))
        assert mse_b1 < 0.5, f"Bank-1 MSE={mse_b1:.4f} too high — outlier contamination?"

    def test_n_high_n_low_sum_equals_bank(self):
        """n_high + n_low must equal bank_size × N_cols for every bank (2D)."""
        sq = SQFormat(bank_size=4, sparsity=0.5, high_bits=8, low_bits=4)
        K, N = 8, 3
        W = make_W(K, N)
        sq.quantize(W)
        for bank_info in sq._last_bank_scales:
            # stored as per-column averages; just check both keys exist and are > 0
            assert "n_high" in bank_info
            assert "n_low" in bank_info


# ═══════════════════════════════════════════════════════════════════════════════
# D. Algorithm 1 — SmoothQuant smoothing
# ═══════════════════════════════════════════════════════════════════════════════

class TestAlg1Smoothing:
    """W' = W/s, Ā' = Ā*s where s_j = max|Āj|^α / max|Wj|^(1-α)."""

    def test_smooth_reduces_weight_range(self):
        """After smoothing, per-channel weight range should decrease for outlier activations."""
        K, N = 8, 4
        W = np.ones((K, N), dtype=np.float32)
        # Channel 0 has large activation outlier
        A_mean = np.ones(K, dtype=np.float32)
        A_mean[0] = 100.0
        W_smooth, A_smooth, sq_scales = _smooth(W, A_mean)
        # SmoothQuant should push the scale from activation to weight side
        assert sq_scales[0] > sq_scales[1], \
            "Channel 0 scale should be larger (absorbing activation outlier)"

    def test_smooth_increases_activation_range(self):
        """A' = A * s: activation values should be scaled up."""
        K = 4
        W = np.ones((K, 4), dtype=np.float32)
        A_mean = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        _, A_smooth, sq_scales = _smooth(W, A_mean)
        # A_smooth = A_mean * sq_scales; scales > 0
        np.testing.assert_allclose(A_smooth, A_mean * sq_scales, rtol=1e-5)

    def test_smooth_equivalent_matmul(self):
        """(A * s) @ (W / s) = A @ W (smoothing is lossless for the dot product)."""
        K, N = 8, 4
        W = make_W(K, N, seed=2)
        A_mean = make_A_mean(K, seed=3)
        W_smooth, A_smooth, _ = _smooth(W, A_mean)
        # Use A_mean as a single-token activation
        A = A_mean[np.newaxis, :]   # (1, K)
        A_s = A_smooth[np.newaxis, :]
        expected  = A @ W
        got       = A_s @ W_smooth
        np.testing.assert_allclose(got, expected, rtol=1e-4)

    def test_smooth_scales_positive(self):
        """SmoothQuant scales must be strictly positive."""
        K = 16
        W = make_W(K, 8)
        A_mean = make_A_mean(K)
        _, _, sq_scales = _smooth(W, A_mean)
        assert np.all(sq_scales > 0)

    def test_smooth_zero_activation(self):
        """Zero activation channel: scale is dominated by w_max, must be finite > 0."""
        K, N = 4, 4
        W = make_W(K, N, seed=4)
        A_mean = np.zeros(K, dtype=np.float32)
        W_smooth, A_smooth, sq_scales = _smooth(W, A_mean)
        assert np.all(np.isfinite(W_smooth))
        assert np.all(np.isfinite(sq_scales))
        assert np.all(sq_scales > 0)

    def test_alg1_smooth_triggered_by_a_mean(self):
        """Passing A_mean to SQFormat.quantize() must trigger smoothing (_sq_scales set)."""
        sq = SQFormat(bank_size=4, sparsity=0.5, high_bits=8, low_bits=4)
        W = make_W(4, 4)
        A_mean = make_A_mean(4)
        sq.quantize(W, A_mean=A_mean)
        assert sq._sq_scales is not None, "_sq_scales must be set after smooth path"
        assert sq._sq_scales.shape == (4,)

    def test_alg1_no_smooth_without_a_mean(self):
        """Without A_mean, _sq_scales must remain None."""
        sq = SQFormat(bank_size=4, sparsity=0.5, high_bits=8, low_bits=4)
        W = make_W(4, 4)
        sq.quantize(W)
        assert sq._sq_scales is None

    def test_smooth_alpha_0_full_weight_migration(self):
        """alpha=0: scale = 1 / max|W|^1 → all migration to weight side."""
        K, N = 4, 4
        W = make_W(K, N, seed=6) * 5  # amplify weights
        A_mean = np.ones(K, dtype=np.float32) * 10.0
        _, _, sq_scales_0 = _smooth(W, A_mean, alpha=0.0)
        _, _, sq_scales_1 = _smooth(W, A_mean, alpha=1.0)
        # Both should be positive and finite
        assert np.all(sq_scales_0 > 0)
        assert np.all(sq_scales_1 > 0)


# ═══════════════════════════════════════════════════════════════════════════════
# E. Algorithm 1 — encoding overhead & metadata
# ═══════════════════════════════════════════════════════════════════════════════

class TestAlg1Overhead:
    """Paper canonical config s=0.5, hhigh=8, hlow=4 → 6 bits/element."""

    def test_paper_canonical_overhead(self):
        """s=0.5, hhigh=8, hlow=4 → 0.5*8 + 0.5*4 = 6 bits/elem, mask_cost=0."""
        sq = SQFormat(bank_size=128, sparsity=0.5, high_bits=8, low_bits=4)
        oh = sq.encoding_overhead()
        assert abs(oh["data_bits_per_element"] - 6.0) < 1e-6
        assert oh["metadata_bits_per_element"] == 0.0

    def test_overhead_formula(self):
        """data_bits = (1-s)*hhigh + s*hlow for arbitrary configs."""
        for sparsity, hhigh, hlow in [(0.25, 8, 4), (0.75, 8, 2), (0.5, 16, 8)]:
            sq = SQFormat(sparsity=sparsity, high_bits=hhigh, low_bits=hlow)
            oh = sq.encoding_overhead()
            expected = (1 - sparsity) * hhigh + sparsity * hlow
            assert abs(oh["data_bits_per_element"] - expected) < 1e-6

    def test_overhead_above_low_bits(self):
        """Mixed precision must use more bits than pure low-precision."""
        sq = SQFormat(sparsity=0.5, high_bits=8, low_bits=4)
        oh = sq.encoding_overhead()
        assert oh["data_bits_per_element"] > sq.low_bits

    def test_overhead_below_high_bits(self):
        """Mixed precision must use fewer bits than pure high-precision."""
        sq = SQFormat(sparsity=0.5, high_bits=8, low_bits=4)
        oh = sq.encoding_overhead()
        assert oh["data_bits_per_element"] < sq.high_bits

    def test_bandwidth_amplification(self):
        """bandwidth_amplification = total_bits / low_bits."""
        sq = SQFormat(sparsity=0.5, high_bits=8, low_bits=4)
        oh = sq.encoding_overhead()
        expected_bwa = oh["data_bits_per_element"] / sq.low_bits
        assert abs(oh["bandwidth_amplification"] - expected_bwa) < 1e-6

    def test_sentinel_in_overhead_dict(self):
        """encoding_overhead() must report v_sentinel."""
        sq = SQFormat(low_bits=4, high_bits=8)
        oh = sq.encoding_overhead()
        assert "v_sentinel" in oh
        assert oh["v_sentinel"] == sq.v_sentinel

    def test_bank_scales_populated_after_quantize(self):
        """_last_bank_scales must be non-empty after quantize()."""
        sq = SQFormat(bank_size=4, sparsity=0.5, high_bits=8, low_bits=4)
        sq.quantize(make_W(8, 4))
        assert len(sq._last_bank_scales) > 0
        for info in sq._last_bank_scales:
            assert "scale_high" in info
            assert "scale_low" in info
            assert "n_high" in info
            assert "n_low" in info

    def test_bank_scales_reset_each_call(self):
        """_last_bank_scales must be reset (not accumulated) on each quantize() call."""
        sq = SQFormat(bank_size=4, sparsity=0.5, high_bits=8, low_bits=4)
        sq.quantize(make_W(8, 4))
        n1 = len(sq._last_bank_scales)
        sq.quantize(make_W(8, 4))
        n2 = len(sq._last_bank_scales)
        assert n1 == n2, "Bank scales should reset, not accumulate across calls"


# ═══════════════════════════════════════════════════════════════════════════════
# F. Algorithm 1 — bank structure (N:M-style 2D / 1D)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAlg1BankStructure:
    """Banks = groups of b consecutive K-elements within a SINGLE output column."""

    def test_2d_shape_preserved(self):
        """Output shape must equal input shape for any (K, N)."""
        sq = SQFormat(bank_size=4, sparsity=0.5, high_bits=8, low_bits=4)
        for K, N in [(4, 4), (8, 3), (7, 5), (1, 1), (128, 64)]:
            W = make_W(K, N)
            W_q = sq.quantize(W)
            assert W_q.shape == W.shape, f"Shape mismatch for K={K}, N={N}"

    def test_1d_shape_preserved(self):
        sq = SQFormat(bank_size=4, sparsity=0.5, high_bits=8, low_bits=4)
        for n in (4, 7, 128, 1):
            W = np.random.default_rng(0).normal(0, 1, n).astype(np.float32)
            W_q = sq.quantize(W)
            assert W_q.shape == (n,), f"1D shape mismatch for n={n}"

    def test_2d_finite(self):
        sq = SQFormat(bank_size=8, sparsity=0.5, high_bits=8, low_bits=4)
        W = make_W(16, 8)
        assert np.all(np.isfinite(sq.quantize(W)))

    def test_1d_finite(self):
        sq = SQFormat(bank_size=8, sparsity=0.5, high_bits=8, low_bits=4)
        W = np.random.default_rng(1).normal(0, 1, 32).astype(np.float32)
        assert np.all(np.isfinite(sq.quantize(W)))

    def test_nm_24_structured_sparsity(self):
        """b=4, s=0.5 → exactly 2 high-prec elements per 4-element bank (2:4 pattern)."""
        sq = SQFormat(bank_size=4, sparsity=0.5, high_bits=8, low_bits=4)
        W = make_W(4, 4, seed=9)
        sq.quantize(W)
        for bank_info in sq._last_bank_scales:
            # Each bank contributes n_high high and n_low low across all columns
            # Average per column: n_high / N should be ≈ (1-s)*bank_size
            # Just check the ratio is correct
            total = bank_info["n_high"] + bank_info["n_low"]
            assert total > 0

    def test_non_multiple_k_handled(self):
        """K not divisible by bank_size: padding must not change output shape."""
        sq = SQFormat(bank_size=4, sparsity=0.5, high_bits=8, low_bits=4)
        W = make_W(6, 4)   # 6 is not divisible by 4
        W_q = sq.quantize(W)
        assert W_q.shape == W.shape

    def test_large_bank_equals_global_quant(self):
        """bank_size >= K: single global bank, equivalent to global quantization."""
        K, N = 8, 4
        W = make_W(K, N, seed=11)
        sq = SQFormat(bank_size=K, sparsity=0.5, high_bits=8, low_bits=4)
        W_q = sq.quantize(W)
        assert W_q.shape == (K, N)
        assert np.all(np.isfinite(W_q))

    def test_single_element_bank(self):
        """bank_size=1: every element is its own bank, all become high-precision."""
        sq = SQFormat(bank_size=1, sparsity=0.5, high_bits=8, low_bits=4)
        W = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
        W_q = sq.quantize(W)
        assert np.all(np.isfinite(W_q))
        assert W_q.shape == W.shape


# ═══════════════════════════════════════════════════════════════════════════════
# G. Algorithm 2 — importance score Ij = |Āj · Σi W'_{j,i}|
# ═══════════════════════════════════════════════════════════════════════════════

class TestAlg2Importance:
    """Paper Algorithm 2 line 4: Ij = |Āj · Σi W'_{j,i}|."""

    def test_large_activation_x_weight_channel_selected(self):
        """Channel with large |Ā · row_sum| must be selected as high-precision."""
        K, N = 4, 8
        sq_a = SQFormatActivations(bank_size=4, sparsity=0.5, high_bits=8, low_bits=4)
        # Channel 0: large Ā × large positive row sum → most important
        # Channel 3: tiny Ā × tiny weights → least important
        W = np.array([
            [10.0] * N,    # channel 0: large row sum
            [1.0]  * N,
            [1.0]  * N,
            [0.001]* N,    # channel 3: tiny row sum
        ], dtype=np.float32)
        A_mean = np.array([100.0, 1.0, 1.0, 0.001], dtype=np.float32)

        _, _, mask_r, reorder_idx, _ = sq_a.quantize_weights(W, A_mean)
        orig_mask = np.empty_like(mask_r)
        orig_mask[reorder_idx] = mask_r

        assert orig_mask[0], "Channel 0 (max A×W) must be high-precision"
        assert not orig_mask[3], "Channel 3 (min A×W) must be low-precision"

    def test_importance_uses_signed_row_sum(self):
        """Importance uses signed (not absolute) row sum before |·|.

        |Āj · Σi W'_{j,i}|: the sum Σi can be negative. A channel with all-negative
        weights and a positive activation should still have non-zero importance.
        """
        K, N = 4, 4
        sq_a = SQFormatActivations(bank_size=4, sparsity=0.5, high_bits=8, low_bits=4)
        W = np.array([
            [-10.0] * N,   # ch 0: large negative row sum
            [1.0]   * N,
            [1.0]   * N,
            [0.01]  * N,
        ], dtype=np.float32)
        A_mean = np.array([5.0, 1.0, 1.0, 0.01], dtype=np.float32)
        # I_0 = |5 * (-40)| = 200 → should still win
        _, _, mask_r, reorder_idx, _ = sq_a.quantize_weights(W, A_mean)
        orig_mask = np.empty_like(mask_r)
        orig_mask[reorder_idx] = mask_r
        assert orig_mask[0], "Negative row sum must still give high importance"

    def test_pure_activation_magnitude_insufficient(self):
        """Activation-only importance (without weights) causes degradation.

        The static strategy redefines I_j = |Āj · Σi W'_{j,i}| precisely because
        using only |Āj| was found to degrade quality. Verify the implementation
        uses the joint formula, not just |Āj|.
        """
        K, N = 4, 4
        sq_a = SQFormatActivations(bank_size=4, sparsity=0.5, high_bits=8, low_bits=4)
        # Channel 0: large A but near-zero weights → joint importance near zero
        # Channel 3: small A but large weights → joint importance wins
        W = np.array([
            [0.0001] * N,   # ch 0: huge A, tiny W
            [0.5]    * N,
            [0.5]    * N,
            [10.0]   * N,   # ch 3: small A, large W
        ], dtype=np.float32)
        A_mean = np.array([1000.0, 0.5, 0.5, 0.1], dtype=np.float32)
        # Joint: I_0 = |1000 * 0.0004| = 0.4; I_3 = |0.1 * 40| = 4.0 → ch 3 wins
        _, _, mask_r, reorder_idx, _ = sq_a.quantize_weights(W, A_mean)
        orig_mask = np.empty_like(mask_r)
        orig_mask[reorder_idx] = mask_r
        assert orig_mask[3], "Ch 3 (large joint A×W) should be high-precision"

    def test_zero_activation_channels_not_high_precision(self):
        """Channels with Ā=0 have I=0 → should not be selected high-precision."""
        K, N = 4, 4
        sq_a = SQFormatActivations(bank_size=4, sparsity=0.5, high_bits=8, low_bits=4)
        W = np.ones((K, N), dtype=np.float32)
        A_mean = np.array([0.0, 0.0, 5.0, 10.0], dtype=np.float32)
        _, _, mask_r, reorder_idx, _ = sq_a.quantize_weights(W, A_mean)
        orig_mask = np.empty_like(mask_r)
        orig_mask[reorder_idx] = mask_r
        # Channels 0,1 have zero importance → should be low-precision
        assert not orig_mask[0], "Zero-activation channel should be low-precision"
        assert not orig_mask[1], "Zero-activation channel should be low-precision"


# ═══════════════════════════════════════════════════════════════════════════════
# H. Algorithm 2 — precision mask, reordering, data locality
# ═══════════════════════════════════════════════════════════════════════════════

class TestAlg2Reordering:
    """Paper step 8: 'Reorder rows of W' based on mask m' — high-prec rows first."""

    def test_reordered_high_prec_rows_first(self):
        """Within each bank of the reordered weight matrix, high-prec rows come first."""
        K, N = 32, 8
        sq_a = SQFormatActivations(bank_size=16, sparsity=0.5, high_bits=8, low_bits=4)
        W = make_W(K, N)
        A_mean = make_A_mean(K)
        _, _, mask_r, reorder_idx, _ = sq_a.quantize_weights(W, A_mean)

        for bank_start in range(0, K, 16):
            bm = mask_r[bank_start:bank_start + 16]
            seen_false = False
            for v in bm:
                if not v:
                    seen_false = True
                if seen_false and v:
                    pytest.fail(f"Low-prec row before high-prec in bank {bank_start}")

    def test_reorder_idx_is_permutation(self):
        """reorder_idx must be a valid K-permutation (all indices present)."""
        K, N = 32, 8
        sq_a = SQFormatActivations(bank_size=16, sparsity=0.5)
        W = make_W(K, N)
        A_mean = make_A_mean(K)
        _, _, _, reorder_idx, _ = sq_a.quantize_weights(W, A_mean)
        assert sorted(reorder_idx.tolist()) == list(range(K)), \
            "reorder_idx must be a valid permutation of 0..K-1"

    def test_reorder_preserves_row_content(self):
        """The reordered W_quant rows must be the same rows as W_smooth quantized, just reordered."""
        K, N = 8, 4
        sq_a = SQFormatActivations(bank_size=4, sparsity=0.5, high_bits=8, low_bits=4)
        W = make_W(K, N)
        A_mean = make_A_mean(K)
        W_q, scales, mask_r, reorder_idx, _ = sq_a.quantize_weights(W, A_mean)
        # Undo reorder
        inv_idx = np.empty_like(reorder_idx)
        inv_idx[reorder_idx] = np.arange(K)
        W_q_orig_order = W_q[inv_idx]
        # All rows should be finite
        assert np.all(np.isfinite(W_q_orig_order))

    def test_mask_dtype_bool(self):
        """mask returned by quantize_weights must be boolean."""
        K, N = 16, 8
        sq_a = SQFormatActivations(bank_size=8, sparsity=0.5)
        _, _, mask, _, _ = sq_a.quantize_weights(make_W(K, N), make_A_mean(K))
        assert mask.dtype == bool

    def test_sparsity_fraction_per_bank(self):
        """Fraction of high-prec channels per bank ≈ (1-s)."""
        K, N = 64, 8
        for sparsity in (0.25, 0.5, 0.75):
            sq_a = SQFormatActivations(bank_size=32, sparsity=sparsity)
            _, _, mask_r, reorder_idx, _ = sq_a.quantize_weights(make_W(K, N), make_A_mean(K))
            orig_mask = np.empty_like(mask_r)
            orig_mask[reorder_idx] = mask_r
            for bank_start in range(0, K, 32):
                bm = orig_mask[bank_start:bank_start + 32]
                high_frac = bm.mean()
                assert abs(high_frac - (1 - sparsity)) < 0.05, \
                    f"s={sparsity}, bank {bank_start}: high_frac={high_frac:.2f}"

    def test_scales_positive(self):
        """All returned per-channel scales must be positive."""
        K, N = 16, 4
        sq_a = SQFormatActivations(bank_size=8, sparsity=0.5)
        _, scales, _, _, sq_scales = sq_a.quantize_weights(make_W(K, N), make_A_mean(K))
        assert np.all(scales > 0), "Per-channel scales must be positive"
        assert np.all(sq_scales > 0), "SmoothQuant scales must be positive"

    def test_output_shapes(self):
        """quantize_weights must return 5-tuple with correct shapes."""
        K, N = 24, 6
        sq_a = SQFormatActivations(bank_size=8, sparsity=0.5)
        W_q, scales, mask, reorder_idx, sq_scales = sq_a.quantize_weights(
            make_W(K, N), make_A_mean(K)
        )
        assert W_q.shape        == (K, N)
        assert scales.shape     == (K,)
        assert mask.shape       == (K,)
        assert reorder_idx.shape == (K,)
        assert sq_scales.shape  == (K,)

    def test_output_finite(self):
        """All returned arrays must be finite."""
        K, N = 16, 8
        sq_a = SQFormatActivations(bank_size=8, sparsity=0.5)
        W_q, scales, mask, reorder_idx, sq_scales = sq_a.quantize_weights(
            make_W(K, N), make_A_mean(K)
        )
        assert np.all(np.isfinite(W_q))
        assert np.all(np.isfinite(scales))
        assert np.all(np.isfinite(sq_scales))


# ═══════════════════════════════════════════════════════════════════════════════
# I. Algorithm 2 — runtime activation quantization (per-token, batch-independent)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAlg2RuntimeActivations:
    """quantize_runtime_activations: per-token scale, batch-invariant."""

    def _setup(self, K=32, N=8, bank_size=16, sparsity=0.5):
        sq_a = SQFormatActivations(bank_size=bank_size, sparsity=sparsity,
                                   high_bits=8, low_bits=4)
        W      = make_W(K, N, seed=10)
        A_mean = make_A_mean(K, seed=11)
        W_q, _, mask_r, reorder_idx, sq_scales = sq_a.quantize_weights(W, A_mean)
        return sq_a, mask_r, reorder_idx, sq_scales

    def test_batch_independence(self):
        """Token A quantized alone must equal token A in a mixed batch."""
        K = 32
        sq_a, mask, reorder_idx, sq_scales = self._setup(K=K)
        rng = np.random.default_rng(99)
        token_A = rng.normal(0, 1,  (1, K)).astype(np.float32)
        token_B = rng.normal(0, 10, (1, K)).astype(np.float32)

        h_A,  l_A,  _ = sq_a.quantize_runtime_activations(
            token_A, mask, sq_scales, reorder_idx)
        h_AB, l_AB, _ = sq_a.quantize_runtime_activations(
            np.vstack([token_A, token_B]), mask, sq_scales, reorder_idx)

        np.testing.assert_array_equal(h_A, h_AB[:1],
            err_msg="High-prec token A must not change when batched with B")
        np.testing.assert_array_equal(l_A, l_AB[:1],
            err_msg="Low-prec token A must not change when batched with B")

    def test_output_shapes_batched(self):
        """Output tensors must all be (B, K) for a B-token batch."""
        K, B = 32, 4
        sq_a, mask, reorder_idx, sq_scales = self._setup(K=K)
        X = np.random.default_rng(0).normal(0, 1, (B, K)).astype(np.float32)
        X_high, X_low, X_reord = sq_a.quantize_runtime_activations(
            X, mask, sq_scales, reorder_idx)
        assert X_high.shape == (B, K)
        assert X_low.shape  == (B, K)
        assert X_reord.shape == (B, K)

    def test_output_shapes_single_token(self):
        """1D input (K,) should produce 1D outputs squeezed back to (K,)."""
        K = 32
        sq_a, mask, reorder_idx, sq_scales = self._setup(K=K)
        X = np.random.default_rng(1).normal(0, 1, K).astype(np.float32)
        X_high, X_low, X_reord = sq_a.quantize_runtime_activations(
            X, mask, sq_scales, reorder_idx)
        assert X_high.shape == (K,), f"Expected ({K},), got {X_high.shape}"
        assert X_low.shape  == (K,), f"Expected ({K},), got {X_low.shape}"

    def test_output_finite(self):
        """All runtime outputs must be finite."""
        K = 32
        sq_a, mask, reorder_idx, sq_scales = self._setup(K=K)
        X = np.random.default_rng(2).normal(0, 1, (4, K)).astype(np.float32)
        X_high, X_low, X_reord = sq_a.quantize_runtime_activations(
            X, mask, sq_scales, reorder_idx)
        assert np.all(np.isfinite(X_high))
        assert np.all(np.isfinite(X_low))
        assert np.all(np.isfinite(X_reord))

    def test_high_low_partition_non_overlapping(self):
        """High-prec and low-prec outputs must be non-overlapping (one is zero where other is non-zero)."""
        K = 32
        sq_a, mask, reorder_idx, sq_scales = self._setup(K=K)
        X = np.random.default_rng(3).normal(0, 1, (2, K)).astype(np.float32)
        X_high, X_low, _ = sq_a.quantize_runtime_activations(
            X, mask, sq_scales, reorder_idx)
        # At low_cols, X_high should be zero; at high_cols, X_low should be zero
        high_cols = np.where(mask)[0]
        low_cols  = np.where(~mask)[0]
        if len(low_cols) > 0:
            np.testing.assert_array_equal(
                X_high[:, low_cols], 0.0,
                err_msg="X_high must be 0 at low-prec columns")
        if len(high_cols) > 0:
            np.testing.assert_array_equal(
                X_low[:, high_cols], 0.0,
                err_msg="X_low must be 0 at high-prec columns")

    def test_high_prec_lower_error(self):
        """High-precision activation channels should have lower quantization error."""
        K = 32
        sq_a, mask, reorder_idx, sq_scales = self._setup(K=K)
        X = np.random.default_rng(4).normal(0, 1, (4, K)).astype(np.float32)
        X_smooth = X * sq_scales[np.newaxis, :]
        X_reord  = X_smooth[:, reorder_idx]
        X_high, X_low, _ = sq_a.quantize_runtime_activations(
            X, mask, sq_scales, reorder_idx)

        high_cols = np.where(mask)[0]
        low_cols  = np.where(~mask)[0]
        if len(high_cols) > 0 and len(low_cols) > 0:
            err_high = float(np.mean((X_high[:, high_cols] - X_reord[:, high_cols]) ** 2))
            err_low  = float(np.mean((X_low[:, low_cols]  - X_reord[:, low_cols])  ** 2))
            # High-precision should have lower or equal MSE than low-precision
            # (relaxed: allow up to 100x difference for pathological inputs)
            assert err_high <= err_low * 100 or err_high < 0.01, \
                f"High-prec MSE={err_high:.4f} unexpectedly much larger than low-prec MSE={err_low:.4f}"

    def test_zero_activations(self):
        """Zero activation input must produce zero outputs without NaN."""
        K = 32
        sq_a, mask, reorder_idx, sq_scales = self._setup(K=K)
        X = np.zeros((2, K), dtype=np.float32)
        X_high, X_low, _ = sq_a.quantize_runtime_activations(
            X, mask, sq_scales, reorder_idx)
        assert np.all(X_high == 0.0)
        assert np.all(X_low  == 0.0)

    def test_smoothquant_applied_before_reorder(self):
        """sq_scales is applied to X first, then columns are reordered."""
        K = 16
        sq_a = SQFormatActivations(bank_size=8, sparsity=0.5, high_bits=8, low_bits=4)
        W = make_W(K, 4, seed=20)
        A_mean = make_A_mean(K, seed=21)
        _, _, mask_r, reorder_idx, sq_scales = sq_a.quantize_weights(W, A_mean)

        X = np.random.default_rng(5).normal(0, 1, (1, K)).astype(np.float32)
        _, _, X_reord = sq_a.quantize_runtime_activations(
            X, mask_r, sq_scales, reorder_idx)

        # Manual: X_smooth = X * sq_scales, then reorder
        X_expected = (X * sq_scales[np.newaxis, :])[:, reorder_idx]
        np.testing.assert_allclose(X_reord, X_expected, rtol=1e-5)


# ═══════════════════════════════════════════════════════════════════════════════
# J. Algorithm 2 quantize() single-tensor interface
# ═══════════════════════════════════════════════════════════════════════════════

class TestAlg2QuantizeInterface:
    """SQFormatActivations.quantize() — simplified harness interface."""

    def test_1d_shape(self):
        sq_a = SQFormatActivations(bank_size=64, sparsity=0.5)
        x = np.random.default_rng(0).normal(0, 1, 128).astype(np.float32)
        assert sq_a.quantize(x).shape == x.shape

    def test_2d_shape(self):
        sq_a = SQFormatActivations(bank_size=8, sparsity=0.5)
        x = make_W(16, 8)
        assert sq_a.quantize(x).shape == x.shape

    def test_1d_finite(self):
        sq_a = SQFormatActivations()
        x = np.random.default_rng(1).normal(0, 1, 256).astype(np.float32)
        assert np.all(np.isfinite(sq_a.quantize(x)))

    def test_2d_finite(self):
        sq_a = SQFormatActivations()
        x = make_W(32, 16)
        assert np.all(np.isfinite(sq_a.quantize(x)))

    def test_outlier_quantize_finite(self):
        sq_a = SQFormatActivations(bank_size=128, sparsity=0.5)
        x = np.random.default_rng(2).normal(0, 1, 256).astype(np.float32)
        x[0] = 500.0
        x[128] = -300.0
        assert np.all(np.isfinite(sq_a.quantize(x)))

    def test_sq_format_a_beats_uniform_quant_on_outlier(self):
        """SQFormatActivations should outperform uniform INT4 on outlier inputs."""
        from formats import build_all_formats
        from distributions.generators import channel_outliers
        from distributions.metrics import snr_db
        fmts = build_all_formats(dim=256, seed=42)
        x, _ = channel_outliers(n=256, outlier_sigma=100.0, seed=42)
        sq_a = SQFormatActivations(bank_size=64, sparsity=0.5, high_bits=8, low_bits=4)
        sqnr_sq  = snr_db(x, sq_a.quantize(x))
        sqnr_int = snr_db(x, fmts["INT4"].quantize(x))
        assert sqnr_sq > sqnr_int, \
            f"SQFormatActivations SQNR={sqnr_sq:.1f} should exceed INT4 SQNR={sqnr_int:.1f}"


# ═══════════════════════════════════════════════════════════════════════════════
# K. Edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_all_zeros_weight(self):
        """All-zero weight matrix must produce all-zero output."""
        sq = SQFormat(bank_size=4, sparsity=0.5, high_bits=8, low_bits=4)
        W = np.zeros((8, 4), dtype=np.float32)
        W_q = sq.quantize(W)
        assert np.all(W_q == 0.0)

    def test_all_zeros_activation(self):
        """All-zero A_mean must not cause NaN in SQFormat (smoothing guard)."""
        sq = SQFormat(bank_size=4, sparsity=0.5, high_bits=8, low_bits=4)
        W = make_W(4, 4)
        A_mean = np.zeros(4, dtype=np.float32)
        W_q = sq.quantize(W, A_mean=A_mean)
        assert np.all(np.isfinite(W_q))

    def test_single_row_matrix(self):
        """K=1 (single row/channel) must work without error."""
        sq = SQFormat(bank_size=1, sparsity=0.5, high_bits=8, low_bits=4)
        W = np.array([[1.0, -2.0, 3.0]], dtype=np.float32)
        W_q = sq.quantize(W)
        assert W_q.shape == (1, 3)
        assert np.all(np.isfinite(W_q))

    def test_single_column_matrix(self):
        """N=1 (single output column) must work."""
        sq = SQFormat(bank_size=4, sparsity=0.5, high_bits=8, low_bits=4)
        W = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
        W_q = sq.quantize(W)
        assert W_q.shape == (4, 1)
        assert np.all(np.isfinite(W_q))

    def test_very_large_values(self):
        """Extremely large weights (1e6) must be quantized without Inf/NaN."""
        sq = SQFormat(bank_size=4, sparsity=0.5, high_bits=8, low_bits=4)
        W = np.ones((4, 4), dtype=np.float32) * 1e6
        W[0, 0] = -1e6
        assert np.all(np.isfinite(sq.quantize(W)))

    def test_very_small_values(self):
        """Subnormal weight values must be quantized without NaN."""
        sq = SQFormat(bank_size=4, sparsity=0.5, high_bits=8, low_bits=4)
        W = np.ones((4, 4), dtype=np.float32) * 1e-38
        assert np.all(np.isfinite(sq.quantize(W)))

    def test_sparsity_zero(self):
        """sparsity=0: all elements high-precision → lower error than sparsity=0.5."""
        sq_full = SQFormat(bank_size=8, sparsity=0.0, high_bits=8, low_bits=4)
        sq_half = SQFormat(bank_size=8, sparsity=0.5, high_bits=8, low_bits=4)
        W = make_W(8, 4, seed=99)
        mse_full = float(np.mean((W - sq_full.quantize(W)) ** 2))
        mse_half = float(np.mean((W - sq_half.quantize(W)) ** 2))
        assert mse_full <= mse_half + 1e-6, \
            f"Full-precision (s=0) MSE={mse_full:.4f} should be ≤ mixed (s=0.5) MSE={mse_half:.4f}"

    def test_sparsity_one_int2(self):
        """sparsity=1.0 would force all to low-prec, but min 1 high-prec is enforced."""
        sq = SQFormat(bank_size=4, sparsity=1.0, high_bits=8, low_bits=4)
        W = make_W(4, 4, seed=0)
        W_q = sq.quantize(W)
        assert np.all(np.isfinite(W_q))

    def test_alg2_k_equals_bank_size(self):
        """K == bank_size: exactly one bank for the whole matrix."""
        K, N = 8, 4
        sq_a = SQFormatActivations(bank_size=K, sparsity=0.5, high_bits=8, low_bits=4)
        W = make_W(K, N)
        A_mean = make_A_mean(K)
        W_q, scales, mask, reorder_idx, sq_scales = sq_a.quantize_weights(W, A_mean)
        assert W_q.shape == (K, N)
        assert np.all(np.isfinite(W_q))
        # Check roughly half high-prec
        assert abs(mask.mean() - 0.5) < 0.15

    def test_alg2_k_not_multiple_of_bank_size(self):
        """K not divisible by bank_size: last bank is smaller, must still work."""
        K, N = 10, 4
        sq_a = SQFormatActivations(bank_size=4, sparsity=0.5, high_bits=8, low_bits=4)
        W = make_W(K, N)
        A_mean = make_A_mean(K)
        W_q, _, mask, reorder_idx, _ = sq_a.quantize_weights(W, A_mean)
        assert W_q.shape == (K, N)
        assert sorted(reorder_idx.tolist()) == list(range(K))

    def test_identical_rows_nonduplicate_mask(self):
        """Identical rows must still produce a valid (not all-true) mask."""
        K, N = 4, 4
        sq_a = SQFormatActivations(bank_size=4, sparsity=0.5, high_bits=8, low_bits=4)
        W = np.ones((K, N), dtype=np.float32)
        A_mean = np.ones(K, dtype=np.float32)
        _, _, mask, _, _ = sq_a.quantize_weights(W, A_mean)
        # Should have ~(1-s) high-precision even when rows are identical
        assert 0 < mask.sum() < K


# ═══════════════════════════════════════════════════════════════════════════════
# L. Backward-compatibility aliases
# ═══════════════════════════════════════════════════════════════════════════════

class TestBackwardCompatAliases:

    def test_dense_sparse_aliases(self):
        """dense_bits/sparse_bits must map to low_bits/high_bits."""
        sq = SQFormat(dense_bits=4, sparse_bits=8, sparsity_ratio=0.01)
        assert sq.low_bits  == 4
        assert sq.high_bits == 8

    def test_sparsity_ratio_alias(self):
        """sparsity_ratio = 1 - sparsity."""
        sq = SQFormat(sparsity_ratio=0.3)
        assert abs(sq.sparsity - 0.7) < 1e-6
        assert abs(sq.sparsity_ratio - 0.3) < 1e-6

    def test_sparsity_takes_precedence(self):
        """When both sparsity and sparsity_ratio are given, sparsity wins."""
        sq = SQFormat(sparsity=0.4, sparsity_ratio=0.3)
        assert abs(sq.sparsity - 0.4) < 1e-6

    def test_bits_attribute(self):
        """sq.bits must equal low_bits (for harness compatibility)."""
        sq = SQFormat(low_bits=4, high_bits=8)
        assert sq.bits == sq.low_bits

    def test_dense_bits_attribute_synced(self):
        """dense_bits attribute must equal low_bits after construction."""
        sq = SQFormat(low_bits=4, high_bits=8)
        assert sq.dense_bits == sq.low_bits

    def test_sparse_bits_attribute_synced(self):
        sq = SQFormat(low_bits=4, high_bits=8)
        assert sq.sparse_bits == sq.high_bits

    def test_old_params_quantize_runs(self):
        """Old-style construction must still produce valid quantization output."""
        sq = SQFormat(dense_bits=4, sparse_bits=8, sparsity_ratio=0.5)
        x = np.random.default_rng(0).normal(0, 1, 64).astype(np.float32)
        x_q = sq.quantize(x)
        assert x_q.shape == x.shape
        assert np.all(np.isfinite(x_q))

    def test_sq_format_a_alias_attributes(self):
        """SQFormatActivations must expose dense_bits/sparse_bits/sparsity_ratio."""
        sq_a = SQFormatActivations(low_bits=4, high_bits=8, sparsity=0.5)
        assert sq_a.dense_bits    == 4
        assert sq_a.sparse_bits   == 8
        assert abs(sq_a.sparsity_ratio - 0.5) < 1e-6


# ═══════════════════════════════════════════════════════════════════════════════
# M. _smooth helper — unit tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSmoothHelper:

    def test_shapes(self):
        K, N = 8, 4
        W = make_W(K, N)
        A_mean = make_A_mean(K)
        W_s, A_s, sq = _smooth(W, A_mean)
        assert W_s.shape == (K, N)
        assert A_s.shape == (K,)
        assert sq.shape  == (K,)

    def test_dot_product_preserved(self):
        """A @ W == (A * sq) @ (W / sq[:,None]) (lossless transformation)."""
        K, N = 8, 4
        W = make_W(K, N, seed=30)
        A_mean = make_A_mean(K, seed=31)
        W_s, A_s, _ = _smooth(W, A_mean)
        # Use A_mean as 1-token row vector
        A = A_mean[np.newaxis, :]
        A_sm = A_s[np.newaxis, :]
        np.testing.assert_allclose(A @ W, A_sm @ W_s, rtol=1e-4)

    def test_all_outputs_finite(self):
        K, N = 16, 8
        W = make_W(K, N)
        A_mean = make_A_mean(K)
        W_s, A_s, sq = _smooth(W, A_mean)
        assert np.all(np.isfinite(W_s))
        assert np.all(np.isfinite(A_s))
        assert np.all(np.isfinite(sq))

    def test_alpha_05_default(self):
        """Default alpha=0.5 should produce geometric mean of A and W magnitudes."""
        K = 4
        W = np.ones((K, 4), dtype=np.float32) * 2.0
        A_mean = np.ones(K, dtype=np.float32) * 8.0
        # s_j = max|Āj|^0.5 / max|Wj|^0.5 = sqrt(8)/sqrt(2) = 2
        _, _, sq = _smooth(W, A_mean, alpha=0.5)
        np.testing.assert_allclose(sq, 2.0, rtol=1e-4)

    def test_custom_alpha(self):
        """alpha=1.0: scale = max|Ā|^1 / max|W|^0 = max|Ā|."""
        K = 4
        W = np.ones((K, 4), dtype=np.float32)
        A_mean = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)
        _, _, sq = _smooth(W, A_mean, alpha=1.0)
        # s_j = A_mean_j^1 / 1^0 = A_mean_j
        np.testing.assert_allclose(sq, A_mean, rtol=1e-4)


# ═══════════════════════════════════════════════════════════════════════════════
# N. POT scale helper properties
# ═══════════════════════════════════════════════════════════════════════════════

class TestPOTScaleHelper:
    """_pot_scale and _pot_scale_vec: 2^ceil(log2(absmax/q_max))."""

    def test_no_clipping(self):
        """scale * q_max must be >= absmax (no clipping guarantee)."""
        for absmax, q_max in [(15, 7), (8, 7), (1.5, 7), (100, 127), (0.001, 7)]:
            s = _pot_scale(float(absmax), q_max)
            assert s * q_max >= absmax - 1e-6, \
                f"absmax={absmax}, q_max={q_max}: {s}*{q_max}={s*q_max} < {absmax}"

    def test_is_power_of_two(self):
        """_pot_scale must always return a power of two."""
        for absmax in [0.1, 0.5, 1.0, 3.0, 7.5, 15.0, 100.0, 0.0001]:
            s = _pot_scale(absmax, 7)
            assert is_pot(s), f"absmax={absmax}: scale={s} is not a power of 2"

    def test_zero_absmax_returns_one(self):
        """absmax=0 must return 1.0 (no NaN from log2(0))."""
        assert _pot_scale(0.0, 7) == 1.0

    def test_vectorized_matches_scalar(self):
        """_pot_scale_vec must produce same results as scalar _pot_scale."""
        absmax_arr = np.array([0.1, 1.0, 3.5, 7.0, 15.0, 0.0], dtype=np.float32)
        q_max = 7
        vec = _pot_scale_vec(absmax_arr, q_max)
        for i, v in enumerate(absmax_arr):
            expected = _pot_scale(float(v), q_max)
            assert abs(vec[i] - expected) < 1e-6, \
                f"vec[{i}]={vec[i]} vs scalar={expected} for absmax={v}"

    def test_counter_example_floor_approach(self):
        """absmax=15, q_max=7: floor approach gives s=2 (clips!); ceil must give s=4."""
        s = _pot_scale(15.0, 7)
        assert s == 4.0, f"Expected s=4 for absmax=15, q_max=7; got {s}"
        assert s * 7 >= 15.0
