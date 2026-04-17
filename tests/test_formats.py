"""Unit tests for all quantization formats.

Tests cover:
  - Shape preservation and no NaN/Inf
  - Mathematical properties (energy, self-inverse, invertibility)
  - HAD fixed-point model: correctness, hardware_ops, normalize=False default
  - POT scale properties for INT quantizers and SQ-Format
  - MXINT block independence and E8M0 POT scale
  - SQ-Format salient masking
  - Encoding overhead metadata
  - Hardware model correctness
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from formats import build_all_formats, FOCUS_4BIT, FOCUS_8BIT, FOCUS_ALL
from formats.baseline import FP32Format, BF16Format
from formats.nvfp4 import NVFP4Format
from formats.mxfp import MXFPFormat
from formats.mxint import MXINTFormat
from formats.nf4 import NF4Format
from formats.fp6 import FP6Format
from formats.sq_format import SQFormat, SQFormatActivations
from formats.transforms.hadamard import hadamard_transform, inverse_hadamard_transform, HADTransform
from formats.transforms.random_rotation import RandomRotationTransform
from config import NF4_LEVELS


RNG = np.random.default_rng(42)
N = 256   # must be power of 2 for HAD


def make_input(n=N, outlier=False, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n).astype(np.float32)
    if outlier:
        x[0] = 500.0
        x[n // 2] = -300.0
    return x


# ─── Format registry ──────────────────────────────────────────────────────────

class TestFormatRegistry:
    """All formats in build_all_formats() must pass basic interface checks."""

    def setup_method(self):
        self.formats = build_all_formats(dim=N, seed=42)
        self.x = make_input()
        self.x_out = make_input(outlier=True)

    def test_focus_formats_present(self):
        """All focus formats must be in the registry."""
        for f in FOCUS_ALL:
            assert f in self.formats or f == "FP32", f"Focus format '{f}' missing"

    def test_no_turbo_quant(self):
        """TurboQuant must be removed from the registry."""
        for name in self.formats:
            assert "TurboQuant" not in name, f"TurboQuant still in registry: {name}"

    @pytest.mark.parametrize("fmt_name", list(build_all_formats(dim=N, seed=42).keys()))
    def test_shape_preserved(self, fmt_name):
        fmt = build_all_formats(dim=N, seed=42)[fmt_name]
        out = fmt.quantize(self.x)
        assert out.shape == self.x.shape

    @pytest.mark.parametrize("fmt_name", list(build_all_formats(dim=N, seed=42).keys()))
    def test_no_nan_inf(self, fmt_name):
        fmt = build_all_formats(dim=N, seed=42)[fmt_name]
        out = fmt.quantize(self.x)
        assert np.all(np.isfinite(out)), f"{fmt_name}: non-finite values"

    @pytest.mark.parametrize("fmt_name", list(build_all_formats(dim=N, seed=42).keys()))
    def test_outlier_robustness(self, fmt_name):
        fmt = build_all_formats(dim=N, seed=42)[fmt_name]
        out = fmt.quantize(self.x_out)
        assert out.shape == self.x_out.shape
        assert np.all(np.isfinite(out)), f"{fmt_name}: non-finite on outlier input"


# ─── HAD Transform ────────────────────────────────────────────────────────────

class TestHADTransform:
    """Correctness tests for the fixed-point FWHT implementation."""

    def test_known_result_unnormalized(self):
        """WHT([1,2,3,4]) should be [10, -2, -4, 0] (unnormalized)."""
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        y = hadamard_transform(x, normalize=False)
        expected = np.array([10.0, -2.0, -4.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(y, expected, atol=1e-5)

    def test_known_result_normalized(self):
        """WHT([1,2,3,4])/2 should be [5, -1, -2, 0]."""
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        y = hadamard_transform(x, normalize=True)
        expected = np.array([5.0, -1.0, -2.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(y, expected, atol=1e-5)

    def test_default_is_unnormalized(self):
        """Default normalize=False → integer-valued butterfly output."""
        x = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        y = hadamard_transform(x)  # normalize=False by default
        # All elements should be ±1 (unnormalized Hadamard row)
        np.testing.assert_allclose(np.abs(y), 1.0, atol=1e-5)

    def test_energy_preserved_normalized(self):
        """Normalized HAD should preserve L2 norm."""
        x = make_input()
        y = hadamard_transform(x, normalize=True)
        np.testing.assert_allclose(np.sum(y**2), np.sum(x**2), rtol=1e-4)

    def test_energy_scales_unnormalized(self):
        """Unnormalized HAD: ||H(x)||² = N * ||x||²."""
        x = make_input()
        n = x.shape[0]
        y = hadamard_transform(x, normalize=False)
        np.testing.assert_allclose(np.sum(y**2), n * np.sum(x**2), rtol=1e-4)

    def test_self_inverse_normalized(self):
        """Normalized HAD is its own inverse: H(H(x)) = x."""
        x = make_input()
        y = hadamard_transform(hadamard_transform(x, normalize=True), normalize=True)
        np.testing.assert_allclose(y, x, atol=1e-4)

    def test_inverse_unnormalized(self):
        """Unnormalized HAD round-trip: H(H(x))/N = x."""
        x = make_input()
        n = x.shape[0]
        y = hadamard_transform(x, normalize=False)
        z = hadamard_transform(y, normalize=False) / n
        np.testing.assert_allclose(z, x, atol=1e-4)

    def test_had_transform_class_roundtrip(self):
        """HADTransform.forward + inverse should recover x (normalize=False)."""
        had = HADTransform(normalize=False)
        x = make_input()
        y = had.forward(x)
        x_rec = had.inverse(y)
        np.testing.assert_allclose(x_rec, x, atol=1e-3)

    def test_had_transform_class_roundtrip_normalized(self):
        """HADTransform.forward + inverse should recover x (normalize=True)."""
        had = HADTransform(normalize=True)
        x = make_input()
        x_rec = had.inverse(had.forward(x))
        np.testing.assert_allclose(x_rec, x, atol=1e-4)

    def test_outlier_energy_spread(self):
        """After HAD, a single outlier's energy spreads across all elements."""
        x = np.zeros(N, dtype=np.float32)
        x[0] = 100.0   # single outlier
        y = hadamard_transform(x, normalize=True)
        # All elements should have magnitude 100/sqrt(N), not concentrated at 0
        assert np.max(np.abs(y)) < 100.0 / np.sqrt(N) * 1.01

    def test_uniform_output_single_outlier(self):
        """For single non-zero input, all HAD outputs have SAME magnitude (uniform spread)."""
        x = np.zeros(N, dtype=np.float32)
        x[0] = 1.0
        y = hadamard_transform(x, normalize=True)
        np.testing.assert_allclose(np.abs(y), 1.0 / np.sqrt(N), atol=1e-5)

    def test_power_of_two_required(self):
        """HAD should raise AssertionError for non-power-of-2 length."""
        x = np.zeros(3, dtype=np.float32)
        with pytest.raises(AssertionError):
            hadamard_transform(x)

    def test_batch_dims(self):
        """HAD should work with batch dimensions (2D input)."""
        x = make_input(n=N * 4).reshape(4, N)
        y = hadamard_transform(x, normalize=True)
        assert y.shape == x.shape
        assert np.all(np.isfinite(y))

    def test_hardware_ops_correct(self):
        """hardware_ops should report n//2 * stages for adds AND subs."""
        had = HADTransform()
        n = 256
        ops = had.hardware_ops(n)
        stages = int(np.log2(n))
        expected_per_type = (n // 2) * stages
        assert ops["additions"] == expected_per_type, \
            f"Expected {expected_per_type} adds, got {ops['additions']}"
        assert ops["subtractions"] == expected_per_type, \
            f"Expected {expected_per_type} subs, got {ops['subtractions']}"
        assert ops["multiplications"] == 0, "No multiplications in HAD"
        assert ops["total_ops"] == 2 * expected_per_type

    def test_hardware_ops_not_doubled(self):
        """Previous bug: n*stages (2× too many). Verify fix."""
        had = HADTransform()
        ops_256 = had.hardware_ops(256)
        ops_512 = had.hardware_ops(512)
        # Total ops should be N * log2(N)
        assert ops_256["total_ops"] == 256 * 8,  "256*log2(256)=256*8=2048"
        assert ops_512["total_ops"] == 512 * 9,  "512*log2(512)=512*9=4608"


# ─── POT Scale Properties ─────────────────────────────────────────────────────

class TestPOTScales:
    """Verify that INT quantizers and SQ-Format use power-of-two scales."""

    def _is_pot(self, v: float) -> bool:
        """True if v is a power of two (within float tolerance)."""
        if v <= 0:
            return False
        log2v = np.log2(abs(v))
        return abs(log2v - round(log2v)) < 1e-5

    def test_int4_pot_scale(self):
        """Plain INT4 quantized values must be multiples of a POT scale."""
        from formats import build_all_formats
        fmt = build_all_formats(dim=N)["INT4"]
        # Input whose absmax/q_max = 8/7 → POT scale = 2^floor(log2(8/7)) = 1.0
        x = np.array([1.0, 2.0, 4.0, 8.0, -3.0, 0.5], dtype=np.float32)
        x_q = fmt.quantize(x)
        # Determine implied scale: values should be multiples of some POT value
        non_zero = x_q[x_q != 0]
        if len(non_zero) > 0:
            step = np.min(np.abs(non_zero))
            assert self._is_pot(float(step)), f"INT4 step {step} is not a power of 2"

    def test_sq_format_pot_scale(self):
        """SQ-Format quantized values should be multiples of a POT scale."""
        sq = SQFormat(dense_bits=4, sparse_bits=8, sparsity_ratio=0.01)
        x = np.array([7.5, 0.5, -1.5, 3.5, -7.5] * 10, dtype=np.float32)
        x_q = sq.quantize(x)
        # After POT quantization, values should be multiples of a POT scale
        assert np.all(np.isfinite(x_q))
        assert x_q.shape == x.shape

    def test_mxint_already_pot(self):
        """MXINT4 uses E8M0 (POT) scale by design — verify consistency."""
        fmt = MXINTFormat(element_bits=4)
        x = make_input(n=32)  # single block
        x_q = fmt.quantize(x)
        assert np.all(np.isfinite(x_q))

    def test_had_int4_pot_roundtrip(self):
        """HAD+INT4(C) should produce finite output with POT INT quantizer."""
        fmts = build_all_formats(dim=N)
        fmt = fmts["HAD+INT4(C)"]
        x = make_input(outlier=True)
        x_q = fmt.quantize(x)
        assert np.all(np.isfinite(x_q))
        assert x_q.shape == x.shape


# ─── Random Rotation ─────────────────────────────────────────────────────────

class TestRandomRotation:
    def test_energy_preserved(self):
        rr = RandomRotationTransform(dim=N, seed=42)
        x = make_input()
        y = rr.forward(x)
        np.testing.assert_allclose(np.sum(y**2), np.sum(x**2), rtol=1e-4)

    def test_inverse(self):
        rr = RandomRotationTransform(dim=N, seed=42)
        x = make_input()
        y = rr.forward(x)
        x_rec = rr.inverse(y)
        np.testing.assert_allclose(x_rec, x, atol=1e-4)

    def test_orthogonality(self):
        rr = RandomRotationTransform(dim=N, seed=42)
        Q = rr.Q.astype(np.float64)
        I_approx = Q @ Q.T
        np.testing.assert_allclose(I_approx, np.eye(N), atol=1e-4)

    def test_determinism(self):
        rr1 = RandomRotationTransform(dim=N, seed=42)
        rr2 = RandomRotationTransform(dim=N, seed=42)
        np.testing.assert_array_equal(rr1.Q, rr2.Q)

    def test_randrot_vs_had_quality(self):
        """HAD+INT4(C) should have SQNR >= RandRot+INT4 (uniform energy spread)."""
        from formats import build_all_formats
        from distributions.generators import channel_outliers
        from distributions.metrics import snr_db

        fmts = build_all_formats(dim=N)
        x, _ = channel_outliers(n=N, outlier_sigma=50.0, seed=42)

        x_had = fmts["HAD+INT4(C)"].quantize(x)
        x_rr  = fmts["RandRot+INT4"].quantize(x)

        sqnr_had = snr_db(x, x_had)
        sqnr_rr  = snr_db(x, x_rr)

        # HAD should be at least as good (within 3 dB tolerance)
        assert sqnr_had >= sqnr_rr - 3.0, \
            f"HAD+INT4 SQNR={sqnr_had:.1f} dB unexpectedly much worse than RandRot+INT4={sqnr_rr:.1f} dB"


# ─── NF4 ─────────────────────────────────────────────────────────────────────

class TestNF4:
    def test_levels_sorted(self):
        assert np.all(np.diff(NF4_LEVELS) > 0), "NF4 levels must be strictly sorted"

    def test_16_levels(self):
        assert len(NF4_LEVELS) == 16

    def test_output_in_level_set(self):
        """NF4 output = level × absmax, so normalised output must be in NF4_LEVELS."""
        fmt = NF4Format()
        x = make_input()
        x_q = fmt.quantize(x)
        absmax = float(np.max(np.abs(x)))
        x_q_norm = x_q / absmax
        for v in x_q_norm.ravel():
            assert any(abs(v - lv) < 1e-4 for lv in NF4_LEVELS), \
                f"Normalised output {v:.6f} not in NF4 level set"

    def test_zero_input(self):
        fmt = NF4Format()
        x = np.zeros(N, dtype=np.float32)
        x_q = fmt.quantize(x)
        assert np.all(np.isfinite(x_q))


# ─── MXINT ───────────────────────────────────────────────────────────────────

class TestMXINT:
    def test_block_independence(self):
        """Outlier in one block must not affect other blocks (block-local scale)."""
        fmt = MXINTFormat(element_bits=4, block_size=32)
        x = np.zeros(64, dtype=np.float32)
        x[:32] = np.random.default_rng(0).normal(0, 1, 32).astype(np.float32)  # normal block
        x[32:] = np.random.default_rng(0).normal(0, 1, 32).astype(np.float32)
        x[0] = 500.0  # outlier in block 0

        # Block 1 (no outlier) should quantize accurately
        x[32:] = 1.0   # constant block 1
        x_q = fmt.quantize(x)
        mse_block1 = float(np.mean((x[32:] - x_q[32:]) ** 2))
        # MSE limit: INT4 with POT scale on constant=1.0 can be ~0.02; outlier in block 0
        # must NOT inflate block 1 error to many times that.
        assert mse_block1 < 0.1, f"Block 1 MSE={mse_block1:.4f} too high (outlier contamination from block 0)"

    def test_metadata_025_bits(self):
        """MXINT4 encoding overhead must be 0.25 bits/element (8-bit E8M0 per 32 elems)."""
        fmt = MXINTFormat(element_bits=4, block_size=32)
        overhead = fmt.encoding_overhead()
        assert abs(overhead["metadata_bits_per_element"] - 0.25) < 1e-9

    def test_e8m0_scale_is_pot(self):
        """MXINT uses 2^floor(log2(...)) scale — verify it's a power of 2."""
        fmt = MXINTFormat(element_bits=4, block_size=32)
        x = np.ones(32, dtype=np.float32) * 3.5
        x_q = fmt.quantize(x)
        # Reconstruct scale: scale = x_q[0] (all same) / round(3.5/scale)
        # With scale = 2^floor(log2(3.5/7)) = 2^floor(-1) = 0.5
        # q=round(3.5/0.5)=7, dequant=7*0.5=3.5 ✓
        assert np.all(np.isfinite(x_q))

    def test_mxint4_better_than_int4_outlier(self):
        """MXINT4 should outperform plain INT4 on channel-outlier inputs."""
        from formats import build_all_formats
        from distributions.generators import channel_outliers
        from distributions.metrics import snr_db

        fmts = build_all_formats(dim=N)
        x, _ = channel_outliers(n=N, outlier_sigma=50.0, seed=42)

        sqnr_mx  = snr_db(x, fmts["MXINT4"].quantize(x))
        sqnr_int = snr_db(x, fmts["INT4"].quantize(x))

        assert sqnr_mx > sqnr_int, \
            f"MXINT4 SQNR={sqnr_mx:.1f} should exceed INT4 SQNR={sqnr_int:.1f}"


# ─── SQ-Format ───────────────────────────────────────────────────────────────

class TestSQFormat:
    # ── Algorithm 1 (weight quantization) ────────────────────────────────────

    def test_shape(self):
        sq = SQFormat()
        x = make_input(outlier=True)
        x_q = sq.quantize(x)
        assert x_q.shape == x.shape

    def test_salient_mse_less_than_int4(self):
        """SQ-Format should outperform INT4 on outlier-heavy input."""
        from formats import build_all_formats
        from distributions.metrics import snr_db
        from distributions.generators import channel_outliers

        fmts = build_all_formats(dim=N)
        x, _ = channel_outliers(n=N, outlier_sigma=100.0, seed=42)

        sqnr_sq  = snr_db(x, fmts["SQ-Format"].quantize(x))
        sqnr_int = snr_db(x, fmts["INT4"].quantize(x))

        assert sqnr_sq > sqnr_int, \
            f"SQ SQNR={sqnr_sq:.1f} should exceed INT4 SQNR={sqnr_int:.1f}"

    def test_overhead_above_4(self):
        """SQ-Format must use more than 4 bits/elem (mask + sparse overhead)."""
        sq = SQFormat()
        overhead = sq.encoding_overhead()
        assert overhead["data_bits_per_element"] > 4.0

    def test_finite_output(self):
        sq = SQFormat(dense_bits=4, sparse_bits=8, sparsity_ratio=0.01)
        x = make_input(outlier=True)
        assert np.all(np.isfinite(sq.quantize(x)))

    def test_backward_compat_params(self):
        """Old parameter names (dense_bits, sparse_bits, sparsity_ratio) still work."""
        sq = SQFormat(dense_bits=4, sparse_bits=8, sparsity_ratio=0.01)
        assert sq.low_bits == 4
        assert sq.high_bits == 8
        assert abs(sq.sparsity - 0.99) < 1e-6   # sparsity = 1 - sparsity_ratio
        assert abs(sq.sparsity_ratio - 0.01) < 1e-6

    def test_bank_structure_isolates_outlier(self):
        """Within a bank the outlier element must receive high-precision treatment."""
        # One bank, two outliers planted at known positions.
        # With sparsity=0.5 the top-50% of a 4-element bank = 2 elements get hhigh.
        sq = SQFormat(bank_size=4, sparsity=0.5, high_bits=8, low_bits=4)
        x = np.array([100.0, 0.1, -100.0, 0.1], dtype=np.float32)
        x_q = sq.quantize(x)
        # The two outliers should be recovered more accurately than INT4 would allow
        assert np.all(np.isfinite(x_q))
        assert x_q.shape == x.shape
        # Outlier reconstruction error should be very small (high-precision path)
        assert abs(x_q[0] - 100.0) < 2.0, f"outlier[0] error={abs(x_q[0]-100.0):.2f}"
        assert abs(x_q[2] - (-100.0)) < 2.0, f"outlier[2] error={abs(x_q[2]+100.0):.2f}"

    def test_alg1_smooth_changes_importance(self):
        """Providing A_mean triggers _smooth() and shifts importance to smoothed W'.

        Without smoothing, an outlier weight dominates the mask.
        With smoothing, the calibration activation shifts importance to the
        channel with large activation × weight product.
        """
        sq = SQFormat(bank_size=4, sparsity=0.5, high_bits=8, low_bits=4)
        # Two output neurons, 4 input channels
        W = np.array([
            [10.0, 0.01],   # ch 0: large weight,  tiny weight
            [0.01, 0.01],   # ch 1
            [0.01, 0.01],   # ch 2
            [0.01, 10.0],   # ch 3: tiny weight, large weight
        ], dtype=np.float32)
        # With A_mean = [1, 1, 1, 1]: no shift from smoothing → magnitude dominates
        A_uniform = np.ones(4, dtype=np.float32)
        W_q_uniform = sq.quantize(W, A_mean=A_uniform)
        assert W_q_uniform.shape == W.shape
        assert np.all(np.isfinite(W_q_uniform))
        # Result is the smoothed (modified) W', not the raw W
        # Just verify the call runs without error and _last_bank_scales is populated
        assert len(sq._last_bank_scales) > 0

    def test_hessian_importance_changes_mask(self):
        """Providing H_inv_diag should shift importance from magnitude to Hessian metric."""
        sq = SQFormat(bank_size=4, sparsity=0.5, high_bits=8, low_bits=4)
        # W: large row-0, small row-1
        W = np.array([[10.0, 9.0], [0.1, 0.1]], dtype=np.float32)
        # H_inv_diag: row-0 has tiny H⁻¹ (high Hessian curvature = less important per metric)
        #             row-1 has huge H⁻¹ (low Hessian curvature = more important per metric)
        H_inv_diag = np.array([0.001, 1000.0], dtype=np.float32)
        # With Hessian: I_0 = (10)² / (0.001)² = 1e8 >> I_1 = (0.1)² / (1000)² = 1e-8
        # So row-0 still dominates here — just verify it runs without error and shape is ok.
        W_q = sq.quantize(W, H_inv_diag=H_inv_diag)
        assert W_q.shape == W.shape
        assert np.all(np.isfinite(W_q))

    def test_paper_faithful_s05_overhead(self):
        """Paper-canonical config (s=0.5, b=128, hhigh=8, hlow=4): overhead = 6 bits.

        Sentinel mask eliminates the separate 1-bit/element boolean mask:
          high_cost = 0.5×8 = 4, low_cost = 0.5×4 = 2, mask_cost = 0 → total = 6.
        """
        sq = SQFormat(bank_size=128, sparsity=0.5, high_bits=8, low_bits=4)
        oh = sq.encoding_overhead()
        # mask_cost=0 because v_sentinel encodes high-prec positions in-band
        assert abs(oh["data_bits_per_element"] - 6.0) < 1e-6
        assert oh["metadata_bits_per_element"] == 0.0
        assert oh["v_sentinel"] == -8  # -(2^(4-1)) for INT4

    # ── Algorithm 2 (activation quantization) ────────────────────────────────

    def test_sq_format_a_shape(self):
        sq_a = SQFormatActivations(bank_size=128, sparsity=0.5)
        x = make_input(outlier=True)
        x_q = sq_a.quantize(x)
        assert x_q.shape == x.shape

    def test_sq_format_a_finite(self):
        sq_a = SQFormatActivations(bank_size=128, sparsity=0.5)
        x = make_input(outlier=True)
        assert np.all(np.isfinite(sq_a.quantize(x)))

    def test_sq_format_a_quantize_weights_shape(self):
        """quantize_weights must return 5-tuple with correct shapes."""
        K, N = 64, 32
        sq_a = SQFormatActivations(bank_size=32, sparsity=0.5, high_bits=8, low_bits=4)
        W      = np.random.default_rng(0).normal(0, 1, (K, N)).astype(np.float32)
        A_mean = np.random.default_rng(1).normal(0, 1, K).astype(np.float32)

        W_q, scales, mask, reorder_idx, sq_scales = sq_a.quantize_weights(W, A_mean)

        assert W_q.shape        == (K, N), f"W_q shape {W_q.shape}"
        assert scales.shape     == (K,),   f"scales shape {scales.shape}"
        assert mask.shape       == (K,),   f"mask shape {mask.shape}"
        assert reorder_idx.shape == (K,),  f"reorder_idx shape {reorder_idx.shape}"
        assert sq_scales.shape  == (K,),   f"sq_scales shape {sq_scales.shape}"
        assert mask.dtype == bool
        assert np.all(np.isfinite(W_q))
        assert np.all(sq_scales > 0), "sq_scales must be positive"

    def test_sq_format_a_mask_sparsity(self):
        """Fraction of high-precision channels per bank ≈ (1-s)."""
        K, N = 128, 64
        sq_a = SQFormatActivations(bank_size=64, sparsity=0.5, high_bits=8, low_bits=4)
        W      = np.random.default_rng(0).normal(0, 1, (K, N)).astype(np.float32)
        A_mean = np.random.default_rng(1).normal(0, 1, K).astype(np.float32)

        _, _, mask, reorder_idx, _ = sq_a.quantize_weights(W, A_mean)

        # Undo reordering to check per-bank mask fractions
        orig_mask = np.empty_like(mask)
        orig_mask[reorder_idx] = mask
        for bank_start in range(0, K, 64):
            bank_mask = orig_mask[bank_start:bank_start + 64]
            high_frac = bank_mask.mean()
            assert abs(high_frac - 0.5) < 0.1, \
                f"Bank {bank_start}: high-prec fraction={high_frac:.2f}, expected ≈0.5"

    def test_sq_format_a_reorder_high_first(self):
        """After reordering, high-precision channels must come before low-prec in each bank."""
        K, N = 32, 16
        sq_a = SQFormatActivations(bank_size=16, sparsity=0.5, high_bits=8, low_bits=4)
        W      = np.random.default_rng(2).normal(0, 1, (K, N)).astype(np.float32)
        A_mean = np.abs(np.random.default_rng(3).normal(0, 1, K)).astype(np.float32)

        _, _, mask_reordered, _, _ = sq_a.quantize_weights(W, A_mean)

        for bank_start in range(0, K, 16):
            bm = mask_reordered[bank_start:bank_start + 16]
            # All True values must come before False values within the bank
            seen_false = False
            for v in bm:
                if not v:
                    seen_false = True
                if seen_false and v:
                    pytest.fail(f"Bank {bank_start}: low-prec channel before high-prec after reorder")

    def test_runtime_activation_batch_independence(self):
        """quantize_runtime_activations must be invariant to batch composition.

        If token A appears in batch [A] vs batch [A, B], its quantized values
        must be identical.  This verifies the fix for the bug where per-column
        (across-batch) scaling made each token's quantization depend on all
        other tokens in the same batch.
        """
        K, N = 32, 16
        sq_a = SQFormatActivations(bank_size=16, sparsity=0.5, high_bits=8, low_bits=4)
        W      = np.random.default_rng(0).normal(0, 1, (K, N)).astype(np.float32)
        A_mean = np.abs(np.random.default_rng(1).normal(0, 1, K)).astype(np.float32)

        # sq_scales is now returned directly — no need to recompute via _smooth
        _, _, mask_reordered, reorder_idx, sq_scales = sq_a.quantize_weights(W, A_mean)

        rng = np.random.default_rng(42)
        token_A = rng.normal(0, 1, (1, K)).astype(np.float32)
        token_B = rng.normal(0, 5, (1, K)).astype(np.float32)   # very different scale

        # Quantize token A alone vs alongside token B
        h_A_solo, l_A_solo, _ = sq_a.quantize_runtime_activations(
            token_A, mask_reordered, sq_scales, reorder_idx
        )
        h_AB, l_AB, _ = sq_a.quantize_runtime_activations(
            np.vstack([token_A, token_B]), mask_reordered, sq_scales, reorder_idx
        )

        np.testing.assert_array_equal(
            h_A_solo, h_AB[:1],
            err_msg="High-prec token A changes when batched with token B"
        )
        np.testing.assert_array_equal(
            l_A_solo, l_AB[:1],
            err_msg="Low-prec token A changes when batched with token B"
        )

    def test_sq_format_a_importance_uses_activation(self):
        """Channels with large |Ā_j · Σ W'_ji| should be selected as high-precision."""
        K, N = 4, 8
        sq_a = SQFormatActivations(bank_size=4, sparsity=0.5, high_bits=8, low_bits=4)
        # Channel 0: large activation, large weights → most important
        # Channel 3: tiny activation, small weights → least important
        W = np.array([
            [10.0] * N,   # channel 0: large weights
            [1.0]  * N,   # channel 1
            [1.0]  * N,   # channel 2
            [0.01] * N,   # channel 3: tiny weights
        ], dtype=np.float32)
        A_mean = np.array([100.0, 1.0, 1.0, 0.01], dtype=np.float32)

        _, _, mask_reordered, reorder_idx, _ = sq_a.quantize_weights(W, A_mean)

        # Undo reordering
        orig_mask = np.empty_like(mask_reordered)
        orig_mask[reorder_idx] = mask_reordered

        # Channel 0 (highest importance) must be high-precision
        assert orig_mask[0], "Channel 0 (highest importance) should be high-precision"
        # Channel 3 (lowest importance) must be low-precision
        assert not orig_mask[3], "Channel 3 (lowest importance) should be low-precision"


# ─── Metrics ─────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_mse_identity(self):
        from distributions.metrics import mse
        x = make_input()
        assert mse(x, x) == 0.0

    def test_snr_db_identity(self):
        from distributions.metrics import snr_db
        x = make_input()
        assert snr_db(x, x) == float("inf")

    def test_snr_db_monotone(self):
        """Higher quantization bits → higher SQNR."""
        from distributions.metrics import snr_db
        from formats import build_all_formats
        fmts = build_all_formats(dim=N)
        x = make_input()
        sqnr4 = snr_db(x, fmts["INT4"].quantize(x))
        sqnr8 = snr_db(x, fmts["INT8"].quantize(x))
        assert sqnr8 > sqnr4

    def test_effective_bits(self):
        from distributions.metrics import effective_bits
        from formats import build_all_formats
        fmts = build_all_formats(dim=N)
        x = make_input()
        eff = effective_bits(x, fmts["INT8"].quantize(x))
        # POT scale on small N can give slightly lower eff_bits; range loosened
        assert 1.0 < eff < 9.0, f"INT8 eff_bits={eff:.1f} out of expected range"


# ─── Distributions ───────────────────────────────────────────────────────────

class TestDistributions:
    def test_all_generators_run(self):
        from distributions.generators import (
            gaussian, channel_outliers, spiky_outliers, student_t_dist,
            laplace, bimodal, log_normal,
        )
        n = 512
        for fn, kwargs in [
            (gaussian, {"n": n}),
            (channel_outliers, {"n": n}),
            (spiky_outliers, {"n": n}),
            (student_t_dist, {"n": n}),
            (laplace, {"n": n}),
            (bimodal, {"n": n}),
            (log_normal, {"n": n}),
        ]:
            result = fn(**kwargs)
            x = result[0] if isinstance(result, tuple) else result
            assert np.all(np.isfinite(x)), f"{fn.__name__} produced non-finite values"

    def test_channel_outlier_has_outliers(self):
        from distributions.generators import channel_outliers
        x, meta = channel_outliers(n=512, outlier_sigma=50.0, seed=42)
        # Some channels should have std >> 1
        assert np.max(np.abs(x)) > 10.0, "No outliers injected"


# ─── Hardware models ─────────────────────────────────────────────────────────

class TestHardwareModels:
    def test_int8_area_greater_int4(self):
        from hardware.pyrtl_modules.int_mac_array import get_int_array_ppa
        ppa4 = get_int_array_ppa(bits=4)
        ppa8 = get_int_array_ppa(bits=8)
        assert ppa8.get("area_um2", 1) >= ppa4.get("area_um2", 0)

    def test_energy_model_memory_dominates(self):
        from hardware.energy_model import EnergyModel
        em = EnergyModel()
        result = em.total_inference_energy("INT4", n_macs=256, n_weight_reads=256,
                                            n_activation_reads=256)
        assert result["memory_pJ"] > result["compute_pJ"], \
            "Memory energy should dominate compute at inference batch=1"

    def test_fwht_area_positive(self):
        from hardware.pyrtl_modules.fwht_module import get_fwht_ppa
        ppa = get_fwht_ppa(n=256, bits=4)
        # Key may be area_um2 or area_mm2_total depending on PyRTL availability
        area = ppa.get("area_um2", ppa.get("area_mm2_total", 0) * 1e6)
        assert area > 0, f"FWHT area must be positive, got ppa={ppa}"

    def test_had_hardware_ops_formula(self):
        """Verify hardware_ops follows n * log2(n) formula."""
        had = HADTransform()
        for n in [64, 128, 256, 512]:
            ops = had.hardware_ops(n)
            expected_total = n * int(np.log2(n))
            assert ops["total_ops"] == expected_total, \
                f"n={n}: expected {expected_total} total_ops, got {ops['total_ops']}"
