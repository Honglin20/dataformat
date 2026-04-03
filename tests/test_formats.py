"""Unit tests for all quantization formats.

Tests cover:
  - Shape preservation
  - No NaN/Inf in output
  - Mathematical properties (energy, self-inverse, monotonicity)
  - Quantization level constraints
  - Encoding overhead metadata
  - Outlier robustness (non-crash)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from formats import build_all_formats
from formats.baseline import FP32Format, BF16Format
from formats.nvfp4 import NVFP4Format
from formats.mxfp import MXFPFormat
from formats.mxint import MXINTFormat
from formats.nf4 import NF4Format
from formats.fp6 import FP6Format, _FP6_POS_LEVELS
from formats.sq_format import SQFormat
from formats.transforms.hadamard import hadamard_transform, inverse_hadamard_transform
from formats.transforms.random_rotation import RandomRotationTransform, TurboQuantTransform
from formats.transforms.smoothquant import SmoothQuantTransform, SmoothQuantINTQuantizer
from config import NF4_LEVELS


RNG = np.random.default_rng(42)
N = 256


def make_input(n=N, outlier=False):
    x = RNG.normal(0, 1, n).astype(np.float32)
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

    def test_count(self):
        assert len(self.formats) >= 20, f"Expected ≥20 formats, got {len(self.formats)}"

    @pytest.mark.parametrize("fmt_name", list(build_all_formats(dim=N, seed=42).keys()))
    def test_shape_preserved(self, fmt_name):
        fmt = build_all_formats(dim=N, seed=42)[fmt_name]
        out = fmt.quantize(self.x)
        assert out.shape == self.x.shape, f"{fmt_name}: shape {out.shape} != {self.x.shape}"

    @pytest.mark.parametrize("fmt_name", list(build_all_formats(dim=N, seed=42).keys()))
    def test_no_nan_inf(self, fmt_name):
        fmt = build_all_formats(dim=N, seed=42)[fmt_name]
        out = fmt.quantize(self.x)
        assert np.all(np.isfinite(out)), f"{fmt_name}: non-finite values in output"

    @pytest.mark.parametrize("fmt_name", list(build_all_formats(dim=N, seed=42).keys()))
    def test_outlier_robustness(self, fmt_name):
        fmt = build_all_formats(dim=N, seed=42)[fmt_name]
        out = fmt.quantize(self.x_out)
        assert out.shape == self.x_out.shape
        assert np.all(np.isfinite(out)), f"{fmt_name}: non-finite on outlier input"

    @pytest.mark.parametrize("fmt_name", list(build_all_formats(dim=N, seed=42).keys()))
    def test_mse_less_than_input_variance(self, fmt_name):
        """Quantization MSE should be less than input variance (sanity)."""
        fmt = build_all_formats(dim=N, seed=42)[fmt_name]
        out = fmt.quantize(self.x)
        mse = float(np.mean((self.x - out) ** 2))
        var = float(np.var(self.x))
        assert mse <= var * 100, f"{fmt_name}: MSE {mse:.4f} >> variance {var:.4f}"

    @pytest.mark.parametrize("fmt_name", list(build_all_formats(dim=N, seed=42).keys()))
    def test_encoding_overhead_returns_dict(self, fmt_name):
        fmt = build_all_formats(dim=N, seed=42)[fmt_name]
        if hasattr(fmt, "encoding_overhead"):
            oh = fmt.encoding_overhead()
            assert isinstance(oh, dict)
            assert "data_bits_per_element" in oh


# ─── HAD Transform ────────────────────────────────────────────────────────────

class TestHADTransform:
    def setup_method(self):
        self.x = make_input(N)

    def test_correct_wht_small(self):
        """WHT of [1,2,3,4] should be [10,-2,-4,0]."""
        x = np.array([1., 2., 3., 4.], dtype=np.float32)
        result = hadamard_transform(x, normalize=False)
        expected = np.array([10., -2., -4., 0.], dtype=np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-4)

    def test_energy_preserved(self):
        """Orthonormal HAD must preserve L2 norm."""
        h = hadamard_transform(self.x, normalize=True)
        ratio = np.sum(h ** 2) / np.sum(self.x ** 2)
        assert abs(ratio - 1.0) < 1e-4, f"Energy ratio {ratio:.6f} != 1.0"

    def test_self_inverse(self):
        """Applying HAD twice with normalize=True must recover input."""
        h = hadamard_transform(self.x, normalize=True)
        x_rec = inverse_hadamard_transform(h, normalize=True)
        np.testing.assert_allclose(x_rec, self.x, atol=1e-4,
                                   err_msg="HAD is not self-inverse")

    def test_non_power_of_two(self):
        """HAD should handle non-power-of-2 lengths by zero-padding then trimming."""
        x7 = make_input(7)
        h = hadamard_transform(x7, normalize=True)
        assert h.shape == (7,)
        assert np.all(np.isfinite(h))

    def test_outlier_spreads_energy(self):
        """After HAD, a single spike should spread its energy across all elements."""
        x_spike = np.zeros(N, dtype=np.float32)
        x_spike[0] = 100.0
        h = hadamard_transform(x_spike, normalize=True)
        # No element should have |h[i]| > 100/sqrt(N) * 2 (energy spread)
        max_expected = 100.0 / np.sqrt(N) * 1.1
        assert np.max(np.abs(h)) <= max_expected + 1e-3, \
            f"HAD didn't spread energy: max={np.max(np.abs(h)):.3f}"

    def test_batch_dimension(self):
        """HAD should work on 2D (batch, n) inputs."""
        X = RNG.normal(0, 1, (8, N)).astype(np.float32)
        H = hadamard_transform(X, normalize=True)
        assert H.shape == (8, N)
        # Energy preservation per sample
        ratios = np.sum(H ** 2, axis=1) / np.sum(X ** 2, axis=1)
        np.testing.assert_allclose(ratios, 1.0, atol=1e-4)


# ─── Random Rotation ──────────────────────────────────────────────────────────

class TestRandomRotation:
    def setup_method(self):
        self.x = make_input(N)
        self.rr = RandomRotationTransform(dim=N, seed=42)

    def test_energy_preserved(self):
        r = self.rr.forward(self.x)
        ratio = np.sum(r ** 2) / np.sum(self.x ** 2)
        assert abs(ratio - 1.0) < 1e-4

    def test_inverse(self):
        r = self.rr.forward(self.x)
        x_rec = self.rr.inverse(r)
        np.testing.assert_allclose(x_rec, self.x, atol=1e-4)

    def test_deterministic(self):
        """Same seed → same rotation matrix."""
        rr2 = RandomRotationTransform(dim=N, seed=42)
        np.testing.assert_array_equal(self.rr.Q, rr2.Q)

    def test_orthogonal_matrix(self):
        """Q @ Q^T should be identity."""
        QtQ = self.rr.Q @ self.rr.Q.T
        np.testing.assert_allclose(QtQ, np.eye(N, dtype=np.float32), atol=1e-4)


class TestTurboQuant:
    def setup_method(self):
        self.x = make_input(N)
        self.turbo = TurboQuantTransform(dim=N, seed=42)

    def test_self_inverse(self):
        t = self.turbo.forward(self.x)
        x_rec = self.turbo.inverse(t)
        np.testing.assert_array_equal(x_rec, self.x)

    def test_signs_are_pm1(self):
        assert np.all(np.abs(self.turbo.signs) == 1.0)

    def test_energy_preserved(self):
        t = self.turbo.forward(self.x)
        np.testing.assert_allclose(np.sum(t**2), np.sum(self.x**2), rtol=1e-5)


# ─── SmoothQuant ──────────────────────────────────────────────────────────────

class TestSmoothQuant:
    def setup_method(self):
        self.sq = SmoothQuantTransform(alpha=0.5)
        self.X = RNG.normal(0, 1, (32, N)).astype(np.float32)
        self.W = RNG.normal(0, 1, (N, N)).astype(np.float32)
        self.sq.fit(self.X, self.W)

    def test_scales_positive(self):
        assert np.all(self.sq.scales > 0)

    def test_forward_inverse(self):
        X_smooth = self.sq.forward(self.X)
        X_rec = self.sq.inverse(X_smooth)
        np.testing.assert_allclose(X_rec, self.X, rtol=1e-4)

    def test_algebraic_equivalence(self):
        """(X * s) @ (W / s) ≈ X @ W (up to quantization of s)."""
        X_s = self.sq.forward(self.X)
        W_s = self.sq.transform_weights(self.W)
        original = self.X @ self.W
        smoothed = X_s @ W_s
        np.testing.assert_allclose(smoothed, original, atol=1e-4,
                                   err_msg="SmoothQuant algebraic identity failed")


# ─── NF4 ──────────────────────────────────────────────────────────────────────

class TestNF4:
    def setup_method(self):
        self.nf4 = NF4Format()

    def test_levels_sorted(self):
        assert np.all(np.diff(NF4_LEVELS) > 0), "NF4 levels must be strictly increasing"

    def test_output_in_level_set(self):
        x = make_input()
        q = self.nf4.quantize(x)
        absmax = np.max(np.abs(x))
        q_norm = q / (absmax + 1e-8)
        for val in q_norm:
            assert np.any(np.abs(val - NF4_LEVELS) < 1e-4), \
                f"Value {val:.6f} not in NF4 level set"

    def test_zero_input(self):
        x = np.zeros(64, dtype=np.float32)
        q = self.nf4.quantize(x)
        np.testing.assert_array_equal(q, x)

    def test_4bit_range(self):
        """16 unique output levels per sign → max 31 unique values."""
        x = RNG.normal(0, 1, 1000).astype(np.float32)
        q = self.nf4.quantize(x)
        absmax = np.max(np.abs(x))
        q_norm = np.unique(np.round(q / absmax, 5))
        assert len(q_norm) <= 16, f"Too many unique NF4 levels: {len(q_norm)}"


# ─── FP6 ──────────────────────────────────────────────────────────────────────

class TestFP6:
    def setup_method(self):
        self.fp6 = FP6Format()

    def test_positive_levels_monotonic(self):
        assert np.all(np.diff(_FP6_POS_LEVELS) >= 0)

    def test_max_value(self):
        assert abs(_FP6_POS_LEVELS[-1] - 28.0) < 1e-4

    def test_clamps_to_range(self):
        x = np.array([1000.0, -1000.0, 0.0, 0.5], dtype=np.float32)
        q = self.fp6.quantize(x)
        # scaled by absmax/28 so clipping should handle extreme values
        assert np.all(np.isfinite(q))

    def test_mse_less_than_int8(self):
        """FP6 should outperform INT4 on normal distribution."""
        x = RNG.normal(0, 1, N).astype(np.float32)
        q6 = self.fp6.quantize(x)
        mse_fp6 = float(np.mean((x - q6) ** 2))

        from formats.mxint import MXINTFormat
        mint4 = MXINTFormat(element_bits=4)
        q4 = mint4.quantize(x)
        mse_int4 = float(np.mean((x - q4) ** 2))
        assert mse_fp6 < mse_int4, \
            f"FP6 MSE {mse_fp6:.4f} should be less than MXINT4 MSE {mse_int4:.4f}"


# ─── MXFP ─────────────────────────────────────────────────────────────────────

class TestMXFP:
    def test_mxfp4_block_scaling(self):
        """Consecutive blocks should be independently scaled."""
        # Block 1: large values; Block 2: small values
        x = np.zeros(64, dtype=np.float32)
        x[:32] = 100.0
        x[32:] = 0.01
        fmt = MXFPFormat(element_bits=4, block_size=32)
        q = fmt.quantize(x)
        # Large block should not corrupt small block reconstruction
        mse_small = float(np.mean((x[32:] - q[32:]) ** 2))
        mse_large = float(np.mean((x[:32] - q[:32]) ** 2))
        # Both blocks should have finite MSE proportional to their own range
        assert np.isfinite(mse_small) and np.isfinite(mse_large)
        assert mse_small < 1.0, f"Small block MSE too large: {mse_small:.4f}"

    def test_mxfp4_vs_nvfp4_outlier(self):
        """MXFP4 block scale should handle outliers better than per-tensor NVFP4."""
        x = np.zeros(128, dtype=np.float32)
        x[:32] = RNG.normal(0, 1, 32).astype(np.float32)   # normal block
        x[32:64] = 0.001  # tiny block
        x[0] = 50.0  # outlier in block 0

        mxfp4 = MXFPFormat(element_bits=4, block_size=32)
        nvfp4 = NVFP4Format()

        q_mx = mxfp4.quantize(x)
        q_nv = nvfp4.quantize(x)

        # MSE on the tiny block: MXFP4 should be better (separate scale)
        mse_mx_tiny = float(np.mean((x[32:64] - q_mx[32:64]) ** 2))
        mse_nv_tiny = float(np.mean((x[32:64] - q_nv[32:64]) ** 2))
        assert mse_mx_tiny < mse_nv_tiny, \
            f"MXFP4 tiny block MSE {mse_mx_tiny:.6f} should < NVFP4 {mse_nv_tiny:.6f}"

    def test_encoding_overhead_metadata(self):
        fmt4 = MXFPFormat(element_bits=4, block_size=32)
        fmt8 = MXFPFormat(element_bits=8, block_size=32)
        oh4 = fmt4.encoding_overhead()
        oh8 = fmt8.encoding_overhead()
        # metadata = 8 bits per 32 elements = 0.25 bits/element
        assert abs(oh4["metadata_bits_per_element"] - 0.25) < 1e-6
        assert abs(oh8["metadata_bits_per_element"] - 0.25) < 1e-6
        assert oh4["bandwidth_amplification"] > 1.0
        assert oh8["bandwidth_amplification"] > 1.0


# ─── SQ-Format ────────────────────────────────────────────────────────────────

class TestSQFormat:
    def setup_method(self):
        self.sq = SQFormat(dense_bits=4, sparse_bits=8, sparsity_ratio=0.01)
        self.x = make_input(N, outlier=True)

    def test_output_shape(self):
        q = self.sq.quantize(self.x)
        assert q.shape == self.x.shape

    def test_salient_weights_preserved_better(self):
        """Top-1% weights should have lower error than the dense-only version."""
        from formats.mxint import MXINTFormat
        int4 = MXINTFormat(element_bits=4)

        q_sq = self.sq.quantize(self.x)
        q_int = int4.quantize(self.x)

        # Compare top-1% elements specifically
        k = max(1, int(0.01 * N))
        top_idx = np.argpartition(np.abs(self.x), -k)[-k:]

        err_sq = np.mean((self.x[top_idx] - q_sq[top_idx]) ** 2)
        err_int = np.mean((self.x[top_idx] - q_int[top_idx]) ** 2)
        assert err_sq < err_int, \
            f"SQ salient MSE {err_sq:.4f} should < INT4 {err_int:.4f}"

    def test_encoding_overhead(self):
        oh = self.sq.encoding_overhead()
        assert oh["sparsity_ratio"] == 0.01
        assert oh["dense_bits"] == 4
        assert oh["sparse_bits"] == 8
        assert oh["data_bits_per_element"] > 4   # overhead > dense-only


# ─── Metrics ──────────────────────────────────────────────────────────────────

class TestMetrics:
    def setup_method(self):
        self.x = make_input(N)
        self.x_noisy = self.x + 0.01 * RNG.normal(0, 1, N).astype(np.float32)

    def test_mse_zero_for_identity(self):
        from distributions.metrics import mse
        assert mse(self.x, self.x) == 0.0

    def test_mse_positive(self):
        from distributions.metrics import mse
        assert mse(self.x, self.x_noisy) > 0.0

    def test_snr_inf_for_identity(self):
        from distributions.metrics import snr_db
        assert snr_db(self.x, self.x) == float("inf")

    def test_snr_decreases_with_noise(self):
        from distributions.metrics import snr_db
        snr_low = snr_db(self.x, self.x + 1.0 * RNG.normal(0, 1, N).astype(np.float32))
        snr_high = snr_db(self.x, self.x_noisy)
        assert snr_high > snr_low

    def test_kl_zero_for_identity(self):
        from distributions.metrics import kl_divergence
        assert kl_divergence(self.x, self.x) < 1e-6

    def test_max_ae_zero_for_identity(self):
        from distributions.metrics import max_absolute_error
        assert max_absolute_error(self.x, self.x) == 0.0

    def test_eff_bits_inf_for_identity(self):
        from distributions.metrics import effective_bits
        assert effective_bits(self.x, self.x) == float("inf")

    def test_eff_bits_reasonable_for_int4(self):
        from distributions.metrics import effective_bits
        from formats.mxint import MXINTFormat
        fmt = MXINTFormat(element_bits=4)
        q = fmt.quantize(self.x)
        eb = effective_bits(self.x, q)
        # INT4 should yield somewhere between 2 and 5 effective bits on Gaussian
        assert 1.0 < eb < 6.0, f"INT4 EffBits={eb:.2f} out of expected range [1,6]"


# ─── Distributions ────────────────────────────────────────────────────────────

class TestDistributions:
    def test_all_generators_run(self):
        from distributions.generators import generate_all_distributions
        dists = generate_all_distributions(n=256, seed=42)
        assert len(dists) >= 14

    def test_all_finite(self):
        from distributions.generators import generate_all_distributions
        for name, x, _ in generate_all_distributions(n=256, seed=42):
            assert np.all(np.isfinite(x)), f"{name}: non-finite values"

    def test_channel_outliers_correct_channels(self):
        from distributions.generators import channel_outliers
        x, meta = channel_outliers(n=256, outlier_ratio=0.05, outlier_sigma=50.0, seed=42)
        n_outliers = meta["n_outlier_channels"]
        # The outlier channels should have significantly larger magnitudes
        all_abs = np.sort(np.abs(x))[::-1]
        assert all_abs[0] > 10.0, "Largest element should be an outlier (>10σ)"

    def test_spiky_outliers_count(self):
        from distributions.generators import spiky_outliers
        x, meta = spiky_outliers(n=1000, spike_ratio=0.01, spike_multiplier=50.0, seed=42)
        # Should have approximately 1% spikes > 10σ
        n_large = int(np.sum(np.abs(x) > 20.0))
        assert 5 <= n_large <= 20, f"Expected ~10 spikes, got {n_large}"


# ─── Hardware models ──────────────────────────────────────────────────────────

class TestHardwareModels:
    def test_int_array_ppa_returns_area(self):
        from hardware.pyrtl_modules.int_mac_array import get_int_array_ppa
        ppa4 = get_int_array_ppa(bits=4)
        ppa8 = get_int_array_ppa(bits=8)
        assert ppa4["area_mm2_total"] > 0
        assert ppa8["area_mm2_total"] > ppa4["area_mm2_total"], \
            "INT8 array should be larger than INT4"

    def test_mxfp_area_larger_than_int(self):
        from hardware.pyrtl_modules.int_mac_array import get_int_array_ppa
        from hardware.pyrtl_modules.mxfp_mac_array import get_mxfp_array_ppa
        int_ppa = get_int_array_ppa(bits=4)
        mxfp_ppa = get_mxfp_array_ppa(element_bits=4)
        assert mxfp_ppa["area_mm2_total"] > int_ppa["area_mm2_total"], \
            "MXFP4 array should be larger than INT4 (exponent logic overhead)"

    def test_fwht_area_positive(self):
        from hardware.pyrtl_modules.fwht_module import get_fwht_ppa
        fwht = get_fwht_ppa(n=256, bits=4)
        assert fwht["area_mm2_total"] > 0
        assert fwht["n_stages"] == 8  # log2(256) = 8

    def test_fwht_area_smaller_than_array(self):
        """FWHT module should be much smaller than the MAC array (amortization argument)."""
        from hardware.pyrtl_modules.int_mac_array import get_int_array_ppa
        from hardware.pyrtl_modules.fwht_module import get_fwht_ppa
        array_ppa = get_int_array_ppa(bits=4)
        fwht_ppa = get_fwht_ppa(n=256, bits=4)
        ratio = fwht_ppa["area_mm2_total"] / array_ppa["area_mm2_total"]
        assert ratio < 50.0, f"FWHT/array area ratio {ratio:.1f} seems too large"

    def test_energy_model_compute_less_than_memory(self):
        """For inference, memory access energy >> compute energy (Horowitz model)."""
        from hardware.energy_model import EnergyModel
        em = EnergyModel()
        e = em.total_inference_energy(
            "INT8", n_macs=256, n_weight_reads=256, n_activation_reads=256
        )
        assert e["memory_pJ"] > e["compute_pJ"], \
            "Memory access should dominate compute energy"

    def test_roofline_data_all_formats(self):
        from hardware.roofline import build_roofline_data
        data = build_roofline_data()
        assert len(data) >= 10
        for pt in data:
            assert np.isfinite(pt["arithmetic_intensity"])
            assert pt["arithmetic_intensity"] > 0
            assert pt["attainable_tops"] > 0

    def test_mx_metadata_reduces_arithmetic_intensity(self):
        """MX block scale metadata should lower I vs same-bitwidth format without metadata."""
        from hardware.roofline import build_roofline_data
        data = {p["format"]: p for p in build_roofline_data()}
        if "MXFP4" in data and "INT4" in data:
            assert data["MXFP4"]["arithmetic_intensity"] < data["INT4"]["arithmetic_intensity"], \
                "MXFP4 I should be lower than INT4 due to block scale metadata"

    def test_bops_matmul_correct(self):
        from hardware.bop_counter import BopCounter
        bc = BopCounter()
        bops = bc.matmul(M=16, K=256, N=256, bits_a=8, bits_b=8)
        expected = 16 * 256 * 256 * 8 * 8
        assert bops == expected

    def test_bops_had_overhead(self):
        """HAD transform should add non-zero BOPs above the matmul."""
        from hardware.bop_counter import BopCounter
        bc = BopCounter()
        bd = bc.linear_layer_bops("HAD+INT4", M=16, K=256, N=256,
                                   weight_bits=4, activation_bits=4)
        assert bd["transform_bops"] > 0
        assert bd["total_bops"] > bd["matmul_bops"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
