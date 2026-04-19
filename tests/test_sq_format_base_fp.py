import numpy as np
import pytest
from formats.sq_format import _ELEMENT_ENCODERS


def test_int4_encoder_round_clip():
    fn, q_max = _ELEMENT_ENCODERS[("int", 4)]
    assert q_max == 7
    x = np.array([0.2, 7.6, -8.3, 0.51], dtype=np.float32)
    # scale=1.0 → round(x) clipped to [-7, 7]
    got = fn(x, scale=1.0)
    np.testing.assert_array_equal(got, np.array([0.0, 7.0, -7.0, 1.0], dtype=np.float32))


def test_fp4_e2m1_encoder_uses_level_set():
    fn, q_max = _ELEMENT_ENCODERS[("fp", 4)]
    assert q_max == 6.0
    # scale=1.0 → nearest E2M1 level (0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6)
    x = np.array([0.3, 1.4, 5.5, -6.5], dtype=np.float32)
    got = fn(x, scale=1.0)
    np.testing.assert_allclose(got, np.array([0.5, 1.5, 6.0, -6.0], dtype=np.float32))


def test_fp8_e4m3_encoder_saturates_at_448():
    fn, q_max = _ELEMENT_ENCODERS[("fp", 8)]
    assert q_max == 448.0
    x = np.array([0.0, 448.0, 500.0, -600.0], dtype=np.float32)
    got = fn(x, scale=1.0)
    assert got[0] == 0.0
    assert got[1] == 448.0
    assert abs(got[2]) <= 448.0
    assert abs(got[3]) <= 448.0


from formats.sq_format import SQFormat


def test_sqformat_base_int_default_unchanged():
    # Regression pin: default SQFormat() behaviour is identical to
    # SQFormat(base="int") — existing golden CSVs depend on this.
    rng = np.random.default_rng(0)
    W = rng.standard_normal((128, 64)).astype(np.float32)
    a = SQFormat()
    b = SQFormat(base="int")
    np.testing.assert_array_equal(a.quantize(W), b.quantize(W))


def test_sqformat_base_fp_produces_fp_levels():
    rng = np.random.default_rng(1)
    W = rng.standard_normal((128, 64)).astype(np.float32)
    fmt = SQFormat(base="fp", high_bits=8, low_bits=4, sparsity=0.5)
    W_q = fmt.quantize(W)
    assert W_q.shape == W.shape
    # FP output must not be on an integer grid — this distinguishes it
    # from base="int" at a coarse scale.
    a_int = SQFormat(base="int", high_bits=8, low_bits=4, sparsity=0.5).quantize(W)
    assert not np.array_equal(a_int, W_q)


def test_sqformat_rejects_unsupported_cell():
    with pytest.raises(ValueError, match="Unsupported SQ-Format cell"):
        SQFormat(base="fp", high_bits=2, low_bits=2)  # no FP2 in registry
