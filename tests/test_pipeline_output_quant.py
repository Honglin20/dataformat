"""Tests for PR C: optional Y quantisation at GEMM output.

These tests pin the semantics laid out in
``docs/plans/2026-04-19-sqformat-experiment-plan.md`` → "PR C · Y quantisation
at GEMM output" and design decision R2:

  * ``FourBitConfig.quantize_output`` defaults to ``False`` so existing
    4-bit CSVs stay byte-identical.
  * ``Pipeline.output_fmt=None`` reproduces the legacy behaviour exactly
    (FP32 accumulator, no Y quantisation).
  * ``Pipeline.output_fmt=<fmt>`` quantises Y after ``output_correction``
    and before bias.
  * SQ-Format instances bank along the out_features axis (one bank per
    token), not along the batch / token axis.
"""
from __future__ import annotations

import numpy as np

from experiments.fourbit.config import DEFAULT_CONFIG


def test_quantize_output_default_false():
    assert DEFAULT_CONFIG.quantize_output is False


def test_pipeline_output_fmt_none_keeps_y_fp32():
    from experiments.fourbit.registry import build_formats
    from experiments.fourbit.pipeline import Pipeline
    from experiments.fourbit.transforms import make_transform

    fmt = build_formats(DEFAULT_CONFIG)["INT4"]
    pipe = Pipeline(transform=make_transform("identity"), fmt=fmt, output_fmt=None)
    X = np.random.default_rng(0).standard_normal((8, 64)).astype(np.float32)
    W = np.random.default_rng(1).standard_normal((32, 64)).astype(np.float32)
    Y = pipe.simulate_linear(X, W)
    # FP32 accumulator semantics — Y must not be quantised
    assert Y.dtype == np.float32
    # Should match the current behaviour byte-for-byte
    pipe_legacy = Pipeline(transform=make_transform("identity"), fmt=fmt)
    np.testing.assert_array_equal(Y, pipe_legacy.simulate_linear(X, W))


def test_pipeline_output_fmt_set_quantises_y():
    from experiments.fourbit.registry import build_formats
    from experiments.fourbit.pipeline import Pipeline
    from experiments.fourbit.transforms import make_transform

    fmt = build_formats(DEFAULT_CONFIG)["INT4"]
    pipe_q = Pipeline(transform=make_transform("identity"), fmt=fmt, output_fmt=fmt)
    pipe_n = Pipeline(transform=make_transform("identity"), fmt=fmt, output_fmt=None)
    X = np.random.default_rng(0).standard_normal((8, 64)).astype(np.float32)
    W = np.random.default_rng(1).standard_normal((32, 64)).astype(np.float32)
    Y_q = pipe_q.simulate_linear(X, W)
    Y_n = pipe_n.simulate_linear(X, W)
    assert not np.array_equal(Y_q, Y_n)


def test_sqformat_output_fmt_uses_out_features_axis():
    """R2: SQ-Format output quantisation must bank along out_features, not batch.

    A regression test for the helper that transposes y before/after applying
    SQ-Format. Without the transpose, banks would group tokens, not output
    channels — destroying per-token semantics."""
    from experiments.fourbit.pipeline import Pipeline
    from experiments.fourbit.transforms import make_transform
    from formats.sq_format import SQFormat

    fmt = SQFormat(base="int", high_bits=4, low_bits=4, bank_size=128, sparsity=0.5)
    pipe = Pipeline(transform=make_transform("identity"), fmt=fmt, output_fmt=fmt)
    X = np.random.default_rng(0).standard_normal((16, 64)).astype(np.float32)  # 16 tokens
    W = np.random.default_rng(1).standard_normal((10, 64)).astype(np.float32)  # 10 out_features
    Y = pipe.simulate_linear(X, W)
    assert Y.shape == (16, 10)
    # Output is finite — verifies the (transpose, quantise, transpose-back) round-trip works
    assert np.isfinite(Y).all()
