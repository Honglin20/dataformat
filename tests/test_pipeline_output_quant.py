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

    Chooses ``bank_size=4`` with ``batch=16, out_features=8`` so banking along
    the two axes yields observably different outputs:
      * transposed path   → K=out_features=8,  N=batch=16  → 2 K-banks of 4
      * no-transpose path → K=batch=16,        N=out_features=8 → 4 K-banks of 4
    Since the per-bank importance ordering depends on which axis is grouped,
    the two outputs will not be array-equal; this pins the transpose."""
    from experiments.fourbit.pipeline import _apply_output_fmt
    from formats.sq_format import SQFormat

    fmt = SQFormat(base="int", high_bits=4, low_bits=2, bank_size=4, sparsity=0.5)
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((16, 8)).astype(np.float32)

    # Helper result (what Pipeline.simulate_linear actually applies to Y).
    Y_helper = _apply_output_fmt(fmt, Y)
    # No-transpose reference: bank along batch instead of out_features.
    Y_no_transpose = fmt.quantize(Y)
    # Manual transpose reference: quantise along out_features by hand.
    Y_manual = fmt.quantize(np.ascontiguousarray(Y.T)).T

    assert Y_helper.shape == Y.shape
    # Helper must differ from the no-transpose path — proves axis is pinned.
    assert not np.array_equal(Y_helper, Y_no_transpose)
    # Helper must match the hand-rolled transpose/un-transpose reference.
    np.testing.assert_array_equal(Y_helper, Y_manual)


def test_sqformat_output_fmt_auto_adapts_bank_size_to_out_features():
    """R2 auto-adapt: when ``out_features < fmt.bank_size``, the helper must
    apply the SQ-Format with ``bank_size = min(orig, out_features)`` rather
    than silently padding the whole row into one bank.

    With ``bank_size=32`` and ``out_features=10, sparsity=0.5``:
      * Without auto-adapt: K_pad=32, one padded bank, ``k_high=round(0.5*32)=16``
        → all 10 real elements rank in top-16 → all get high-precision cell.
      * With auto-adapt to bank_size=10:   ``k_high=round(0.5*10)=5`` → only
        top 5 get high-precision; bottom 5 get low-precision.
    The two outputs therefore must not be equal, and the helper's output must
    match a format with bank_size pre-shrunk to 10.
    """
    from experiments.fourbit.pipeline import _apply_output_fmt
    from formats.sq_format import SQFormat

    fmt_big    = SQFormat(base="int", high_bits=4, low_bits=2, bank_size=32, sparsity=0.5)
    fmt_shrunk = SQFormat(base="int", high_bits=4, low_bits=2, bank_size=10, sparsity=0.5)

    rng = np.random.default_rng(0)
    Y = rng.standard_normal((8, 10)).astype(np.float32)

    Y_helper = _apply_output_fmt(fmt_big, Y)
    # Reference: apply the hand-shrunk format through the same transpose dance.
    Y_ref = fmt_shrunk.quantize(np.ascontiguousarray(Y.T)).T

    np.testing.assert_array_equal(Y_helper, Y_ref)
    # Sanity: auto-adapt actually changed behaviour vs. the padded path.
    Y_padded = fmt_big.quantize(np.ascontiguousarray(Y.T)).T
    assert not np.array_equal(Y_helper, Y_padded)
