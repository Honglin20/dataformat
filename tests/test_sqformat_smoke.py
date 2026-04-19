"""Smoke tests for the SQ-Format study package (PR E).

These tests do NOT run the full CLI — Part 1 with 16 cells × 3 transforms
× 24 distributions takes well over a minute and is not appropriate for
unit-test CI.  Instead they pin three cheap invariants:

  * ``FORMAT_FACTORIES`` contains the three SQ-Format entries and each
    cell declared in ``DEFAULT_CONFIG.formats`` instantiates.
  * ``DEFAULT_CONFIG`` has exactly 16 SQ-cells + 1 legacy hybrid = 17
    formats, the SQ-metrics set, and both ``quantize_output`` and
    ``use_quantizable_mha`` enabled.
  * ``build_pipelines(DEFAULT_CONFIG)`` returns the full Cartesian
    product and every pipeline has a non-None ``output_fmt``.

Byte-identical regression against committed golden CSVs is the job of
PR F (``tests/test_regression.py``).
"""
from __future__ import annotations


def test_sqformat_factories_registered():
    from experiments.fourbit.formats import FORMAT_FACTORIES

    for key in ("sqformat_alg1", "sqformat_alg2", "sqformat_fp_hybrid"):
        assert key in FORMAT_FACTORIES, f"missing factory: {key}"


def test_default_config_matrix_has_17_cells():
    from experiments.sqformat.config import DEFAULT_CONFIG

    assert len(DEFAULT_CONFIG.formats) == 17  # 2 × (5+3) + 1 hybrid
    assert DEFAULT_CONFIG.quantize_output is True
    assert DEFAULT_CONFIG.use_quantizable_mha is True
    assert DEFAULT_CONFIG.output_dir == "results/sqformat"
    assert DEFAULT_CONFIG.profile_samples == 128

    names = {m.name for m in DEFAULT_CONFIG.metrics}
    assert names == {"qsnr_db", "snr_db", "mse", "fp16_qsnr_db"}


def test_every_sqformat_cell_instantiates():
    from experiments.sqformat.config import DEFAULT_CONFIG
    from experiments.fourbit.formats import make_format

    for spec in DEFAULT_CONFIG.formats:
        fmt = make_format(spec.factory, **spec.kwargs)
        # Every SQ-Format instance exposes bank_size (live attribute)
        # and a ``quantize`` method that the pipeline uses.
        assert hasattr(fmt, "quantize")
        assert hasattr(fmt, "bank_size")


def test_build_pipelines_has_output_fmt_when_quantize_output_true():
    from experiments.sqformat.config import DEFAULT_CONFIG
    from experiments.fourbit.registry import build_pipelines

    pipelines = build_pipelines(DEFAULT_CONFIG)
    assert len(pipelines) == 17 * 3  # 17 formats × 3 transforms
    assert all(p.output_fmt is not None for p in pipelines)
