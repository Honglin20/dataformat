"""SQ-Format study configuration: 16 cells × 3 transforms × metrics set.

Matrix
------
  * Algorithm 1 (:class:`formats.sq_format.SQFormat`) × INT × {8&8, 4&8, 4&4, 4&2, 2&2}
  * Algorithm 1                                       × FP  × {8&8, 4&8, 4&4}
  * Algorithm 2 (:class:`formats.sq_format.SQFormatActivations`) × INT × same 5 pairs
  * Algorithm 2                                                   × FP  × same 3 pairs
  * Legacy ``SQFormatFP`` (FP8 high / INT low) — kept as a single reference cell.

Metrics default to the SQ-Format set (``qsnr_db`` + ``snr_db`` + ``mse`` +
``fp16_qsnr_db``) and ``quantize_output=True`` / ``use_quantizable_mha=True``
are set so every GEMM's W, A and Y and every attention Linear are
quantised as required by the study's core rule.
"""
from __future__ import annotations

from experiments.fourbit.config import (
    FourBitConfig,
    FormatSpec,
    TransformSpec,
    MetricSpec,
)


def _cell(alg: str, base: str, hi: int, lo: int) -> FormatSpec:
    algo_factory = {"alg1": "sqformat_alg1", "alg2": "sqformat_alg2"}[alg]
    name = f"SQ-{base.upper()}-{hi}{lo}-{alg.upper()}"
    return FormatSpec(
        name,
        algo_factory,
        kwargs={"base": base, "high_bits": hi, "low_bits": lo},
    )


INT_PAIRS = [(8, 8), (4, 8), (4, 4), (4, 2), (2, 2)]
FP_PAIRS  = [(8, 8), (4, 8), (4, 4)]

SQFORMAT_CELLS = (
    [_cell("alg1", "int", hi, lo) for hi, lo in INT_PAIRS] +
    [_cell("alg1", "fp",  hi, lo) for hi, lo in FP_PAIRS]  +
    [_cell("alg2", "int", hi, lo) for hi, lo in INT_PAIRS] +
    [_cell("alg2", "fp",  hi, lo) for hi, lo in FP_PAIRS]  +
    # Legacy FP8-high / INT-low hybrid — independent reference cell.
    [FormatSpec("SQ-FP8-INT4-hybrid", "sqformat_fp_hybrid", kwargs={"low_bits": 4})]
)


SQ_METRICS = [
    MetricSpec("qsnr_db",      "qsnr_db"),
    MetricSpec("snr_db",       "snr_db"),
    MetricSpec("mse",          "mse"),
    MetricSpec("fp16_qsnr_db", "fp16_qsnr_db"),
]


DEFAULT_CONFIG = FourBitConfig(
    formats=SQFORMAT_CELLS,
    transforms=[
        TransformSpec("base",   "identity"),
        TransformSpec("smooth", "smoothquant", kwargs={"alpha": 0.5}),
        TransformSpec("had",    "hadamard"),
    ],
    metrics=SQ_METRICS,
    quantize_output=True,
    use_quantizable_mha=True,
    output_dir="results/sqformat",
    profile_samples=128,  # R3: default to 128 to keep Part 2 run-time bounded.
)
