# CLAUDE.md

Guidance for Claude Code when working in this repo. Keep it short; prefer linking to source over duplicating it.

## Purpose

Research codebase comparing low-bit (4/6/8) numeric formats for neural-network inference. Numerical primitives live under `formats/`, `distributions/`, and `profiler/`. Study scripts live under `experiments/`. Never change output numerics without updating the golden regression fixtures.

## Architecture (post-refactor, 2026-04-19)

```
formats/          # Canonical format primitives (one module = one family)
  _pot.py         #   POT scale helpers: pot_scale{,_vec} (floor) and pot_scale_ceil{,_vec}
  baseline.py     #   FP32 / BF16 / FP16
  mxint.py mxfp.py nvfp4.py nf4.py fp6.py
  sq_format.py    #   SQFormat (Alg 1) / SQFormatActivations (Alg 2) / SQFormatFP (legacy hybrid);
                  #   element encoders keyed by (base, bits) ∈ {int,fp}×{2,4,8}
  int_variants.py #   INT4-FP / APoT4 / LOG4 / NF4-FP8
  transforms/     #   hadamard.py, smoothquant.py

distributions/
  generators.py   # Scalar generators (gaussian, laplace, bimodal, outlier, …)
  linear_pairs.py # Paired (X, W) generators (weight_transformer / moe / attention / smooth_friendly_*)
  metrics.py      # METRIC_REGISTRY (pair: qsnr_db, snr_db, mse, fp16_qsnr_db)
                  # TENSOR_STAT_REGISTRY (single: mean, std, min, max, crest, kurtosis, abs_max)
                  # register_metric(name, fn, kind=...) extends both at runtime

profiler/         # PyTorch runtime profiler (forward-hook based, streaming stats)
  profiler.py     #   ModelProfiler
  stats.py        #   WelfordStats / RunningHistogram / QuantStats

utils/
  qsnr_table.py   # Builds results/qsnr_summary.html from exp1 CSVs

examples/
  model.py                #   MNISTTransformer — accepts use_quantizable_mha flag
  model_quantizable.py    #   QuantizedMHA: drop-in for nn.MultiheadAttention whose q/k/v/out
                          #   route through Module.forward so QuantLinear can intercept them

experiments/
  config.py defaults.py runner.py          # Shared ExperimentRunner
  exp1_common_distributions.py             # 9 formats × 24 distributions (4-bit + 8-bit)
  exp2_crest_factor.py                     # SQNR vs crest factor sweep
  robustness.py bitwidth_ablation.py
  fourbit/        # 4-bit study (Part 1 numpy + Part 2 MNIST transformer)
    cli.py        #   argparse entry (called by run_4bit_study.py shim)
    config.py     #   FourBitConfig: formats, transforms, metrics (MetricSpec),
                  #   tensor_stats, quantize_output, use_quantizable_mha
    registry.py pipeline.py
    part1.py part2.py accuracy.py profiler_v2.py reporter.py
    formats.py transforms.py distribution_sets.py
  sqformat/       # SQ-Format study (reuses fourbit/ Part 1 + Part 2 via FourBitConfig)
    cli.py config.py   # 17-cell matrix (Alg1/Alg2 × INT 5 pairs + FP 3 pairs + legacy hybrid)

tests/
  test_regression.py             # Golden CLI regression (exp1/exp2/fourbit/sqformat part1 = 5)
  test_pot_scale_equivalence.py  # Pins floor vs ceil POT semantics
  test_sq_format_base_fp.py      # Element-encoder registry + base={int,fp} semantics
  test_pipeline_output_quant.py  # Y quantisation + R2 auto-adapt for SQ-Format
  test_quantizable_mha.py        # QuantizedMHA ≈ nn.MultiheadAttention in FP32
  test_metrics_registry.py test_sqformat_smoke.py
  fixtures/golden/               # Committed reference CSVs — DO NOT edit casually
  test_*.py                      # Unit tests

run_all.py              # Master pipeline (Phases 1–5)
run_4bit_study.py       # Shim → experiments.fourbit.cli.main
run_sqformat_study.py   # Shim → experiments.sqformat.cli.main
generate_qsnr_table.py  # Shim → utils.qsnr_table.main
```

## Usage

```bash
# Exp 1 (standalone, 4-bit + 8-bit)
python experiments/exp1_common_distributions.py

# Exp 2 (SQNR vs crest factor)
python experiments/exp2_crest_factor.py

# Full 4-bit study (Part 1 + Part 2)
python run_4bit_study.py              # all
python run_4bit_study.py --part 1     # numpy-only, no model
python run_4bit_study.py --part 2 --model-path results/mnist/model.pt

# SQ-Format study (17-cell matrix, reuses fourbit infra; Y-quant + QuantizedMHA on)
python run_sqformat_study.py --part 1
python run_sqformat_study.py --part 2 --model-path results/mnist/model.pt

# Master pipeline (robustness + ablation + figures)
python run_all.py --fast --skip-hw

# QSNR summary table
python generate_qsnr_table.py         # writes results/qsnr_summary.html

# Tests
pytest tests/ -q                      # full suite (374)
pytest tests/test_regression.py -v    # golden CLI regression (5)
```

CLI flags, output paths, and CSV schemas are the **interface boundary** — do not change them without an explicit interface-change task.

## Extension

- **New format** — add a module under `formats/` (follow `baseline.py`/`mxint.py` shape). Export a factory and register it in `formats/__init__.py::FORMAT_FACTORIES`. For INT-family formats needing POT scaling, import `pot_scale` / `pot_scale_vec` from `formats._pot`; for no-clip ceil semantics, use `pot_scale_ceil{,_vec}`. For the 4-bit study, also add an entry to `experiments/fourbit/config.py::DEFAULT_CONFIG.formats`.
- **New SQ-Format cell** — no code change: append a `FormatSpec("NAME", "sqformat_alg1|alg2", kwargs={"base": "int|fp", "high_bits": k, "low_bits": m, "bank_size": 128})` to `experiments/sqformat/config.py::DEFAULT_CONFIG.formats`. Element encoders are keyed by `(base, bits)` in `formats/sq_format.py`.
- **New transform** — add a module under `formats/transforms/` implementing the Transform protocol. Register it in `experiments/fourbit/transforms.py::TRANSFORM_FACTORIES` if the 4-bit study should see it.
- **New distribution** — scalar: add to `distributions/generators.py` and list in `experiments/fourbit/distribution_sets.py::COMMON_DISTRIBUTIONS`. Paired (X, W): add to `distributions/linear_pairs.py` and list in `LINEAR_WEIGHT_ACTIVATION` / `SMOOTH_FRIENDLY`.
- **New metric / tensor stat** — add the function to `distributions/metrics.py` and call `register_metric(name, fn, kind="pair"|"tensor_stat")`. Part-1 CSVs pick it up via `FourBitConfig.metrics` / `FourBitConfig.tensor_stats` — no part1.py edits needed.
- **New experiment** — add under `experiments/`, reuse `experiments/runner.py::ExperimentRunner` where possible. If the script emits a CSV that needs regression protection, add a fixture under `tests/fixtures/golden/` and a test in `tests/test_regression.py`.

## Conventions

- **POT scale** — `formats/_pot.py` is the single source. `formats/__init__.py::_pot_scale` and `formats/sq_format.py::_pot_scale` are module-internal aliases (left in place intentionally; they are the modules' own private name for the canonical helpers).
- **Regression protection** — every main CLI writes a CSV that `tests/test_regression.py` compares byte-identically against `tests/fixtures/golden/`. If you intentionally change output numerics, regenerate the fixtures in the same commit.
- **Import paths** — import from the canonical modules (`from formats._pot import pot_scale`, `from distributions.metrics import qsnr_db`, `from experiments.fourbit.config import …`). Top-level `fourbit/` no longer exists.
- **No top-level scripts import each other** — `run_all.py`, `run_4bit_study.py`, `generate_qsnr_table.py` are entry points / shims only; real logic lives under the packages.

## When in doubt

- Run `pytest tests/test_regression.py -v` after any change that touches format, transform, metric, or distribution code. Byte-identical CSVs is the primary correctness signal.
- Run `pytest tests/test_pot_scale_equivalence.py -v` after touching anything in `formats/_pot.py` or the modules that alias it.
