# CLAUDE.md

Guidance for Claude Code when working in this repo. Keep it short; prefer linking to source over duplicating it.

## Purpose

Research codebase comparing low-bit (4/6/8) numeric formats for neural-network inference. Numerical primitives live under `formats/`, `distributions/`, and `profiler/`. Study scripts live under `experiments/`. Never change output numerics without updating the golden regression fixtures.

## Architecture (post-refactor, 2026-04-18)

```
formats/          # Canonical format primitives (one module = one family)
  _pot.py         #   POT scale helpers: pot_scale{,_vec} (floor) and pot_scale_ceil{,_vec}
  baseline.py     #   FP32 / BF16 / FP16
  mxint.py mxfp.py nvfp4.py nf4.py fp6.py
  sq_format.py    #   SQ-Format (ceil-variant POT)
  int_variants.py #   INT4-FP / APoT4 / LOG4 / NF4-FP8
  transforms/     #   hadamard.py, smoothquant.py

distributions/
  generators.py   # Scalar generators (gaussian, laplace, bimodal, outlier, …)
  linear_pairs.py # Paired (X, W) generators (weight_transformer / moe / attention / smooth_friendly_*)
  metrics.py      # mse, snr_db, qsnr_db (alias), fp16_qsnr_db, crest_factor, tensor_summary

profiler/         # PyTorch runtime profiler (forward-hook based, streaming stats)
  profiler.py     #   ModelProfiler
  stats.py        #   WelfordStats / RunningHistogram / QuantStats

utils/
  qsnr_table.py   # Builds results/qsnr_summary.html from exp1 CSVs

experiments/
  config.py defaults.py runner.py          # Shared ExperimentRunner
  exp1_common_distributions.py             # 9 formats × 24 distributions (4-bit + 8-bit)
  exp2_crest_factor.py                     # SQNR vs crest factor sweep
  robustness.py bitwidth_ablation.py
  fourbit/        # 4-bit study (Part 1 numpy + Part 2 MNIST transformer)
    cli.py        #   argparse entry (called by run_4bit_study.py shim)
    registry.py pipeline.py config.py
    part1.py part2.py accuracy.py profiler_v2.py reporter.py
    formats.py transforms.py distribution_sets.py

tests/
  test_regression.py             # Golden CLI regression (exp1/exp2/fourbit part1)
  test_pot_scale_equivalence.py  # Pins floor vs ceil POT semantics
  fixtures/golden/               # Committed reference CSVs — DO NOT edit casually
  test_*.py                      # Unit tests

run_all.py              # Master pipeline (Phases 1–5)
run_4bit_study.py       # Shim → experiments.fourbit.cli.main
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

# Master pipeline (robustness + ablation + figures)
python run_all.py --fast --skip-hw

# QSNR summary table
python generate_qsnr_table.py         # writes results/qsnr_summary.html

# Tests
pytest tests/ -q                      # full suite (347)
pytest tests/test_regression.py -v    # golden CLI regression (4)
```

CLI flags, output paths, and CSV schemas are the **interface boundary** — do not change them without an explicit interface-change task.

## Extension

- **New format** — add a module under `formats/` (follow `baseline.py`/`mxint.py` shape). Export a factory and register it in `formats/__init__.py::FORMAT_FACTORIES`. For INT-family formats needing POT scaling, import `pot_scale` / `pot_scale_vec` from `formats._pot`; for no-clip ceil semantics, use `pot_scale_ceil{,_vec}`. For the 4-bit study, also add an entry to `experiments/fourbit/config.py::DEFAULT_CONFIG.formats`.
- **New transform** — add a module under `formats/transforms/` implementing the Transform protocol. Register it in `experiments/fourbit/transforms.py::TRANSFORM_FACTORIES` if the 4-bit study should see it.
- **New distribution** — scalar: add to `distributions/generators.py` and list in `experiments/fourbit/distribution_sets.py::COMMON_DISTRIBUTIONS`. Paired (X, W): add to `distributions/linear_pairs.py` and list in `LINEAR_WEIGHT_ACTIVATION` / `SMOOTH_FRIENDLY`.
- **New metric** — add to `distributions/metrics.py`. Wire into `evaluate_all()` if it should flow into the standard report.
- **New experiment** — add under `experiments/`, reuse `experiments/runner.py::ExperimentRunner` where possible. If the script emits a CSV that needs regression protection, add a fixture under `tests/fixtures/golden/` and a test in `tests/test_regression.py`.

## Conventions

- **POT scale** — `formats/_pot.py` is the single source. `formats/__init__.py::_pot_scale` and `formats/sq_format.py::_pot_scale` are module-internal aliases (left in place intentionally; they are the modules' own private name for the canonical helpers).
- **Regression protection** — every main CLI writes a CSV that `tests/test_regression.py` compares byte-identically against `tests/fixtures/golden/`. If you intentionally change output numerics, regenerate the fixtures in the same commit.
- **Import paths** — import from the canonical modules (`from formats._pot import pot_scale`, `from distributions.metrics import qsnr_db`, `from experiments.fourbit.config import …`). Top-level `fourbit/` no longer exists.
- **No top-level scripts import each other** — `run_all.py`, `run_4bit_study.py`, `generate_qsnr_table.py` are entry points / shims only; real logic lives under the packages.

## When in doubt

- Run `pytest tests/test_regression.py -v` after any change that touches format, transform, metric, or distribution code. Byte-identical CSVs is the primary correctness signal.
- Run `pytest tests/test_pot_scale_equivalence.py -v` after touching anything in `formats/_pot.py` or the modules that alias it.
