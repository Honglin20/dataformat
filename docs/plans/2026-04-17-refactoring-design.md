# Refactoring Design — Consolidation & Layering

**Date:** 2026-04-17
**Status:** Approved (design phase)
**Goal:** Remove redundancy and apply software-design principles (SRP, DRY) across
the codebase while preserving behavior and the externally-observable interface.

---

## 1. Context & Goals

The repo is a ~17k-line quantization-format research codebase with ~60 Python
files. It grew organically, producing three recurring problems:

1. **Duplicated primitives.** `_pot_scale` is reimplemented in at least three
   files; distribution generators, metrics, and profilers exist in both the
   top-level packages and under `fourbit/` (which declared itself
   "self-contained").
2. **Mixed layers.** Study-specific code (`fourbit/`) sits at the same tree
   level as generic primitives (`formats/`, `distributions/`, `profiler/`),
   and post-processing utilities (`generate_qsnr_table.py`) sit at the same
   level as CLI entry points (`run_all.py`).
3. **Inconsistent configuration.** Three config modules (`config.py`,
   `experiments/config.py`, `fourbit/config.py`) with overlapping concerns.

Tests (320) pass and cover ~2.5k lines; this is the regression anchor.

---

## 2. Constraints

Agreed with the user during brainstorming:

| Dimension | Choice |
|---|---|
| External-interface boundary | **A** — keep CLI commands, output file paths, CSV schemas and figure filenames. Internal Python structure is free. |
| Dead-code handling | **A (aggressive)** — delete files unreferenced after reachability analysis; merge duplicated live code into a single canonical implementation. |
| Delivery cadence | **B (phased multi-PR)** — seven independent PRs, each mergeable and revertable in isolation. |
| Regression oracle | **C (light golden baseline)** — snapshot CSV output of the most-touched CLI entry points, run regression on every PR; `run_all.py` stays on manual smoke. |

---

## 3. Target Directory Structure

Categorization rule:

| Layer | What | Examples |
|---|---|---|
| CLI entry (top-level scripts) | argparse + dispatch only | `run_all.py`, `run_4bit_study.py`, `generate_qsnr_table.py` |
| Core primitives | reusable, study-agnostic | `formats/`, `distributions/`, `profiler/`, `hardware/`, `visualization/` |
| Experiments | study-specific orchestration | `experiments/` (including `experiments/fourbit/`) |
| Utils | post-processing / reports | `utils/qsnr_table.py` |
| Apps / examples | end-to-end demonstrations | `examples/` |
| Global config | constants | `config.py` |

Target tree after refactor:

```
dataformat/
├── config.py                           # global constants (single source)
├── requirements.txt  README.md  RESEARCH_PLAN.md  analysis.md
│
├── run_all.py                          # CLI (unchanged)
├── run_4bit_study.py                   # CLI (unchanged) — thin shim
├── generate_qsnr_table.py              # CLI (unchanged) — thin shim
│
├── formats/                            # core primitives: format impls
│   ├── __init__.py    _pot.py          # ← single pot_scale / pot_scale_vec
│   ├── baseline.py    int_pot.py
│   ├── mxint.py  mxfp.py  nvfp4.py  nf4.py  fp6.py
│   ├── int_variants.py                 # ← APoT4 / LOG4 / NF4_FP8 / INT4_FP
│   ├── sq_format.py
│   └── transforms/
│       ├── hadamard.py  random_rotation.py  smoothquant.py
│       └── identity.py                 # ← extracted from fourbit/transforms.py
│
├── distributions/                      # core primitives: generators & metrics
│   ├── generators.py
│   ├── linear_pairs.py                 # ← _weight_transformer / _smooth_friendly_*
│   └── metrics.py                      # ← SQNR / MSE / KL / MaxAE (single)
│
├── profiler/                           # core primitives: model profiler
│   └── profiler.py  stats.py  formats.py  export.py
│
├── hardware/                           # unchanged subsystem
├── visualization/                      # unchanged subsystem
│
├── experiments/                        # experiment orchestration
│   ├── __init__.py
│   ├── config.py  defaults.py  runner.py
│   ├── robustness.py  bitwidth_ablation.py
│   ├── exp1_common_distributions.py
│   ├── exp2_crest_factor.py
│   └── fourbit/                        # ← moved from top-level fourbit/
│       ├── __init__.py                 # public API: Pipeline, build_formats, …
│       ├── config.py                   # FourBitConfig, DEFAULT_CONFIG
│       ├── registry.py    pipeline.py
│       ├── part1.py  part2.py  accuracy.py
│       ├── reporter.py
│       └── distribution_sets.py        # ← DistSpec / LinearSpec / curated lists
│
├── utils/                              # post-processing
│   ├── __init__.py
│   └── qsnr_table.py                   # ← content of generate_qsnr_table.py
│
├── examples/                           # unchanged
│
└── tests/
    ├── (existing 320 tests, import paths adjusted)
    ├── fixtures/golden/                # ← CSV snapshots
    └── test_regression.py              # ← CLI regression harness
```

**Externally unchanged:** every CLI command and flag, every output path under
`results/`, every CSV column schema, every figure filename.

---

## 4. Phased PR Plan

Seven PRs. Each must pass `pytest tests/ -q` (≥ 320 tests) and
`pytest tests/test_regression.py` before merging. Each merges independently to
`main` (branch-protected, 1 approval).

| # | PR | Purpose | Risk |
|---|---|---|---|
| 0 | **Golden regression baseline** | snapshot CLI CSVs under `tests/fixtures/golden/`, add `tests/test_regression.py`; no code change | very low |
| 1 | **`utils/` extraction** | move `generate_qsnr_table.py` logic to `utils/qsnr_table.py`; top-level becomes shim | very low |
| 2 | **Consolidate distributions & metrics** | merge `fourbit/distributions.py` (unique parts) → `distributions/linear_pairs.py`; merge `fourbit/metrics.py` → `distributions/metrics.py` | low |
| 3 | **Consolidate formats & transforms** | single `formats/_pot.py`; move APoT4/LOG4/NF4_FP8/INT4_FP to `formats/int_variants.py`; extract identity transform; delete `fourbit/formats.py`, `fourbit/transforms.py` | medium (most touch points) |
| 4 | **Consolidate profiler** | analyze `fourbit/profiler_v2.py`; merge unique logic into `profiler/` or delete | medium |
| 5 | **Relocate `fourbit/` → `experiments/fourbit/`** | mechanical file move + import updates; `run_4bit_study.py` becomes shim | medium-low |
| 6 | **Final cleanup** | naming consistency, dead-code sweep, README tree update | low |

#### PR 0 — golden baseline details

Five CLI entries are snapshotted with a pinned seed and reduced sample sizes so
fixtures stay small and CI can run them:

| Entry | Command | Snapshot |
|---|---|---|
| exp1 | `python experiments/exp1_common_distributions.py --bits 4 --bits 8` | `exp1/results_{4,8}bit.csv` |
| exp2 | `python experiments/exp2_crest_factor.py` (fast) | `exp2/results_{4,8}bit.csv` |
| fourbit Part 1 | `python run_4bit_study.py --part 1` | `fourbit/part1/exp1{1,2,3}_*.csv` |
| qsnr table | `python generate_qsnr_table.py` | `qsnr_summary.html` (asserted on key cells) |
| mnist profile | `python examples/profile_mnist.py --n-samples 64` | `mnist/profiler_results.csv` |

Comparison: `rtol=1e-6, atol=1e-8` for numeric columns; exact equality for
text; strict column name and order check.

`run_all.py` is **not** auto-regressed: I will manually run
`python run_all.py --fast --skip-hw` before each structural PR and record the
stdout summary in the PR description.

#### Definition of Done per PR

1. `pytest tests/ -q` green
2. `pytest tests/test_regression.py` green
3. `--help` output identical to pre-PR for every affected CLI
4. PR description lists removed files, added files, and manual smoke result
5. One code-review approval (`superpowers:requesting-code-review`)

---

## 5. Module-Level Symbol Migration

| Source | Destination | Notes |
|---|---|---|
| `fourbit/metrics.py` | `distributions/metrics.py` | equivalence check required before merge |
| `fourbit/distributions.py` · `DistSpec`, `LinearSpec`, curated lists | `experiments/fourbit/distribution_sets.py` | study-configuration scope |
| `fourbit/distributions.py` · `_weight_*`, `_smooth_friendly_*` | `distributions/linear_pairs.py` (new) | generic primitive |
| `fourbit/formats.py::_pot_scale_for_qmax` (vectorized) | `formats/_pot.py::pot_scale_vec` (new) | canonical vectorized pot_scale |
| `formats/__init__.py::_pot_scale` (scalar) | `formats/_pot.py::pot_scale` (new) | canonical scalar pot_scale |
| `formats/sq_format.py::_pot_scale` | `formats/_pot.py::pot_scale` | import changes |
| `fourbit/formats.py` · INT4/FP4/NF4/NVFP4/MXINT4/MXFP4 wrappers | `formats/int_variants.py` (new) | incl. APoT4, LOG4, NF4_FP8, INT4_FP |
| `fourbit/transforms.py` · identity wrapper | `formats/transforms/identity.py` (new) | smoothquant/hadamard already canonical |
| `fourbit/profiler_v2.py` | phase 4 decision — merge into `profiler/` or delete | reachability analysis first |
| `fourbit/config.py` | `experiments/fourbit/config.py` | path only |
| `fourbit/{pipeline,registry,part1,part2,accuracy,reporter}.py` | `experiments/fourbit/` | path + import updates |
| `generate_qsnr_table.py` (logic) | `utils/qsnr_table.py::main()` | top-level becomes 5-line shim |
| `run_4bit_study.py` (logic) | `experiments/fourbit/cli.py` (or `__main__.py`) | top-level becomes shim |

### Known gotchas

1. `distributions.generators.student_t_dist` calls `np.random.seed(seed)`, a
   global side effect. Not modified during this refactor (would change
   observable numerics, violating constraint A). Attach a `# TODO(post-refactor)`
   comment for a future cleanup.
2. `_pot_scale` exists as both a scalar and a vectorized function. The
   consolidated module exposes both `pot_scale(absmax: float)` and
   `pot_scale_vec(absmax: np.ndarray)`.
3. `fourbit/metrics.py` and `distributions/metrics.py` may differ subtly
   (e.g. KL-approximation choice). PR 2 must add an equivalence test
   (`np.allclose`) before merging; any drift must be resolved by picking one
   implementation and documenting the decision.
4. Roughly ten `from formats import _pot_scale` imports across `tests/` and
   `formats/` will need updating to `from formats._pot import pot_scale`.
5. The shim `run_4bit_study.py` must forward every CLI flag (`--part`,
   `--out`, `--model-path`, `--data-dir`) exactly; golden regression on
   Part 1 output catches any drift.

---

## 6. Risks & Rollback

| # | Risk | Mitigation |
|---|---|---|
| R1 | Floating-point drift breaks golden comparison across machines | use `rtol=1e-6, atol=1e-8`; confirm three runs stable before committing fixtures |
| R2 | Duplicate implementations are not actually equivalent | equivalence `pytest` before each merge |
| R3 | Circular imports after restructure | shared primitive modules (`_pot.py`, `metrics.py`) depend only on `numpy`/`scipy`; no same-layer imports |
| R4 | External consumers import `fourbit.*` | excluded by constraint A; state the rename in the PR description anyway |
| R5 | `run_all.py` bug slips through (no auto-regression) | run `python run_all.py --fast --skip-hw` manually before each structural PR |
| R6 | PR 3 diff is large, hard to review | pre-enumerate every consumer of `_pot_scale` / `_POTINTQuantizer` and attach the list to the PR |
| R7 | Golden fixtures bloat git history | small N, short distribution lists; each CSV under ~20 KB |
| R8 | `results/` run can overwrite goldens | goldens live under `tests/fixtures/golden/`, isolated from `results/` |

### Rollback

- Per-PR revertable: each PR is based on the post-merge `main`, so
  `git revert <merge-sha>` is safe.
- Emergency: a whole PR may be reverted rather than forward-fixed when a
  regression appears after merge.
- Branch protection allows revert via PR (1 approval, no force-push needed).
- Intentional output change: if a later PR deliberately changes a CSV (e.g.
  bug fix), fixtures are regenerated only via a documented `--update-golden`
  flow with the reason recorded in the PR description.

### Out of scope (YAGNI)

- No new abstractions (no `QuantFormat` ABC unless already in use)
- No algorithm rewrites
- No new third-party dependencies
- No type-annotation overhaul
- No restructuring inside `hardware/`, `examples/`, `visualization/` beyond
  import-path fixes
- No CLI argument additions, removals, or renames
- No docstring rewrites unless content is stale

---

## 7. Completion Criteria

1. All seven PRs merged into `main` (each with one approval)
2. `pytest tests/ -q` green (≥ 320 tests); `pytest tests/test_regression.py` green
3. In a clean venv, these all run to completion with unchanged output paths:
   - `python run_all.py --fast --skip-hw`
   - `python run_4bit_study.py --part 1`
   - `python experiments/exp1_common_distributions.py --bits 4`
   - `python experiments/exp2_crest_factor.py`
   - `python generate_qsnr_table.py`
   - `python examples/profile_mnist.py --n-samples 64`
4. Every CLI flag and every `--help` text matches pre-refactor
5. Every output CSV column schema and file path matches pre-refactor
6. Numeric output within `rtol=1e-6, atol=1e-8` of pre-refactor
7. `README.md` "Project Structure" section updated to match the new tree

### Quantitative targets (informational, not blocking)

- `_pot_scale` implementations: 3+ → 1
- Distribution generators: 2 → 1
- Python file count: ~60 → ~52
- Python line count: ~17k → ~15k (excluding golden fixtures)

### Deliverables

| Type | Artifact | Location |
|---|---|---|
| Design doc | This file | `docs/plans/2026-04-17-refactoring-design.md` |
| Implementation plan | Seven-PR step list | `docs/plans/2026-04-17-refactoring.md` (from `writing-plans`) |
| Golden baseline | CSV/HTML snapshots + harness | `tests/fixtures/golden/`, `tests/test_regression.py` |
| Code changes | Seven PRs | main branch history |
| Updated README | Revised structure tree | `README.md` |
