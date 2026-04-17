# Refactoring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Consolidate 17k-line codebase by removing duplicated primitives
(`_pot_scale` ×3, distribution generators ×2, metrics ×2, profilers ×2,
transforms, configs) and relayer the tree (CLI entries / core primitives /
experiments / utils), with zero change to CLI behavior, output paths, CSV
schemas, figure filenames, or numeric outputs within
`rtol=1e-6, atol=1e-8`.

**Architecture:** Seven sequential PRs against `main`. PR 0 establishes a
golden regression harness (no code change). PRs 1–6 each remove one axis of
duplication or relocate one subtree. Each PR is independently revertable
because every PR branches from merged `main`.

**Tech Stack:** Python 3, numpy, scipy, pytorch, pytest, pandas, numpy.allclose
for numeric comparison, `gh` CLI for PRs.

**Reference:** See `docs/plans/2026-04-17-refactoring-design.md` for design
rationale, constraints, and quantitative targets.

---

## Ground Rules (apply to every PR)

1. **Always branch from merged `main`.** Never stack PRs.
2. **Before starting work**, run `git fetch origin && git checkout main && git pull` and `pytest tests/ -q` (must be green).
3. **Small commits** within a PR are OK; keep commit messages imperative and scoped (`refactor:`, `test:`, `chore:`, `docs:`).
4. **Never skip hooks** (`--no-verify` forbidden) and never force-push.
5. **`superpowers:verification-before-completion`** is required before claiming any task complete. Concretely: run the listed command, read the output, confirm it matches "Expected" text, and only then mark done.
6. **If a test fails unexpectedly**, STOP, use `superpowers:systematic-debugging`, do not patch around it.
7. **If the equivalence check in PR 2/3/4 shows the two implementations are NOT equivalent**, STOP, surface the divergence to the user, and let the user decide which implementation to keep. Do not silently pick one.
8. **PR creation** uses this title format: `refactor(pr-N): <short>` (e.g. `refactor(pr-1): extract utils/qsnr_table`). Body must include:
   - Summary of changes
   - Files removed / added / modified counts
   - `pytest` + regression result ("320 passed, regression 5 passed")
   - For structural PRs: manual `run_all.py --fast --skip-hw` summary
9. **After each PR merges to main**, the next PR's first task is to rebase off the new main and re-verify pytest green before starting new work.

---

## Pre-flight: Environment Check

### Task P.1: Verify environment

**Step 1:** Run: `python -c "import numpy, scipy, torch, pandas, pytest; print('ok')"`
Expected: `ok`

**Step 2:** Run: `pytest tests/ -q`
Expected: `320 passed` (time ~12s)

**Step 3:** Run: `git status` — must be clean, on `main`, up-to-date with `origin/main`.

**Step 4:** Confirm branch protection on `main`: `gh api repos/Honglin20/dataformat/branches/main/protection --jq '.allow_deletions.enabled'`
Expected: `false`

If any step fails, stop and report to user.

---

# PR 0 — Golden Regression Baseline

**Goal:** Add `tests/fixtures/golden/` snapshots and `tests/test_regression.py` so every subsequent PR has an automatic oracle for CLI output. No production code changes.

**Branch:** `refactor/pr0-golden-baseline`

### Task 0.1: Branch off main

**Step 1:** `git checkout main && git pull origin main`
**Step 2:** `git checkout -b refactor/pr0-golden-baseline`
**Step 3:** `pytest tests/ -q` → `320 passed`

### Task 0.2: Create directories

**Step 1:** Create: `tests/fixtures/golden/` (with `.gitkeep` inside)
**Step 2:** Create: `tests/fixtures/golden/exp1/`, `tests/fixtures/golden/exp2/`, `tests/fixtures/golden/fourbit/part1/`, `tests/fixtures/golden/mnist/`
**Step 3:** `git add tests/fixtures/golden/` (only .gitkeep files yet)

### Task 0.3: Generate exp1 golden fixtures

**Step 1:** Record the current results path:
```bash
ls results/exp1/results_4bit.csv results/exp1/results_8bit.csv
```
Expected: both exist (they are in the repo).

**Step 2:** Re-run with the same defaults to confirm determinism:
```bash
python experiments/exp1_common_distributions.py --bits 4 --bits 8
```
Expected: exits 0 in ≤ 2 min. CSVs at `results/exp1/results_{4,8}bit.csv`.

**Step 3:** Copy to golden:
```bash
cp results/exp1/results_4bit.csv tests/fixtures/golden/exp1/
cp results/exp1/results_8bit.csv tests/fixtures/golden/exp1/
```

**Step 4:** If file size > 30 KB, reduce by editing the script's default `N_SAMPLES` / distribution list down to a fast-CI subset. For this plan we accept current sizes; revisit in Task 0.9 if CI is too slow.

### Task 0.4: Generate exp2 golden fixtures

```bash
python experiments/exp2_crest_factor.py
cp results/exp2/results_4bit.csv tests/fixtures/golden/exp2/
cp results/exp2/results_8bit.csv tests/fixtures/golden/exp2/
```
Expected: both CSVs exist.

### Task 0.5: Generate fourbit Part 1 golden fixtures

```bash
python run_4bit_study.py --part 1
cp results/fourbit/part1/exp11_direct_quant.csv    tests/fixtures/golden/fourbit/part1/
cp results/fourbit/part1/exp12_linear_wa.csv       tests/fixtures/golden/fourbit/part1/
cp results/fourbit/part1/exp13_smooth_transforms.csv tests/fixtures/golden/fourbit/part1/
```

### Task 0.6: Generate mnist profile golden fixture

**Step 1:** Check model exists: `ls results/mnist/model.pt`
If missing, run: `python examples/train_mnist.py --epochs 1` (1-epoch is enough for a determinism check) and record this in the fixture README.

**Step 2:**
```bash
python examples/profile_mnist.py --n-samples 64
cp results/mnist/profiler_results.csv tests/fixtures/golden/mnist/
```

### Task 0.7: Generate qsnr_summary.html golden fixture

```bash
python generate_qsnr_table.py
# Copy just header + known rows (file is ~700 lines). For regression we
# extract specific format × distribution cells rather than byte-comparing the
# entire HTML.
cp results/qsnr_summary.html tests/fixtures/golden/qsnr_summary.html
```

### Task 0.8: Write the regression test

**Files:**
- Create: `tests/test_regression.py`

**Step 1:** Write the harness:

```python
"""Regression tests: re-run key CLI entries, compare CSV output to golden fixtures.

Fixtures live under tests/fixtures/golden/. To refresh a fixture after an
intentional behavior change, delete the target file and re-run the generating
CLI (see docs/plans/2026-04-17-refactoring.md PR 0 for commands).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).parent.parent
GOLDEN    = REPO_ROOT / "tests" / "fixtures" / "golden"
RESULTS   = REPO_ROOT / "results"

RTOL = 1e-6
ATOL = 1e-8


def _run(cmd: list[str]) -> None:
    """Run a CLI command from REPO_ROOT; fail the test on non-zero exit."""
    result = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True,
        timeout=600,
    )
    if result.returncode != 0:
        pytest.fail(
            f"Command failed: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def _compare_csv(got: Path, expected: Path) -> None:
    """Compare two CSVs: identical columns, numeric cells within tolerance,
    text cells exactly equal."""
    assert got.exists(),       f"missing output: {got}"
    assert expected.exists(),  f"missing golden: {expected}"

    df_got = pd.read_csv(got)
    df_exp = pd.read_csv(expected)

    assert list(df_got.columns) == list(df_exp.columns), (
        f"columns changed\n got: {list(df_got.columns)}\n exp: {list(df_exp.columns)}"
    )
    assert len(df_got) == len(df_exp), (
        f"row count changed: got {len(df_got)}, expected {len(df_exp)}"
    )

    for col in df_got.columns:
        if pd.api.types.is_numeric_dtype(df_got[col]):
            g = df_got[col].to_numpy(dtype=np.float64)
            e = df_exp[col].to_numpy(dtype=np.float64)
            # Treat inf/nan positionally identical.
            inf_g, inf_e = np.isinf(g), np.isinf(e)
            nan_g, nan_e = np.isnan(g), np.isnan(e)
            assert np.array_equal(inf_g, inf_e), f"inf pattern changed in {col}"
            assert np.array_equal(nan_g, nan_e), f"nan pattern changed in {col}"
            mask = ~(inf_g | nan_g)
            if not np.allclose(g[mask], e[mask], rtol=RTOL, atol=ATOL):
                diffs = np.abs(g[mask] - e[mask])
                pytest.fail(
                    f"numeric column '{col}' exceeds tol: "
                    f"max abs diff = {diffs.max():g}"
                )
        else:
            assert (df_got[col].astype(str) == df_exp[col].astype(str)).all(), (
                f"text column '{col}' changed"
            )


# ── exp1 ─────────────────────────────────────────────────────────────────────

def test_exp1_4bit(tmp_path):
    _run([sys.executable, "experiments/exp1_common_distributions.py", "--bits", "4"])
    _compare_csv(RESULTS / "exp1" / "results_4bit.csv",
                 GOLDEN  / "exp1" / "results_4bit.csv")


def test_exp1_8bit(tmp_path):
    _run([sys.executable, "experiments/exp1_common_distributions.py", "--bits", "8"])
    _compare_csv(RESULTS / "exp1" / "results_8bit.csv",
                 GOLDEN  / "exp1" / "results_8bit.csv")


# ── exp2 ─────────────────────────────────────────────────────────────────────

def test_exp2(tmp_path):
    _run([sys.executable, "experiments/exp2_crest_factor.py"])
    for bits in (4, 8):
        _compare_csv(
            RESULTS / "exp2" / f"results_{bits}bit.csv",
            GOLDEN  / "exp2" / f"results_{bits}bit.csv",
        )


# ── fourbit Part 1 ───────────────────────────────────────────────────────────

def test_fourbit_part1(tmp_path):
    _run([sys.executable, "run_4bit_study.py", "--part", "1"])
    for name in ("exp11_direct_quant.csv",
                 "exp12_linear_wa.csv",
                 "exp13_smooth_transforms.csv"):
        _compare_csv(
            RESULTS / "fourbit" / "part1" / name,
            GOLDEN  / "fourbit" / "part1" / name,
        )


# ── mnist profile ────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not (REPO_ROOT / "results" / "mnist" / "model.pt").exists(),
    reason="needs trained MNIST model (run examples/train_mnist.py first)",
)
def test_mnist_profile(tmp_path):
    _run([sys.executable, "examples/profile_mnist.py", "--n-samples", "64"])
    _compare_csv(
        RESULTS / "mnist" / "profiler_results.csv",
        GOLDEN  / "mnist" / "profiler_results.csv",
    )
```

**Step 2:** Run regression once: `pytest tests/test_regression.py -v`
Expected: 5 passed (or 4 passed + 1 skipped if no model).

**Step 3:** Run regression TWICE more to confirm stability:
```bash
pytest tests/test_regression.py -v && pytest tests/test_regression.py -v
```
Expected: same results both times; no flakiness.

### Task 0.9: Size check

```bash
du -sh tests/fixtures/golden/
find tests/fixtures/golden/ -name '*.csv' -exec wc -l {} +
```
Expected: total < 500 KB; each CSV < 250 lines typically.

If > 500 KB total: flag to user, ask whether to reduce fixture coverage or sample sizes. Do not proceed until resolved.

### Task 0.10: Commit + push + PR

**Step 1:**
```bash
git add tests/fixtures/golden/ tests/test_regression.py
git commit -m "test: add golden CLI regression harness for PR 0

Snapshots CSV output of exp1, exp2, fourbit Part 1, and mnist profile CLI
entries. tests/test_regression.py re-runs each, compares columns exactly and
numeric cells at rtol=1e-6, atol=1e-8. Protects subsequent refactor PRs
against behavior drift.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

**Step 2:** Confirm full suite:
```bash
pytest tests/ -q
```
Expected: ≥ 325 passed (320 + 4 or 5 regression).

**Step 3:**
```bash
git push -u origin refactor/pr0-golden-baseline
gh pr create --title "refactor(pr-0): golden CLI regression harness" \
  --body "$(cat <<'EOF'
## Summary
- Add `tests/fixtures/golden/` snapshots for exp1, exp2, fourbit Part 1, mnist profile
- Add `tests/test_regression.py` comparing CSV output at `rtol=1e-6, atol=1e-8`
- Foundation for PR 1–6 refactoring series (see `docs/plans/2026-04-17-refactoring-design.md`)

## No production code change
Only files in `tests/fixtures/golden/` and `tests/test_regression.py`.

## Test plan
- [x] `pytest tests/ -q` → 325 passed (320 + 5 regression, or 324 + 1 skipped)
- [x] `pytest tests/test_regression.py` → 5 passed (stable across 3 runs)
- [x] Fixture total size < 500 KB

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

**Step 4:** Wait for user to approve PR. On approval, user merges; then proceed to PR 1.

---

# PR 1 — Extract `utils/qsnr_table.py`

**Goal:** Move `generate_qsnr_table.py` logic into `utils/qsnr_table.py`; top-level becomes a 5-line shim. No behavior change.

**Branch:** `refactor/pr1-utils-extraction`

### Task 1.1: Branch off merged main

```bash
git checkout main && git pull origin main
git checkout -b refactor/pr1-utils-extraction
pytest tests/ -q                 # 325+ passed
pytest tests/test_regression.py  # 5 passed
```

### Task 1.2: Create `utils/` package

**Files:**
- Create: `utils/__init__.py` — single line: `"""Post-processing utilities for experiment outputs."""`

### Task 1.3: Move logic to `utils/qsnr_table.py`

**Files:**
- Create: `utils/qsnr_table.py`
- Modify: `generate_qsnr_table.py` → shim

**Step 1:** Read current `generate_qsnr_table.py` (294 lines). Copy its entire content verbatim to `utils/qsnr_table.py`. Wrap the top-level code (argparse + main flow) in a function `main(argv=None) -> int` that accepts `argv` list.

**Step 2:** Rewrite `generate_qsnr_table.py` as:
```python
"""Shim: keeps `python generate_qsnr_table.py` working.

Actual logic lives in utils/qsnr_table.py.
"""
from __future__ import annotations
import sys
from utils.qsnr_table import main

if __name__ == "__main__":
    sys.exit(main())
```

**Step 3:** Verify nothing else imports from `generate_qsnr_table`:
```bash
grep -rn "from generate_qsnr_table\|import generate_qsnr_table" --include="*.py"
```
Expected: no matches. If any match exists, update it to `from utils.qsnr_table import …`.

### Task 1.4: Verify

**Step 1:** `python generate_qsnr_table.py --help`
Expected: identical `--help` output to pre-PR (stash pre-PR output in `/tmp/help-before.txt` before Task 1.3 for comparison).

**Step 2:** `python generate_qsnr_table.py`
Expected: exits 0, regenerates `results/qsnr_summary.html`.

**Step 3:** `pytest tests/ -q` → all green (325+).

**Step 4:** `pytest tests/test_regression.py -v` → 5 passed (qsnr_table cell assertions still valid if any; CSV-based tests unchanged).

### Task 1.5: Commit + push + PR

```bash
git add utils/ generate_qsnr_table.py
git commit -m "refactor(pr-1): extract qsnr_table logic to utils/

generate_qsnr_table.py becomes a 5-line shim delegating to
utils.qsnr_table.main(). CLI command, arguments, and output unchanged.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
git push -u origin refactor/pr1-utils-extraction
gh pr create --title "refactor(pr-1): extract utils/qsnr_table" --body ...
```

Wait for merge.

---

# PR 2 — Consolidate Distributions & Metrics

**Goal:**
- Move unique generators from `fourbit/distributions.py` into `distributions/linear_pairs.py`.
- Merge `fourbit/metrics.py` unique functions into `distributions/metrics.py`.
- Move `DistSpec` / `LinearSpec` / curated lists to `fourbit/distribution_sets.py` (still inside `fourbit/` — will move to `experiments/fourbit/` in PR 5).
- Delete `fourbit/distributions.py` and `fourbit/metrics.py`.

**Branch:** `refactor/pr2-distributions-metrics`

### Task 2.1: Branch + pre-check

```bash
git checkout main && git pull origin main
git checkout -b refactor/pr2-distributions-metrics
pytest tests/ -q                 # all green
pytest tests/test_regression.py  # all green
```

### Task 2.2: Equivalence test (FIRST — before any deletion)

**Files:**
- Create: `tests/test_metrics_equivalence.py`

**Step 1:** Write equivalence tests that both implementations of `mse` and `snr_db`/`qsnr_db` produce identical output for 5 fixed-seed tensors (Gaussian, Laplace, bimodal, outlier, zero-tensor):

```python
import numpy as np
import pytest

from distributions.metrics import mse as mse_top, snr_db as snr_top
from fourbit.metrics     import mse as mse_fb,  qsnr_db as qsnr_fb

RNG_SEEDS = [0, 1, 42, 100, 2025]

@pytest.mark.parametrize("seed", RNG_SEEDS)
def test_mse_agrees(seed):
    rng = np.random.default_rng(seed)
    x  = rng.normal(0, 1, size=4096).astype(np.float32)
    xq = (x + rng.normal(0, 0.01, size=x.shape)).astype(np.float32)
    a = mse_top(x, xq)
    b = mse_fb(x, xq)
    assert np.isclose(a, b, rtol=1e-12, atol=1e-15), f"{a} vs {b}"

@pytest.mark.parametrize("seed", RNG_SEEDS)
def test_snr_agrees(seed):
    rng = np.random.default_rng(seed)
    x  = rng.normal(0, 1, size=4096).astype(np.float32)
    xq = (x + rng.normal(0, 0.01, size=x.shape)).astype(np.float32)
    a = snr_top(x, xq)
    b = qsnr_fb(x, xq)
    assert np.isclose(a, b, rtol=1e-12, atol=1e-15), f"{a} vs {b}"
```

**Step 2:** Run: `pytest tests/test_metrics_equivalence.py -v`
Expected: 10 passed.

**If any fails**: STOP. Report the divergence to the user. Do not proceed.

### Task 2.3: Consolidate metrics in `distributions/metrics.py`

**Files:**
- Modify: `distributions/metrics.py` — add `fp16_quantize`, `fp16_qsnr_db`, `crest_factor`, `tensor_summary` from `fourbit/metrics.py`
- Keep both `snr_db` and add alias `qsnr_db = snr_db` for drop-in replacement

**Step 1:** Append to `distributions/metrics.py`:
```python
# ── Aliases and FP16 baseline helpers ──────────────────────────────────────────

qsnr_db = snr_db  # alias used by fourbit code paths

def fp16_quantize(x: np.ndarray) -> np.ndarray:
    """Round-trip ``x`` through float16. FP16 is the upper-bound baseline
    for 4-bit quantization QSNR comparisons."""
    return np.asarray(x, dtype=np.float32).astype(np.float16).astype(np.float32)

def fp16_qsnr_db(x: np.ndarray) -> float:
    """QSNR (dB) of FP16-rounded ``x`` vs FP32 ``x``."""
    return snr_db(x, fp16_quantize(x))

def crest_factor(x: np.ndarray) -> float:
    """Peak-to-RMS ratio ``max(|x|) / rms(x)``. Returns 0 for all-zero tensor."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return 0.0
    peak = float(np.max(np.abs(x)))
    rms  = float(np.sqrt(np.mean(x * x)))
    return peak / rms if rms > 0 else 0.0

def tensor_summary(x: np.ndarray) -> dict:
    """Compact stat bundle: std, max_abs, crest (peak/std), crest_rms
    (peak/rms including mean), kurtosis, n."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return {"std": 0.0, "max_abs": 0.0, "crest": 0.0,
                "crest_rms": 0.0, "kurtosis": 0.0, "n": 0}
    std  = float(np.std(x))
    peak = float(np.max(np.abs(x)))
    mean = float(np.mean(x))
    dev  = x - mean
    var  = float(np.mean(dev * dev))
    kurt = float(np.mean(dev ** 4) / (var ** 2)) - 3.0 if var > 0 else 0.0
    return {
        "std":       std,
        "max_abs":   peak,
        "crest":     peak / std if std > 0 else 0.0,
        "crest_rms": crest_factor(x),
        "kurtosis":  kurt,
        "n":         int(x.size),
    }
```

**Step 2:** Update `fourbit/part1.py` line 39, `fourbit/profiler_v2.py` line 57, and any other consumers: change `from fourbit.metrics import …` → `from distributions.metrics import …`.

```bash
grep -rn "from fourbit.metrics\|import fourbit.metrics" --include="*.py" | \
  cut -d: -f1 | sort -u
```
Edit each file, replace import.

**Step 3:** Delete `fourbit/metrics.py`.

**Step 4:** Run: `pytest tests/ -q` → all green. Run regression → all green.

**Step 5:** Commit:
```bash
git add distributions/metrics.py fourbit/ tests/test_metrics_equivalence.py
git commit -m "refactor(pr-2a): consolidate quantization metrics in distributions.metrics

Move fp16_quantize, fp16_qsnr_db, crest_factor, tensor_summary from
fourbit/metrics.py to distributions/metrics.py. Add qsnr_db alias for
snr_db. Equivalence test (test_metrics_equivalence.py) verifies mse and
snr_db are numerically identical to the fourbit versions before removal.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

### Task 2.4: Move paired generators to `distributions/linear_pairs.py`

**Files:**
- Create: `distributions/linear_pairs.py`
- Modify: `fourbit/distributions.py` → delete; replaced by new files
- Create: `fourbit/distribution_sets.py`

**Step 1:** Create `distributions/linear_pairs.py` with the 6 paired generators from `fourbit/distributions.py` lines 109–233:
- `weight_transformer(batch, in_features, out_features, seed)` (rename `_weight_transformer`, make public)
- `weight_moe(...)`
- `weight_attention(...)`
- `smooth_friendly_mild(...)`
- `smooth_friendly_severe(...)`
- `smooth_friendly_balanced(...)`

Drop the leading underscore: they are now a module-level public API.

**Step 2:** Create `fourbit/distribution_sets.py` (content identical to current `fourbit/distributions.py` minus the `_weight_*` and `_smooth_friendly_*` definitions; imports come from `distributions.linear_pairs` instead):

```python
"""Curated distribution sets for the 4-bit study (Part 1)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List
import numpy as np

from distributions.generators import (
    gaussian, laplace, student_t_dist, bimodal,
    channel_outliers, spiky_outliers, log_normal,
)
from distributions.linear_pairs import (
    weight_transformer, weight_moe, weight_attention,
    smooth_friendly_mild, smooth_friendly_severe, smooth_friendly_balanced,
)

@dataclass
class DistSpec:
    name: str
    fn: Callable
    tags: List[str] = field(default_factory=list)
    def generate(self, n: int, seed: int):
        return self.fn(n=n, seed=seed)

@dataclass
class LinearSpec:
    name: str
    fn: Callable
    tags: List[str] = field(default_factory=list)
    def generate(self, batch: int, in_features: int, out_features: int, seed: int):
        return self.fn(batch=batch, in_features=in_features,
                       out_features=out_features, seed=seed)

# COMMON_DISTRIBUTIONS = [...]        # copy verbatim from old fourbit/distributions.py
# LINEAR_WEIGHT_ACTIVATION = [...]    # copy verbatim, referencing weight_transformer etc.
# SMOOTH_FRIENDLY = [...]             # copy verbatim, referencing smooth_friendly_* funcs
```

**Step 3:** Update `fourbit/part1.py` line 36:
```python
# OLD:
from fourbit.distributions import (COMMON_DISTRIBUTIONS, LINEAR_WEIGHT_ACTIVATION, SMOOTH_FRIENDLY)
# NEW:
from fourbit.distribution_sets import (COMMON_DISTRIBUTIONS, LINEAR_WEIGHT_ACTIVATION, SMOOTH_FRIENDLY)
```
Check any other file: `grep -rn "fourbit.distributions" --include="*.py"` — update all.

**Step 4:** Delete `fourbit/distributions.py`.

**Step 5:** Run: `pytest tests/ -q` → all green. Run regression → all green.

**Step 6:** Commit:
```bash
git add distributions/linear_pairs.py fourbit/
git commit -m "refactor(pr-2b): move paired generators to distributions.linear_pairs

Extract the six (weight, activation) generators from fourbit/distributions.py
into distributions/linear_pairs.py as public primitives. Curated list definitions
(DistSpec, LinearSpec, COMMON_DISTRIBUTIONS, LINEAR_WEIGHT_ACTIVATION,
SMOOTH_FRIENDLY) moved to fourbit/distribution_sets.py. fourbit/distributions.py
removed. Output numerics unchanged (golden regression verifies).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

### Task 2.5: Push + PR

Same pattern as PR 1. Title: `refactor(pr-2): consolidate distributions & metrics`. Body must list:
- Files removed: `fourbit/distributions.py`, `fourbit/metrics.py`
- Files added: `distributions/linear_pairs.py`, `fourbit/distribution_sets.py`, `tests/test_metrics_equivalence.py`
- Files modified: ~3 (fourbit/part1.py, fourbit/profiler_v2.py, distributions/metrics.py)
- `pytest` + regression green.

---

# PR 3 — Consolidate Formats & Transforms

**Goal:**
- Create `formats/_pot.py` with canonical scalar `pot_scale` + vectorized `pot_scale_vec`.
- Move 4 new format wrappers (APoT4, LOG4, NF4_FP8, INT4_FP) from `fourbit/formats.py` into `formats/int_variants.py`.
- Update consumers; delete `fourbit/formats.py` (wrappers merged into `fourbit/registry.py` where needed).
- Leave `fourbit/transforms.py` IN PLACE for now (it is study-specific, will move to `experiments/fourbit/` in PR 5; no duplication with `formats/transforms/` so no dedup needed here).

**Note revision from design doc §5:** After re-reading `fourbit/transforms.py` (246 lines), its `Transform` base class and `SmoothQuantTransform`/`HadamardTransform` are pre-processing transforms for the study pipeline, NOT duplicates of `formats/transforms/smoothquant.py::SmoothQuantINTQuantizer` (which is an integrated quantizer). They are different abstractions. Leave `fourbit/transforms.py` for PR 5 (relocation only).

**Branch:** `refactor/pr3-formats`

### Task 3.1: Branch + pre-check

```bash
git checkout main && git pull origin main
git checkout -b refactor/pr3-formats
pytest tests/ -q && pytest tests/test_regression.py
```

### Task 3.2: Equivalence test for pot_scale

**Files:**
- Create: `tests/test_pot_scale_equivalence.py`

```python
"""Equivalence: scalar and vectorized _pot_scale implementations agree."""
import numpy as np
import pytest

# Current locations
from formats import _pot_scale as _pot_top      # formats/__init__.py
from fourbit.formats import _pot_scale_for_qmax as _pot_vec_fb

@pytest.mark.parametrize("absmax,q_max", [
    (0.0, 7), (1e-30, 7), (0.5, 7), (1.0, 7), (3.14, 7),
    (100.0, 7), (1e10, 7),
    (1.0, 127), (0.01, 127), (1e-5, 127),
])
def test_scalar_matches_vectorized_at_one_point(absmax, q_max):
    scalar  = _pot_top(absmax, q_max)
    vectord = float(_pot_vec_fb(np.array([absmax], dtype=np.float64), q_max)[0])
    assert scalar == vectord, f"mismatch at {absmax}, {q_max}: {scalar} vs {vectord}"

def test_vectorized_batch():
    absmax = np.array([0.0, 0.5, 1.0, 3.14, 100.0], dtype=np.float64)
    got = _pot_vec_fb(absmax, 7)
    exp = np.array([_pot_top(float(a), 7) for a in absmax], dtype=np.float64)
    assert np.array_equal(got, exp)
```

Run: `pytest tests/test_pot_scale_equivalence.py -v`
Expected: 10+ passed. If any fails, STOP and report.

### Task 3.3: Create `formats/_pot.py`

**Files:**
- Create: `formats/_pot.py`

```python
"""Canonical Power-of-Two scale helpers.

OCP-aligned formula:
    scale = 2^(floor(log2(absmax)) - floor(log2(q_max)))

Both scalar and vectorized versions share the same semantics; see
formats/__init__.py for the full mathematical discussion that predates
this extraction.
"""
from __future__ import annotations
import numpy as np


def pot_scale(absmax: float, q_max: int) -> float:
    """Scalar POT scale. absmax <= 0 returns 1.0."""
    if absmax <= 0:
        return 1.0
    log2_absmax = int(np.floor(np.log2(float(absmax) + 1e-38)))
    log2_qmax   = int(np.floor(np.log2(float(q_max))))
    return float(2.0 ** (log2_absmax - log2_qmax))


def pot_scale_vec(absmax: np.ndarray, q_max: float) -> np.ndarray:
    """Vectorized POT scale, safe for zero / subnormal absmax."""
    absmax = np.asarray(absmax, dtype=np.float64)
    q_max  = float(q_max)
    log2_qmax = int(np.floor(np.log2(q_max)))
    # Safe log2: replace <=0 with 1 to avoid warnings, mask afterwards.
    safe = np.where(absmax > 0, absmax, 1.0)
    log2_absmax = np.floor(np.log2(safe + 1e-38)).astype(np.int64)
    scale = np.power(2.0, log2_absmax - log2_qmax)
    return np.where(absmax > 0, scale, 1.0)
```

### Task 3.4: Update consumers of _pot_scale

**Step 1:** List consumers:
```bash
grep -rn "_pot_scale\b" --include="*.py" .
```
Expected: 5 files touched (from earlier grep):
- `formats/__init__.py`: defines `_pot_scale` (remove; re-export from `_pot`)
- `formats/sq_format.py`: has its own `_pot_scale` (remove duplicate, import from `_pot`)
- `fourbit/formats.py`: has `_pot_scale_for_qmax` (remove, import from `_pot`)
- `tests/test_formats.py`: references `_pot_scale` (update import)
- `tests/test_sq_format.py`: references `_pot_scale` (update import)

**Step 2:** Edit each consumer:

In `formats/__init__.py`: delete local `def _pot_scale(...)` definition. Near the top, add:
```python
from formats._pot import pot_scale as _pot_scale   # backward-compat alias
```
(The underscore-prefixed alias is internal — `_pot_scale` is referenced by both tests. Keep the alias to limit diff in tests and to be removed in PR 6.)

In `formats/sq_format.py`: remove local `_pot_scale` definition; add `from formats._pot import pot_scale as _pot_scale` at module top.

In `fourbit/formats.py`: remove local `_pot_scale_for_qmax` definition; add `from formats._pot import pot_scale_vec as _pot_scale_for_qmax`.

In `tests/test_formats.py`, `tests/test_sq_format.py`: no code change yet; imports via re-export alias keep working. Verify with:
```bash
pytest tests/test_formats.py tests/test_sq_format.py -v
```
Expected: all green.

**Step 3:** Run full suite + regression.

**Step 4:** Commit:
```bash
git add formats/_pot.py formats/__init__.py formats/sq_format.py \
        fourbit/formats.py tests/test_pot_scale_equivalence.py
git commit -m "refactor(pr-3a): consolidate pot_scale into formats/_pot

Single canonical implementation of the OCP power-of-two scale formula.
formats/__init__.py::_pot_scale, formats/sq_format.py::_pot_scale, and
fourbit/formats.py::_pot_scale_for_qmax now import from formats._pot.
Backward-compat aliases retained in each source module (to be removed in
PR 6). Equivalence test guards numeric identity.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

### Task 3.5: Extract 4 new format wrappers to `formats/int_variants.py`

**Files:**
- Create: `formats/int_variants.py`
- Modify: `fourbit/formats.py` — delete the 4 variant definitions; import from `formats.int_variants`

**Step 1:** Identify the 4 variant class/function names in `fourbit/formats.py`:
Read the file (350 lines), find:
- `INT4_FP` (FP absmax per-channel INT4)
- `APoT4` (Additive Power-of-Two 4-bit)
- `LOG4` (pure logarithmic 4-bit)
- `NF4_FP8` (NF4 with FP8 E4M3 scale)

**Step 2:** Move these four definitions (plus any private helpers they depend on) verbatim into `formats/int_variants.py`. Keep internal imports consistent (e.g., `from formats._pot import pot_scale_vec`).

**Step 3:** In `fourbit/formats.py`, replace the removed definitions with:
```python
from formats.int_variants import INT4_FP, APoT4, LOG4, NF4_FP8
```

**Step 4:** Test: `pytest tests/ -q && pytest tests/test_regression.py`

**Step 5:** Commit:
```bash
git add formats/int_variants.py fourbit/formats.py
git commit -m "refactor(pr-3b): move new 4-bit variants to formats.int_variants

INT4_FP, APoT4, LOG4, NF4_FP8 are now in formats.int_variants and can be
consumed by any experiment (previously locked inside fourbit.formats).
fourbit/formats.py imports them unchanged."
```

### Task 3.6: Push + PR

Title: `refactor(pr-3): consolidate pot_scale and 4-bit variants`.

---

# PR 4 — Consolidate Profiler

**Goal:**
- Reachability-analyze `fourbit/profiler_v2.py`. If the only external consumers are inside `fourbit/` itself, no merge into `profiler/` is required — mark it as study-specific and leave for PR 5.
- If it has utilities that overlap with `profiler/`, extract them.

**Branch:** `refactor/pr4-profiler`

### Task 4.1: Reachability analysis

```bash
grep -rn "from fourbit.profiler_v2\|import fourbit.profiler_v2" --include="*.py"
```
Expected consumers (from earlier grep):
- `fourbit/accuracy.py`
- `fourbit/part2.py`

No consumers outside `fourbit/`. Record this in a commit message.

### Task 4.2: Compare `LayerRecord`, `LayerCollector`, `analyse_all` vs `profiler/stats.py` and `profiler/profiler.py`

Read:
- `fourbit/profiler_v2.py` (286 lines)
- `profiler/profiler.py` (348 lines)
- `profiler/stats.py` (211 lines)

**Decision rule:**
- If `LayerRecord`/`LayerCollector`/`analyse_all` are algorithmically similar to any `profiler/*.py` class, document the overlap; propose the cleaner implementation; surface to user for approval.
- If they are genuinely study-specific (e.g., record schema tailored to 4-bit part2 reporter), leave them in place and PR 5 just relocates.

**Expected outcome:** `fourbit/profiler_v2.py` is study-specific (different data shape, feeds `fourbit/reporter.py`). No merge. PR 4 becomes a no-op PR that records the finding.

### Task 4.3: If no-op, skip this PR

If analysis shows no consolidation opportunity, do NOT create an empty PR. Instead, add a one-paragraph note to `docs/plans/2026-04-17-refactoring.md` (this file) under PR 4 stating:

> Reachability analysis showed `fourbit/profiler_v2.py` is consumed only by `fourbit/accuracy.py` and `fourbit/part2.py`; its `LayerRecord`/`LayerCollector` schema is specific to the 4-bit study's per-layer×format table and does not overlap with `profiler/profiler.py`'s PyTorch hook-based ModelProfiler. No consolidation. File relocates as-is in PR 5.

Commit that doc change on branch `refactor/pr4-profiler-analysis`, push, and merge as a docs-only PR (1 commit, ~10 lines).

### Task 4.4: If consolidation possible, proceed

(Populate steps based on actual findings during Task 4.2.)

---

# PR 5 — Relocate `fourbit/` → `experiments/fourbit/`

**Goal:** Mechanical file move + import path updates. `run_4bit_study.py` becomes a shim.

**Branch:** `refactor/pr5-relocate-fourbit`

### Task 5.1: Branch + pre-check

```bash
git checkout main && git pull origin main
git checkout -b refactor/pr5-relocate-fourbit
pytest tests/ -q && pytest tests/test_regression.py
```

### Task 5.2: Move files

```bash
git mv fourbit/__init__.py                experiments/fourbit/__init__.py
git mv fourbit/config.py                  experiments/fourbit/config.py
git mv fourbit/registry.py                experiments/fourbit/registry.py
git mv fourbit/pipeline.py                experiments/fourbit/pipeline.py
git mv fourbit/part1.py                   experiments/fourbit/part1.py
git mv fourbit/part2.py                   experiments/fourbit/part2.py
git mv fourbit/accuracy.py                experiments/fourbit/accuracy.py
git mv fourbit/profiler_v2.py             experiments/fourbit/profiler_v2.py
git mv fourbit/reporter.py                experiments/fourbit/reporter.py
git mv fourbit/formats.py                 experiments/fourbit/formats.py
git mv fourbit/transforms.py              experiments/fourbit/transforms.py
git mv fourbit/distribution_sets.py       experiments/fourbit/distribution_sets.py
rmdir fourbit   # must be empty now
```

### Task 5.3: Update all internal imports

```bash
# In-place replace, limited to files we're moving and their consumers
grep -rln "from fourbit\.\|import fourbit\." --include="*.py" .
```

For each file found (expect ~15–20), rewrite:
- `from fourbit.X import …` → `from experiments.fourbit.X import …`
- `from fourbit import …` → `from experiments.fourbit import …`

Use a single `python -c "..."` or sed command, verify with `grep -rn "fourbit\." --include="*.py"` expecting only `experiments.fourbit.…` matches afterward.

### Task 5.4: Rewrite `run_4bit_study.py` as shim

Replace the 78-line current CLI with:
```python
"""Shim: keeps `python run_4bit_study.py` working.

Actual logic lives in experiments.fourbit.cli.
"""
from __future__ import annotations
import sys
from experiments.fourbit.cli import main

if __name__ == "__main__":
    sys.exit(main())
```

Create `experiments/fourbit/cli.py` with the contents of old `run_4bit_study.py` (78 lines) wrapped in a `main(argv=None) -> int` function.

### Task 5.5: Verify

**Step 1:** `python run_4bit_study.py --help`
Expected: identical to pre-PR (compare via saved `/tmp/help-before.txt`).

**Step 2:** `python run_4bit_study.py --part 1`
Expected: exits 0, all CSVs generated under `results/fourbit/part1/`.

**Step 3:** `pytest tests/ -q && pytest tests/test_regression.py`
Expected: all green.

**Step 4:** Manual `run_all.py` smoke:
```bash
python run_all.py --fast --skip-hw 2>&1 | tee /tmp/runall-after-pr5.log
```
Compare stdout summary lines (Phase 1..5 completion, format counts) vs a pre-PR reference run captured in Task 5.1.

### Task 5.6: Commit + push + PR

```bash
git add -A
git commit -m "refactor(pr-5): relocate fourbit/ → experiments/fourbit/

fourbit/ is study-specific orchestration, not a peer of core primitives.
Move all files to experiments/fourbit/ and update internal imports.
run_4bit_study.py becomes a 5-line shim delegating to
experiments.fourbit.cli.main(). CLI arguments and output unchanged; golden
regression + run_all.py --fast --skip-hw smoke verified.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

# PR 6 — Final Cleanup

**Goal:** Remove leftover backward-compat aliases, verify nothing unused remains, update README.

**Branch:** `refactor/pr6-final`

### Task 6.1: Remove backward-compat aliases

In `formats/__init__.py` and `formats/sq_format.py`: remove `from formats._pot import pot_scale as _pot_scale` aliases if their only consumers were the tests.

Check:
```bash
grep -rn "_pot_scale\b" --include="*.py" .
```
- If only `tests/test_formats.py` and `tests/test_sq_format.py` match, update those test files to `from formats._pot import pot_scale` and remove both aliases.
- Otherwise, leave the alias and note why in the commit.

### Task 6.2: Unused-symbol sweep

```bash
python -m pyflakes formats/ distributions/ profiler/ experiments/ utils/ 2>&1 | head -30
```
Address any "imported but unused" reported in files we touched. Do not mass-fix pre-existing warnings from untouched files.

### Task 6.3: Update `README.md`

Replace the "Project Structure" tree block (README.md lines ~77–125) with the new tree from `docs/plans/2026-04-17-refactoring-design.md` § 3. Preserve all other text.

### Task 6.4: Line count check

```bash
find . -name '*.py' -not -path '*/\.*' -not -path '*__pycache__*' | xargs wc -l | tail -1
```
Record the total. Compare to pre-refactor ~17,079. Record in PR description.

### Task 6.5: Verify + push + PR

```bash
pytest tests/ -q && pytest tests/test_regression.py
python run_all.py --fast --skip-hw  # final smoke
```

PR title: `refactor(pr-6): final cleanup & docs`. Body must include line count delta and an "Out of scope followups" section listing any `TODO(post-refactor)` comments added during earlier PRs.

---

## Post-Series Actions

After PR 6 merges:

1. **Close the design PR** (`refactor/design-doc`) if still open — its content is merged.
2. **Delete all `refactor/*` branches** on remote:
   ```bash
   git fetch --prune
   git branch -a | grep refactor/ | grep remotes/origin | \
     sed 's#remotes/origin/##' | xargs -I{} git push origin --delete {}
   ```
3. **Update `MEMORY.md`** (memory system) if any durable conventions came out of this refactor (e.g., "New format wrappers go in `formats/int_variants.py`").
4. **Write a one-paragraph retrospective** in `docs/plans/2026-04-17-refactoring-retro.md`: what worked, what didn't, any scope cuts made, quantitative line-count delta.

---

## Summary Checklist (all PRs)

- [ ] PR 0 — golden baseline
- [ ] PR 1 — utils extraction
- [ ] PR 2 — distributions & metrics
- [ ] PR 3 — formats & pot_scale
- [ ] PR 4 — profiler analysis (expected no-op)
- [ ] PR 5 — relocate fourbit → experiments/fourbit
- [ ] PR 6 — final cleanup & README
- [ ] Retrospective
