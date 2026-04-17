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


# NOTE: These tests are not parallel-safe (pytest-xdist). The invoked CLIs
# write to the repo-root `results/` directory by design.
def test_exp1_4bit():
    _run([sys.executable, "experiments/exp1_common_distributions.py", "--bits", "4"])
    _compare_csv(RESULTS / "exp1" / "results_4bit.csv",
                 GOLDEN  / "exp1" / "results_4bit.csv")


def test_exp1_8bit():
    _run([sys.executable, "experiments/exp1_common_distributions.py", "--bits", "8"])
    _compare_csv(RESULTS / "exp1" / "results_8bit.csv",
                 GOLDEN  / "exp1" / "results_8bit.csv")


def test_exp2():
    _run([sys.executable, "experiments/exp2_crest_factor.py"])
    for bits in (4, 8):
        _compare_csv(
            RESULTS / "exp2" / f"results_{bits}bit.csv",
            GOLDEN  / "exp2" / f"results_{bits}bit.csv",
        )


def test_fourbit_part1():
    _run([sys.executable, "run_4bit_study.py", "--part", "1"])
    for name in ("exp11_direct_quant.csv",
                 "exp12_linear_wa.csv",
                 "exp13_smooth_transforms.csv"):
        _compare_csv(
            RESULTS / "fourbit" / "part1" / name,
            GOLDEN  / "fourbit" / "part1" / name,
        )


# NOTE: test_mnist_profile intentionally omitted.
# examples/profile_mnist.py:93 uses torch.randperm without a seed, so the
# 64-image subset differs per run. Fixing it is out of scope for the refactor
# series (no production-code change in PR 0). The mnist golden CSV is retained
# at tests/fixtures/golden/mnist/profiler_results.csv as a reference only.
