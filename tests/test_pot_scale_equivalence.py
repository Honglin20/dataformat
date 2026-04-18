"""Equivalence tests for the consolidated pot_scale helpers.

Before PR 3 there were three near-duplicate implementations:
  - formats/__init__.py::_pot_scale          (OCP floor, scalar)
  - fourbit/formats.py::_pot_scale_for_qmax  (OCP floor, vectorized)
  - formats/sq_format.py::_pot_scale         (ceil-of-ratio, scalar)
  - formats/sq_format.py::_pot_scale_vec     (ceil-of-ratio, vectorized)

After PR 3, all four are aliases for two canonical implementations in
formats/_pot.py: pot_scale{,_vec} (floor) and pot_scale_ceil{,_vec} (ceil).
These tests pin both families so future edits cannot silently diverge.
"""
import numpy as np
import pytest

from formats._pot import (
    pot_scale,
    pot_scale_vec,
    pot_scale_ceil,
    pot_scale_ceil_vec,
)


POT_POINTS = [
    (0.0, 7), (1e-30, 7), (0.5, 7), (1.0, 7), (3.14, 7),
    (100.0, 7), (1e10, 7),
    (1.0, 127), (0.01, 127), (1e-5, 127),
]


@pytest.mark.parametrize("absmax,q_max", POT_POINTS)
def test_floor_scalar_matches_vectorized(absmax, q_max):
    scalar = pot_scale(absmax, q_max)
    vectord = float(pot_scale_vec(np.array([absmax], dtype=np.float64), q_max)[0])
    assert scalar == vectord, (
        f"floor mismatch at {absmax}, {q_max}: {scalar} vs {vectord}"
    )


@pytest.mark.parametrize("absmax,q_max", POT_POINTS)
def test_ceil_scalar_matches_vectorized(absmax, q_max):
    scalar = pot_scale_ceil(absmax, q_max)
    vectord = float(pot_scale_ceil_vec(np.array([absmax], dtype=np.float64), q_max)[0])
    assert scalar == vectord, (
        f"ceil mismatch at {absmax}, {q_max}: {scalar} vs {vectord}"
    )


def test_floor_batch():
    absmax = np.array([0.0, 0.5, 1.0, 3.14, 100.0], dtype=np.float64)
    got = pot_scale_vec(absmax, 7)
    exp = np.array([pot_scale(float(a), 7) for a in absmax], dtype=np.float64)
    assert np.array_equal(got, exp)


def test_ceil_batch():
    absmax = np.array([0.5, 1.0, 3.14, 100.0], dtype=np.float64)  # omit 0: ratio undefined
    got = pot_scale_ceil_vec(absmax, 7)
    exp = np.array([pot_scale_ceil(float(a), 7) for a in absmax], dtype=np.float64)
    assert np.array_equal(got, exp)


def test_floor_vs_ceil_differ_when_expected():
    """absmax=15, q_max=7 is the canonical counter-example (sq_format docstring)."""
    assert pot_scale(15.0, 7) == 2.0
    assert pot_scale_ceil(15.0, 7) == 4.0
