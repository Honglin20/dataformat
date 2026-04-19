"""PR B — metrics registry unit tests.

See ``docs/plans/2026-04-19-sqformat-experiment-plan.md`` §PR B.
"""
from __future__ import annotations

import numpy as np
import pytest

from distributions.metrics import (
    METRIC_REGISTRY,
    TENSOR_STAT_REGISTRY,
    register_metric,
)


def test_default_metrics_present():
    for key in ("qsnr_db", "snr_db", "mse", "fp16_qsnr_db"):
        assert key in METRIC_REGISTRY


def test_default_tensor_stats_present():
    for key in ("crest", "kurtosis"):
        assert key in TENSOR_STAT_REGISTRY


def test_register_metric_pair_kind():
    register_metric(
        "custom_mse",
        lambda r, q: float(np.mean((r - q) ** 2)),
        kind="pair",
    )
    assert "custom_mse" in METRIC_REGISTRY


def test_register_metric_rejects_unknown_kind():
    with pytest.raises(ValueError):
        register_metric("nope", lambda *_: 0.0, kind="bogus")
