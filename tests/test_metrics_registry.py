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


def test_fourbit_config_default_metrics_match_current_columns():
    from experiments.fourbit.config import DEFAULT_CONFIG

    names = {(m.name, tuple(m.roles)) for m in DEFAULT_CONFIG.metrics}
    # Exact set must equal the four columns that profiler_v2 currently emits
    assert names == {
        ("qsnr_db",      ("W", "X", "Y")),
        ("fp16_qsnr_db", ("W", "X", "Y")),
    }


def test_profiler_emits_custom_metric_when_configured():
    import numpy as np
    from experiments.fourbit.config import DEFAULT_CONFIG, MetricSpec
    from experiments.fourbit.profiler_v2 import analyse_layer
    from distributions.metrics import register_metric

    register_metric("unit_metric", lambda r, q: 42.0, kind="pair")

    cfg = DEFAULT_CONFIG.__class__(**{
        **DEFAULT_CONFIG.__dict__,
        "metrics": [MetricSpec("unit_metric", "unit_metric", roles=["W"])],
    })
    # W is (out, in) = (8, 4); X is (batch, in) = (4, 4); Y = X @ W.T => (4, 8).
    # The plan's example used mismatched shapes (W=(8,4), X=(4,8)) which caused
    # every pipeline call to raise and land in the failure branch. We use
    # matching shapes so the metric is evaluated normally.
    rec = {
        "W":    np.ones((8, 4), dtype=np.float32),
        "X":    np.ones((4, 4), dtype=np.float32),
        "Y":    np.ones((4, 8), dtype=np.float32),
        "bias": None,
    }
    df = analyse_layer(rec, "layer0", cfg)
    assert "unit_metric_w" in df.columns
    assert (df["unit_metric_w"] == 42.0).all()
