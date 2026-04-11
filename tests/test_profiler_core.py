"""Tests for ModelProfiler hook management, start/stop, and format iteration."""
import numpy as np
import pytest
import torch
import torch.nn as nn

from profiler.profiler import ModelProfiler


def _make_model():
    return nn.Sequential(
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
    )


def _run_batches(model, n=3):
    for _ in range(n):
        x = torch.randn(2, 16)
        with torch.no_grad():
            model(x)


class TestProfilerInit:
    def test_done_false_initially(self):
        p = ModelProfiler(_make_model())
        assert not p.done

    def test_done_true_after_all_formats(self):
        model = _make_model()
        p = ModelProfiler(model)
        while not p.done:
            p.start()
            _run_batches(model, n=1)
            p.stop()
        assert p.done

    def test_current_format_name(self):
        p = ModelProfiler(_make_model())
        assert p.current_format_name == "FP32"

    def test_format_advances_after_stop(self):
        model = _make_model()
        p = ModelProfiler(model)
        p.start()
        _run_batches(model, n=1)
        p.stop()
        assert p.current_format_name == "FP16"


class TestProfilerHooks:
    def test_hooks_registered_after_start(self):
        model = _make_model()
        p = ModelProfiler(model)
        p.start()
        n_leaf = sum(1 for m in model.modules() if len(list(m.children())) == 0)
        assert len(p._hooks) == n_leaf * 2

    def test_hooks_removed_after_stop(self):
        model = _make_model()
        p = ModelProfiler(model)
        p.start()
        _run_batches(model, n=1)
        p.stop()
        assert len(p._hooks) == 0

    def test_no_hooks_before_start(self):
        p = ModelProfiler(_make_model())
        assert len(p._hooks) == 0


class TestProfilerStats:
    def test_stats_populated_after_stop(self):
        model = _make_model()
        p = ModelProfiler(model)
        p.start()
        _run_batches(model, n=2)
        p.stop()
        assert "FP32" in p._data
        assert len(p._data["FP32"]) >= 2

    def test_input_output_captured(self):
        model = _make_model()
        p = ModelProfiler(model)
        p.start()
        _run_batches(model, n=1)
        p.stop()
        fmt_data = p._data["FP32"]
        for layer_name, tensors in fmt_data.items():
            assert "input" in tensors or "output" in tensors

    def test_weight_captured_for_linear(self):
        model = _make_model()
        p = ModelProfiler(model)
        p.start()
        _run_batches(model, n=3)
        p.stop()
        fmt_data = p._data["FP32"]
        linear_layers = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
        for layer_name in linear_layers:
            assert "weight" in fmt_data[layer_name]

    def test_target_layers_filter(self):
        model = _make_model()
        p = ModelProfiler(model, target_layers=[nn.Linear])
        p.start()
        _run_batches(model, n=1)
        p.stop()
        fmt_data = p._data["FP32"]
        for layer_name in fmt_data:
            module = dict(model.named_modules())[layer_name]
            assert isinstance(module, nn.Linear)

    def test_n_batches_tracked(self):
        model = _make_model()
        p = ModelProfiler(model)
        p.start()
        _run_batches(model, n=5)
        p.stop()
        assert p._n_batches["FP32"] == 5


class TestWrapFunctional:
    def test_wrap_captures_input_and_output(self):
        model = nn.Linear(16, 16)
        p = ModelProfiler(model)
        p.start()
        with torch.no_grad():
            x = torch.randn(4, 16)
            a = model(x)
            _ = p.wrap(torch.matmul, "custom_matmul")(a, a.T)
        p.stop()
        assert "custom_matmul" in p._data.get("FP32", {})
