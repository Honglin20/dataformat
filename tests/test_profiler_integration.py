"""End-to-end tests: profile a small model through all 14 formats."""
import os
import tempfile

import pandas as pd
import torch
import torch.nn as nn

from profiler import ModelProfiler
from profiler.formats import PROFILER_FORMAT_NAMES


class TinyTransformerLayer(nn.Module):
    """Minimal model with Linear + LayerNorm (common in LLMs)."""

    def __init__(self):
        super().__init__()
        self.q = nn.Linear(16, 16)
        self.k = nn.Linear(16, 16)
        self.norm = nn.LayerNorm(16)
        self.out = nn.Linear(16, 8)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        scores = torch.softmax(q * k, dim=-1)
        return self.out(self.norm(scores))


class TestFullProfileRun:
    def test_all_14_formats_profiled(self):
        model = TinyTransformerLayer()
        p = ModelProfiler(model)
        while not p.done:
            p.start()
            for _ in range(3):
                with torch.no_grad():
                    model(torch.randn(4, 16))
            p.stop()
        assert p.done
        assert len(p._data) == 14

    def test_csv_contains_all_formats(self):
        model = TinyTransformerLayer()
        p = ModelProfiler(model)
        while not p.done:
            p.start()
            with torch.no_grad():
                model(torch.randn(4, 16))
            p.stop()
        with tempfile.TemporaryDirectory() as d:
            path = p.export_csv(d)
            df = pd.read_csv(path)
        assert set(df["format"].unique()) == set(PROFILER_FORMAT_NAMES)

    def test_quantization_error_increases_with_lower_bits(self):
        """FP32 MSE == 0; lower-bit formats have higher MSE."""
        model = nn.Linear(64, 64)
        p = ModelProfiler(model, target_layers=[nn.Linear])
        while not p.done:
            p.start()
            for _ in range(5):
                with torch.no_grad():
                    model(torch.randn(8, 64))
            p.stop()
        with tempfile.TemporaryDirectory() as d:
            path = p.export_csv(d)
            df = pd.read_csv(path)
        weight_df = df[df["tensor_type"] == "weight"]
        fp32_mse = weight_df[weight_df["format"] == "FP32"]["mse"].mean()
        int4_mse = weight_df[weight_df["format"] == "INT4(TENSOR)"]["mse"].mean()
        assert fp32_mse == 0.0
        assert int4_mse > fp32_mse

    def test_wrap_functional(self):
        """profiler.wrap() captures stats for torch.matmul."""
        model = nn.Linear(16, 16)
        p = ModelProfiler(model)
        p.start()
        with torch.no_grad():
            x = torch.randn(4, 16)
            a = model(x)
            _ = p.wrap(torch.matmul, "custom_matmul")(a, a.T)
        p.stop()
        assert "custom_matmul" in p._data.get("FP32", {})

    def test_context_manager_cleans_up_on_exception(self):
        """Hooks are removed even if inference raises."""
        model = nn.Linear(8, 8)
        p = ModelProfiler(model)
        try:
            with p:
                raise RuntimeError("simulated inference failure")
        except RuntimeError:
            pass
        # Hooks must be cleaned up
        assert len(p._hooks) == 0
        # Format should NOT have advanced (exception path)
        assert p.current_format_name == "FP32"
