"""Smoke tests for profiling script — uses a tiny dummy model, no MNIST download."""
import sys, os, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
import torch.nn as nn
import torch.utils.data as data
import pandas as pd
import pytest

from profiler import ModelProfiler


def _make_tiny_model():
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 32), nn.ReLU(), nn.Linear(32, 10))


def _make_tiny_loader(n=16, batch_size=8):
    xs = torch.randn(n, 1, 28, 28)
    ys = torch.randint(0, 10, (n,))
    ds = data.TensorDataset(xs, ys)
    return data.DataLoader(ds, batch_size=batch_size)


class TestProfileMnist:
    def test_profiler_runs_and_produces_csv(self):
        from examples.profile_mnist import run_profiling
        model = _make_tiny_model()
        loader = _make_tiny_loader()
        with tempfile.TemporaryDirectory() as d:
            csv_path = run_profiling(model, loader, out_dir=d)
            assert os.path.exists(csv_path)
            df = pd.read_csv(csv_path)
            assert len(df) > 0
            assert "format" in df.columns
            assert "mse" in df.columns

    def test_all_14_formats_in_csv(self):
        from examples.profile_mnist import run_profiling
        from profiler.formats import PROFILER_FORMAT_NAMES
        model = _make_tiny_model()
        loader = _make_tiny_loader()
        with tempfile.TemporaryDirectory() as d:
            csv_path = run_profiling(model, loader, out_dir=d)
            df = pd.read_csv(csv_path)
            assert set(df["format"].unique()) == set(PROFILER_FORMAT_NAMES)

    def test_fp32_mse_is_zero(self):
        from examples.profile_mnist import run_profiling
        model = _make_tiny_model()
        loader = _make_tiny_loader()
        with tempfile.TemporaryDirectory() as d:
            csv_path = run_profiling(model, loader, out_dir=d)
            df = pd.read_csv(csv_path)
            fp32 = df[df["format"] == "FP32"]["mse"].dropna()
            assert (fp32 == 0.0).all(), "FP32 should have zero MSE"
