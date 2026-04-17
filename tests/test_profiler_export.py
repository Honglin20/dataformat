import os
import tempfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from profiler.profiler import ModelProfiler


def _profile_tiny_model(n_formats=2):
    model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))
    p = ModelProfiler(model)
    for _ in range(n_formats):
        if p.done:
            break
        p.start()
        for _ in range(2):
            with torch.no_grad():
                model(torch.randn(4, 8))
        p.stop()
    return p


class TestExportCSV:
    def test_export_creates_csv(self):
        p = _profile_tiny_model()
        with tempfile.TemporaryDirectory() as d:
            p.export_csv(d)
            files = os.listdir(d)
            assert any(f.endswith(".csv") for f in files)

    def test_csv_has_required_columns(self):
        p = _profile_tiny_model()
        with tempfile.TemporaryDirectory() as d:
            p.export_csv(d)
            csv_path = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".csv")][0]
            df = pd.read_csv(csv_path)
            required = {
                "format", "layer_name", "layer_type", "tensor_type", "bits",
                "mse", "snr_db", "eff_bits", "max_ae",
                "mean", "std", "outlier_ratio", "n_batches", "n_elements",
            }
            assert required.issubset(set(df.columns)), f"Missing: {required - set(df.columns)}"

    def test_csv_has_one_row_per_format_layer_tensortype(self):
        p = _profile_tiny_model(n_formats=2)
        with tempfile.TemporaryDirectory() as d:
            p.export_csv(d)
            csv_path = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".csv")][0]
            df = pd.read_csv(csv_path)
            dupes = df.duplicated(["format", "layer_name", "tensor_type"])
            assert not dupes.any(), f"Duplicate rows found:\n{df[dupes]}"

    def test_fp32_mse_is_zero(self):
        p = _profile_tiny_model(n_formats=1)
        with tempfile.TemporaryDirectory() as d:
            p.export_csv(d)
            csv_path = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".csv")][0]
            df = pd.read_csv(csv_path)
            fp32_rows = df[df["format"] == "FP32"]
            assert (fp32_rows["mse"].fillna(0.0) == 0.0).all(), \
                "FP32 should have zero quantization error"
