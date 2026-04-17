"""Smoke tests for HTML report generation using synthetic CSV + log data."""
import sys, os, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import numpy as np
import pytest


def _make_sample_csv(path):
    """Write a minimal profiler_results.csv with 2 formats × 2 layers × 2 tensor types."""
    from profiler.formats import PROFILER_FORMAT_NAMES
    rows = []
    for fmt in ["FP32", "INT4(TENSOR)", "MXINT4", "HAD+INT4(C)"]:
        bits = 32 if fmt == "FP32" else 4
        for layer in ["0", "2"]:
            for tt in ["weight", "input"]:
                mse = 0.0 if fmt == "FP32" else np.random.uniform(1e-4, 1e-2)
                rows.append({
                    "format": fmt, "layer_name": layer, "layer_type": "Linear",
                    "tensor_type": tt, "bits": bits,
                    "mse": mse, "snr_db": 0.0 if mse == 0 else 10 * np.log10(0.1 / mse),
                    "eff_bits": 0.0 if mse == 0 else 0.5 * np.log2(0.1 / mse),
                    "max_ae": np.sqrt(mse), "mean": 0.0, "std": 0.3,
                    "outlier_ratio": 0.01, "n_batches": 5, "n_elements": 1000,
                })
    pd.DataFrame(rows).to_csv(path, index=False)


def _make_sample_log(path):
    log = {
        "epoch": list(range(1, 6)),
        "train_loss": [1.2, 0.5, 0.3, 0.2, 0.15],
        "test_loss":  [0.6, 0.4, 0.25, 0.18, 0.14],
        "train_acc":  [60.0, 85.0, 90.0, 94.0, 96.0],
        "test_acc":   [75.0, 87.0, 91.0, 94.5, 96.5],
    }
    with open(path, "w") as f:
        json.dump(log, f)


class TestGenerateReport:
    def test_report_html_created(self):
        from examples.generate_report import generate_report
        with tempfile.TemporaryDirectory() as d:
            csv_path = os.path.join(d, "profiler_results.csv")
            log_path = os.path.join(d, "training_log.json")
            _make_sample_csv(csv_path)
            _make_sample_log(log_path)
            out_path = generate_report(csv_path, log_path, d, open_browser=False)
            assert os.path.exists(out_path)
            assert out_path.endswith(".html")

    def test_report_is_self_contained_html(self):
        from examples.generate_report import generate_report
        with tempfile.TemporaryDirectory() as d:
            csv_path = os.path.join(d, "profiler_results.csv")
            log_path = os.path.join(d, "training_log.json")
            _make_sample_csv(csv_path)
            _make_sample_log(log_path)
            out_path = generate_report(csv_path, log_path, d, open_browser=False)
            content = open(out_path).read()
            assert "<!DOCTYPE html>" in content
            assert "data:image/png;base64," in content   # at least one inline image
            assert "FP32" in content                      # format names present
            assert "href" not in content or "http" not in content  # no external links

    def test_report_contains_all_sections(self):
        from examples.generate_report import generate_report
        with tempfile.TemporaryDirectory() as d:
            csv_path = os.path.join(d, "profiler_results.csv")
            log_path = os.path.join(d, "training_log.json")
            _make_sample_csv(csv_path)
            _make_sample_log(log_path)
            out_path = generate_report(csv_path, log_path, d, open_browser=False)
            content = open(out_path).read()
            for section in ["Training Curves", "EffBits", "Heatmap", "SNR", "Summary"]:
                assert section in content, f"Missing section: {section}"
