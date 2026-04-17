"""CSV and JSON export for ModelProfiler results.

Output schema — profiler_results.csv (one row per format × layer × tensor_type):
  format, layer_name, layer_type, tensor_type, bits,
  mse, snr_db, eff_bits, max_ae, mare, saturation_rate,
  mean, std, skewness, kurtosis, abs_max, min, max, outlier_ratio,
  domain_kurtosis, domain_skewness, domain_std,
  e2e_snr_db, e2e_mse, e2e_eff_bits, e2e_max_ae,
  n_batches, n_elements
"""
from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from profiler.profiler import ModelProfiler

_FORMAT_BITS: dict[str, int] = {
    "FP32": 32, "FP16": 16,
    "SQ-FORMAT-INT": 4, "SQ-FORMAT-FP": 4,
    "INT4(CHANNEL)": 4, "INT8(CHANNEL)": 8,
    "INT4(TENSOR)": 4,  "INT8(TENSOR)": 8,
    "HAD+INT4(C)": 4,   "HAD+INT8(C)": 8,
    "HAD+INT4(T)": 4,   "HAD+INT8(T)": 8,
    "MXINT4": 4,        "MXINT8": 8,
}

_NAN = float("nan")


def _safe_finalize(obj):
    """Call finalize() on a stats object, returning {} on error."""
    try:
        return obj.finalize()
    except Exception:
        return {}


def export_csv(
    profiler: "ModelProfiler",
    output_dir: str,
    filename: str = "profiler_results.csv",
) -> str:
    """Write all recorded stats to a single CSV file.

    Parameters
    ----------
    profiler : ModelProfiler
        Profiler instance after one or more start/stop cycles.
    output_dir : str
        Directory to write the CSV into (created if absent).
    filename : str
        Output filename.

    Returns
    -------
    str
        Absolute path to the written CSV file.
    """
    layer_types: dict[str, str] = {
        name: type(mod).__name__
        for name, mod in profiler._model.named_modules()
    }

    # Pre-finalize e2e stats (QuantStats objects → dicts)
    e2e_finalized: dict[str, dict[str, dict]] = {}
    for fmt_name, layer_dict in profiler._e2e_data.items():
        e2e_finalized[fmt_name] = {}
        for layer_name, qs in layer_dict.items():
            e2e_finalized[fmt_name][layer_name] = _safe_finalize(qs)

    rows = []
    for fmt_name, layer_dict in profiler._data.items():
        n_batches = profiler._n_batches.get(fmt_name, 0)
        bits = _FORMAT_BITS.get(fmt_name, -1)
        e2e_fmt = e2e_finalized.get(fmt_name, {})

        for layer_name, tensor_dict in layer_dict.items():
            layer_type = layer_types.get(layer_name, "unknown")
            e2e = e2e_fmt.get(layer_name, {})

            for tensor_type, ts in tensor_dict.items():
                w_stats = _safe_finalize(ts.welford)
                h_stats = _safe_finalize(ts.hist)
                q_stats = _safe_finalize(ts.quant)

                if not w_stats:
                    continue  # no data captured — skip row

                # Transform-domain stats (e.g. HAD domain for HAD+INT)
                d_stats = _safe_finalize(ts.domain_welford) if ts.domain_welford else {}

                rows.append({
                    # ── Identity ───────────────────────────────────────────────
                    "format":          fmt_name,
                    "layer_name":      layer_name,
                    "layer_type":      layer_type,
                    "tensor_type":     tensor_type,
                    "bits":            bits,
                    # ── Quantization quality ───────────────────────────────────
                    "mse":             q_stats.get("mse",             _NAN),
                    "snr_db":          q_stats.get("snr_db",          _NAN),
                    "eff_bits":        q_stats.get("eff_bits",        _NAN),
                    "max_ae":          q_stats.get("max_ae",          _NAN),
                    "mare":            q_stats.get("mare",            _NAN),
                    "saturation_rate": q_stats.get("saturation_rate", _NAN),
                    # ── Original-domain distribution ───────────────────────────
                    "mean":            w_stats.get("mean",      _NAN),
                    "std":             w_stats.get("std",       _NAN),
                    "skewness":        w_stats.get("skewness",  _NAN),
                    "kurtosis":        w_stats.get("kurtosis",  _NAN),
                    "abs_max":         w_stats.get("abs_max",   _NAN),
                    "min":             w_stats.get("min",       _NAN),
                    "max":             w_stats.get("max",       _NAN),
                    "outlier_ratio":   h_stats.get("outlier_ratio", _NAN),
                    # ── Transform-domain distribution (HAD domain, etc.) ───────
                    "domain_kurtosis": d_stats.get("kurtosis",  _NAN),
                    "domain_skewness": d_stats.get("skewness",  _NAN),
                    "domain_std":      d_stats.get("std",       _NAN),
                    # ── End-to-end layer output SQNR (Linear layers only) ──────
                    "e2e_snr_db":      e2e.get("snr_db",   _NAN),
                    "e2e_mse":         e2e.get("mse",      _NAN),
                    "e2e_eff_bits":    e2e.get("eff_bits", _NAN),
                    "e2e_max_ae":      e2e.get("max_ae",   _NAN),
                    # ── Book-keeping ───────────────────────────────────────────
                    "n_batches":       n_batches,
                    "n_elements":      w_stats.get("n_elements", 0),
                })

    df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False)
    return os.path.abspath(path)


def export_histograms(
    profiler: "ModelProfiler",
    output_dir: str,
    filename: str = "profiler_histograms.json",
) -> str:
    """Write per-layer histogram data to a JSON file.

    Output schema:
      {format_name: {layer_name: {tensor_type: {hist_counts, hist_edges, outlier_ratio}}}}
    """
    data: dict = {}
    for fmt_name, layer_dict in profiler._data.items():
        data[fmt_name] = {}
        for layer_name, tensor_dict in layer_dict.items():
            data[fmt_name][layer_name] = {}
            for tensor_type, ts in tensor_dict.items():
                try:
                    h = ts.hist.finalize()
                    data[fmt_name][layer_name][tensor_type] = {
                        "hist_counts":   h["hist_counts"],
                        "hist_edges":    h["hist_edges"],
                        "outlier_ratio": h["outlier_ratio"],
                    }
                except Exception:
                    pass

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        json.dump(data, f)
    return os.path.abspath(path)
