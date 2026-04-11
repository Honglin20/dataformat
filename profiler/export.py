"""CSV export for ModelProfiler results.

Output schema (one row per format × layer × tensor_type):
  format, layer_name, layer_type, tensor_type, bits,
  mse, snr_db, eff_bits, max_ae,
  mean, std, outlier_ratio, n_batches, n_elements
"""
from __future__ import annotations

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

    rows = []
    for fmt_name, layer_dict in profiler._data.items():
        n_batches = profiler._n_batches.get(fmt_name, 0)
        bits = _FORMAT_BITS.get(fmt_name, -1)

        for layer_name, tensor_dict in layer_dict.items():
            layer_type = layer_types.get(layer_name, "unknown")

            for tensor_type, ts in tensor_dict.items():
                try:
                    w_stats = ts.welford.finalize()
                    h_stats = ts.hist.finalize()
                    q_stats = ts.quant.finalize()
                except RuntimeError:
                    continue

                rows.append({
                    "format":        fmt_name,
                    "layer_name":    layer_name,
                    "layer_type":    layer_type,
                    "tensor_type":   tensor_type,
                    "bits":          bits,
                    "mse":           q_stats["mse"],
                    "snr_db":        q_stats["snr_db"],
                    "eff_bits":      q_stats["eff_bits"],
                    "max_ae":        q_stats["max_ae"],
                    "mean":          w_stats["mean"],
                    "std":           w_stats["std"],
                    "outlier_ratio": h_stats["outlier_ratio"],
                    "n_batches":     n_batches,
                    "n_elements":    w_stats["n_elements"],
                })

    df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False)
    return os.path.abspath(path)
