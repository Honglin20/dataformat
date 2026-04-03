"""Phase 3: Bit-width ablation experiment (4-bit vs 8-bit paradigm).

Two sub-experiments:
  A. 8-bit Efficiency Regime (W8A8):
     Measures where HAD transform becomes redundant overhead vs. plain INT8/MXFP8.
     Finds the "precision saturation point" — the distribution where HAD stops helping.

  B. 4-bit Survival Regime (W4A4, W4A8):
     Tests which approach best survives the precision cliff of ultra-low-bit
     quantization under extreme outlier distributions.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import N_SAMPLES, RANDOM_SEED, OUTPUT_DIR
from distributions.generators import (
    gaussian, channel_outliers, spiky_outliers, student_t_dist, bimodal
)
from distributions.metrics import evaluate_all
from formats import build_all_formats


# Formats relevant to 8-bit efficiency regime
_8BIT_FORMATS = [
    "INT8",
    "MXFP8", "MXINT8",
    "FP6",          # included as Pareto middle-ground
    "SmoothQuant+INT8",
    "HAD+INT8",
    "RandRot+INT8",
    "TurboQuant+INT8",
    "FP32",         # reference
]

# Formats relevant to 4-bit survival regime
_4BIT_FORMATS = [
    "INT4",
    "NVFP4", "MXFP4", "MXINT4",
    "NF4",
    "SQ-Format",
    "SmoothQuant+INT4",
    "HAD+INT4",
    "HAD+LUT4",
    "HAD+SQ",
    "RandRot+INT4",
    "TurboQuant+INT4",
    "FP32",         # reference
]

# Representative distributions for each regime
_SURVIVAL_DISTS = [
    ("Gaussian(σ=1)",      lambda n, s: gaussian(n, sigma=1.0, seed=s)),
    ("Student-t(ν=3)",     lambda n, s: student_t_dist(n, nu=3.0, seed=s)),
    ("Bimodal",            lambda n, s: bimodal(n, seed=s)),
    ("ChannelOutlier(σ=50)", lambda n, s: channel_outliers(n, outlier_sigma=50.0, seed=s)),
    ("Spiky(10×)",         lambda n, s: spiky_outliers(n, spike_multiplier=10.0, seed=s)),
    ("Spiky(50×)",         lambda n, s: spiky_outliers(n, spike_multiplier=50.0, seed=s)),
    ("Spiky(100×)",        lambda n, s: spiky_outliers(n, spike_multiplier=100.0, seed=s)),
]


def _run_subset(
    dist_list: list,
    fmt_names: list,
    all_formats: dict,
    n: int,
    seed: int,
    tag: str,
) -> pd.DataFrame:
    rows = []
    total = len(dist_list) * len(fmt_names)
    count = 0
    for dist_name, dist_fn in dist_list:
        x, dist_meta = dist_fn(n, seed)
        for fmt_name in fmt_names:
            count += 1
            print(f"  [{tag}] [{count}/{total}] {fmt_name} × {dist_name}", end="\r")
            if fmt_name not in all_formats:
                continue
            fmt = all_formats[fmt_name]
            try:
                x_q = fmt.quantize(x)
                metrics = evaluate_all(x, x_q)
            except Exception as e:
                metrics = {"mse": np.nan, "snr_db": np.nan, "kl_div": np.nan,
                           "max_ae": np.nan, "eff_bits": np.nan}
            rows.append({
                "regime": tag,
                "format": fmt_name,
                "dist_name": dist_name,
                "bits": getattr(fmt, "bits", np.nan),
                **metrics,
            })
    print()
    return pd.DataFrame(rows)


def run_bitwidth_ablation(
    n: int = N_SAMPLES,
    seed: int = RANDOM_SEED,
    verbose: bool = True,
) -> dict:
    """Run 4-bit vs 8-bit ablation.

    Returns
    -------
    dict with keys "efficiency_8bit" and "survival_4bit", each a DataFrame.
    """
    dim = min(n, 256)
    all_formats = build_all_formats(dim=dim, seed=seed)

    if verbose:
        print("=== 8-bit Efficiency Regime ===")
    df_8bit = _run_subset(
        _SURVIVAL_DISTS, _8BIT_FORMATS, all_formats, n, seed, "8bit"
    )

    if verbose:
        print("=== 4-bit Survival Regime ===")
    df_4bit = _run_subset(
        _SURVIVAL_DISTS, _4BIT_FORMATS, all_formats, n, seed, "4bit"
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_8bit.to_csv(os.path.join(OUTPUT_DIR, "ablation_8bit.csv"), index=False)
    df_4bit.to_csv(os.path.join(OUTPUT_DIR, "ablation_4bit.csv"), index=False)

    if verbose:
        print("8-bit regime (mean eff_bits per format):")
        print(df_8bit.groupby("format")["eff_bits"].mean().sort_values(ascending=False).round(3))
        print("\n4-bit regime (mean eff_bits per format):")
        print(df_4bit.groupby("format")["eff_bits"].mean().sort_values(ascending=False).round(3))

    return {"efficiency_8bit": df_8bit, "survival_4bit": df_4bit}


if __name__ == "__main__":
    results = run_bitwidth_ablation(verbose=True)
