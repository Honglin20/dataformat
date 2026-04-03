"""Phase 2+3 experiment: distribution robustness analysis.

Runs all formats × all distributions → 5-metric evaluation matrix.
Outputs results as a pandas DataFrame and saves to CSV.
"""

import os
import sys
import numpy as np
import pandas as pd

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import N_SAMPLES, RANDOM_SEED, OUTPUT_DIR
from distributions.generators import generate_all_distributions
from distributions.metrics import evaluate_all
from formats import build_all_formats


def run_robustness_experiment(
    n: int = N_SAMPLES,
    seed: int = RANDOM_SEED,
    formats_subset: list = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run robustness experiment across all distributions and formats.

    Parameters
    ----------
    n : int
        Tensor size.
    seed : int
        Random seed.
    formats_subset : list of str, optional
        If given, only evaluate these format names. Otherwise all formats.
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame
        Columns: format, dist_name, mse, snr_db, kl_div, max_ae, eff_bits,
                 bits, dist_type, [distribution metadata keys...]
    """
    distributions = generate_all_distributions(n=n, seed=seed)

    # Use dimension consistent with rotation transforms
    dim = min(n, 256)   # cap at 256 for RandRot memory
    all_formats = build_all_formats(dim=dim, seed=seed)

    if formats_subset is not None:
        all_formats = {k: v for k, v in all_formats.items() if k in formats_subset}

    rows = []
    total = len(distributions) * len(all_formats)
    count = 0

    for dist_name, x, dist_meta in distributions:
        for fmt_name, fmt in all_formats.items():
            count += 1
            if verbose:
                print(f"[{count}/{total}] {fmt_name} × {dist_name}", end="\r")

            try:
                # For rotation-based transforms, work on the full tensor
                x_q = fmt.quantize(x)
                metrics = evaluate_all(x, x_q)
            except Exception as e:
                metrics = {"mse": np.nan, "snr_db": np.nan, "kl_div": np.nan,
                           "max_ae": np.nan, "eff_bits": np.nan}
                if verbose:
                    print(f"\n  WARNING: {fmt_name} × {dist_name} failed: {e}")

            overhead = {}
            try:
                overhead = fmt.encoding_overhead()
            except Exception:
                pass

            row = {
                "format": fmt_name,
                "dist_name": dist_name,
                "bits": getattr(fmt, "bits", np.nan),
                **metrics,
                **{f"enc_{k}": v for k, v in overhead.items()
                   if isinstance(v, (int, float))},
                **{f"dist_{k}": v for k, v in dist_meta.items()
                   if isinstance(v, (int, float, str))},
            }
            rows.append(row)

    if verbose:
        print(f"\nDone. {len(rows)} evaluations.")

    df = pd.DataFrame(rows)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "robustness.csv")
    df.to_csv(out_path, index=False)
    if verbose:
        print(f"Saved → {out_path}")
    return df


if __name__ == "__main__":
    df = run_robustness_experiment(verbose=True)
    print(df.groupby("format")[["mse", "snr_db", "eff_bits"]].mean().round(4))
