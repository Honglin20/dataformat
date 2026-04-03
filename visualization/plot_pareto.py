"""Figures 3 & 4: Dual Pareto Frontier Charts.

Figure 3 (Pareto A): Bit-width vs. Quantization Quality (EffBits / MSE)
  - Identifies which formats dominate in the 4-bit survival and 8-bit efficiency regimes.

Figure 4 (Pareto B): Bit-width vs. Memory Bandwidth Amplification
  - Reveals the hidden bandwidth cost of MX block scales.
  - A format with high bandwidth amplification may be memory-bound despite lower bit-width.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from distributions.generators import channel_outliers, gaussian, student_t_dist
from distributions.metrics import effective_bits, mse as compute_mse
from formats import build_all_formats
from visualization.style import save_fig, PALETTE, MARKERS, get_color, get_marker


# Representative formats for Pareto plots (cover all families)
_PARETO_FORMATS = {
    # (format_name, nominal_bits, metadata_bpe)
    "FP32":        (32, 0.0),
    "BF16":        (16, 0.0),
    "INT8":        (8,  0.0),
    "MXFP8":       (8,  0.25),
    "MXINT8":      (8,  0.25),
    "FP6":         (6,  0.0),
    "INT4":        (4,  0.0),
    "MXFP4":       (4,  0.25),
    "MXINT4":      (4,  0.25),
    "NVFP4":       (4,  0.0),
    "NF4":         (4,  0.0),
    "SQ-Format":   (4,  1.0),      # 4 dense + 1-bit mask
    "SmoothQuant+INT8": (8, 0.0),
    "SmoothQuant+INT4": (4, 0.0),
    "HAD+INT8":    (8,  0.0),
    "HAD+INT4":    (4,  0.0),
    "HAD+LUT4":    (4,  0.0),
    "HAD+SQ":      (4,  1.0),
    "TurboQuant+INT4": (4, 0.0),
    "RandRot+INT4": (4, 0.0),
}


def _compute_pareto_data(n: int = 4096, seed: int = 42) -> pd.DataFrame:
    """Compute quality and bandwidth metrics for all formats on a challenging distribution."""
    all_formats = build_all_formats(dim=256, seed=seed)

    # Use channel outlier as the representative challenging distribution
    x_hard, _ = channel_outliers(n=n, outlier_ratio=0.01, outlier_sigma=50.0, seed=seed)
    x_easy, _ = gaussian(n=n, sigma=1.0, seed=seed)

    rows = []
    for fmt_name, (nominal_bits, meta_bpe) in _PARETO_FORMATS.items():
        if fmt_name not in all_formats:
            continue
        fmt = all_formats[fmt_name]

        # Quality on hard distribution
        try:
            x_q_hard = fmt.quantize(x_hard)
            eff_bits_hard = effective_bits(x_hard, x_q_hard)
            mse_hard = compute_mse(x_hard, x_q_hard)
        except Exception:
            eff_bits_hard, mse_hard = np.nan, np.nan

        # Quality on easy distribution
        try:
            x_q_easy = fmt.quantize(x_easy)
            eff_bits_easy = effective_bits(x_easy, x_q_easy)
        except Exception:
            eff_bits_easy = np.nan

        # Effective bits (storage) = nominal_bits + metadata
        storage_bits = nominal_bits + meta_bpe
        bandwidth_amplification = storage_bits / nominal_bits if nominal_bits > 0 else 1.0

        rows.append({
            "format": fmt_name,
            "nominal_bits": nominal_bits,
            "metadata_bpe": meta_bpe,
            "storage_bits": storage_bits,
            "bandwidth_amplification": bandwidth_amplification,
            "eff_bits_hard": eff_bits_hard,
            "eff_bits_easy": eff_bits_easy,
            "mse_hard": mse_hard,
        })

    return pd.DataFrame(rows)


def _is_pareto_optimal(points: np.ndarray, minimize_x: bool = False) -> np.ndarray:
    """Return boolean mask of Pareto-optimal points.

    Assumes higher-y and lower-x is better (or higher-y and higher-x based on context).
    Here: lower x (storage_bits) is better, higher y (eff_bits) is better.
    """
    n = len(points)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j or not mask[j]:
                continue
            # Point j dominates point i if it has smaller x AND larger y
            if points[j, 0] <= points[i, 0] and points[j, 1] >= points[i, 1]:
                if points[j, 0] < points[i, 0] or points[j, 1] > points[i, 1]:
                    mask[i] = False
                    break
    return mask


def plot_pareto_charts(n: int = 4096, seed: int = 42, out_dir: str = "results/figures"):
    """Plot Figure 3 (quality Pareto) and Figure 4 (bandwidth Pareto)."""
    df = _compute_pareto_data(n=n, seed=seed)

    # ── Figure 3: EffBits vs. Storage Bit-Width ──────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(10, 6))

    valid = df.dropna(subset=["eff_bits_hard", "storage_bits"])
    points = valid[["storage_bits", "eff_bits_hard"]].values

    pareto_mask = _is_pareto_optimal(points)

    for _, row in valid.iterrows():
        c = get_color(row["format"])
        m = get_marker(row["format"])
        ax3.scatter(row["storage_bits"], row["eff_bits_hard"],
                    c=c, marker=m, s=90, zorder=5, edgecolors="white", linewidths=0.5)
        # Label offset
        ax3.annotate(
            row["format"],
            (row["storage_bits"], row["eff_bits_hard"]),
            fontsize=7.5, ha="left", va="bottom",
            xytext=(3, 2), textcoords="offset points", color=c
        )

    # Draw Pareto frontier line
    pareto_pts = valid[pareto_mask].sort_values("storage_bits")
    if len(pareto_pts) > 1:
        ax3.plot(pareto_pts["storage_bits"], pareto_pts["eff_bits_hard"],
                 "k--", linewidth=1.2, alpha=0.5, label="Pareto frontier", zorder=3)

    # Regime annotations
    ax3.axvspan(3.5, 5.5, alpha=0.06, color="green", label="4-bit Survival Regime")
    ax3.axvspan(7.0, 9.0, alpha=0.06, color="blue", label="8-bit Efficiency Regime")
    ax3.axvline(4, color="green", linewidth=0.8, linestyle=":", alpha=0.5)
    ax3.axvline(8, color="blue", linewidth=0.8, linestyle=":", alpha=0.5)

    ax3.set_xlabel("Storage Bit-Width (nominal bits + metadata bits per element)")
    ax3.set_ylabel("Effective Bit-Width (EffBits, higher = better)")
    ax3.set_title(
        "Figure 3 (Pareto A): Quantization Quality vs. Storage Cost\n"
        "(Channel Outlier σ=50 distribution; lower-right → lower storage, higher quality)"
    )
    ax3.legend(loc="upper left", fontsize=8)
    save_fig(fig3, "fig03_pareto_quality", out_dir)

    # ── Figure 4: BW Amplification vs. Nominal Bit-Width ─────────────────────
    fig4, ax4 = plt.subplots(figsize=(10, 6))

    for _, row in df.iterrows():
        if not np.isfinite(row.get("eff_bits_hard", np.nan)):
            continue
        c = get_color(row["format"])
        m = get_marker(row["format"])
        ax4.scatter(row["nominal_bits"], row["bandwidth_amplification"],
                    c=c, marker=m, s=90, zorder=5, edgecolors="white", linewidths=0.5)
        ax4.annotate(
            row["format"],
            (row["nominal_bits"], row["bandwidth_amplification"]),
            fontsize=7.5, ha="left", va="bottom",
            xytext=(3, 2), textcoords="offset points", color=c
        )

    # Reference line: no overhead
    ax4.axhline(1.0, color="gray", linewidth=1.0, linestyle="--",
                alpha=0.7, label="BW amplification = 1.0 (no metadata overhead)")

    # Highlight MX formats bandwidth cost
    mx_fmts = df[df["format"].str.startswith("MX")]
    if len(mx_fmts):
        ax4.annotate(
            "MX block-scale\nmetadata overhead",
            xy=(mx_fmts["nominal_bits"].mean(), mx_fmts["bandwidth_amplification"].mean()),
            xytext=(10, 1.12),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
            fontsize=9, color="red"
        )

    ax4.set_xlabel("Nominal Bit-Width (data bits per element)")
    ax4.set_ylabel("Memory Bandwidth Amplification\n(effective bytes / data bytes)")
    ax4.set_title(
        "Figure 4 (Pareto B): Bandwidth Cost vs. Nominal Bit-Width\n"
        "(Values > 1.0 mean metadata overhead increases effective memory traffic)"
    )
    ax4.set_ylim(0.9, max(df["bandwidth_amplification"].max() + 0.1, 1.5))
    ax4.legend(loc="upper right", fontsize=9)
    save_fig(fig4, "fig04_pareto_bandwidth", out_dir)

    return fig3, fig4


if __name__ == "__main__":
    plot_pareto_charts()
