"""Figure 2: Precision-Outlier Sensitivity Heatmap.

X-axis: outlier severity (type × magnitude).
Y-axis: format / technique stack.
Color:  MSE (log scale) — darker = higher error.

Rows:  all formats in the study.
Columns: spiky outlier multipliers (1×, 10×, 50×, 100×) and
          channel outlier sigmas (σ=10, 30, 100).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from distributions.generators import spiky_outliers, channel_outliers, gaussian
from distributions.metrics import mse as compute_mse
from formats import build_all_formats
from visualization.style import save_fig, PALETTE


# Formats to show in the heatmap (ordered for readability)
_HEATMAP_FORMATS = [
    "FP32",
    "INT4", "INT8",
    "MXFP4", "MXFP8", "MXINT4", "MXINT8",
    "NVFP4", "NF4", "FP6",
    "SQ-Format",
    "SmoothQuant+INT4", "SmoothQuant+INT8",
    "HAD+INT4", "HAD+INT8",
    "HAD+LUT4", "HAD+SQ",
    "TurboQuant+INT4",
    "RandRot+INT4",
]


def build_heatmap_data(n: int = 2048, seed: int = 42) -> pd.DataFrame:
    """Build the MSE matrix for all formats × outlier conditions."""
    all_formats = build_all_formats(dim=256, seed=seed)
    formats = {k: v for k, v in all_formats.items() if k in _HEATMAP_FORMATS}

    # Define outlier conditions
    conditions = []
    # Spiky outliers
    for mult in [1.0, 5.0, 10.0, 25.0, 50.0, 100.0]:
        label = f"Spiky\n{int(mult)}×" if mult > 1 else "Gaussian\nbaseline"
        conditions.append((label, lambda n, s, m=mult: spiky_outliers(n, spike_multiplier=m, seed=s)[0]))
    # Channel outliers
    for sig in [10.0, 30.0, 50.0, 100.0]:
        conditions.append((
            f"Channel\nσ={int(sig)}",
            lambda n, s, sg=sig: channel_outliers(n, outlier_sigma=sg, seed=s)[0]
        ))

    rows = []
    for fmt_name in _HEATMAP_FORMATS:
        if fmt_name not in formats:
            continue
        fmt = formats[fmt_name]
        row = {"format": fmt_name}
        for cond_label, cond_fn in conditions:
            x = cond_fn(n, seed)
            try:
                x_q = fmt.quantize(x)
                row[cond_label] = compute_mse(x, x_q)
            except Exception:
                row[cond_label] = np.nan
        rows.append(row)

    return pd.DataFrame(rows).set_index("format")


def plot_outlier_heatmap(
    n: int = 2048, seed: int = 42, out_dir: str = "results/figures"
):
    """Plot Figure 2: Precision-Outlier Sensitivity Heatmap."""
    df = build_heatmap_data(n=n, seed=seed)

    # Log transform for color scale (MSE spans many orders of magnitude)
    df_log = np.log10(df.replace(0, 1e-10) + 1e-10)

    # Reorder formats for visual grouping
    ordered_fmts = [f for f in _HEATMAP_FORMATS if f in df_log.index]
    df_log = df_log.loc[ordered_fmts]

    with plt.rc_context({"figure.constrained_layout.use": False}):
        fig, ax = plt.subplots(figsize=(14, 8))

    cmap = "RdYlGn_r"   # Red=high error, Green=low error
    sns.heatmap(
        df_log,
        ax=ax,
        cmap=cmap,
        annot=False,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "log₁₀(MSE)", "shrink": 0.7},
        vmin=df_log.values[np.isfinite(df_log.values)].min(),
        vmax=df_log.values[np.isfinite(df_log.values)].max(),
    )

    # Annotate with actual MSE values (2 sig figs)
    for i, fmt in enumerate(df_log.index):
        for j, col in enumerate(df_log.columns):
            val = df.loc[fmt, col]
            if np.isfinite(val):
                text = f"{val:.2e}" if val >= 0.01 else f"{val:.1e}"
                ax.text(j + 0.5, i + 0.5, text,
                        ha="center", va="center", fontsize=6.5, color="black")

    # Visual separator between format families
    separator_positions = [2, 10, 11, 13, 15, 17, 18]   # row indices
    for pos in separator_positions:
        if pos < len(ordered_fmts):
            ax.axhline(pos, color="navy", linewidth=1.5, alpha=0.6)

    ax.set_title(
        "Figure 2: Precision-Outlier Sensitivity Heatmap\n"
        "(Color = log₁₀MSE; darker red = higher error; MX formats vs. Transform-based)",
        fontsize=12, pad=12
    )
    ax.set_xlabel("Outlier Condition")
    ax.set_ylabel("Quantization Format / Pipeline")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    # Add family labels on the right
    family_labels = {
        "Baselines": (0, 1),
        "Plain INT": (1, 3),
        "Hardware-Native (MX/FP)": (3, 10),
        "SQ-Format": (10, 11),
        "SmoothQuant": (11, 13),
        "HAD-based": (13, 17),
        "Random Rotation": (17, 19),
    }
    for family, (start, end) in family_labels.items():
        if end > len(ordered_fmts):
            end = len(ordered_fmts)
        if start < end:
            mid = (start + end) / 2
            ax.annotate(
                family, xy=(len(df_log.columns) + 0.15, mid),
                xycoords=("data", "data"),
                fontsize=7.5, rotation=-90, va="center", ha="left",
                color="navy", fontstyle="italic",
                annotation_clip=False,
            )

    fig.subplots_adjust(left=0.18, right=0.88, top=0.92, bottom=0.12)
    save_fig(fig, "fig02_outlier_sensitivity_heatmap", out_dir)
    return fig


if __name__ == "__main__":
    plot_outlier_heatmap()
