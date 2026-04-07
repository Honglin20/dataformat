"""Figure 2: Precision-Outlier Sensitivity Heatmap.

X-axis: outlier condition (spiky 1×/10×/50×/100×/200× + channel σ=10/30/50/100).
Y-axis: focus formats.
Color:  SQNR in dB (RdYlGn: green=high SQNR=good, red=low SQNR=bad).
Cells annotated with integer SQNR values (font 8).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from distributions.generators import spiky_outliers, channel_outliers
from distributions.metrics import snr_db
from formats import build_all_formats, FOCUS_4BIT, FOCUS_8BIT, FOCUS_ALL
from visualization.style import save_fig, PALETTE

# Focus formats for the heatmap (ordered for visual grouping)
_HEATMAP_FORMATS = [
    "FP32",
    "INT4", "INT8",
    "MXINT4", "MXINT8",
    "NVFP4", "NF4",
    "SQ-Format",
    "HAD+INT4(C)", "HAD+INT4(T)",
    "HAD+INT8(C)", "HAD+INT8(T)",
    "HAD+SQ",
    "RandRot+INT4", "RandRot+INT8",
]

# Family separators: row index AFTER which to draw a line
_SEPARATORS = [
    1,   # after FP32
    3,   # after INT4, INT8
    5,   # after MXINT4, MXINT8
    7,   # after NVFP4, NF4
    8,   # after SQ-Format
    12,  # after HAD+INT4(C/T), HAD+INT8(C/T)
    13,  # after HAD+SQ
]

# Family label positions: (label, start_row, end_row)
_FAMILY_LABELS = [
    ("Baseline",    0, 1),
    ("Plain INT",   1, 3),
    ("MXINT",       3, 5),
    ("HW-Native 4b",5, 7),
    ("SQ-Format",   7, 8),
    ("HAD+INT",     8, 12),
    ("HAD+SQ",      12, 13),
    ("RandRot",     13, 15),
]


_BATCH = 16
_FEATURES = 128  # power of 2 for HAD; features must match dim in build_all_formats


def build_heatmap_data(n: int = 2048, seed: int = 42) -> pd.DataFrame:
    """Build the SQNR matrix for focus formats × outlier conditions.

    Uses 2D tensors of shape (16, 128) = 2048 elements so that
    HAD+INT(C) ≠ HAD+INT(T) for row-outlier (Channel) conditions:
      - Spiky: spikes randomly distributed across ALL elements → (C) ≈ (T)
      - Channel/Row outlier: row 0 has high σ, rows 1-15 are N(0,1)
          → (C) per-row scale adapts to row 0 independently → (C) >> (T)

    Parameters
    ----------
    n : int
        Kept for API compatibility; actual tensor size is always _BATCH * _FEATURES = 2048.
    seed : int
        Random seed for reproducibility.
    """
    all_formats = build_all_formats(dim=_FEATURES, seed=seed)

    def _make_spiky_2d(spike_mult: float, s: int) -> np.ndarray:
        """2D (16, 128) with spikes randomly distributed across all positions."""
        rng = np.random.default_rng(s)
        x = rng.normal(0, 1, (_BATCH, _FEATURES)).astype(np.float32)
        n_total = _BATCH * _FEATURES
        n_spikes = max(1, int(np.ceil(0.001 * n_total)))
        flat = x.ravel()
        spike_idx = rng.choice(n_total, size=n_spikes, replace=False)
        signs = rng.choice([-1.0, 1.0], size=n_spikes).astype(np.float32)
        flat[spike_idx] = signs * spike_mult
        return flat.reshape(_BATCH, _FEATURES)

    def _make_row_outlier_2d(outlier_sigma: float, s: int) -> np.ndarray:
        """2D (16, 128) where row 0 has σ=outlier_sigma, rows 1-15 are N(0,1)."""
        rng = np.random.default_rng(s)
        x = rng.normal(0, 1, (_BATCH, _FEATURES)).astype(np.float32)
        x[0, :] = rng.normal(0, outlier_sigma, _FEATURES).astype(np.float32)
        return x

    # Outlier conditions: 5 spiky + 4 row-outlier (channel)
    conditions = []
    for mult in [1, 10, 50, 100, 200]:
        label = f"Spiky\n{mult}×"
        m = float(mult)
        conditions.append((label, lambda s, _m=m: _make_spiky_2d(_m, s)))
    for sig in [10, 30, 50, 100]:
        label = f"Channel\nσ={sig}"
        sg = float(sig)
        conditions.append((label, lambda s, _sg=sg: _make_row_outlier_2d(_sg, s)))

    rows = []
    for fmt_name in _HEATMAP_FORMATS:
        if fmt_name not in all_formats:
            continue
        fmt = all_formats[fmt_name]
        row = {"format": fmt_name}
        for cond_label, cond_fn in conditions:
            x = cond_fn(seed)
            try:
                x_q = fmt.quantize(x)
                row[cond_label] = snr_db(x.ravel(), x_q.ravel())
            except Exception:
                row[cond_label] = np.nan
        rows.append(row)

    return pd.DataFrame(rows).set_index("format")


def plot_outlier_heatmap(
    n: int = 2048,
    seed: int = 42,
    out_dir: str = "results/figures",
):
    """Plot Figure 2: Precision-Outlier Sensitivity Heatmap (SQNR)."""
    df = build_heatmap_data(n=n, seed=seed)

    # Reorder rows to match _HEATMAP_FORMATS
    ordered_fmts = [f for f in _HEATMAP_FORMATS if f in df.index]
    df = df.loc[ordered_fmts]

    valid_vals = df.values[np.isfinite(df.values)]
    vmin = float(np.nanmin(valid_vals)) if len(valid_vals) else 0
    vmax = float(np.nanmax(valid_vals)) if len(valid_vals) else 60

    with plt.rc_context({"figure.constrained_layout.use": False}):
        fig, ax = plt.subplots(figsize=(13, 9))

    sns.heatmap(
        df,
        ax=ax,
        cmap="RdYlGn",
        annot=False,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "SQNR (dB) — higher = better", "shrink": 0.7},
        vmin=vmin,
        vmax=vmax,
    )

    # Annotate cells with integer SQNR values (font 8)
    for i, fmt in enumerate(df.index):
        for j, col in enumerate(df.columns):
            val = df.loc[fmt, col]
            if np.isfinite(val):
                ax.text(j + 0.5, i + 0.5, f"{int(round(val))}",
                        ha="center", va="center", fontsize=8, color="black",
                        fontweight="bold")

    # Horizontal family separators between families
    for sep in _SEPARATORS:
        if sep < len(ordered_fmts):
            ax.axhline(sep, color="navy", linewidth=1.8, alpha=0.7)

    # Family labels on the right
    n_cols = len(df.columns)
    for family, start, end in _FAMILY_LABELS:
        end = min(end, len(ordered_fmts))
        if start >= end:
            continue
        mid = (start + end) / 2.0
        ax.annotate(
            family,
            xy=(n_cols + 0.1, mid),
            xycoords="data",
            fontsize=7.5, rotation=-90, va="center", ha="left",
            color="navy", fontstyle="italic",
            annotation_clip=False,
        )

    ax.set_title(
        "Figure 2: Precision-Outlier Sensitivity Heatmap\n"
        "(SQNR in dB — green = high quality, red = low quality)",
        fontsize=12, pad=12,
    )
    ax.set_xlabel("Outlier Condition")
    ax.set_ylabel("Quantization Format")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    fig.subplots_adjust(left=0.16, right=0.88, top=0.92, bottom=0.10)
    save_fig(fig, "fig02_outlier_sensitivity_heatmap", out_dir)
    return fig


if __name__ == "__main__":
    plot_outlier_heatmap()
