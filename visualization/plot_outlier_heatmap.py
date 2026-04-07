"""Figure 2: Precision-Outlier Sensitivity Heatmap.

Two panels (top=4-bit, bottom=8-bit), same outlier conditions on X-axis.
X-axis: outlier condition (spiky 1×/10×/50×/100×/200× + channel σ=10/30/50/100).
Y-axis: focus formats per bit-width.
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

# 4-bit formats panel (ordered for visual grouping)
_HEATMAP_FORMATS_4BIT = [
    "FP32",
    "INT4",
    "MXINT4",
    "NVFP4", "NF4",
    "SQ-Format",
    "HAD+INT4(C)", "HAD+INT4(T)",
    "HAD+SQ",
    "RandRot+INT4",
]

# 8-bit formats panel
_HEATMAP_FORMATS_8BIT = [
    "FP32",
    "INT8",
    "MXINT8",
    "SQ-Format(8b)",
    "HAD+INT8(C)", "HAD+INT8(T)",
    "RandRot+INT8",
]

# Family separators per panel: row index AFTER which to draw a line
_SEPARATORS_4BIT = [1, 2, 3, 5, 6, 8, 9]  # after FP32/INT4/MXINT4/NF4/SQ/HAD(T)/HAD+SQ
_SEPARATORS_8BIT = [1, 2, 3, 4, 6]

# Family labels: (label, start_row, end_row)
_FAMILY_LABELS_4BIT = [
    ("FP32",        0, 1),
    ("Plain INT",   1, 2),
    ("MXINT",       2, 3),
    ("HW-Native",   3, 5),
    ("SQ-Format",   5, 6),
    ("HAD+INT",     6, 8),
    ("HAD+SQ",      8, 9),
    ("RandRot",     9, 10),
]

_FAMILY_LABELS_8BIT = [
    ("FP32",        0, 1),
    ("Plain INT",   1, 2),
    ("MXINT",       2, 3),
    ("SQ-Format",   3, 4),
    ("HAD+INT",     4, 6),
    ("RandRot",     6, 7),
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
    all_fmt_names = list(dict.fromkeys(_HEATMAP_FORMATS_4BIT + _HEATMAP_FORMATS_8BIT))
    for fmt_name in all_fmt_names:
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


def _draw_heatmap_panel(ax, df: pd.DataFrame, ordered_fmts: list,
                        separators: list, family_labels: list,
                        vmin: float, vmax: float, title: str) -> None:
    """Draw one heatmap panel (4-bit or 8-bit) onto ax."""
    sub = df.loc[[f for f in ordered_fmts if f in df.index]]
    sns.heatmap(
        sub,
        ax=ax,
        cmap="RdYlGn",
        annot=False,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "SQNR (dB)", "shrink": 0.8},
        vmin=vmin,
        vmax=vmax,
    )
    # Annotate cells
    for i, fmt in enumerate(sub.index):
        for j, col in enumerate(sub.columns):
            val = sub.loc[fmt, col]
            if np.isfinite(val):
                ax.text(j + 0.5, i + 0.5, f"{int(round(val))}",
                        ha="center", va="center", fontsize=7.5,
                        color="black", fontweight="bold")
    # Separators
    n_rows = len(sub)
    for sep in separators:
        if sep < n_rows:
            ax.axhline(sep, color="navy", linewidth=1.5, alpha=0.7)
    # Family labels on the right
    n_cols = len(sub.columns)
    for family, start, end in family_labels:
        end = min(end, n_rows)
        if start >= end:
            continue
        mid = (start + end) / 2.0
        ax.annotate(
            family,
            xy=(n_cols + 0.1, mid),
            xycoords="data",
            fontsize=7, rotation=-90, va="center", ha="left",
            color="navy", fontstyle="italic",
            annotation_clip=False,
        )
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Outlier Condition", fontsize=9)
    ax.set_ylabel("Format", fontsize=9)
    ax.tick_params(axis="x", rotation=0, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)


def plot_outlier_heatmap(
    n: int = 2048,
    seed: int = 42,
    out_dir: str = "results/figures",
):
    """Plot Figure 2: Precision-Outlier Sensitivity Heatmap — 4-bit (top) and 8-bit (bottom)."""
    df = build_heatmap_data(n=n, seed=seed)

    # Shared color scale across both panels for fair comparison
    valid_vals = df.values[np.isfinite(df.values)]
    vmin = float(np.nanmin(valid_vals)) if len(valid_vals) else 0
    vmax = float(np.nanmax(valid_vals)) if len(valid_vals) else 60

    n4 = len([f for f in _HEATMAP_FORMATS_4BIT if f in df.index])
    n8 = len([f for f in _HEATMAP_FORMATS_8BIT if f in df.index])

    with plt.rc_context({"figure.constrained_layout.use": False}):
        fig, (ax4, ax8) = plt.subplots(
            2, 1, figsize=(14, 4 + n4 * 0.55 + n8 * 0.55),
            gridspec_kw={"height_ratios": [n4, n8]},
        )

    _draw_heatmap_panel(
        ax4, df, _HEATMAP_FORMATS_4BIT,
        _SEPARATORS_4BIT, _FAMILY_LABELS_4BIT,
        vmin, vmax,
        "4-bit Formats — SQNR (dB): MXINT4 vs SQ-Format vs HAD+INT4(C/T)",
    )
    _draw_heatmap_panel(
        ax8, df, _HEATMAP_FORMATS_8BIT,
        _SEPARATORS_8BIT, _FAMILY_LABELS_8BIT,
        vmin, vmax,
        "8-bit Formats — SQNR (dB): MXINT8 vs SQ-Format(8b) vs HAD+INT8(C/T)",
    )

    fig.suptitle(
        "Figure 2: Precision-Outlier Sensitivity Heatmap\n"
        "(green = high SQNR = good  ·  red = low SQNR = bad  ·  "
        "Channel conditions expose HAD+INT(C) vs (T) difference)",
        fontsize=11, y=1.01,
    )
    fig.subplots_adjust(left=0.18, right=0.88, top=0.95, bottom=0.08, hspace=0.45)
    save_fig(fig, "fig02_outlier_sensitivity_heatmap", out_dir)
    return fig


if __name__ == "__main__":
    plot_outlier_heatmap()
