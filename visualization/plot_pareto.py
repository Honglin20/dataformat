"""Figures 3 & 4: Dual Pareto Frontier Charts — 4-bit (left) and 8-bit (right) panels.

Figure 3 (Quality Pareto):
  Left panel: 4-bit formats — X: storage bits, Y: SQNR (dB).
  Right panel: 8-bit formats — same axes.
  Pareto-optimal frontier drawn as dashed line per panel.
  Highlights MXINT vs HAD+INT difference at each bit-width.

Figure 4 (Bandwidth Pareto):
  Left panel: 4-bit — X: SQNR, Y: storage bits/elem.
  Right panel: 8-bit — same axes.
  Ideal corner = high SQNR + low bits (lower-right).
  Iso-cost brackets show MXINT vs HAD+INT overhead per bit-width.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from distributions.metrics import snr_db
from formats import build_all_formats, FOCUS_4BIT, FOCUS_8BIT, FOCUS_ALL
from visualization.style import save_fig, PALETTE, get_color, get_marker

# Focus formats with their effective bits per element:
#   storage_bits + metadata_bpe
# INT4=4, INT8=8, MXINT4=4.25, MXINT8=8.25, NVFP4=4, NF4=4, SQ-Format=5.01,
# HAD+INT4(C)=4, HAD+INT4(T)=4, HAD+INT8(C)=8, HAD+INT8(T)=8,
# HAD+SQ=5.01, RandRot+INT4=4, RandRot+INT8=8, FP32=32
_PARETO_FORMATS = {
    "FP32":          (32,   0.0),
    "INT4":          (4,    0.0),
    "INT8":          (8,    0.0),
    "MXINT4":        (4,    0.25),
    "MXINT8":        (8,    0.25),
    "NVFP4":         (4,    0.0),
    "NF4":           (4,    0.0),
    "SQ-Format":     (4,    1.01),
    "SQ-Format(8b)": (8,    1.01),
    "HAD+INT4(C)":   (4,    0.0),
    "HAD+INT4(T)":   (4,    0.0),
    "HAD+INT8(C)":   (8,    0.0),
    "HAD+INT8(T)":   (8,    0.0),
    "HAD+SQ":        (4,    1.01),
    "RandRot+INT4":  (4,    0.0),
    "RandRot+INT8":  (8,    0.0),
}


def _is_pareto_optimal_high_y_low_x(points: np.ndarray) -> np.ndarray:
    """Return boolean mask of Pareto-optimal points (lower x, higher y is better)."""
    n = len(points)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j or not mask[j]:
                continue
            # j dominates i if j has <= x AND >= y with at least one strict
            if points[j, 0] <= points[i, 0] and points[j, 1] >= points[i, 1]:
                if points[j, 0] < points[i, 0] or points[j, 1] > points[i, 1]:
                    mask[i] = False
                    break
    return mask


def _make_2d_outlier_tensor(batch: int = 16, features: int = 256, outlier_sigma: float = 50.0, seed: int = 42) -> np.ndarray:
    """2D (batch × features) tensor with row 0 as systematic outlier row.

    HAD+INT(C) per-row scale adapts independently; HAD+INT(T) global scale
    is dominated by the outlier row, hurting precision on clean rows.
    features must be a power of 2 for HAD compatibility.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (batch, features)).astype(np.float32)
    x[0, :] = rng.normal(0, outlier_sigma, features).astype(np.float32)
    return x


def _compute_pareto_data(seed: int = 42) -> pd.DataFrame:
    """Compute SQNR and storage metrics for all focus formats.

    Uses a 2D (16×256) tensor so HAD+INT(C) and HAD+INT(T) produce
    different SQNR values and appear as distinct points on the Pareto chart.
    """
    all_formats = build_all_formats(dim=256, seed=seed)
    x = _make_2d_outlier_tensor(batch=16, features=256, outlier_sigma=50.0, seed=seed)

    rows = []
    for fmt_name, (nominal_bits, meta_bpe) in _PARETO_FORMATS.items():
        if fmt_name not in all_formats:
            continue
        fmt = all_formats[fmt_name]
        storage_bpe = nominal_bits + meta_bpe
        try:
            x_q = fmt.quantize(x)
            sqnr = snr_db(x.ravel(), x_q.ravel())
        except Exception:
            sqnr = np.nan
        rows.append({
            "format":       fmt_name,
            "nominal_bits": nominal_bits,
            "metadata_bpe": meta_bpe,
            "storage_bpe":  storage_bpe,
            "sqnr_db":      sqnr,
        })
    return pd.DataFrame(rows)


_4BIT_FMTS  = ["FP32", "INT4", "MXINT4", "NVFP4", "NF4",
               "SQ-Format", "HAD+INT4(C)", "HAD+INT4(T)", "HAD+SQ", "RandRot+INT4"]
_8BIT_FMTS  = ["FP32", "INT8", "MXINT8",
               "SQ-Format(8b)", "HAD+INT8(C)", "HAD+INT8(T)", "RandRot+INT8"]


def _draw_pareto_quality(ax, df: pd.DataFrame, fmt_list: list, title: str) -> None:
    """Draw quality Pareto (X=storage_bpe, Y=SQNR) for fmt_list onto ax."""
    sub = df[df["format"].isin(fmt_list)].dropna(subset=["sqnr_db", "storage_bpe"])
    if sub.empty:
        return
    points = sub[["storage_bpe", "sqnr_db"]].values
    pareto_mask = _is_pareto_optimal_high_y_low_x(points)

    for _, row in sub.iterrows():
        c = get_color(row["format"])
        m = get_marker(row["format"])
        ax.scatter(row["storage_bpe"], row["sqnr_db"],
                   color=c, marker=m, s=100, zorder=5,
                   edgecolors="white", linewidths=0.6)
        ax.annotate(row["format"], (row["storage_bpe"], row["sqnr_db"]),
                    fontsize=7, ha="left", va="bottom",
                    xytext=(4, 3), textcoords="offset points", color=c)

    pareto_pts = sub.iloc[pareto_mask].sort_values("storage_bpe")
    if len(pareto_pts) > 1:
        ax.plot(pareto_pts["storage_bpe"], pareto_pts["sqnr_db"],
                "k--", linewidth=1.3, alpha=0.5, label="Pareto frontier", zorder=3)

    # Annotate MXINT vs HAD+INT arrow
    for fmt_hi, fmt_lo in [("HAD+INT4(C)", "MXINT4"), ("HAD+INT8(C)", "MXINT8")]:
        rhi = sub[sub["format"] == fmt_hi]
        rlo = sub[sub["format"] == fmt_lo]
        if len(rhi) and len(rlo):
            hi, lo = rhi.iloc[0], rlo.iloc[0]
            ax.annotate("", xy=(hi["storage_bpe"], hi["sqnr_db"]),
                        xytext=(lo["storage_bpe"], lo["sqnr_db"]),
                        arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.5))
            ax.text((hi["storage_bpe"] + lo["storage_bpe"]) / 2 + 0.05,
                    (hi["sqnr_db"] + lo["sqnr_db"]) / 2,
                    "HAD > MX\n(same bits)", fontsize=7, color="darkgreen", va="center")

    ax.set_xlabel("Storage Bits/Elem (nominal + metadata)", fontsize=9)
    ax.set_ylabel("SQNR (dB) — higher is better", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)


def _draw_pareto_bandwidth(ax, df: pd.DataFrame, fmt_list: list, title: str) -> None:
    """Draw bandwidth Pareto (X=SQNR, Y=storage_bpe) for fmt_list onto ax."""
    sub = df[df["format"].isin(fmt_list)].dropna(subset=["sqnr_db", "storage_bpe"])
    if sub.empty:
        return

    for _, row in sub.iterrows():
        c = get_color(row["format"])
        m = get_marker(row["format"])
        ax.scatter(row["sqnr_db"], row["storage_bpe"],
                   color=c, marker=m, s=100, zorder=5,
                   edgecolors="white", linewidths=0.6)
        ax.annotate(row["format"], (row["sqnr_db"], row["storage_bpe"]),
                    fontsize=7, ha="left", va="bottom",
                    xytext=(4, 3), textcoords="offset points", color=c)

    for fmt_mx, fmt_had in [("MXINT4", "HAD+INT4(C)"), ("MXINT8", "HAD+INT8(C)")]:
        rmx  = sub[sub["format"] == fmt_mx]
        rhad = sub[sub["format"] == fmt_had]
        if len(rmx) and len(rhad):
            mx, had = rmx.iloc[0], rhad.iloc[0]
            x_mid = min(mx["sqnr_db"], had["sqnr_db"]) - 1.5
            ax.annotate("", xy=(x_mid, mx["storage_bpe"]),
                        xytext=(x_mid, had["storage_bpe"]),
                        arrowprops=dict(arrowstyle="<->", color="steelblue", lw=1.3))
            ax.text(x_mid - 0.3, (mx["storage_bpe"] + had["storage_bpe"]) / 2,
                    f"+{mx['storage_bpe'] - had['storage_bpe']:.2f} bpe",
                    fontsize=7, color="steelblue", ha="right", va="center")

    ax.set_xlabel("SQNR (dB) — higher is better →", fontsize=9)
    ax.set_ylabel("Storage Bits/Elem (actual memory cost)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.text(0.98, 0.04, "← lower-right = better",
            transform=ax.transAxes, fontsize=8, color="darkgreen",
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="honeydew",
                      edgecolor="darkgreen", alpha=0.8))


def plot_pareto_charts(seed: int = 42, out_dir: str = "results/figures"):
    """Plot Figure 3 (quality Pareto) and Figure 4 (bandwidth Pareto), each with 4-bit/8-bit panels."""
    df = _compute_pareto_data(seed=seed)

    # ── Figure 3: Quality Pareto — 4-bit (left) and 8-bit (right) ─────────────
    fig3, (ax3l, ax3r) = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=False)
    _draw_pareto_quality(
        ax3l, df, _4BIT_FMTS,
        "4-bit: Quality Pareto\n(Channel Outlier σ=50, 2D 16×256)",
    )
    _draw_pareto_quality(
        ax3r, df, _8BIT_FMTS,
        "8-bit: Quality Pareto\n(Channel Outlier σ=50, 2D 16×256)",
    )
    fig3.suptitle(
        "Figure 3: Quality Pareto Frontier — SQNR vs. Storage Bits/Element\n"
        "(upper-left is better; dashed = Pareto frontier)",
        fontsize=12,
    )
    fig3.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.10, wspace=0.30)
    save_fig(fig3, "fig03_pareto_quality", out_dir)

    # ── Figure 4: Bandwidth Pareto — 4-bit (left) and 8-bit (right) ──────────
    fig4, (ax4l, ax4r) = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=False)
    _draw_pareto_bandwidth(
        ax4l, df, _4BIT_FMTS,
        "4-bit: Bandwidth Pareto\n(lower-right = high quality + low memory)",
    )
    _draw_pareto_bandwidth(
        ax4r, df, _8BIT_FMTS,
        "8-bit: Bandwidth Pareto\n(lower-right = high quality + low memory)",
    )
    fig4.suptitle(
        "Figure 4: Bandwidth Pareto — Memory Cost vs. Quality\n"
        "(MXINT pays +0.25 bpe metadata vs HAD+INT at similar SQNR)",
        fontsize=12,
    )
    fig4.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.10, wspace=0.30)
    save_fig(fig4, "fig04_pareto_bandwidth", out_dir)

    return fig3, fig4


if __name__ == "__main__":
    plot_pareto_charts()
