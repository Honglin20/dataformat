"""Figures 3 & 4: Dual Pareto Frontier Charts.

Figure 3 (Quality Pareto):
  X-axis: Effective bits per element (storage_bits = nominal + metadata_bpe).
  Y-axis: SQNR in dB (higher is better).
  Pareto-optimal frontier drawn as dashed line.
  Highlights MXINT4 vs HAD+INT4(C) with annotation arrows.

Figure 4 (Bandwidth Pareto):
  X-axis: SQNR in dB (quality).
  Y-axis: Effective bits per element (actual memory cost).
  Ideal corner = high SQNR AND low effective bits (lower-right).
  Iso-SQNR comparison lines between MXINT and HAD+INT.
  Highlights that MXINT4 pays +0.25 bpe overhead vs HAD+INT4(C) at similar quality.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from distributions.generators import channel_outliers
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


def _compute_pareto_data(n: int = 4096, seed: int = 42) -> pd.DataFrame:
    """Compute SQNR and storage metrics for all focus formats."""
    all_formats = build_all_formats(dim=256, seed=seed)
    x, _ = channel_outliers(n=n, outlier_sigma=50.0, seed=seed)

    rows = []
    for fmt_name, (nominal_bits, meta_bpe) in _PARETO_FORMATS.items():
        if fmt_name not in all_formats:
            continue
        fmt = all_formats[fmt_name]
        storage_bpe = nominal_bits + meta_bpe
        try:
            x_q = fmt.quantize(x)
            sqnr = snr_db(x, x_q)
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


def plot_pareto_charts(n: int = 4096, seed: int = 42, out_dir: str = "results/figures"):
    """Plot Figure 3 (quality Pareto) and Figure 4 (bandwidth Pareto)."""
    df = _compute_pareto_data(n=n, seed=seed)

    # ── Figure 3: SQNR vs. Effective Bits ─────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(10, 7))

    valid3 = df.dropna(subset=["sqnr_db", "storage_bpe"]).copy()
    points3 = valid3[["storage_bpe", "sqnr_db"]].values
    pareto_mask3 = _is_pareto_optimal_high_y_low_x(points3)

    for _, row in valid3.iterrows():
        c = get_color(row["format"])
        m = get_marker(row["format"])
        ax3.scatter(row["storage_bpe"], row["sqnr_db"],
                    color=c, marker=m, s=100, zorder=5,
                    edgecolors="white", linewidths=0.6)
        ax3.annotate(
            row["format"],
            (row["storage_bpe"], row["sqnr_db"]),
            fontsize=7.5, ha="left", va="bottom",
            xytext=(4, 3), textcoords="offset points", color=c,
        )

    # Pareto frontier dashed line
    pareto_pts3 = valid3.iloc[pareto_mask3].sort_values("storage_bpe")
    if len(pareto_pts3) > 1:
        ax3.plot(
            pareto_pts3["storage_bpe"], pareto_pts3["sqnr_db"],
            "k--", linewidth=1.4, alpha=0.55, label="Pareto frontier", zorder=3,
        )

    # Highlight MXINT4 vs HAD+INT4(C) with annotation arrows
    for fmt_hi, fmt_lo in [("HAD+INT4(C)", "MXINT4")]:
        row_hi = valid3[valid3["format"] == fmt_hi]
        row_lo = valid3[valid3["format"] == fmt_lo]
        if len(row_hi) and len(row_lo):
            hi = row_hi.iloc[0]
            lo = row_lo.iloc[0]
            ax3.annotate(
                "",
                xy=(hi["storage_bpe"], hi["sqnr_db"]),
                xytext=(lo["storage_bpe"], lo["sqnr_db"]),
                arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.5),
            )
            ax3.text(
                (hi["storage_bpe"] + lo["storage_bpe"]) / 2 + 0.05,
                (hi["sqnr_db"] + lo["sqnr_db"]) / 2,
                "HAD > MXINT\n(same bits)",
                fontsize=7.5, color="darkgreen", va="center",
            )

    ax3.set_xlabel("Effective Bits per Element (nominal + metadata)")
    ax3.set_ylabel("SQNR (dB)  — higher is better")
    ax3.set_title(
        "Figure 3: Quality Pareto Frontier\n"
        "(Channel Outlier σ=50; lower-left = worse; upper-left = better)",
        fontsize=12,
    )
    ax3.legend(loc="lower right", fontsize=9)
    save_fig(fig3, "fig03_pareto_quality", out_dir)

    # ── Figure 4: Effective Bits vs. SQNR (Bandwidth Pareto) ─────────────────
    fig4, ax4 = plt.subplots(figsize=(10, 7))

    valid4 = df.dropna(subset=["sqnr_db", "storage_bpe"]).copy()

    for _, row in valid4.iterrows():
        c = get_color(row["format"])
        m = get_marker(row["format"])
        ax4.scatter(row["sqnr_db"], row["storage_bpe"],
                    color=c, marker=m, s=100, zorder=5,
                    edgecolors="white", linewidths=0.6)
        ax4.annotate(
            row["format"],
            (row["sqnr_db"], row["storage_bpe"]),
            fontsize=7.5, ha="left", va="bottom",
            xytext=(4, 3), textcoords="offset points", color=c,
        )

    # Iso-SQNR comparison lines between MX and HAD pairs
    # Highlights MXINT4 pays +0.25 bpe overhead vs HAD+INT4(C)
    pairs = [
        ("MXINT4",   "HAD+INT4(C)", "4-bit"),
        ("MXINT8",   "HAD+INT8(C)", "8-bit"),
    ]
    for fmt_mx, fmt_had, label in pairs:
        row_mx  = valid4[valid4["format"] == fmt_mx]
        row_had = valid4[valid4["format"] == fmt_had]
        if len(row_mx) and len(row_had):
            mx  = row_mx.iloc[0]
            had = row_had.iloc[0]
            # Draw a horizontal bracket showing bandwidth overhead
            y_mx  = mx["storage_bpe"]
            y_had = had["storage_bpe"]
            x_mid = min(mx["sqnr_db"], had["sqnr_db"]) - 1.5
            ax4.annotate(
                "",
                xy=(x_mid, y_mx),
                xytext=(x_mid, y_had),
                arrowprops=dict(arrowstyle="<->", color="steelblue", lw=1.3),
            )
            ax4.text(
                x_mid - 0.2, (y_mx + y_had) / 2,
                f"+{y_mx - y_had:.2f} bpe\n({label})",
                fontsize=7, color="steelblue", ha="right", va="center",
            )

    ax4.set_xlabel("SQNR (dB)  — higher is better →")
    ax4.set_ylabel("Effective Bits per Element (actual memory cost)")
    ax4.set_title(
        "Figure 4: Bandwidth Pareto — Memory Cost vs. Quality\n"
        "(Ideal corner = high SQNR + low effective bits = lower-right)",
        fontsize=12,
    )
    ax4.text(
        0.98, 0.04,
        "← lower-right = better\n(high quality, low memory)",
        transform=ax4.transAxes,
        fontsize=9, color="darkgreen", ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="honeydew",
                  edgecolor="darkgreen", alpha=0.8),
    )
    save_fig(fig4, "fig04_pareto_bandwidth", out_dir)

    return fig3, fig4


if __name__ == "__main__":
    plot_pareto_charts()
