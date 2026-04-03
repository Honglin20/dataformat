"""Figure 7: Roofline Model Analysis.

Plots the classic Roofline chart with format working points.

X-axis: Arithmetic Intensity (FLOPs/Byte, log scale).
Y-axis: Attainable Performance (TOPs, log scale).
Diagonal: Peak BW line (memory-bound region).
Horizontal: Peak Compute ceilings per precision.

Key insight to visualize:
  - MX formats: despite 4-bit data, E8M0 block scale metadata pushes them
    LEFT (lower arithmetic intensity), potentially back into memory-bound territory.
  - HAD+INT: transform adds compute ops (numerator) but no memory overhead →
    moves RIGHT (higher arithmetic intensity) while maintaining compute ceiling.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from hardware.roofline import build_roofline_data, ridge_point
from config import PEAK_COMPUTE_TOPS, PEAK_BW_TB_S
from visualization.style import save_fig, PALETTE, MARKERS, get_color, get_marker


def plot_roofline(out_dir: str = "results/figures"):
    """Plot Figure 7: Roofline Model."""
    roofline_data = build_roofline_data(
        matmul_m=128, matmul_k=1024, matmul_n=1024
    )

    fig, ax = plt.subplots(figsize=(11, 7))

    # ── Draw roofline ceilings ────────────────────────────────────────────────
    I_range = np.logspace(-2, 4, 500)
    bw_line = PEAK_BW_TB_S * 1e12 * I_range / 1e12   # TOPs

    precision_ceilings = {
        "INT4 Peak": (PEAK_COMPUTE_TOPS["int4"], PALETTE["INT4"],     ":"),
        "INT8 Peak": (PEAK_COMPUTE_TOPS["int8"], PALETTE["INT8"],     "-."),
        "FP8 Peak":  (PEAK_COMPUTE_TOPS["fp8"],  PALETTE["MXFP8"],   "--"),
        "FP16 Peak": (PEAK_COMPUTE_TOPS["fp16"], PALETTE["BF16"],    "-"),
        "FP32 Peak": (PEAK_COMPUTE_TOPS["fp32"], PALETTE["FP32"],    "-"),
    }

    for label, (peak, color, ls) in precision_ceilings.items():
        ax.axhline(peak, color=color, linestyle=ls, linewidth=1.2,
                   alpha=0.6, label=f"{label} ({peak} TOPs)")

    # BW-limited diagonal
    ax.plot(I_range, bw_line, "k-", linewidth=2, label=f"Mem BW = {PEAK_BW_TB_S} TB/s")

    # Shade memory-bound region
    # For each precision, BW-limited up to ridge point
    for label, (peak, color, ls) in precision_ceilings.items():
        r = peak / (PEAK_BW_TB_S * 1e12 / 1e12)
        ax.fill_between(
            I_range[I_range < r],
            bw_line[I_range < r],
            peak,
            alpha=0.03, color=color
        )

    # ── Plot format working points ────────────────────────────────────────────
    for pt in roofline_data:
        fmt = pt["format"]
        I = pt["arithmetic_intensity"]
        perf = pt["attainable_tops"]
        is_mem_bound = pt["is_memory_bound"]

        color = get_color(fmt)
        marker = get_marker(fmt)

        edge_color = "red" if is_mem_bound else "white"
        ax.scatter(I, perf, c=color, marker=marker, s=120, zorder=6,
                   edgecolors=edge_color, linewidths=1.2)

        # Label
        ax.annotate(
            fmt, (I, perf),
            xytext=(5, 4), textcoords="offset points",
            fontsize=7.5, color=color,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none")
        )

    # ── Annotations ───────────────────────────────────────────────────────────
    ax.annotate(
        "← Memory-Bound\n(BW-limited)",
        xy=(0.02, 5), xycoords="data", fontsize=9, color="gray",
        style="italic"
    )
    ax.annotate(
        "Compute-Bound →",
        xy=(200, 5), xycoords="data", fontsize=9, color="gray",
        style="italic"
    )

    # Highlight that MX metadata shifts formats LEFT
    mx_pts = [p for p in roofline_data if "MX" in p["format"]]
    if mx_pts:
        xs = [p["arithmetic_intensity"] for p in mx_pts]
        ys = [p["attainable_tops"] for p in mx_pts]
        ax.annotate(
            "MX Block Scale\nmetadata pushes\nI leftward →",
            xy=(min(xs), min(ys)),
            xytext=(min(xs) * 5, min(ys) * 0.5),
            fontsize=8.5, color="darkred",
            arrowprops=dict(arrowstyle="->", color="darkred", lw=1.2)
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOPs / Byte)", fontsize=12)
    ax.set_ylabel("Attainable Performance (TOPs)", fontsize=12)
    ax.set_title(
        "Figure 7: Roofline Analysis — Hardware Formats vs. Transform-Based Approaches\n"
        "(Red border = memory-bound; markers inside BW diagonal are memory-limited)",
        fontsize=12
    )
    ax.set_xlim(1e-2, 1e4)
    ax.set_ylim(0.5, 1000)

    # Legend
    mem_bound_patch = mpatches.Patch(edgecolor="red", facecolor="none", label="Memory-Bound (red border)")
    comp_bound_patch = mpatches.Patch(edgecolor="white", facecolor="gray", label="Compute-Bound")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [mem_bound_patch, comp_bound_patch],
              labels + ["Memory-Bound (red border)", "Compute-Bound"],
              loc="lower right", fontsize=8, ncol=2)

    save_fig(fig, "fig07_roofline", out_dir)
    return fig


if __name__ == "__main__":
    plot_roofline()
