"""Figure 10: Hardware Pipeline Latency Breakdown.

Stacked horizontal bar chart showing the clock-cycle breakdown of each pipeline stage
for Scheme A (MXFP array) and Scheme B (INT + FWHT) side by side.

Stages modeled:
  Scheme A (MXFP4/8):
    1. Weight load from SRAM
    2. E8M0 scale read + broadcast
    3. E2M1/E4M3 element decode
    4. Exponent alignment (barrel shifter)
    5. Mantissa multiply
    6. Accumulate
    7. Write back

  Scheme B (INT + FWHT):
    1. Weight/Activation load from SRAM
    2. FWHT butterfly (pipelined, overlaps with load)
    3. INT quantize (per-channel scale)
    4. INT multiply
    5. INT accumulate
    6. Inverse FWHT
    7. Write back

Key point: Scheme A's barrel shifter (stage 4) is the dominant critical-path
contributor. Scheme B's FWHT is pipelined and overlaps with memory load,
making it effectively "free" in throughput terms.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from visualization.style import save_fig, PALETTE


# ── Pipeline stage definitions (in picoseconds, 45nm model) ──────────────────
# Gate delay = 40ps per logic stage. Values are approximate critical-path
# contributions for each pipeline stage.

SCHEME_A_4BIT = {
    "name": "Scheme A: MXFP4 Array",
    "color_base": PALETTE["MXFP4"],
    "stages": [
        ("SRAM Weight Load",          80,  "#94a3b8"),
        ("E8M0 Scale Read+Broadcast", 120, "#f87171"),   # scale broadcast: wider logic
        ("E2M1 Element Decode",        80, "#fb923c"),
        ("Exponent Alignment (Shift)", 200, "#ef4444"),  # DOMINANT: barrel shifter
        ("Mantissa Multiply",          80, "#fca5a5"),
        ("Accumulate (Add)",           80, "#86efac"),
        ("Write Back",                 40, "#94a3b8"),
    ]
}

SCHEME_A_8BIT = {
    "name": "Scheme A: MXFP8 Array",
    "color_base": PALETTE["MXFP8"],
    "stages": [
        ("SRAM Weight Load",           80, "#94a3b8"),
        ("E8M0 Scale Read+Broadcast",  120, "#f87171"),
        ("E4M3 Element Decode",         80, "#fb923c"),
        ("Exponent Alignment (Shift)", 240, "#ef4444"),  # 5-bit barrel shifter, DOMINANT
        ("Mantissa Multiply (4b×4b)",   80, "#fca5a5"),
        ("Accumulate",                  80, "#86efac"),
        ("Write Back",                  40, "#94a3b8"),
    ]
}

SCHEME_B_4BIT = {
    "name": "Scheme B: INT4 + FWHT",
    "color_base": PALETTE["HAD+INT4"],
    "stages": [
        ("SRAM Weight/Act Load",       80,  "#94a3b8"),
        ("FWHT Butterfly (pipelined)", 40,  "#4ade80"),  # overlaps with load → ~free in throughput
        ("INT4 Quant Scale Apply",      40,  "#86efac"),
        ("INT4 Multiply",               80,  "#22c55e"),  # FAST: no exponent alignment
        ("INT Accumulate",              80,  "#16a34a"),
        ("Inverse FWHT (pipelined)",    40,  "#4ade80"),  # also pipelined
        ("Write Back",                  40,  "#94a3b8"),
    ]
}

SCHEME_B_8BIT = {
    "name": "Scheme B: INT8 + FWHT",
    "color_base": PALETTE["HAD+INT8"],
    "stages": [
        ("SRAM Weight/Act Load",        80,  "#94a3b8"),
        ("FWHT Butterfly (pipelined)",  40,  "#4ade80"),
        ("INT8 Quant Scale Apply",       40,  "#86efac"),
        ("INT8 Multiply",                80,  "#22c55e"),
        ("INT Accumulate",               80,  "#16a34a"),
        ("Inverse FWHT (pipelined)",     40,  "#4ade80"),
        ("Write Back",                   40,  "#94a3b8"),
    ]
}

SCHEME_B_PLUS = {
    "name": "Scheme B+: INT4 + FWHT + SQ",
    "color_base": PALETTE["HAD+SQ"],
    "stages": [
        ("SRAM Weight/Act Load",        80,  "#94a3b8"),
        ("FWHT Butterfly (pipelined)",  40,  "#4ade80"),
        ("SQ Gather (salient 1%)",       80,  "#0d9488"),   # Gather adds some latency
        ("INT4 Multiply (dense 99%)",    80,  "#22c55e"),
        ("INT8 Multiply (sparse 1%)",    80,  "#059669"),
        ("Accumulate (merged)",          80,  "#16a34a"),
        ("Inverse FWHT + Scatter",       80,  "#0d9488"),
        ("Write Back",                   40,  "#94a3b8"),
    ]
}


def plot_pipeline_breakdown(out_dir: str = "results/figures"):
    """Plot Figure 10: Pipeline Latency Breakdown."""
    schemes = [
        SCHEME_A_4BIT,
        SCHEME_A_8BIT,
        SCHEME_B_4BIT,
        SCHEME_B_8BIT,
        SCHEME_B_PLUS,
    ]

    fig, ax = plt.subplots(figsize=(14, 7))

    y_positions = np.arange(len(schemes))
    bar_height = 0.55

    for y_idx, scheme in enumerate(schemes):
        x_start = 0
        stage_patches = []
        for stage_name, duration_ps, color in scheme["stages"]:
            bar = ax.barh(
                y_positions[y_idx],
                duration_ps, left=x_start, height=bar_height,
                color=color, edgecolor="white", linewidth=0.5,
                label=stage_name
            )
            # Label inside bar if wide enough
            if duration_ps >= 60:
                ax.text(
                    x_start + duration_ps / 2, y_positions[y_idx],
                    f"{stage_name[:18]}\n{duration_ps}ps",
                    ha="center", va="center", fontsize=6.5,
                    color="white" if color != "#94a3b8" else "black",
                    fontweight="bold"
                )
            x_start += duration_ps

        # Total latency annotation
        total_ps = sum(d for _, d, _ in scheme["stages"])
        ax.text(
            total_ps + 10, y_positions[y_idx],
            f"{total_ps}ps\n({total_ps/1000:.2f}ns)",
            va="center", ha="left", fontsize=8.5, fontweight="bold",
            color=scheme["color_base"]
        )

    # Y-axis labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels([s["name"] for s in schemes], fontsize=10)

    # Vertical lines at key latencies for comparison
    ax.axvline(400, color="navy", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.text(400, len(schemes) - 0.3, "400ps\n(2.5GHz budget)",
            ha="center", fontsize=8, color="navy", alpha=0.7)

    ax.set_xlabel("Pipeline Stage Latency (ps, critical path) — 45nm Technology Node")
    ax.set_title(
        "Figure 10: Hardware Pipeline Latency Breakdown — Scheme A vs. Scheme B\n"
        "(Exponent Alignment in MXFP is the dominant bottleneck; "
        "FWHT butterflies overlap with memory load in Scheme B)",
        fontsize=11
    )

    # Legend for stage types
    legend_elements = [
        mpatches.Patch(facecolor="#94a3b8", label="Memory Access"),
        mpatches.Patch(facecolor="#ef4444", label="Exponent Logic (MXFP only)"),
        mpatches.Patch(facecolor="#4ade80", label="FWHT Butterfly (pipelined)"),
        mpatches.Patch(facecolor="#22c55e", label="INT Compute"),
        mpatches.Patch(facecolor="#fb923c", label="Format Decode"),
        mpatches.Patch(facecolor="#0d9488", label="SQ Gather/Scatter"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8, ncol=3)

    # Horizontal separator between Scheme A and B
    ax.axhline(1.5, color="black", linewidth=1.5, alpha=0.5, linestyle="-")
    ax.text(-20, 1.55, "↑ Scheme A\n(MXFP)", ha="right", fontsize=9,
            color="darkblue", fontweight="bold")
    ax.text(-20, 1.45, "↓ Scheme B\n(INT+HAD)", ha="right", fontsize=9,
            color="darkgreen", fontweight="bold")

    ax.invert_yaxis()
    ax.set_xlim(0, max(
        sum(d for _, d, _ in s["stages"]) for s in schemes
    ) * 1.15)

    save_fig(fig, "fig10_pipeline_breakdown", out_dir)
    return fig


if __name__ == "__main__":
    plot_pipeline_breakdown()
