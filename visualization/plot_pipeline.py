"""Figure 10: Hardware Pipeline Latency Breakdown.

Stacked horizontal bar chart for key schemes:
  Scheme A:  MXINT4 (block-scale integer array)
  Scheme A8: MXINT8
  Scheme B:  INT4 + Hadamard (FWHT butterfly unit)
  Scheme B8: INT8 + Hadamard
  Scheme B+: INT4 + Hadamard + SQ Gather/Scatter
  SQ-only:   INT4 + SQ (no Hadamard, reference)

Stage latencies modelled at 45nm:
  Gate delay ≈ 40 ps, SRAM read ≈ 80 ps.
  Hadamard butterfly is pipelined and overlaps with memory access.
  MXINT block-max comparator tree is the dominant latency in Scheme A.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from visualization.style import save_fig, PALETTE


# Stage color scheme (shared across all schemes)
_CLR = {
    "mem":      "#94A3B8",   # gray
    "scale":    "#F87171",   # red   — block-max / scale logic (MXINT bottleneck)
    "had":      "#4ADE80",   # light green — Hadamard butterfly (pipelined)
    "int_mul":  "#22C55E",   # green — INT multiply
    "accum":    "#16A34A",   # dark green — accumulate
    "sq":       "#0D9488",   # teal  — SQ Gather/Scatter
    "write":    "#94A3B8",   # gray
}

_SCHEMES = [
    {
        "name": "Scheme A:  MXINT4",
        "color_key": "MXINT4",
        "stages": [
            ("SRAM Weight Load",             80,  _CLR["mem"]),
            ("Block-Max Comparator Tree",    180, _CLR["scale"]),   # DOMINANT bottleneck
            ("E8M0 Scale Broadcast",          80, _CLR["scale"]),
            ("INT4 Multiply",                 80, _CLR["int_mul"]),
            ("Accumulate",                    80, _CLR["accum"]),
            ("Write Back",                    40, _CLR["write"]),
        ],
    },
    {
        "name": "Scheme A8: MXINT8",
        "color_key": "MXINT8",
        "stages": [
            ("SRAM Weight Load",              80, _CLR["mem"]),
            ("Block-Max Comparator Tree",    200, _CLR["scale"]),
            ("E8M0 Scale Broadcast",          80, _CLR["scale"]),
            ("INT8 Multiply",                 80, _CLR["int_mul"]),
            ("Accumulate",                    80, _CLR["accum"]),
            ("Write Back",                    40, _CLR["write"]),
        ],
    },
    {
        "name": "Scheme B:  INT4 + Hadamard",
        "color_key": "HAD+INT4(C)",
        "stages": [
            ("SRAM Weight/Act Load",          80, _CLR["mem"]),
            ("Hadamard Butterfly (pipeln'd)", 40, _CLR["had"]),    # overlaps with load
            ("INT4 Quant Scale (POT)",         40, _CLR["scale"]),
            ("INT4 Multiply",                  80, _CLR["int_mul"]),
            ("Accumulate",                     80, _CLR["accum"]),
            ("Inverse Hadamard (pipeln'd)",    40, _CLR["had"]),
            ("Write Back",                     40, _CLR["write"]),
        ],
    },
    {
        "name": "Scheme B8: INT8 + Hadamard",
        "color_key": "HAD+INT8(C)",
        "stages": [
            ("SRAM Weight/Act Load",           80, _CLR["mem"]),
            ("Hadamard Butterfly (pipeln'd)",  40, _CLR["had"]),
            ("INT8 Quant Scale (POT)",          40, _CLR["scale"]),
            ("INT8 Multiply",                   80, _CLR["int_mul"]),
            ("Accumulate",                      80, _CLR["accum"]),
            ("Inverse Hadamard (pipeln'd)",     40, _CLR["had"]),
            ("Write Back",                      40, _CLR["write"]),
        ],
    },
    {
        "name": "Scheme B+: INT4 + Hadamard + SQ",
        "color_key": "HAD+SQ",
        "stages": [
            ("SRAM Weight/Act Load",            80, _CLR["mem"]),
            ("Hadamard Butterfly (pipeln'd)",   40, _CLR["had"]),
            ("SQ Gather (salient 1%)",           80, _CLR["sq"]),
            ("INT4 Multiply (dense 99%)",        80, _CLR["int_mul"]),
            ("INT8 Multiply (sparse 1%)",        80, _CLR["int_mul"]),
            ("Accumulate (merged)",              80, _CLR["accum"]),
            ("Inverse Hadamard + Scatter",       80, _CLR["sq"]),
            ("Write Back",                       40, _CLR["write"]),
        ],
    },
    {
        "name": "SQ-only:  INT4 + SQ (no HAD)",
        "color_key": "SQ-Format",
        "stages": [
            ("SRAM Weight/Act Load",            80, _CLR["mem"]),
            ("SQ Gather (salient 1%)",           80, _CLR["sq"]),
            ("INT4 Multiply (dense 99%)",        80, _CLR["int_mul"]),
            ("INT8 Multiply (sparse 1%)",        80, _CLR["int_mul"]),
            ("Accumulate (merged)",              80, _CLR["accum"]),
            ("SQ Scatter",                       40, _CLR["sq"]),
            ("Write Back",                       40, _CLR["write"]),
        ],
    },
]


def plot_pipeline_breakdown(out_dir: str = "results/figures"):
    """Plot Figure 10: Hardware Pipeline Latency Breakdown."""
    fig, ax = plt.subplots(figsize=(15, 7))
    with plt.rc_context({"figure.constrained_layout.use": False}):
        fig, ax = plt.subplots(figsize=(15, 7))

    y_pos = np.arange(len(_SCHEMES))
    bar_h = 0.55

    for yi, scheme in enumerate(_SCHEMES):
        x = 0
        for stage_name, dur_ps, color in scheme["stages"]:
            ax.barh(y_pos[yi], dur_ps, left=x, height=bar_h,
                    color=color, edgecolor="white", linewidth=0.5)
            if dur_ps >= 55:
                ax.text(x + dur_ps / 2, y_pos[yi],
                        f"{stage_name[:20]}\n{dur_ps}ps",
                        ha="center", va="center", fontsize=6.2,
                        color="white" if color != _CLR["mem"] else "black",
                        fontweight="bold")
            x += dur_ps

        total_ps = sum(d for _, d, _ in scheme["stages"])
        c = PALETTE.get(scheme["color_key"], "#888888")
        ax.text(x + 12, y_pos[yi],
                f"{total_ps} ps  ({total_ps/1000:.2f} ns)",
                va="center", ha="left", fontsize=9, color=c, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([s["name"] for s in _SCHEMES], fontsize=10)
    ax.invert_yaxis()

    # Separator between Scheme A and Scheme B
    ax.axhline(1.5, color="black", linewidth=1.5, alpha=0.5)
    ax.text(-15, 0.9, "Scheme A\n(MXINT)", ha="right", fontsize=9,
            color="navy", fontweight="bold")
    ax.text(-15, 2.1, "Scheme B\n(INT+HAD)", ha="right", fontsize=9,
            color="darkgreen", fontweight="bold")

    ax.axvline(400, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.text(400, len(_SCHEMES) - 0.3, "400 ps\n(2.5 GHz budget)",
            ha="center", fontsize=7.5, color="gray")

    ax.set_xlabel("Pipeline Stage Latency (ps, critical path) — 45nm", fontsize=11)
    ax.set_title(
        "Figure 10: Hardware Pipeline Latency Breakdown\n"
        "(Block-Max Comparator Tree in MXINT is dominant bottleneck; "
        "Hadamard butterfly is pipelined and nearly free)",
        fontsize=12,
    )

    legend_elements = [
        mpatches.Patch(facecolor=_CLR["mem"],     label="Memory Access"),
        mpatches.Patch(facecolor=_CLR["scale"],   label="Scale / Block-Max Logic"),
        mpatches.Patch(facecolor=_CLR["had"],     label="Hadamard Butterfly (pipelined)"),
        mpatches.Patch(facecolor=_CLR["int_mul"], label="INT Multiply"),
        mpatches.Patch(facecolor=_CLR["accum"],   label="Accumulate"),
        mpatches.Patch(facecolor=_CLR["sq"],      label="SQ Gather/Scatter"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8.5, ncol=3)

    max_total = max(sum(d for _, d, _ in s["stages"]) for s in _SCHEMES)
    ax.set_xlim(0, max_total * 1.18)

    fig.subplots_adjust(left=0.22, right=0.97, top=0.90, bottom=0.10)
    save_fig(fig, "fig10_pipeline_breakdown", out_dir)
    return fig


if __name__ == "__main__":
    plot_pipeline_breakdown()
