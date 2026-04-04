"""Figure 10: Hardware Pipeline Latency Breakdown.

Stacked horizontal bar chart for 8-bit key schemes
(4-bit and 8-bit pipeline structure is the same; only latency scales differ):
  Scheme A8: MXINT8
  Scheme B8: INT8 + Hadamard (HAD+INT8)
  Scheme B+: HAD + SQ Gather/Scatter (dual-prec INT dense‖sparse)
  SQ-only:   SQ-Format (no Hadamard, dense INT4 ‖ sparse INT8)

Stage latencies modelled at 45nm:
  Gate delay ≈ 40 ps, SRAM read ≈ 80 ps.
  Hadamard butterfly is pipelined and overlaps with memory access.
  MXINT block-max comparator tree is the dominant latency in Scheme A.

Pipeline correctness notes:
  - Scheme B+ and SQ-only: INT4 (dense 99%) and INT8 (sparse 1%) multiplications
    execute IN PARALLEL on separate hardware paths. They are shown as a single
    "INT MAC (dual prec)" stage with latency = max(INT4, INT8) ≈ 80 ps.
    Modelling them as sequential would overestimate latency by 80 ps.
  - MXINT: Block-max comparator tree (180–200 ps) is the critical-path bottleneck.
    It cannot be pipelined to overlap with MAC because the scale must be known
    before quantized multiplication can begin.
  - HAD inverse: pipelined to overlap with write-back; adds 0 ps to critical path
    in back-to-back tile processing.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from visualization.style import save_fig, PALETTE


# Stage color scheme (shared across all schemes)
_CLR = {
    "mem":      "#94A3B8",   # slate gray   — memory access
    "scale":    "#F87171",   # red          — block-max / scale logic
    "had":      "#4ADE80",   # light green  — Hadamard butterfly (pipelined)
    "int_mul":  "#22C55E",   # green        — INT multiply
    "dual_mul": "#059669",   # emerald      — dual-precision INT MAC (parallel paths)
    "accum":    "#16A34A",   # dark green   — accumulate
    "sq":       "#0D9488",   # teal         — SQ Gather/Scatter
    "write":    "#94A3B8",   # slate gray   — write-back
}

# Each stage: (label, duration_ps, color)
# 4-bit and 8-bit pipeline structure is the same; showing 8-bit as representative.
_SCHEMES = [
    {
        "name": "A8: MXINT8",
        "color_key": "MXINT8",
        "stages": [
            ("SRAM Load",            80,  _CLR["mem"]),
            ("Block-Max Tree",       200, _CLR["scale"]),   # wider comparators → slower
            ("E8M0 Broadcast",        40, _CLR["scale"]),
            ("INT8 Multiply",         80, _CLR["int_mul"]),
            ("Accumulate",            80, _CLR["accum"]),
            ("Write Back",            40, _CLR["write"]),
        ],
    },
    {
        "name": "B8: HAD+INT8",
        "color_key": "HAD+INT8(C)",
        "stages": [
            ("SRAM Load",            80, _CLR["mem"]),
            ("HAD Butterfly",        40, _CLR["had"]),
            ("POT Scale (INT8)",     40, _CLR["scale"]),
            ("INT8 Multiply",        80, _CLR["int_mul"]),
            ("Accumulate",           80, _CLR["accum"]),
            ("Inv HAD (pipelined)",  40, _CLR["had"]),
            ("Write Back",           40, _CLR["write"]),
        ],
    },
    {
        "name": "B+: INT4 + HAD + SQ",
        "color_key": "HAD+SQ",
        "stages": [
            # INT4 (99%) and INT8 (1%) MACs execute in parallel on dedicated HW units.
            # Latency = max(INT4, INT8) ≈ 80 ps. Modelling as sequential is WRONG.
            ("SRAM Load",            80, _CLR["mem"]),
            ("HAD Butterfly",        40, _CLR["had"]),
            ("SQ Gather (1%)",       80, _CLR["sq"]),
            ("INT MAC (dual prec)",  80, _CLR["dual_mul"]),  # INT4‖INT8 in parallel
            ("Accumulate",           80, _CLR["accum"]),
            ("Inv HAD + Scatter",    80, _CLR["sq"]),
            ("Write Back",           40, _CLR["write"]),
        ],
    },
    {
        "name": "SQ: INT4 + SQ (no HAD)",
        "color_key": "SQ-Format",
        "stages": [
            # Same parallel INT4‖INT8 execution model as B+.
            ("SRAM Load",            80, _CLR["mem"]),
            ("SQ Gather (1%)",       80, _CLR["sq"]),
            ("INT MAC (dual prec)",  80, _CLR["dual_mul"]),  # INT4‖INT8 in parallel
            ("Accumulate",           80, _CLR["accum"]),
            ("SQ Scatter",           40, _CLR["sq"]),
            ("Write Back",           40, _CLR["write"]),
        ],
    },
]


def _bar_text(ax, x_start, dur, y, color, label, fontsize=6.5):
    """Draw text inside a bar segment, with smart truncation to avoid overflow."""
    if dur < 40:
        return  # too narrow for any label
    cx = x_start + dur / 2
    if dur <= 40:
        # Only show duration
        text = f"{dur}ps"
    elif dur <= 60:
        # Short label: duration only
        text = f"{dur}ps"
    else:
        # Full label: name + duration on two lines
        # Truncate name to fit within segment width (heuristic)
        max_chars = max(8, int(dur / 8))
        name = label[:max_chars] + ("…" if len(label) > max_chars else "")
        text = f"{name}\n{dur}ps"

    txt_color = "white" if color not in (_CLR["mem"], _CLR["write"]) else "#1e293b"
    ax.text(cx, y, text,
            ha="center", va="center", fontsize=fontsize,
            color=txt_color, fontweight="bold",
            clip_on=True)


def plot_pipeline_breakdown(out_dir: str = "results/figures"):
    """Plot Figure 10: Hardware Pipeline Latency Breakdown."""
    n = len(_SCHEMES)
    fig, ax = plt.subplots(figsize=(15, 6.5), constrained_layout=False)

    y_pos = np.arange(n)
    bar_h = 0.52

    for yi, scheme in enumerate(_SCHEMES):
        x = 0
        for stage_name, dur_ps, color in scheme["stages"]:
            ax.barh(y_pos[yi], dur_ps, left=x, height=bar_h,
                    color=color, edgecolor="white", linewidth=0.6)
            _bar_text(ax, x, dur_ps, y_pos[yi], color, stage_name, fontsize=6.2)
            x += dur_ps

        total_ps = sum(d for _, d, _ in scheme["stages"])
        c = PALETTE.get(scheme["color_key"], "#888888")
        ax.text(x + 10, y_pos[yi],
                f"{total_ps} ps  ({total_ps / 1000:.2f} ns)",
                va="center", ha="left", fontsize=9, color=c, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([s["name"] for s in _SCHEMES], fontsize=9.5)
    ax.invert_yaxis()

    # Scheme-group separator: after MXINT8 (row 0), before HAD+INT8 (row 1)
    ax.axhline(0.5, color="black", linewidth=1.2, alpha=0.45, linestyle="--")

    # Side annotations using axes-fraction coordinates to avoid clipping
    ax.annotate("MXINT\nParadigm", xy=(0, 0.875), xycoords="axes fraction",
                fontsize=8.5, color="navy", fontweight="bold",
                ha="right", va="center",
                xytext=(-8, 0), textcoords="offset points")
    ax.annotate("HAD+INT /\nSQ Paradigm", xy=(0, 0.375), xycoords="axes fraction",
                fontsize=8.5, color="darkgreen", fontweight="bold",
                ha="right", va="center",
                xytext=(-8, 0), textcoords="offset points")

    # 2.5 GHz budget reference line
    budget_ps = 400
    ax.axvline(budget_ps, color="gray", linewidth=0.9, linestyle="--", alpha=0.55)
    ax.text(budget_ps + 4, n - 0.6, f"{budget_ps} ps\n(2.5 GHz)",
            ha="left", fontsize=7.5, color="gray", va="top")

    ax.set_xlabel("Pipeline Stage Latency (ps, critical path) — 45 nm", fontsize=11)
    ax.set_title(
        "Figure 10: Hardware Pipeline Latency Breakdown (8-bit representative)\n"
        "Block-Max Comparator Tree (MXINT) = dominant bottleneck; "
        "Hadamard butterfly pipelined & nearly free  ·  "
        "4-bit pipeline structure is identical\n"
        "B+/SQ-only: INT4 dense ‖ INT8 sparse MACs execute in parallel "
        "(single 'dual prec' stage)",
        fontsize=10,
    )

    legend_elements = [
        mpatches.Patch(facecolor=_CLR["mem"],      label="Memory Access"),
        mpatches.Patch(facecolor=_CLR["scale"],    label="Scale / Block-Max Logic"),
        mpatches.Patch(facecolor=_CLR["had"],      label="Hadamard Butterfly (pipelined)"),
        mpatches.Patch(facecolor=_CLR["int_mul"],  label="INT Multiply (single prec)"),
        mpatches.Patch(facecolor=_CLR["dual_mul"], label="INT MAC dual prec (INT4‖INT8 parallel)"),
        mpatches.Patch(facecolor=_CLR["accum"],    label="Accumulate"),
        mpatches.Patch(facecolor=_CLR["sq"],       label="SQ Gather/Scatter"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8, ncol=2,
              framealpha=0.9)

    max_total = max(sum(d for _, d, _ in s["stages"]) for s in _SCHEMES)
    ax.set_xlim(0, max_total * 1.22)

    fig.subplots_adjust(left=0.20, right=0.97, top=0.87, bottom=0.10)
    save_fig(fig, "fig10_pipeline_breakdown", out_dir)
    return fig


if __name__ == "__main__":
    plot_pipeline_breakdown()
