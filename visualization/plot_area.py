"""Figure 11: Hardware Area Breakdown — 4-bit and 8-bit panels.

Stacked horizontal bar chart showing relative silicon area decomposed into
functional components. Two panels: 4-bit paradigms (top) and 8-bit paradigms
(bottom). Comparison focuses on the three key hardware paradigms:

  • MXINT  — Block-scaled integer (block-max comparator tree)
  • BFP    — Butterfly FP / HAD+INT (Hadamard butterfly unit)
  • SQ     — Sparse Quantization (Gather/Scatter unit)

Area normalised to INT4 MAC array = 1.0× (analytical NAND2 at 45 nm).

Component definitions
---------------------
  mac_array     Core INT multiply + accumulate PEs.
                INT4 = 1.0× (reference). INT8 = 2.2× (wider multiplier tree).
  block_scale   Block-max comparator tree + E8M0 encoder (MXINT only).
                O(N) comparators; cannot be pipelined off critical path.
  fwht          Hadamard butterfly network (N=256). Add/subtract only —
                no multipliers. Amortised over all tokens.
  sq_gs         SQ Gather/Scatter: compaction network + priority encoder
                + scatter mux array. O(N log N) mux gates.
  decoder       Format-specific decode logic:
                  NVFP4: FP32 outer scale FP32 mul (*hardware-unfriendly*)
                  NF4  : FP32 absmax dequant mul (**hardware-unfriendly**)
                  MX   : E8M0 scale broadcast (simple shift, shared)
  hw_fp_penalty Explicitly isolated FP-scale hardware penalty for
                hardware-unfriendly formats (*/**).

To add a new format: add an entry to _SCHEMES_4BIT or _SCHEMES_8BIT.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from visualization.style import save_fig, PALETTE


# ── Component definitions ────────────────────────────────────────────────────

_COMPONENTS = ["mac_array", "block_scale", "fwht", "sq_gs", "decoder", "hw_fp_penalty"]

_COMP_COLORS = {
    "mac_array":      "#1E40AF",  # blue    — core compute
    "block_scale":    "#F87171",  # red     — MXINT comparator tree (bottleneck)
    "fwht":           "#4ADE80",  # green   — FWHT butterfly (cheap)
    "sq_gs":          "#0D9488",  # teal    — SQ Gather/Scatter
    "decoder":        "#A78BFA",  # violet  — format decode (LUT, barrel)
    "hw_fp_penalty":  "#F97316",  # orange  — hardware-unfriendly FP scale ops
}

_COMP_LABELS = {
    "mac_array":     "MAC Array (INT multiply + accumulate)",
    "block_scale":   "Block-Max Comparator Tree (MXINT scale logic)",
    "fwht":          "Hadamard Butterfly Unit (FWHT, add/sub only)",
    "sq_gs":         "SQ Gather/Scatter (compaction + priority encoder)",
    "decoder":       "Format Decoder (LUT / barrel-shift / E8M0 broadcast)",
    "hw_fp_penalty": "HW-Unfriendly: FP Scale Mul (* NVFP4 FP32  ** NF4 FP32)",
}


# ── Scheme tables ─────────────────────────────────────────────────────────────
# Each entry: (display_label, palette_key, {component: area_value})
# All values relative to INT4 MAC array = 1.0.
#
# Focus paradigms are listed first; reference baselines follow with lighter styling.

_SCHEMES_4BIT: list[tuple[str, str, dict, bool]] = [
    # (label, palette_key, components, is_focus)
    # ── Three key paradigms ────────────────────────────────────────────────
    ("MXINT4",               "MXINT4",      {"mac_array": 1.00, "block_scale": 0.30},                      True),
    ("BFP / HAD+INT4",       "HAD+INT4(C)", {"mac_array": 1.00, "fwht": 0.15},                              True),
    ("SQ-Format (4b dense)", "SQ-Format",   {"mac_array": 1.00, "sq_gs": 0.20},                             True),
    # ── Reference baselines ────────────────────────────────────────────────
    ("INT4 (plain)",         "INT4",        {"mac_array": 1.00},                                             False),
    ("MXFP4",                "MXFP4",       {"mac_array": 1.00, "block_scale": 0.30, "decoder": 0.05},      False),
    ("NVFP4 (*)",            "NVFP4",       {"mac_array": 1.00, "decoder": 0.05, "hw_fp_penalty": 0.08},    False),
    ("NF4 (**)",             "NF4",         {"mac_array": 1.00, "decoder": 0.05, "hw_fp_penalty": 0.13},    False),
    ("HAD+SQ",               "HAD+SQ",      {"mac_array": 1.00, "fwht": 0.15, "sq_gs": 0.20},               False),
]

_SCHEMES_8BIT: list[tuple[str, str, dict, bool]] = [
    # ── Three key paradigms ────────────────────────────────────────────────
    ("MXINT8",               "MXINT8",      {"mac_array": 2.20, "block_scale": 0.35},                      True),
    ("BFP / HAD+INT8",       "HAD+INT8(C)", {"mac_array": 2.20, "fwht": 0.18},                              True),
    ("SQ-Format (8b dense)", "SQ-Format",   {"mac_array": 2.20, "sq_gs": 0.20},                             True),
    # ── Reference baselines ────────────────────────────────────────────────
    ("INT8 (plain)",         "INT8",        {"mac_array": 2.20},                                             False),
    ("MXFP8",                "MXFP8",       {"mac_array": 2.20, "block_scale": 0.35, "decoder": 0.05},      False),
]


# ── Drawing helper ────────────────────────────────────────────────────────────

def _draw_area_panel(
    ax: plt.Axes,
    schemes: list[tuple[str, str, dict, bool]],
    title: str,
) -> None:
    """Draw one stacked-bar area breakdown panel."""
    n = len(schemes)
    y_pos  = np.arange(n)
    bar_h  = 0.52
    totals: list[float] = []

    for yi, (label, palette_key, comp_dict, is_focus) in enumerate(schemes):
        x = 0.0
        alpha = 1.0 if is_focus else 0.55

        for comp in _COMPONENTS:
            val = comp_dict.get(comp, 0.0)
            if val <= 0:
                continue
            ax.barh(y_pos[yi], val, left=x, height=bar_h,
                    color=_COMP_COLORS[comp], edgecolor="white", linewidth=0.5,
                    alpha=alpha, zorder=3)
            if val >= 0.12:
                txt_color = "white"
                ax.text(x + val / 2, y_pos[yi], f"{val:.2f}×",
                        ha="center", va="center", fontsize=7.0,
                        color=txt_color, fontweight="bold", clip_on=True, alpha=alpha)
            x += val

        total = sum(comp_dict.values())
        totals.append(total)

        c = PALETTE.get(palette_key, "#888888")
        weight = "bold" if is_focus else "normal"
        ax.text(x + 0.06, y_pos[yi], f"{total:.2f}×",
                va="center", ha="left", fontsize=9 if is_focus else 8,
                color=c, fontweight=weight)

    ax.set_yticks(y_pos)
    y_labels = []
    for label, _, _, is_focus in schemes:
        prefix = "▶ " if is_focus else "   "
        y_labels.append(prefix + label)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.invert_yaxis()

    # Separator line after focus paradigms
    n_focus = sum(1 for _, _, _, f in schemes if f)
    if n_focus < n:
        ax.axhline(n_focus - 0.5, color="black", linewidth=1.0,
                   alpha=0.4, linestyle="--", zorder=5)
        ax.text(0.02, n_focus - 0.5, "▲ focus  ▼ reference",
                ha="left", va="bottom", fontsize=7.5, color="gray",
                transform=ax.get_yaxis_transform())

    # Reference lines
    for ref_val, ref_label, ls in [(1.0, "INT4=1.0×", ":"), (2.2, "INT8=2.2×", "--")]:
        ax.axvline(ref_val, color="gray", linewidth=0.9, linestyle=ls,
                   alpha=0.45, zorder=2)
        ax.text(ref_val + 0.02, -0.8, ref_label,
                ha="left", va="bottom", fontsize=7, color="gray")

    max_total = max(totals) if totals else 3.0
    ax.set_xlim(0, max_total * 1.22)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=5)
    ax.set_xlabel("Relative Area (INT4 MAC array = 1.0×, NAND2 45 nm)", fontsize=8.5)
    ax.grid(axis="x", alpha=0.25, zorder=1)


def plot_area_breakdown(out_dir: str = "results/figures") -> plt.Figure:
    """Plot Figure 11: two-panel hardware area breakdown (4-bit top, 8-bit bottom)."""

    fig, (ax4, ax8) = plt.subplots(2, 1, figsize=(13, 10), constrained_layout=False)

    _draw_area_panel(ax4, _SCHEMES_4BIT, "4-bit: MXINT4 vs BFP (HAD+INT4) vs SQ-Format")
    _draw_area_panel(ax8, _SCHEMES_8BIT, "8-bit: MXINT8 vs BFP (HAD+INT8) vs SQ-Format(8b)")

    # Shared component legend at bottom
    legend_patches = [
        mpatches.Patch(color=_COMP_COLORS[c], label=_COMP_LABELS[c])
        for c in _COMPONENTS
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center", fontsize=8, ncol=3,
        bbox_to_anchor=(0.5, 0.0),
        title="Area component", title_fontsize=8.5,
        framealpha=0.95,
    )

    fig.suptitle(
        "Figure 11: Hardware Area Breakdown per Scheme\n"
        "HAD+INT (C) and (T) share identical hardware — (C)/(T) scale granularity "
        "does not change silicon area  ·  ▶ = focus paradigm (MXINT / HAD+INT / SQ)\n"
        "(*) NVFP4: FP32 outer scale → FP32 decode mul (hardware-unfriendly)  "
        "(**) NF4: FP32 absmax dequant mul per element (hardware-unfriendly)",
        fontsize=9.0,
    )
    fig.subplots_adjust(left=0.22, right=0.97, top=0.91, bottom=0.14, hspace=0.38)

    save_fig(fig, "fig11_area_breakdown", out_dir)
    return fig


if __name__ == "__main__":
    plot_area_breakdown()
