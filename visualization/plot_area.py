"""Figure 11: Hardware Area Breakdown.

Stacked horizontal bar chart showing relative silicon area for each hardware
scheme, decomposed into functional components. Area is normalised to the
INT4 MAC array baseline = 1.0× (analytical NAND2-equivalent, 45 nm).

Component definitions:
  mac_array   — Core MAC systolic array (multiplier + accumulator PEs).
                INT4: 1.0× (reference). INT8: 2.2× (wider multiply tree).
  block_scale — Block-max comparator tree + E8M0 encoder (MXINT/MXFP only).
                Comparator tree is O(N) in array width; ~0.30× for MXINT4.
  fwht        — Hadamard butterfly network (N=256 transform, amortised over
                all tokens). Pure add/subtract → much cheaper than MAC array.
                ~0.15× for INT4, ~0.18× for INT8 (wider adders).
  sq_gs       — SQ Gather/Scatter unit: compaction network + priority encoder
                + scatter mux array. O(N log N) mux gates.  ~0.20×.
  decoder     — Format-specific decode logic:
                  NVFP4: E2M1 → FP32 decode LUT + BF16 outer scale FP16 mul
                  NF4:   16-level LUT + FP32 absmax multiply
                  FP6:   6→8b barrel-shift decode + FP32 scale multiply
                  MXINT: E8M0 scale broadcast (simple shift, shared with block_scale)
                  INT4/8, HAD: none (POT scale = free right-shift)
  rot_rom     — Dense N×N rotation-matrix ROM (RandRot only).
                N=256: 256×256×32b ≈ 2 Mbit of SRAM, extremely area-expensive.

All entries marked (*) contain hardware-unfriendly operations (non-POT FP scale
multipliers). Their area and energy penalties are explicitly broken out.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from visualization.style import save_fig, PALETTE


# ── Area breakdown per scheme ─────────────────────────────────────────────────
# Each entry: (scheme_label, color_key, {component: relative_area})
# All values relative to INT4 MAC array = 1.0.
#
# Sources:
#   mac_array  — analytical NAND2 model (int_mac_array.py _analytical_int_ppa)
#   block_scale— 32-element comparator tree (log₂(32)=5 stages × 32 comparators)
#   fwht       — analytical NAND2 model (fwht_module.py _analytical_fwht_ppa)
#   sq_gs      — analytical NAND2 model (sq_gather_scatter.py _analytical_sq_ppa)
#   decoder    — estimated from literature / cell-count extrapolation
#   rot_rom    — SRAM macro: N²×32b bits / (SRAM bit density ~8 Mbit/mm² at 45nm)

_AREA_DATA = [
    # ── INT baselines (no overhead beyond MAC array) ──────────────────────────
    ("INT4\n(baseline)",      "INT4",   {"mac_array": 1.00}),
    ("INT8\n(reference)",     "INT8",   {"mac_array": 2.20}),

    # ── MXINT: block-max comparator tree is the dominant overhead ─────────────
    ("MXINT4",                "MXINT4", {"mac_array": 1.00, "block_scale": 0.30}),
    ("MXINT8",                "MXINT8", {"mac_array": 2.20, "block_scale": 0.35}),

    # ── NVFP4 (*): E2M1 decode + BF16 outer-scale FP16 multiply (hw-unfriendly)
    ("NVFP4 (*)\n[FP16 outer scale]",
                              "NVFP4",  {"mac_array": 1.00, "block_scale": 0.05,
                                         "decoder": 0.08, "hw_unfriendly": 0.08}),

    # ── NF4 (*): 16-level LUT + FP32 absmax dequant multiply (hw-unfriendly) ──
    ("NF4 (**)\n[FP32 dequant mul]",
                              "NF4",    {"mac_array": 1.00, "decoder": 0.05,
                                         "hw_unfriendly": 0.13}),

    # ── HAD+INT: FWHT butterfly amortised (very small fraction of array) ──────
    ("HAD+INT4",              "HAD+INT4(C)", {"mac_array": 1.00, "fwht": 0.15}),
    ("HAD+INT8",              "HAD+INT8(C)", {"mac_array": 2.20, "fwht": 0.18}),

    # ── SQ-Format (4-bit dense): Gather/Scatter overhead ─────────────────────
    ("SQ-Format\n(4b dense)", "SQ-Format",   {"mac_array": 1.00, "sq_gs": 0.20}),

    # ── SQ-Format (8-bit dense): larger array + Gather/Scatter ───────────────
    ("SQ-Format(8b)\n(8b dense)", "INT8",    {"mac_array": 2.20, "sq_gs": 0.20}),

    # ── HAD+SQ: FWHT + Gather/Scatter ────────────────────────────────────────
    ("HAD+SQ",                "HAD+SQ",      {"mac_array": 1.00, "fwht": 0.15,
                                               "sq_gs": 0.20}),

    # ── RandRot+INT4: dense N×N ROM dominates (area horror reference) ─────────
    ("RandRot+INT4\n(N=256 ROM)",
                              "RandRot+INT4", {"mac_array": 1.00, "rot_rom": 8.00}),
]

# Component display order (bottom → top of stacked bar)
_COMPONENTS = ["mac_array", "block_scale", "fwht", "sq_gs", "decoder",
               "hw_unfriendly", "rot_rom"]

_COMP_COLORS = {
    "mac_array":      "#1E40AF",   # blue       — core compute (always present)
    "block_scale":    "#F87171",   # red        — MXINT block-max overhead
    "fwht":           "#4ADE80",   # green      — FWHT butterfly (cheap)
    "sq_gs":          "#0D9488",   # teal       — SQ Gather/Scatter
    "decoder":        "#A78BFA",   # violet     — format decode logic (LUT, barrel)
    "hw_unfriendly":  "#F97316",   # orange     — hardware-UNFRIENDLY FP ops (*)
    "rot_rom":        "#DC2626",   # red        — rotation ROM (very expensive)
}

_COMP_LABELS = {
    "mac_array":      "MAC Array (INT multiply + accumulate)",
    "block_scale":    "Block-Max Comparator Tree (MXINT scale logic)",
    "fwht":           "Hadamard Butterfly Unit (FWHT, add/sub only)",
    "sq_gs":          "SQ Gather/Scatter (compaction + priority encoder)",
    "decoder":        "Format Decoder (LUT / barrel-shift)",
    "hw_unfriendly":  "HW-Unfriendly: FP Scale Multiply (*/**)",
    "rot_rom":        "Rotation Matrix ROM (N×N SRAM)",
}


def plot_area_breakdown(out_dir: str = "results/figures"):
    """Plot Figure 11: Hardware Area Breakdown per scheme."""
    n = len(_AREA_DATA)
    fig, ax = plt.subplots(figsize=(13, 7), constrained_layout=False)

    y_pos  = np.arange(n)
    bar_h  = 0.55
    totals = []

    for yi, (label, color_key, comp_dict) in enumerate(_AREA_DATA):
        x = 0.0
        for comp in _COMPONENTS:
            val = comp_dict.get(comp, 0.0)
            if val <= 0:
                continue
            ax.barh(y_pos[yi], val, left=x, height=bar_h,
                    color=_COMP_COLORS[comp], edgecolor="white", linewidth=0.5,
                    zorder=3)
            # Label inside segment if wide enough
            if val >= 0.12:
                cx = x + val / 2
                ax.text(cx, y_pos[yi], f"{val:.2f}×",
                        ha="center", va="center", fontsize=7.0,
                        color="white", fontweight="bold", clip_on=True)
            x += val

        total = sum(comp_dict.values())
        totals.append(total)
        c = PALETTE.get(color_key, "#888888")
        ax.text(x + 0.08, y_pos[yi], f"{total:.2f}×",
                va="center", ha="left", fontsize=9.5,
                color=c, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([d[0] for d in _AREA_DATA], fontsize=9)
    ax.invert_yaxis()

    # Reference lines
    for ref, label, ls in [(1.0, "INT4 = 1.0×", "-"), (2.2, "INT8 = 2.2×", "--")]:
        ax.axvline(ref, color="gray", linewidth=0.9, linestyle=ls, alpha=0.5, zorder=2)
        ax.text(ref + 0.03, -0.6, label, ha="left", va="bottom",
                fontsize=7.5, color="gray")

    # Legend
    legend_patches = [
        mpatches.Patch(color=_COMP_COLORS[c], label=_COMP_LABELS[c])
        for c in _COMPONENTS
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=7.5, ncol=1,
              framealpha=0.9, title="Area component", title_fontsize=8)

    ax.set_xlabel("Relative Silicon Area (INT4 MAC array = 1.0×, analytical NAND2 at 45 nm)",
                  fontsize=11)
    ax.set_title(
        "Figure 11: Hardware Area Breakdown per Scheme\n"
        "(*) NVFP4 BF16 outer scale → FP16 multiplier in decode path (hardware-unfriendly)\n"
        "(**) NF4 FP32 absmax dequant multiply per element (hardware-unfriendly)",
        fontsize=10,
    )

    ax.set_xlim(0, max(totals) * 1.18)
    ax.grid(axis="x", alpha=0.3, zorder=1)

    fig.subplots_adjust(left=0.22, right=0.97, top=0.88, bottom=0.09)
    save_fig(fig, "fig11_area_breakdown", out_dir)
    return fig


if __name__ == "__main__":
    plot_area_breakdown()
