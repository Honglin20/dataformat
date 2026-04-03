"""Figure 6: Hardware PPA Bubble Chart.

All hardware schemes shown as bubbles:
  X-axis: Relative hardware area (normalised to MXINT8 = 1.0).
  Y-axis: SQNR on channel-outlier distribution (quality).
  Bubble size: Relative energy (larger = more power).
  Color: Format family.

Schemes modelled:
  Plain INT:    INT4, INT8 (reference, minimal HW)
  MXINT:        MXINT4, MXINT8 (block-scale HW overhead)
  HAD+INT:      HAD+INT4(C), HAD+INT8(C) (FWHT butterfly unit)
  HAD+SQ:       HAD+SQ (FWHT + Gather/Scatter)
  SQ-Format:    SQ-Format (Gather/Scatter only)
  4-bit extras: NVFP4, NF4
  Upper-bound:  RandRot+INT4 (dense ROM, expensive)

Area model (relative, analytical NAND2 at 45nm):
  INT4  array:  1.0× (reference)
  INT8  array:  2.2× (8/4 ratio + wider adders)
  MXINT overhead: +0.3× (comparator tree for block max + E8M0 encode)
  FWHT butterfly: +0.15× per INT array (small pipelined unit)
  SQ Gather/Scatter: +0.2×
  RandRot ROM (N=1024): +8× (dense N²=1M bits of ROM)
  NF4 LUT decoder: +0.05×
  NVFP4 decoder: +0.05×

Energy model (Horowitz 45nm, pJ per MAC):
  INT4 MAC: 0.05 pJ multiply + 0.05 pJ accumulate
  INT8 MAC: 0.20 pJ multiply + 0.10 pJ accumulate
  SRAM read 8-bit: 1.56 pJ / element (weights)
  FWHT add/sub: 0.01 pJ / op (integer butterfly, ~free)
  SQ overhead: 0.15 pJ / element (gather + scatter)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from distributions.generators import channel_outliers
from distributions.metrics import snr_db
from formats import build_all_formats
from visualization.style import save_fig, PALETTE, get_color, get_marker


# ── Analytical hardware estimates ─────────────────────────────────────────────
# (area_rel: relative to INT4 array; energy_rel: relative to INT4 MAC energy)
_HW_PARAMS = {
    "INT4":          {"area_rel": 1.00, "energy_rel": 1.0,  "family": "Plain INT"},
    "INT8":          {"area_rel": 2.20, "energy_rel": 4.0,  "family": "Plain INT"},
    "MXINT4":        {"area_rel": 1.30, "energy_rel": 1.2,  "family": "MXINT"},
    "MXINT8":        {"area_rel": 2.55, "energy_rel": 4.4,  "family": "MXINT"},
    "NVFP4":         {"area_rel": 1.08, "energy_rel": 1.1,  "family": "HW-Native 4b"},
    "NF4":           {"area_rel": 1.05, "energy_rel": 1.05, "family": "HW-Native 4b"},
    "HAD+INT4(C)":   {"area_rel": 1.15, "energy_rel": 1.05, "family": "HAD+INT"},
    "HAD+INT8(C)":   {"area_rel": 2.38, "energy_rel": 4.1,  "family": "HAD+INT"},
    "HAD+SQ":        {"area_rel": 1.35, "energy_rel": 1.25, "family": "HAD+SQ"},
    "SQ-Format":     {"area_rel": 1.20, "energy_rel": 1.15, "family": "SQ-Format"},
    "RandRot+INT4":  {"area_rel": 9.00, "energy_rel": 2.0,  "family": "RandRot (ref)"},
    "FP32":          {"area_rel": 12.0, "energy_rel": 40.0, "family": "Baseline"},
}

_FAMILY_COLORS = {
    "Plain INT":      "#B45309",
    "MXINT":          "#1E40AF",
    "HW-Native 4b":   "#7C3AED",
    "HAD+INT":        "#15803D",
    "HAD+SQ":         "#0D9488",
    "SQ-Format":      "#D97706",
    "RandRot (ref)":  "#DC2626",
    "Baseline":       "#6B7280",
}


def _get_quality(seed: int = 42, n: int = 2048) -> dict:
    all_formats = build_all_formats(dim=256, seed=seed)
    x, _ = channel_outliers(n=n, outlier_sigma=50.0, seed=seed)
    quality = {}
    for fmt_name in _HW_PARAMS:
        if fmt_name not in all_formats:
            quality[fmt_name] = np.nan
            continue
        try:
            x_q = all_formats[fmt_name].quantize(x)
            quality[fmt_name] = snr_db(x, x_q)
        except Exception:
            quality[fmt_name] = np.nan
    return quality


def plot_ppa_bubble(out_dir: str = "results/figures", seed: int = 42):
    """Plot Figure 6: Hardware PPA Bubble Chart."""
    quality = _get_quality(seed=seed)

    # Normalise energy for bubble size
    all_e = [v["energy_rel"] for v in _HW_PARAMS.values()]
    e_min, e_max = min(all_e), max(all_e)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Build legend handles by family
    family_handles = {}
    plotted = []

    for fmt_name, hw in _HW_PARAMS.items():
        sqnr = quality.get(fmt_name, np.nan)
        if not np.isfinite(sqnr):
            continue

        area = hw["area_rel"]
        energy = hw["energy_rel"]
        family = hw["family"]

        # Bubble size proportional to energy
        bubble_s = 80 + (energy - e_min) / max(e_max - e_min, 1e-9) * 1800

        color = _FAMILY_COLORS.get(family, "#888888")

        sc = ax.scatter(area, sqnr, s=bubble_s, color=color, alpha=0.75,
                        zorder=5, edgecolors="white", linewidths=1.0)

        ax.annotate(
            fmt_name,
            (area, sqnr),
            xytext=(6, 5), textcoords="offset points",
            fontsize=8.5, color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"),
        )
        plotted.append((fmt_name, area, sqnr, energy, family))

        if family not in family_handles:
            family_handles[family] = mpatches.Patch(color=color, label=family)

    # Ideal region annotation
    if plotted:
        ax.annotate(
            "← Ideal Region\n(small area + high quality)",
            xy=(1.05, max(q for _, _, q, _, _ in plotted if np.isfinite(q)) * 0.97),
            fontsize=9, color="darkgreen", style="italic",
            bbox=dict(boxstyle="round", fc="#e8f5e9", ec="darkgreen", alpha=0.6),
        )

    # Bubble size legend
    for e_level, label in [(e_min, "Low energy"), ((e_min + e_max) / 2, "Med energy"), (e_max, "High energy")]:
        bsize = 80 + (e_level - e_min) / max(e_max - e_min, 1e-9) * 1800
        ax.scatter([], [], s=bsize, c="gray", alpha=0.5, label=f"{label} ({e_level:.1f}×)")

    ax.set_xlabel("Relative Hardware Area (INT4 array = 1.0×)", fontsize=12)
    ax.set_ylabel("SQNR (dB) on Channel-Outlier σ=50", fontsize=12)
    ax.set_title(
        "Figure 6: Hardware PPA Bubble Chart\n"
        "(Bubble size ∝ energy; upper-left small bubble = best system design)",
        fontsize=13,
    )

    # Combine legends
    handles_family = list(family_handles.values())
    handles_energy = [h for h in ax.get_legend_handles_labels()[0]]
    ax.legend(
        handles=handles_family + handles_energy,
        loc="lower right", fontsize=8,
        title="Format family  |  Energy scale",
        title_fontsize=8,
        ncol=2,
    )

    ax.set_xlim(0, max(hw["area_rel"] for hw in _HW_PARAMS.values()) * 1.1)

    save_fig(fig, "fig06_ppa_bubble", out_dir)
    return fig


if __name__ == "__main__":
    plot_ppa_bubble()
