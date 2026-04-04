"""Figure 6: Hardware PPA Bubble Chart.

All hardware schemes shown as bubbles:
  X-axis: Relative hardware area (normalised to INT4 array = 1.0).
  Y-axis: SQNR on channel-outlier distribution (quality).
  Bubble size: Effective bandwidth (bits/element incl. metadata overhead).
  Color: Format family.

Schemes modelled:
  Plain INT:    INT4, INT8 (reference, minimal HW)
  MXINT:        MXINT4, MXINT8 (block-scale HW overhead)
  HAD+INT:      HAD+INT4(C), HAD+INT8(C) (FWHT butterfly unit)
  HAD+SQ:       HAD+SQ (FWHT + Gather/Scatter)
  SQ-Format:    SQ-Format 4b dense (Gather/Scatter only)
  SQ-Format(8b): SQ-Format 8b dense variant
  4-bit extras: NVFP4 (*), NF4 (**)
  Upper-bound:  RandRot+INT4 (dense ROM, expensive)

Area model (relative, analytical NAND2 at 45nm, INT4 array = 1.0×):
  INT4  array:  1.0× (reference)
  INT8  array:  2.2× (8/4 ratio + wider adders)
  MXINT overhead: +0.30× (comparator tree for block max + E8M0 encode)
  FWHT butterfly: +0.15× per INT array (small pipelined unit)
  SQ Gather/Scatter: +0.20×
  RandRot ROM (N=1024): +8× (dense N²=1M bits of ROM)
  NF4 LUT decoder: +0.05× + FP32 dequant multiply: +0.08× (**hardware-unfriendly**)
  NVFP4 (*): +0.05× (E2M1 decoder) + BF16 outer scale FP16 mul: +0.08×
  FP6: +0.05× (barrel-shift decoder) + FP32 scale mul: +0.08× (**hardware-unfriendly**)

Bandwidth model (bits per weight element including metadata):
  INT4:          4.00 b/elem (no metadata)
  INT8:          8.00 b/elem
  MXINT4:        4.25 b/elem (+0.25 for E8M0 per 32 elements)
  MXINT8:        8.25 b/elem
  NVFP4 (*):     4.50 b/elem (+0.5 for E8M0 per 16 elements, real Blackwell spec)
  NF4 (**):      4.00 b/elem (per-tensor scale negligible)
  SQ-Format:     5.04 b/elem (1%×8b sparse + 4b dense + 1b mask)
  SQ-Format(8b): 9.08 b/elem (1%×8b sparse + 8b dense + 1b mask)
  HAD+INT4:      4.00 b/elem (no extra metadata; HAD is compute, not storage)
  RandRot+INT4:  4.00 b/elem data + N×N ROM (not counted in per-element BW)

(*) NVFP4 real hardware has BF16 outer scale requiring an FP16 mul in decode path —
    hardware-unfriendly overhead not present in pure-INT or MX formats.
(**) NF4 dequantize requires one FP32 multiply per element (absmax × norm_level) —
    hardware-unfriendly vs POT-scale INT formats.

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
# area_rel: relative to INT4 MAC array = 1.0 (analytical NAND2, 45nm)
# energy_rel: relative to INT4 MAC energy = 1.0 (Horowitz model)
# bw_bpe: effective bits per weight element (data + metadata)
#
# Hardware-unfriendly operations increase area/energy beyond what bit-width alone
# would suggest. Marked with (*) in comments below.
_HW_PARAMS = {
    # ── Plain INT (POT scale, hardware-friendly) ──────────────────────────────
    "INT4":   {"area_rel": 1.00, "energy_rel": 1.0,  "bw_bpe": 4.00, "family": "Plain INT"},
    "INT8":   {"area_rel": 2.20, "energy_rel": 4.0,  "bw_bpe": 8.00, "family": "Plain INT"},

    # ── MX block-scaled (E8M0 per 32 elements — hardware-friendly POT scale) ──
    "MXINT4": {"area_rel": 1.30, "energy_rel": 1.2,  "bw_bpe": 4.25, "family": "MXINT"},
    "MXINT8": {"area_rel": 2.55, "energy_rel": 4.4,  "bw_bpe": 8.25, "family": "MXINT"},

    # ── NVFP4: E2M1 element + E8M0 per-16 block scale + BF16 outer scale (*)
    # (*) BF16 outer scale requires FP16 multiplier in decode → +0.08× area,
    #     +0.12× energy vs pure-INT decode. E8M0 block scale: +0.5 b/elem BW.
    "NVFP4":  {"area_rel": 1.21, "energy_rel": 1.22, "bw_bpe": 4.50, "family": "HW-Native 4b"},

    # ── NF4: LUT quantile decode + FP32 dequant scale multiply per element (*)
    # (*) Dequantize = q_norm × absmax (one FP32 mul per element). Not POT.
    #     LUT decode: small ROM. FP32 mul: +0.08× area, +0.12× energy.
    "NF4":    {"area_rel": 1.18, "energy_rel": 1.20, "bw_bpe": 4.00, "family": "HW-Native 4b"},

    # ── HAD+INT: FWHT butterfly (add/sub only, pipelined, nearly free) ────────
    "HAD+INT4(C)": {"area_rel": 1.15, "energy_rel": 1.05, "bw_bpe": 4.00, "family": "HAD+INT"},
    "HAD+INT8(C)": {"area_rel": 2.38, "energy_rel": 4.10, "bw_bpe": 8.00, "family": "HAD+INT"},

    # ── HAD+SQ: FWHT + Gather/Scatter (1% sparse @ INT8, bitmask overhead) ───
    "HAD+SQ":      {"area_rel": 1.35, "energy_rel": 1.25, "bw_bpe": 5.04, "family": "HAD+SQ"},

    # ── SQ-Format 4-bit dense (INT4 dense 99% + INT8 sparse 1% + 1-bit mask) ─
    "SQ-Format":   {"area_rel": 1.20, "energy_rel": 1.15, "bw_bpe": 5.04, "family": "SQ-Format"},

    # ── SQ-Format 8-bit dense (INT8 dense 99% + INT8 sparse 1% + 1-bit mask) ─
    # Larger BW overhead: 8-bit dense dominates. Area adds INT8 array cost.
    "SQ-Format(8b)": {"area_rel": 2.40, "energy_rel": 4.20, "bw_bpe": 9.08, "family": "SQ-Format"},

    # ── RandRot: N×N dense rotation ROM — area dominated by ROM, not array ───
    "RandRot+INT4": {"area_rel": 9.00, "energy_rel": 2.0,  "bw_bpe": 4.00, "family": "RandRot (ref)"},

    # ── FP32 baseline ─────────────────────────────────────────────────────────
    "FP32":         {"area_rel": 12.0, "energy_rel": 40.0, "bw_bpe": 32.0, "family": "Baseline"},
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

# Minimum bits/element for bubble size scaling reference (plain INT4 = 4.0)
_BW_REF_MIN = 4.0   # INT4, no metadata
_BW_REF_MAX = 10.0  # upper reference for scaling (exclude FP32 outlier)


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
    """Plot Figure 6: Hardware PPA Bubble Chart.

    Bubble size ∝ effective memory bandwidth (bits/element including metadata).
    This metric better differentiates formats than energy, because:
      - Energy varies mainly with bit-width (4b vs 8b clusters dominate).
      - Bandwidth overhead is format-specific: SQ-Format has 1-bit bitmask
        overhead, NVFP4 has E8M0 group scale (+0.5 b/elem), MXINT has +0.25
        b/elem from block scales, while NF4/HAD+INT4 have no per-element metadata.
    """
    quality = _get_quality(seed=seed)

    fig, ax = plt.subplots(figsize=(13, 8))

    family_handles = {}
    plotted = []

    for fmt_name, hw in _HW_PARAMS.items():
        sqnr = quality.get(fmt_name, np.nan)
        if not np.isfinite(sqnr):
            continue

        area = hw["area_rel"]
        bw   = hw["bw_bpe"]    # effective bits per element (bandwidth metric)
        family = hw["family"]
        color  = _FAMILY_COLORS.get(family, "#888888")

        # Bubble size: proportional to bandwidth overhead.
        # Clamp FP32 (32 b/elem) to avoid extreme bubble — it's already far right.
        bw_plot = min(bw, _BW_REF_MAX)
        bubble_s = 80 + (bw_plot - _BW_REF_MIN) / (_BW_REF_MAX - _BW_REF_MIN) * 1600

        ax.scatter(area, sqnr, s=bubble_s, color=color, alpha=0.75,
                   zorder=5, edgecolors="white", linewidths=1.0)

        ax.annotate(
            fmt_name,
            (area, sqnr),
            xytext=(6, 5), textcoords="offset points",
            fontsize=8.5, color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"),
        )
        plotted.append((fmt_name, area, sqnr, bw, family))

        if family not in family_handles:
            family_handles[family] = mpatches.Patch(color=color, label=family)

    # Ideal region annotation
    if plotted:
        max_sqnr = max(q for _, _, q, _, _ in plotted if np.isfinite(q))
        ax.annotate(
            "← Ideal Region\n(small area, high quality,\nsmall bubble)",
            xy=(1.05, max_sqnr * 0.97),
            fontsize=9, color="darkgreen", style="italic",
            bbox=dict(boxstyle="round", fc="#e8f5e9", ec="darkgreen", alpha=0.6),
        )

    # Bubble size legend: 3 representative bandwidth points
    for bw_level, label in [
        (_BW_REF_MIN,                   f"4.0 b/elem (INT4, no metadata)"),
        (5.0,                            f"5.0 b/elem (SQ-Format w/ mask)"),
        (min(8.25, _BW_REF_MAX),         f"8.25 b/elem (MXINT8 w/ E8M0)"),
    ]:
        bw_plot = min(bw_level, _BW_REF_MAX)
        bsize = 80 + (bw_plot - _BW_REF_MIN) / (_BW_REF_MAX - _BW_REF_MIN) * 1600
        ax.scatter([], [], s=bsize, c="gray", alpha=0.5, label=label)

    ax.set_xlabel("Relative Hardware Area (INT4 MAC array = 1.0×)", fontsize=12)
    ax.set_ylabel("SQNR (dB) on Channel-Outlier σ=50", fontsize=12)
    ax.set_title(
        "Figure 6: Hardware PPA Bubble Chart\n"
        "(Bubble size ∝ memory bandwidth b/elem; upper-left small bubble = best design point)\n"
        "(*) NVFP4/NF4: hardware-unfriendly FP scale in decode path — see area penalty",
        fontsize=11,
    )

    handles_family = list(family_handles.values())
    handles_bw     = ax.get_legend_handles_labels()[0]
    ax.legend(
        handles=handles_family + handles_bw,
        loc="lower right", fontsize=8,
        title="Format family  |  Bandwidth scale",
        title_fontsize=8,
        ncol=2,
    )

    ax.set_xlim(0, max(hw["area_rel"] for hw in _HW_PARAMS.values()) * 1.08)

    save_fig(fig, "fig06_ppa_bubble", out_dir)
    return fig


if __name__ == "__main__":
    plot_ppa_bubble()
