"""Figure 6: Hardware PPA Bubble Chart — 4-bit and 8-bit panels.

Three hardware paradigms compared side-by-side at 4-bit and 8-bit:
  • MXINT  — Block-scaled integer (E8M0 per 32 elements)
  • BFP    — Butterfly FP: INT + Hadamard transform (HAD+INT)
  • SQ     — Sparse Quantization: INT4/8 dense + INT8 sparse 1% + bitmask

X-axis : Relative hardware area (INT4 MAC array = 1.0×, analytical NAND2 45nm)
Y-axis : SQNR (dB) on channel-outlier σ=50 distribution
Bubble  : Effective memory bandwidth (bits/element incl. metadata overhead)
           — more informative than energy alone, which mainly tracks bit-width

Hardware cost model
-------------------
All entries are analytical NAND2-equivalent estimates at 45 nm.
Entries marked (*) include hardware-unfriendly FP decode operations whose
area/energy penalty is explicitly broken out.

To add a new format: add an entry to _HW_PARAMS_4BIT or _HW_PARAMS_8BIT
and ensure the format exists in build_all_formats().
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from distributions.metrics import snr_db
from formats import build_all_formats
from visualization.style import save_fig, PALETTE


# ── Hardware parameter tables ─────────────────────────────────────────────────
# Keys: area_rel (INT4 array = 1.0), bw_bpe (bits/element incl. metadata),
#       energy_rel (INT4 MAC = 1.0), paradigm ("MXINT" | "BFP" | "SQ" | "ref")
#
# "ref" entries are plotted lightly as grey reference points, not as main
# comparison targets. Remove from _*_FORMATS lists to hide them entirely.

_HW_PARAMS_4BIT: dict[str, dict] = {
    # ── Three focus paradigms ──────────────────────────────────────────────
    "MXINT4": {
        "area_rel": 1.30, "bw_bpe": 4.25, "energy_rel": 1.20,
        "paradigm": "MXINT",
        "note": "Block-max comparator tree (+0.30×); E8M0 per 32 elems (+0.25 b/elem)",
    },
    "HAD+INT4(C)": {
        "area_rel": 1.15, "bw_bpe": 4.00, "energy_rel": 1.05,
        "paradigm": "HAD+INT",
        "note": "FWHT butterfly; per-row POT scale; no metadata BW overhead",
    },
    "HAD+INT4(T)": {
        "area_rel": 1.15, "bw_bpe": 4.00, "energy_rel": 1.05,
        "paradigm": "HAD+INT",
        "note": "FWHT butterfly; single global POT scale; same HW area as (C); lower quality on mixed distributions",
    },
    "SQ-Format": {
        "area_rel": 1.20, "bw_bpe": 5.04, "energy_rel": 1.15,
        "paradigm": "SQ",
        "note": "1-bit mask + 1%×INT8 sparse → 5.04 b/elem; Gather/Scatter unit +0.20×",
    },
    # ── Reference baselines (grey) ─────────────────────────────────────────
    "INT4": {
        "area_rel": 1.00, "bw_bpe": 4.00, "energy_rel": 1.00,
        "paradigm": "ref",
        "note": "Baseline INT4 array, no outlier handling",
    },
    "MXFP4": {
        "area_rel": 1.35, "bw_bpe": 4.25, "energy_rel": 1.30,
        "paradigm": "ref",
        "note": "MXFP4 E2M1; FP element decode adds barrel-shift + exp align logic",
    },
    "NVFP4": {
        # (*) FP32 outer scale → FP32 mul in decode path (hardware-unfriendly)
        "area_rel": 1.21, "bw_bpe": 4.50, "energy_rel": 1.22,
        "paradigm": "ref",
        "note": "(*) FP32 outer scale: FP32 mul in decode (+0.08× area); E8M0/16 (+0.5 b/elem)",
    },
    "NF4": {
        # (**) FP32 absmax dequant multiply per element (hardware-unfriendly)
        "area_rel": 1.18, "bw_bpe": 4.00, "energy_rel": 1.20,
        "paradigm": "ref",
        "note": "(**) FP32 absmax dequant mul per element (+0.13× area)",
    },
    "HAD+SQ": {
        "area_rel": 1.35, "bw_bpe": 5.04, "energy_rel": 1.25,
        "paradigm": "ref",
        "note": "FWHT + SQ Gather/Scatter combined",
    },
    "RandRot+INT4": {
        "area_rel": 9.00, "bw_bpe": 4.00, "energy_rel": 2.00,
        "paradigm": "ref",
        "note": "N×N dense rotation ROM dominates area — impractical reference",
    },
}

_HW_PARAMS_8BIT: dict[str, dict] = {
    # ── Three focus paradigms ──────────────────────────────────────────────
    "MXINT8": {
        "area_rel": 2.55, "bw_bpe": 8.25, "energy_rel": 4.40,
        "paradigm": "MXINT",
        "note": "INT8 array + block-max comparator tree (+0.35×); E8M0 per 32 elems",
    },
    "HAD+INT8(C)": {
        "area_rel": 2.38, "bw_bpe": 8.00, "energy_rel": 4.10,
        "paradigm": "HAD+INT",
        "note": "INT8 FWHT butterfly; per-row POT scale; no metadata BW",
    },
    "HAD+INT8(T)": {
        "area_rel": 2.38, "bw_bpe": 8.00, "energy_rel": 4.10,
        "paradigm": "HAD+INT",
        "note": "INT8 FWHT butterfly; single global POT scale; same HW area as (C); lower quality on mixed distributions",
    },
    "SQ-Format(8b)": {
        "area_rel": 2.40, "bw_bpe": 9.08, "energy_rel": 4.20,
        "paradigm": "SQ",
        "note": "8-bit dense + 1%×INT8 sparse + 1-bit mask → 9.08 b/elem",
    },
    # ── Reference baselines (grey) ─────────────────────────────────────────
    "INT8": {
        "area_rel": 2.20, "bw_bpe": 8.00, "energy_rel": 4.00,
        "paradigm": "ref",
        "note": "Plain INT8, no outlier handling",
    },
    "MXFP8": {
        "area_rel": 2.60, "bw_bpe": 8.25, "energy_rel": 4.50,
        "paradigm": "ref",
        "note": "MXFP8 E4M3; FP decode on critical path",
    },
    "FP32": {
        "area_rel": 12.0, "bw_bpe": 32.0, "energy_rel": 40.0,
        "paradigm": "ref",
        "note": "FP32 baseline — far outside practical region",
    },
}

# Paradigm display properties
_PARADIGM_STYLE: dict[str, dict] = {
    "MXINT":   {"color": "#1E40AF", "marker": "P", "zorder": 8, "label": "MXINT (block-scale INT)"},
    "HAD+INT": {"color": "#15803D", "marker": "o", "zorder": 8, "label": "HAD+INT (Butterfly FP / FWHT)"},
    "SQ":      {"color": "#D97706", "marker": "8", "zorder": 8, "label": "SQ-Format (sparse quant)"},
    "ref":     {"color": "#94A3B8", "marker": "s", "zorder": 4, "label": "Reference"},
}

# Bandwidth range for bubble scaling (clamp FP32 outlier)
_BW_MIN, _BW_MAX = 4.0, 10.0


def _get_quality(
    hw_params: dict[str, dict],
    seed: int = 42,
) -> dict[str, float]:
    """Compute SQNR using a 2D (8×256) row-outlier tensor.

    Row 0 has σ=50 outlier values; rows 1-7 are N(0,1).
    This makes HAD+INT(C) and HAD+INT(T) produce different SQNR values:
      (C) per-row scale → outlier row handled independently → high SQNR
      (T) global scale → outlier row dominates scale → clean rows lose precision
    """
    registry = build_all_formats(dim=256, seed=seed)
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (8, 256)).astype(np.float32)
    x[0, :] = rng.normal(0, 50.0, 256).astype(np.float32)  # row 0 = outlier

    quality: dict[str, float] = {}
    for fmt_name in hw_params:
        if fmt_name not in registry:
            quality[fmt_name] = np.nan
            continue
        try:
            x_q = registry[fmt_name].quantize(x)
            quality[fmt_name] = snr_db(x.ravel(), x_q.ravel())
        except Exception:
            quality[fmt_name] = np.nan
    return quality


def _bubble_size(bw: float) -> float:
    bw_clamped = min(bw, _BW_MAX)
    return 100 + (bw_clamped - _BW_MIN) / (_BW_MAX - _BW_MIN) * 1400


def _draw_panel(
    ax: plt.Axes,
    hw_params: dict[str, dict],
    quality: dict[str, float],
    title: str,
    bits: int,
) -> None:
    """Draw one bubble-chart panel."""
    legend_handles: dict[str, mpatches.Patch] = {}

    for fmt_name, hw in hw_params.items():
        sqnr = quality.get(fmt_name, np.nan)
        if not np.isfinite(sqnr):
            continue

        paradigm = hw["paradigm"]
        style    = _PARADIGM_STYLE[paradigm]
        alpha    = 0.85 if paradigm != "ref" else 0.40
        bsize    = _bubble_size(hw["bw_bpe"])

        ax.scatter(
            hw["area_rel"], sqnr, s=bsize,
            color=style["color"], marker=style["marker"],
            alpha=alpha, zorder=style["zorder"],
            edgecolors="white", linewidths=1.0,
        )

        # Label — only focus paradigms get prominent labels
        fontsize  = 9.0 if paradigm != "ref" else 7.5
        fontweight = "bold" if paradigm != "ref" else "normal"
        # Clean up label: remove (C) suffix for compactness but keep HAD+INT
        label_text = fmt_name.replace("(C)", "").replace("(T)", "")
        # Stagger offsets slightly to reduce overlap
        x_off = 8 if paradigm != "ref" else 5
        y_off = 6 if paradigm != "ref" else 3
        ax.annotate(
            label_text,
            (hw["area_rel"], sqnr),
            xytext=(x_off, y_off), textcoords="offset points",
            fontsize=fontsize, color=style["color"],
            fontweight=fontweight,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85, ec=style["color"],
                      linewidth=0.5 if paradigm != "ref" else 0),
        )

        if paradigm not in legend_handles:
            legend_handles[paradigm] = mpatches.Patch(
                color=style["color"],
                label=style["label"],
            )

    # Ideal region annotation (lower-left corner — away from legend)
    finite_sqnr = [q for q in quality.values() if np.isfinite(q)]
    if finite_sqnr:
        ax.annotate(
            "Ideal: small area\nhigh quality\nsmall bubble →",
            xy=(0.99, 0.04), xycoords="axes fraction",
            fontsize=7.5, color="darkgreen", style="italic", ha="right", va="bottom",
            bbox=dict(boxstyle="round", fc="#e8f5e9", ec="darkgreen", alpha=0.6),
        )

    # Budget reference line at INT4=1.0×
    ax.axvline(1.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.4)

    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel("Relative Area (INT4 array = 1.0×)", fontsize=9)
    ax.set_ylabel("SQNR dB — ChannelOutlier σ=50", fontsize=9)

    # Bandwidth bubble-size legend inside panel
    for bw_val, lbl in [(_BW_MIN, f"{_BW_MIN:.1f} b/elem"), (6.0, "6.0 b/elem"), (_BW_MAX, f"≥{_BW_MAX:.0f} b/elem")]:
        ax.scatter([], [], s=_bubble_size(bw_val), c="gray", alpha=0.4, label=lbl)

    # Build legend — paradigm patches first, then BW sizes
    paradigm_patches = [legend_handles[p] for p in ["MXINT", "HAD+INT", "SQ", "ref"] if p in legend_handles]
    bw_handles, bw_labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=paradigm_patches + bw_handles,
        labels=[h.get_label() for h in paradigm_patches] + bw_labels,
        loc="upper left", fontsize=7.5, ncol=1,
        title="Paradigm  |  BW bubble", title_fontsize=7.5,
        framealpha=0.95,
        bbox_to_anchor=(0.01, 0.99),
    )

    ax.set_xlim(0, max(hw["area_rel"] for hw in hw_params.values()) * 1.12)


def plot_ppa_bubble(out_dir: str = "results/figures", seed: int = 42) -> plt.Figure:
    """Plot Figure 6: two-panel PPA bubble chart (4-bit left, 8-bit right).

    Parameters
    ----------
    out_dir : str
        Output directory for PNG/PDF.
    seed : int
        Random seed for quality evaluation.
    """
    q4 = _get_quality(_HW_PARAMS_4BIT, seed=seed)
    q8 = _get_quality(_HW_PARAMS_8BIT, seed=seed)

    fig, (ax4, ax8) = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=False)

    _draw_panel(ax4, _HW_PARAMS_4BIT, q4, "4-bit: MXINT4 vs HAD+INT4(C/T) vs SQ-Format", bits=4)
    _draw_panel(ax8, _HW_PARAMS_8BIT, q8, "8-bit: MXINT8 vs HAD+INT8(C/T) vs SQ-Format(8b)", bits=8)

    fig.suptitle(
        "Figure 6: Hardware PPA Bubble Chart\n"
        "Bubble ∝ memory bandwidth (b/elem incl. metadata)  ·  "
        "(*) NVFP4 FP32 outer scale  (**) NF4 FP32 dequant — hardware-unfriendly",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.09, wspace=0.28)

    save_fig(fig, "fig06_ppa_bubble", out_dir)
    return fig


if __name__ == "__main__":
    plot_ppa_bubble()
