"""Figure 7: Roofline Model — Key Formats Only.

Classical roofline analysis with a focused set of formats:
  FP32, INT8, INT4, MXINT4, MXINT8, HAD+INT4(C), HAD+INT8(C), SQ-Format

X-axis: Arithmetic Intensity (FLOPs / Byte, log scale).
Y-axis: Attainable Performance (TOPs, log scale).

Key insight:
  MXINT4 arithmetic intensity < MXINT8 and < HAD+INT4, because:
    - MXINT4 pays +0.25 bits/element metadata (block scale) → more bytes
    - HAD+INT adds compute ops (FWHT: N log N additions) but NO extra bytes
    - Moving right on the roofline = more compute per byte = better bandwidth util
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import PEAK_COMPUTE_TOPS, PEAK_BW_TB_S
from visualization.style import save_fig, PALETTE, get_color, get_marker


# ── Roofline working points for focus formats ─────────────────────────────────
# Matmul shape: M=128, K=1024, N=1024
# FLOPs = 2 * M * K * N (multiply-add)
# Bytes = bytes_weights + bytes_activations + bytes_output
# MXINT adds 0.25 bits/elem metadata → 0.25/8 extra bytes per weight element
# HAD adds N*log2(N) additions per activation tile but no extra memory

_MATMUL_M = 128
_MATMUL_K = 1024
_MATMUL_N = 1024
_FLOPS = 2 * _MATMUL_M * _MATMUL_K * _MATMUL_N

def _ai(bits_w: float, bits_a: float, bits_o: float = 32) -> float:
    """Arithmetic intensity for a matmul (FLOPs / Byte)."""
    bytes_w = _MATMUL_M * _MATMUL_K * bits_w / 8
    bytes_a = _MATMUL_K * _MATMUL_N * bits_a / 8
    bytes_o = _MATMUL_M * _MATMUL_N * bits_o / 8
    total_bytes = bytes_w + bytes_a + bytes_o
    return _FLOPS / total_bytes

def _had_fwht_extra_flops(k: int, log2_k: int) -> float:
    """FWHT adds n/2 * log2(n) add + sub per vector of length k."""
    return 2 * (k // 2) * log2_k  # additions + subtractions


# Format: (label, arithmetic_intensity, peak_compute_key, color_key)
_FORMATS = [
    ("FP32",         _ai(32, 32),   "fp32",  "FP32"),
    ("INT8",         _ai(8,  8),    "int8",  "INT8"),
    ("INT4",         _ai(4,  4),    "int4",  "INT4"),
    ("MXINT8",       _ai(8.25, 8.25), "int8", "MXINT8"),
    ("MXINT4",       _ai(4.25, 4.25), "int4", "MXINT4"),
    ("HAD+INT8(C)",  _ai(8, 8),     "int8",  "HAD+INT8(C)"),   # per-row scale; same bytes
    ("HAD+INT8(T)",  _ai(8, 8),     "int8",  "HAD+INT8(T)"),   # global scale; same bytes, same AI
    ("HAD+INT4(C)",  _ai(4, 4),     "int4",  "HAD+INT4(C)"),   # per-row scale; same bytes
    ("HAD+INT4(T)",  _ai(4, 4),     "int4",  "HAD+INT4(T)"),   # global scale; same bytes, same AI
    ("SQ-Format",    _ai(5.01, 4),  "int4",  "SQ-Format"),     # ~5 effective bits
]

# Adjust HAD arithmetic intensity to account for extra FWHT ops
_FWHT_FLOPS = _had_fwht_extra_flops(_MATMUL_K, int(np.log2(_MATMUL_K)))
_FWHT_FLOPS_N = _had_fwht_extra_flops(_MATMUL_N, int(np.log2(_MATMUL_N)))

def _ai_had(bits: float) -> float:
    flops_matmul = _FLOPS
    flops_fwht = _FWHT_FLOPS + _FWHT_FLOPS_N  # transform weights + activations
    bytes_w = _MATMUL_M * _MATMUL_K * bits / 8
    bytes_a = _MATMUL_K * _MATMUL_N * bits / 8
    bytes_o = _MATMUL_M * _MATMUL_N * 32 / 8
    return (flops_matmul + flops_fwht) / (bytes_w + bytes_a + bytes_o)


def plot_roofline(out_dir: str = "results/figures"):
    """Plot Figure 7: Roofline Model."""
    fig, ax = plt.subplots(figsize=(11, 7))

    # ── Roofline ceilings ─────────────────────────────────────────────────────
    I_range = np.logspace(-1, 4, 500)
    bw_line = PEAK_BW_TB_S * 1e12 * I_range / 1e12   # in TOPs

    # Peak compute ceilings per precision
    ceiling_styles = {
        "INT4 Peak":  (PEAK_COMPUTE_TOPS["int4"],  PALETTE["INT4"],     ":"),
        "INT8 Peak":  (PEAK_COMPUTE_TOPS["int8"],  PALETTE["INT8"],     "-."),
        "FP32 Peak":  (PEAK_COMPUTE_TOPS["fp32"],  PALETTE["FP32"],    "-"),
    }
    for label, (peak, color, ls) in ceiling_styles.items():
        ax.axhline(peak, color=color, linestyle=ls, linewidth=1.2, alpha=0.55,
                   label=f"{label} ({peak} TOPs)")

    # Memory bandwidth line
    ax.plot(I_range, bw_line, "k-", linewidth=2.0,
            label=f"Mem BW = {PEAK_BW_TB_S} TB/s", zorder=3)

    # ── Working points ────────────────────────────────────────────────────────
    ai_had4 = _ai_had(4)
    ai_had8 = _ai_had(8)

    # Override HAD arithmetic intensity with FWHT-adjusted values
    format_ai = {
        "FP32":        _ai(32, 32),
        "INT8":        _ai(8, 8),
        "INT4":        _ai(4, 4),
        "MXINT8":      _ai(8.25, 8.25),
        "MXINT4":      _ai(4.25, 4.25),
        "HAD+INT8(C)": ai_had8,
        "HAD+INT8(T)": ai_had8,   # same AI as (C); quality differs (see SQNR figs)
        "HAD+INT4(C)": ai_had4,
        "HAD+INT4(T)": ai_had4,   # same AI as (C); quality differs (see SQNR figs)
        "SQ-Format":   _ai(5.01, 4),
    }
    format_peak_key = {
        "FP32":        "fp32",
        "INT8":        "int8",
        "INT4":        "int4",
        "MXINT8":      "int8",
        "MXINT4":      "int4",
        "HAD+INT8(C)": "int8",
        "HAD+INT8(T)": "int8",
        "HAD+INT4(C)": "int4",
        "HAD+INT4(T)": "int4",
        "SQ-Format":   "int4",
    }

    for fmt_name, ai in format_ai.items():
        peak = PEAK_COMPUTE_TOPS[format_peak_key[fmt_name]]
        attainable = min(peak, PEAK_BW_TB_S * 1e12 * ai / 1e12)
        is_mem_bound = (PEAK_BW_TB_S * 1e12 * ai / 1e12) < peak

        c = get_color(fmt_name)
        m = get_marker(fmt_name)
        edge = "red" if is_mem_bound else "white"

        ax.scatter(ai, attainable, color=c, marker=m, s=130, zorder=6,
                   edgecolors=edge, linewidths=1.5)
        ax.annotate(fmt_name, (ai, attainable),
                    xytext=(5, 4), textcoords="offset points",
                    fontsize=8.5, color=c, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75, ec="none"))

    # ── Annotation: MXINT vs HAD+INT AI comparison ────────────────────────────
    ax.annotate(
        "MXINT4 ← pushed LEFT\nby +0.25 bpe metadata",
        xy=(format_ai["MXINT4"], min(PEAK_COMPUTE_TOPS["int4"],
            PEAK_BW_TB_S * 1e12 * format_ai["MXINT4"] / 1e12)),
        xytext=(format_ai["MXINT4"] * 0.3,
                PEAK_COMPUTE_TOPS["int4"] * 0.6),
        fontsize=8.5, color="navy",
        arrowprops=dict(arrowstyle="->", color="navy", lw=1.2),
    )
    ax.annotate(
        "HAD+INT4 → extra FLOPs\nmove point rightward",
        xy=(format_ai["HAD+INT4(C)"], min(PEAK_COMPUTE_TOPS["int4"],
            PEAK_BW_TB_S * 1e12 * format_ai["HAD+INT4(C)"] / 1e12)),
        xytext=(format_ai["HAD+INT4(C)"] * 2.5,
                PEAK_COMPUTE_TOPS["int4"] * 0.4),
        fontsize=8.5, color="darkgreen",
        arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.2),
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-1, 1e4)
    ax.set_ylim(0.5, 1000)
    ax.set_xlabel("Arithmetic Intensity (FLOPs / Byte)", fontsize=12)
    ax.set_ylabel("Attainable Performance (TOPs)", fontsize=12)
    ax.set_title(
        "Figure 7: Roofline Model — Key Formats\n"
        "(Red border = memory-bound; MXINT metadata pushes AI leftward)",
        fontsize=12,
    )

    mem_bound_patch = mpatches.Patch(edgecolor="red", facecolor="none",
                                     linewidth=1.5, label="Memory-Bound (red border)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [mem_bound_patch],
              labels + ["Memory-Bound (red border)"],
              loc="lower right", fontsize=8.5, ncol=2)

    save_fig(fig, "fig07_roofline", out_dir)
    return fig


if __name__ == "__main__":
    plot_roofline()
