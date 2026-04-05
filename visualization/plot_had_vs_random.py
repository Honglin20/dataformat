"""Figure 5: MXINT vs HAD+INT vs SQ — SQNR vs. Outlier Severity.

Main ablation figure for the research question:
  "For handling outliers, should hardware use MXINT block-scale or HAD+INT?"

X-axis: Outlier severity (spiky multiplier, log scale).
Y-axis: SQNR in dB (higher = better).

Two panels:
  Left:  4-bit formats: INT4, MXINT4, NVFP4, NF4, HAD+INT4(C), HAD+INT4(T),
                        SQ-Format, HAD+SQ, RandRot+INT4
  Right: 8-bit formats: INT8, MXINT8, HAD+INT8(C), HAD+INT8(T), RandRot+INT8

Key insight box: WHY HAD >= RandRot precision:
  HAD spreads energy PERFECTLY UNIFORMLY — for a single-channel-outlier input,
  every output element has the same absolute value (|H_ij| = 1/sqrt(N) * amplitude).
  Random rotation produces approximately Normal output where the max element is
  ~sqrt(log N) / sqrt(N) * amplitude — slightly larger than HAD's 1/sqrt(N).
  This means HAD's quantizer scale is smaller → finer steps → better SQNR.
"""

import numpy as np
import matplotlib.pyplot as plt

from formats import build_all_formats, FOCUS_4BIT, FOCUS_8BIT
from distributions.metrics import snr_db
from visualization.style import save_fig, PALETTE, MARKERS, LINESTYLES, get_color, get_marker, get_linestyle


_SWEEP_MULTS = [1, 2, 5, 10, 20, 50, 100, 200]
# 2D tensor shape: (batch=16, features=64) = 1024 total elements.
# Row-level outliers: row 0 is multiplied by the sweep multiplier.
# This is the minimal setup to distinguish HAD+INT(C) from HAD+INT(T):
#   - (C) per-row scale: row 0 gets a large scale; rows 1-15 keep small scales → clean rows stay precise
#   - (T) per-tensor scale: global scale dominated by row 0 → all rows lose precision
_BATCH = 16
_FEATURES = 64   # power of 2; HAD transform acts on last axis
_N = _BATCH * _FEATURES  # 1024
_SEED = 42

# Formats to show in each panel
_FMT_4BIT = ["INT4", "MXINT4", "NVFP4", "NF4",
             "HAD+INT4(C)", "HAD+INT4(T)", "SQ-Format", "HAD+SQ", "RandRot+INT4"]
_FMT_8BIT = ["INT8", "MXINT8", "HAD+INT8(C)", "HAD+INT8(T)", "SQ-Format(8b)", "RandRot+INT8"]


def _make_2d_outlier(mult: float, seed: int) -> np.ndarray:
    """2D (batch=16, features=64) tensor where row 0 has `mult`× magnitude.

    HAD+INT(C) uses per-row scale → adapts to outlier row independently.
    HAD+INT(T) uses global scale → normal rows (1-15) lose precision.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (_BATCH, _FEATURES)).astype(np.float32)
    x[0, :] *= float(mult)
    return x


def _run_sweep(fmt_names: list, seed: int = _SEED) -> dict:
    """Sweep outlier severity; use 2D row-outlier tensor so (C) ≠ (T)."""
    all_formats = build_all_formats(dim=_FEATURES, seed=seed)
    results = {f: [] for f in fmt_names if f in all_formats}

    for mult in _SWEEP_MULTS:
        x = _make_2d_outlier(mult, seed)
        for fmt_name in results:
            fmt = all_formats[fmt_name]
            try:
                x_q = fmt.quantize(x)
                results[fmt_name].append(snr_db(x.ravel(), x_q.ravel()))
            except Exception:
                results[fmt_name].append(np.nan)

    return results


def _plot_panel(ax, results: dict, title: str, fmt_names: list):
    """Draw one panel of the SQNR sweep."""
    for fmt_name in fmt_names:
        if fmt_name not in results:
            continue
        vals = np.array(results[fmt_name], dtype=float)
        c = get_color(fmt_name)
        m = get_marker(fmt_name)
        ls = get_linestyle(fmt_name)
        ax.plot(_SWEEP_MULTS, vals,
                color=c, linestyle=ls, marker=m, markersize=6,
                linewidth=2.0, label=fmt_name, zorder=5)

    # "Useful quality" threshold
    ax.axhline(20, color="gray", linewidth=1.0, linestyle=":",
               alpha=0.7, label="20 dB threshold")

    ax.set_xscale("log")
    ax.set_xticks(_SWEEP_MULTS)
    ax.set_xticklabels([str(m) for m in _SWEEP_MULTS])
    ax.set_xlabel("Spike Multiplier (×)", fontsize=11)
    ax.set_ylabel("SQNR (dB)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, loc="upper right", ncol=1)


def plot_had_vs_mxint(out_dir: str = "results/figures", seed: int = _SEED):
    """Plot Figure 5: MXINT vs HAD+INT vs SQ — SQNR vs. Outlier Severity."""
    results_4bit = _run_sweep(_FMT_4BIT, seed=seed)
    results_8bit = _run_sweep(_FMT_8BIT, seed=seed)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    _plot_panel(
        axes[0], results_4bit,
        "4-bit: SQNR vs. Outlier Row Magnitude\n"
        "(2D 16×64, row 0 = outlier; (C)=per-row scale, (T)=global scale)",
        _FMT_4BIT,
    )
    _plot_panel(
        axes[1], results_8bit,
        "8-bit: SQNR vs. Outlier Row Magnitude\n"
        "(2D 16×64, row 0 = outlier; (C)=per-row scale, (T)=global scale)",
        _FMT_8BIT,
    )

    # Explanatory annotation on the 4-bit panel
    axes[0].text(
        0.03, 0.05,
        "Why HAD ≥ RandRot precision:\n"
        "HAD → all outputs same |magnitude|\n"
        "→ tighter scale → finer steps.\n"
        "RandRot → Gaussian output, slightly\n"
        "larger max → coarser scale.",
        transform=axes[0].transAxes,
        fontsize=7.5, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                  edgecolor="goldenrod", alpha=0.85),
    )

    fig.suptitle(
        "Figure 5: MXINT vs. HAD+INT vs. SQ-Format — Quantization Quality vs. Outlier Severity",
        fontsize=13, y=1.02,
    )

    save_fig(fig, "fig05_had_vs_mxint_comparison", out_dir)
    return fig


if __name__ == "__main__":
    plot_had_vs_mxint()
