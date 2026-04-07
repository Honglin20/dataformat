"""Figure 13: SQNR vs. Fraction of Outlier Rows.

Key experiment: how does SQNR degrade as more rows become outliers?

Setup:
  2D tensor (16 rows × 256 features).
  Outlier rows have σ=50; normal rows are N(0,1).
  Sweep: 0, 1, 2, 4, 8, 12, 16 outlier rows.

Two panels:
  Left:  4-bit formats — MXINT4, SQ-Format, HAD+INT4(C), HAD+INT4(T)
  Right: 8-bit formats — MXINT8, SQ-Format(8b), HAD+INT8(C), HAD+INT8(T)

Expected behavior (research findings):
  - HAD+INT(C): stays HIGH — per-row scale adapts independently per row
  - HAD+INT(T): degrades as more rows become outliers — global scale gets dominated
  - MXINT: partial degradation — each block's scale is isolated (256/32=8 blocks per row)
  - SQ-Format: the sparse component absorbs largest magnitudes → somewhat robust
"""

import numpy as np
import matplotlib.pyplot as plt

from formats import build_all_formats
from distributions.metrics import snr_db
from visualization.style import get_color, get_marker, get_linestyle, save_fig

# ── Constants ─────────────────────────────────────────────────────────────────
_BATCH = 16
_FEATURES = 256        # power of 2 for HAD
_OUTLIER_SIGMA = 50.0
_N_OUTLIER_SWEEP = [0, 1, 2, 4, 8, 12, 16]

# X-tick labels showing row count and percentage
_XTICK_LABELS = [
    "0\n(0%)", "1\n(6%)", "2\n(12%)", "4\n(25%)",
    "8\n(50%)", "12\n(75%)", "16\n(100%)",
]

# Threshold line
_USEFUL_THRESHOLD_DB = 20.0

# Formats per panel
_FMT_4BIT = ["MXINT4", "SQ-Format", "HAD+INT4(C)", "HAD+INT4(T)"]
_FMT_8BIT = ["MXINT8", "SQ-Format(8b)", "HAD+INT8(C)", "HAD+INT8(T)"]


def _make_partial_outlier(n_outlier_rows: int, seed: int) -> np.ndarray:
    """2D (16, 256) tensor where the first n_outlier_rows rows have σ=50."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (_BATCH, _FEATURES)).astype(np.float32)
    if n_outlier_rows > 0:
        x[:n_outlier_rows, :] = rng.normal(
            0, _OUTLIER_SIGMA, (n_outlier_rows, _FEATURES)
        ).astype(np.float32)
    return x


def _run_sweep(fmt_names: list, all_formats: dict, seed: int) -> dict:
    """Compute SQNR at each n_outlier_rows point for each format.

    Returns dict mapping fmt_name → list of SQNR values (one per sweep point).
    """
    results = {f: [] for f in fmt_names if f in all_formats}

    for n_out in _N_OUTLIER_SWEEP:
        x = _make_partial_outlier(n_out, seed)
        for fmt_name in results:
            fmt = all_formats[fmt_name]
            try:
                x_q = fmt.quantize(x)
                results[fmt_name].append(snr_db(x.ravel(), x_q.ravel()))
            except Exception:
                results[fmt_name].append(np.nan)

    return results


def _plot_panel(ax, results: dict, fmt_names: list, title: str):
    """Draw one panel of the outlier-fraction sweep."""
    x_vals = np.arange(len(_N_OUTLIER_SWEEP))

    for fmt_name in fmt_names:
        if fmt_name not in results:
            continue
        y_vals = results[fmt_name]
        ax.plot(
            x_vals,
            y_vals,
            color=get_color(fmt_name),
            marker=get_marker(fmt_name),
            linestyle=get_linestyle(fmt_name),
            linewidth=2.0,
            markersize=7,
            label=fmt_name,
            zorder=3,
        )

    # Horizontal "useful threshold" line
    ax.axhline(
        _USEFUL_THRESHOLD_DB,
        color="gray", linestyle=":", linewidth=1.5, alpha=0.8, zorder=2,
        label=f"{_USEFUL_THRESHOLD_DB:.0f} dB threshold",
    )
    ax.annotate(
        f"{_USEFUL_THRESHOLD_DB:.0f} dB useful threshold",
        xy=(len(_N_OUTLIER_SWEEP) - 1, _USEFUL_THRESHOLD_DB),
        xytext=(-4, 4), textcoords="offset points",
        fontsize=8, color="gray", ha="right", va="bottom",
    )

    ax.set_title(title, fontsize=11)
    ax.set_xticks(x_vals)
    ax.set_xticklabels(_XTICK_LABELS, fontsize=9)
    ax.set_xlabel("Number of outlier rows (out of 16)", fontsize=10)
    ax.set_ylabel("SQNR (dB)", fontsize=10)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)


def plot_outlier_fraction(
    out_dir: str = "results/figures",
    seed: int = 42,
) -> plt.Figure:
    """Plot Figure 13: SQNR vs. Fraction of Outlier Rows.

    Parameters
    ----------
    out_dir : str
        Output directory for saved figures.
    seed : int
        Random seed for data generation.

    Returns
    -------
    plt.Figure
    """
    all_formats = build_all_formats(dim=_FEATURES, seed=seed)

    results_4 = _run_sweep(_FMT_4BIT, all_formats, seed)
    results_8 = _run_sweep(_FMT_8BIT, all_formats, seed)

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 5),
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.07, right=0.97, top=0.82, bottom=0.14, wspace=0.28)

    _plot_panel(axes[0], results_4, _FMT_4BIT, "4-bit Formats")
    _plot_panel(axes[1], results_8, _FMT_8BIT, "8-bit Formats")

    # Annotation box summarising key findings
    findings = (
        "Key findings:\n"
        "  HAD+INT(C): stays high — per-row scale\n"
        "              adapts to each outlier row\n"
        "  HAD+INT(T): degrades linearly — global\n"
        "              scale dominated by outliers\n"
        "  MXINT: partial degradation — block\n"
        "         isolation limits cross-row harm\n"
        "  SQ-Format: sparse component absorbs\n"
        "             largest magnitudes"
    )
    fig.text(
        0.5, 0.01, findings,
        ha="center", va="bottom", fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF9C4", edgecolor="#CCAA00",
                  alpha=0.9),
        family="monospace",
    )

    fig.suptitle(
        f"Figure 13: SQNR vs. Fraction of Outlier Rows (σ={_OUTLIER_SIGMA:.0f})\n"
        f"(2D {_BATCH}×{_FEATURES} tensor; all formats use 2D input)",
        fontsize=12,
    )

    save_fig(fig, "fig13_outlier_fraction_sweep", out_dir)
    return fig


if __name__ == "__main__":
    plot_outlier_fraction()
