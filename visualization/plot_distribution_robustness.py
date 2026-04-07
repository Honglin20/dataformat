"""Figure 12: Cross-Distribution Robustness Comparison.

Grouped bar chart showing SQNR (dB) across 8 distribution types for key formats.

Two panels:
  Left:  4-bit formats  — INT4, MXINT4, SQ-Format, HAD+INT4(C), HAD+INT4(T)
  Right: 8-bit formats  — INT8, MXINT8, SQ-Format(8b), HAD+INT8(C), HAD+INT8(T)

Key finding:
  - Distributions 1-5 (no row outlier): HAD+INT(C) ≈ HAD+INT(T)
  - Distributions 6-8 (row outlier):    HAD+INT(C) >> HAD+INT(T)
    The per-row scale in (C) adapts independently; global scale in (T) is dominated
    by the outlier row, hurting all other rows.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as scipy_t

from formats import build_all_formats
from distributions.metrics import snr_db
from visualization.style import get_color, save_fig

# ── Format lists ──────────────────────────────────────────────────────────────
_FMT_4BIT = ["INT4", "MXINT4", "SQ-Format", "HAD+INT4(C)", "HAD+INT4(T)"]
_FMT_8BIT = ["INT8", "MXINT8", "SQ-Format(8b)", "HAD+INT8(C)", "HAD+INT8(T)"]

# ── Tensor shape ─────────────────────────────────────────────────────────────
_BATCH = 16
_FEATURES = 256  # power of 2 for HAD; must match dim in build_all_formats

# ── Distribution definitions ──────────────────────────────────────────────────
# Each entry: (display_label, generator_fn(seed) -> np.ndarray of shape (16,256))
_DIST_LABELS = [
    "Gaussian",
    "Laplace",
    "Student-t",
    "Bimodal",
    "LogNormal",
    "RowOutlier\n(σ=10)",
    "RowOutlier\n(σ=50)",
    "RowOutlier\n(σ=100)",
]

# Index of first "row outlier" distribution (0-indexed) — used for separator line
_OUTLIER_START = 5


def _make_gaussian(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (_BATCH, _FEATURES)).astype(np.float32)


def _make_laplace(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.laplace(0, 1, (_BATCH, _FEATURES)).astype(np.float32)


def _make_student_t(seed: int) -> np.ndarray:
    np.random.seed(seed)
    return scipy_t.rvs(df=3, size=(_BATCH, _FEATURES)).astype(np.float32)


def _make_bimodal(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    half = _FEATURES // 2
    x1 = rng.normal(-3.0, 0.5, (_BATCH, half)).astype(np.float32)
    x2 = rng.normal(3.0, 0.5, (_BATCH, _FEATURES - half)).astype(np.float32)
    x = np.concatenate([x1, x2], axis=1)
    # Shuffle columns so modes are intermixed
    rng.shuffle(x, axis=1)
    return x


def _make_lognormal(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x_pos = rng.lognormal(0, 1, (_BATCH, _FEATURES)).astype(np.float32)
    signs = rng.choice([-1.0, 1.0], size=(_BATCH, _FEATURES)).astype(np.float32)
    return signs * x_pos


def _make_row_outlier(sigma: float, seed: int) -> np.ndarray:
    """2D (16, 256): row 0 has σ=sigma, rows 1-15 are N(0,1)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (_BATCH, _FEATURES)).astype(np.float32)
    x[0, :] = rng.normal(0, sigma, _FEATURES).astype(np.float32)
    return x


_DIST_GENERATORS = [
    _make_gaussian,
    _make_laplace,
    _make_student_t,
    _make_bimodal,
    _make_lognormal,
    lambda s: _make_row_outlier(10.0, s),
    lambda s: _make_row_outlier(50.0, s),
    lambda s: _make_row_outlier(100.0, s),
]


def _compute_sqnr_matrix(fmt_names: list, all_formats: dict, seed: int) -> np.ndarray:
    """Return (n_formats × n_distributions) SQNR matrix."""
    n_fmt = len(fmt_names)
    n_dist = len(_DIST_LABELS)
    matrix = np.full((n_fmt, n_dist), np.nan)

    for j, gen_fn in enumerate(_DIST_GENERATORS):
        x = gen_fn(seed)
        for i, fmt_name in enumerate(fmt_names):
            if fmt_name not in all_formats:
                continue
            try:
                x_q = all_formats[fmt_name].quantize(x)
                matrix[i, j] = snr_db(x.ravel(), x_q.ravel())
            except Exception:
                matrix[i, j] = np.nan

    return matrix


def _draw_panel(
    ax,
    fmt_names: list,
    matrix: np.ndarray,
    title: str,
    bar_width: float = 0.15,
):
    """Draw one grouped-bar panel."""
    n_fmt = len(fmt_names)
    n_dist = len(_DIST_LABELS)
    x_centers = np.arange(n_dist)

    # Center the group of bars around each x position
    offsets = (np.arange(n_fmt) - (n_fmt - 1) / 2.0) * bar_width

    # Hatch patterns: (C) formats get no hatch, (T) formats get '//'
    def _hatch(name: str) -> str:
        return "//" if "(T)" in name else ""

    for i, fmt_name in enumerate(fmt_names):
        color = get_color(fmt_name)
        hatch = _hatch(fmt_name)
        bars = ax.bar(
            x_centers + offsets[i],
            matrix[i],
            width=bar_width,
            color=color,
            hatch=hatch,
            edgecolor="white",
            linewidth=0.5,
            label=fmt_name,
            zorder=3,
        )
        # Value labels on bars (rotated 90°, fontsize 6)
        for bar, val in zip(bars, matrix[i]):
            if np.isfinite(val) and val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.3,
                    f"{val:.0f}",
                    ha="center", va="bottom",
                    fontsize=6, rotation=90, color="black",
                )

    # Red dashed separator between no-outlier and row-outlier distributions
    sep_x = _OUTLIER_START - 0.5
    ax.axvline(sep_x, color="red", linestyle="--", linewidth=1.5, zorder=5, alpha=0.8)
    ax.annotate(
        "← no row outlier | row outlier (C≠T) →",
        xy=(sep_x, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1),
        xycoords=("data", "axes fraction"),
        xytext=(0, -14),
        textcoords="offset points",
        fontsize=7, color="red", ha="center",
        annotation_clip=False,
    )

    ax.set_title(title, fontsize=11)
    ax.set_xticks(x_centers)
    ax.set_xticklabels(_DIST_LABELS, fontsize=8)
    ax.set_ylabel("SQNR (dB)", fontsize=10)
    ax.set_xlabel("Distribution", fontsize=10)
    ax.yaxis.grid(True, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)


def plot_distribution_robustness(
    out_dir: str = "results/figures",
    seed: int = 42,
) -> plt.Figure:
    """Plot Figure 12: Cross-Distribution Robustness Comparison.

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

    mat4 = _compute_sqnr_matrix(_FMT_4BIT, all_formats, seed)
    mat8 = _compute_sqnr_matrix(_FMT_8BIT, all_formats, seed)

    fig, axes = plt.subplots(
        1, 2, figsize=(18, 6),
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.14, wspace=0.28)

    _draw_panel(axes[0], _FMT_4BIT, mat4, "4-bit Formats")
    _draw_panel(axes[1], _FMT_8BIT, mat8, "8-bit Formats")

    # After drawing, add separator annotation properly (axes fraction y is now set)
    for ax in axes:
        sep_x = _OUTLIER_START - 0.5
        ymax = ax.get_ylim()[1]
        # Reposition separator annotation at top of axes
        for child in ax.get_children():
            if hasattr(child, "get_text") and "no row outlier" in child.get_text():
                child.set_position((sep_x, ymax * 0.97))
                break

    fig.suptitle(
        "Figure 12: Cross-Distribution Robustness Comparison\n"
        "(2D 16×256 tensors; SQNR dB — higher = better)",
        fontsize=12,
    )

    save_fig(fig, "fig12_distribution_robustness", out_dir)
    return fig


if __name__ == "__main__":
    plot_distribution_robustness()
