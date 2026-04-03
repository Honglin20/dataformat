"""Figure 1: Distribution Evolution — Histogram + Q-Q plots.

Three-panel layout:
  Left:   Normalized histogram comparison (log y-axis, x clipped to [-15, 15]).
  Middle: Q-Q plots (sorted values vs standard normal quantiles) for all 4.
  Right:  Zoom-in Q-Q showing tail behavior difference (HAD uniform vs RandRot Gaussian).

Key insight annotation: HAD spreads energy perfectly uniformly → equal output
magnitudes for single-channel-outlier input → tighter quantizer scale →
finer quantization steps → better SQNR despite RandRot looking more Gaussian.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm as scipy_norm

from distributions.generators import channel_outliers
from formats.transforms.hadamard import hadamard_transform
from formats.transforms.random_rotation import RandomRotationTransform
from visualization.style import save_fig, PALETTE


def plot_distribution_evolution(
    n: int = 1024,
    seed: int = 42,
    out_dir: str = "results/figures",
):
    """Plot Figure 1: Distribution evolution under HAD and RandRot transforms."""

    # Generate channel-outlier input (σ=50, the most challenging case)
    x_orig, _ = channel_outliers(n=n, outlier_sigma=50.0, seed=seed)

    # Apply HAD transform (normalize=False: hardware-faithful, no division by sqrt(N))
    # Then divide by sqrt(N) for display so energies are comparable
    x_had_raw = hadamard_transform(x_orig, normalize=False)
    x_had_display = x_had_raw / np.sqrt(float(n))

    # Apply RandRot (dense random orthogonal matrix)
    rand_rot = RandomRotationTransform(dim=n, seed=seed)
    x_randrot = rand_rot.forward(x_orig)

    # Ideal Gaussian reference matched to RandRot std
    rng = np.random.default_rng(seed + 1)
    sigma_ref = float(np.std(x_randrot))
    x_gaussian = rng.normal(0.0, sigma_ref, size=n).astype(np.float32)

    # Labels, colors, linestyles for the 4 distributions
    labels = [
        "Original (Channel Outlier σ=50)",
        "After HAD",
        "After RandRot",
        "Ideal Gaussian",
    ]
    arrays = [x_orig, x_had_display, x_randrot, x_gaussian]
    colors = [
        PALETTE["INT4"],          # brown — original
        PALETTE["HAD+INT4(C)"],   # green — HAD
        PALETTE["RandRot+INT4"],  # red   — RandRot
        PALETTE["FP32"],          # grey  — Gaussian
    ]
    linestyles = ["--", "-", "-.", ":"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── Left: Normalized histogram (log y-axis) ──────────────────────────────
    ax = axes[0]
    bins = np.linspace(-15, 15, 80)
    for label, arr, col, ls in zip(labels, arrays, colors, linestyles):
        arr_clipped = np.clip(arr, -15, 15)
        counts, edges = np.histogram(arr_clipped, bins=bins, density=True)
        centres = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centres, np.maximum(counts, 1e-6), label=label,
                color=col, linestyle=ls, linewidth=1.8)

    ax.set_yscale("log")
    ax.set_xlim(-15, 15)
    ax.set_ylim(1e-4, 10)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density (log scale)")
    ax.set_title("(a) Histogram after Transform")
    ax.legend(fontsize=8, loc="upper right")

    # ── Middle: Q-Q plot (full range) ────────────────────────────────────────
    ax2 = axes[1]
    theoretical = np.sort(scipy_norm.ppf(
        np.linspace(1 / (2 * n), 1 - 1 / (2 * n), n)
    ))
    for label, arr, col, ls in zip(labels, arrays, colors, linestyles):
        x_norm = arr.copy().astype(np.float64)
        std = np.std(x_norm)
        if std > 0:
            x_norm = (x_norm - np.mean(x_norm)) / std
        sample_q = np.sort(x_norm)
        ax2.plot(theoretical, sample_q, label=label,
                 color=col, linestyle=ls, linewidth=1.5, alpha=0.85)

    lim = 4.5
    ax2.plot([-lim, lim], [-lim, lim], "k--", linewidth=1.0, alpha=0.4,
             label="y = x (perfect Gaussian)")
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    ax2.set_xlabel("Theoretical Normal Quantiles")
    ax2.set_ylabel("Sample Quantiles (z-normalised)")
    ax2.set_title("(b) Q-Q Plot vs. Standard Normal")
    ax2.legend(fontsize=7, loc="upper left")

    # ── Right: Zoom-in Q-Q showing tail behavior ──────────────────────────────
    ax3 = axes[2]
    theoretical_tail = np.sort(scipy_norm.ppf(
        np.linspace(1 / (2 * n), 1 - 1 / (2 * n), n)
    ))
    # Focus on HAD vs RandRot vs Gaussian in the upper tail
    tail_labels = ["After HAD", "After RandRot", "Ideal Gaussian"]
    tail_arrays = [x_had_display, x_randrot, x_gaussian]
    tail_colors = [PALETTE["HAD+INT4(C)"], PALETTE["RandRot+INT4"], PALETTE["FP32"]]
    tail_ls     = ["-", "-.", ":"]

    for label, arr, col, ls in zip(tail_labels, tail_arrays, tail_colors, tail_ls):
        x_norm = arr.copy().astype(np.float64)
        std = np.std(x_norm)
        if std > 0:
            x_norm = (x_norm - np.mean(x_norm)) / std
        sample_q = np.sort(x_norm)
        ax3.plot(theoretical_tail, sample_q, label=label,
                 color=col, linestyle=ls, linewidth=2.0)

    zoom_lo, zoom_hi = 1.5, 4.5
    ax3.plot([zoom_lo, zoom_hi], [zoom_lo, zoom_hi], "k--",
             linewidth=1.0, alpha=0.4, label="y = x (Gaussian)")
    ax3.set_xlim(zoom_lo, zoom_hi)
    ax3.set_ylim(zoom_lo - 0.5, zoom_hi + 1.5)
    ax3.set_xlabel("Theoretical Normal Quantiles")
    ax3.set_ylabel("Sample Quantiles (z-normalised)")
    ax3.set_title("(c) Tail Zoom: HAD vs RandRot")
    ax3.legend(fontsize=8, loc="upper left")

    # Key insight annotation: HAD >= RandRot precision despite RandRot looking
    # more Gaussian. HAD spreads energy PERFECTLY UNIFORMLY → all output elements
    # have equal magnitude for a single-channel-outlier input → scale = sqrt(N)*sigma
    # (tighter) → finer quantization steps → better SQNR.
    # RandRot output is approximately Normal with slightly larger max (by sqrt(log N))
    # → coarser scale → slightly worse SQNR.
    ax3.text(
        0.03, 0.97,
        "HAD: all outputs equal magnitude\n"
        "→ scale ∝ √N·σ (tighter)\n"
        "→ finer quant steps → better SQNR\n\n"
        "RandRot: ≈ Gaussian, max ∝ √(logN)·σ_had\n"
        "→ slightly coarser scale → slightly\n"
        "  worse SQNR (despite looking Gaussian)",
        transform=ax3.transAxes,
        fontsize=7.5, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                  edgecolor="goldenrod", alpha=0.85),
    )

    fig.suptitle(
        "Figure 1: Distribution Evolution Under Transforms\n"
        "(Channel Outlier σ=50  →  HAD  /  RandRot  /  Ideal Gaussian)",
        fontsize=12, y=1.01,
    )

    save_fig(fig, "fig01_distribution_evolution", out_dir)
    return fig


if __name__ == "__main__":
    plot_distribution_evolution()
