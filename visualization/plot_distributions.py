"""Figure 1: Distribution evolution — CDF + Q-Q Plot.

Shows how HAD, RandRot, and TurboQuant transform outlier distributions
into near-Gaussian distributions amenable to uniform quantization.

Two-panel layout:
  Left:  CDFs of original vs. transformed distributions (log-scale tails).
  Right: Q-Q plot (sorted values vs. standard normal quantiles).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import probplot

from distributions.generators import (
    channel_outliers, spiky_outliers, student_t_dist
)
from formats.transforms.hadamard import hadamard_transform
from formats.transforms.random_rotation import RandomRotationTransform, TurboQuantTransform
from visualization.style import save_fig, fig_and_ax, PALETTE


def plot_distribution_evolution(
    n: int = 1024,
    seed: int = 42,
    out_dir: str = "results/figures",
):
    """Plot Figure 1: CDF + Q-Q Plot for distribution evolution under transforms."""

    # Generate test distribution: channel outliers (most challenging)
    x_outlier, _ = channel_outliers(n=n, outlier_ratio=0.01, outlier_sigma=50.0, seed=seed)
    x_gaussian = np.random.default_rng(seed).normal(0, 1, n).astype(np.float32)
    x_student, _ = student_t_dist(n=n, nu=3, seed=seed)

    # Apply transforms to channel_outlier distribution
    had_out = hadamard_transform(x_outlier, normalize=True)
    rand_rot = RandomRotationTransform(dim=n, seed=seed)
    turbo = TurboQuantTransform(dim=n, seed=seed)
    randrot_out = rand_rot.forward(x_outlier)
    turbo_out = turbo.forward(x_outlier)

    distributions = {
        "Original (ChannelOutlier σ=50)": x_outlier,
        "Student-t (ν=3)": x_student,
        "HAD transform":  had_out,
        "TurboQuant":     turbo_out,
        "RandRot":        randrot_out,
        "Ideal Gaussian": x_gaussian,
    }

    colors = {
        "Original (ChannelOutlier σ=50)": PALETTE["INT4"],
        "Student-t (ν=3)": PALETTE["MXFP4"],
        "HAD transform":  PALETTE["HAD+INT4"],
        "TurboQuant":     PALETTE["TurboQuant+INT4"],
        "RandRot":        PALETTE["RandRot+INT4"],
        "Ideal Gaussian": PALETTE["FP32"],
    }
    linestyles = {
        "Original (ChannelOutlier σ=50)": "--",
        "Student-t (ν=3)": "-.",
        "HAD transform":  "-",
        "TurboQuant":     "-",
        "RandRot":        "-",
        "Ideal Gaussian": ":",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: CDF plot ────────────────────────────────────────────────────────
    ax = axes[0]
    for label, x in distributions.items():
        x_sorted = np.sort(x)
        cdf = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
        ax.plot(x_sorted, cdf, label=label,
                color=colors[label], linestyle=linestyles[label], linewidth=1.8)

    ax.set_xlabel("Tensor Value")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Figure 1a: CDF — Distribution Evolution Under Transforms")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(-20, 20)

    # ── Right: Q-Q plot ───────────────────────────────────────────────────────
    ax2 = axes[1]
    for label, x in distributions.items():
        # Normalize x for Q-Q comparison
        x_norm = (x - np.mean(x)) / (np.std(x) + 1e-8)
        theoretical_q = np.sort(
            np.random.default_rng(seed).standard_normal(len(x_norm))
        )
        sample_q = np.sort(x_norm)
        ax2.plot(theoretical_q, sample_q, label=label,
                 color=colors[label], linestyle=linestyles[label],
                 linewidth=1.2, alpha=0.85)

    # Reference line: y=x (perfect Gaussian)
    lim = 4.0
    ax2.plot([-lim, lim], [-lim, lim], "k--", linewidth=1.0, alpha=0.5, label="y=x (Gaussian)")
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    ax2.set_xlabel("Theoretical Normal Quantiles")
    ax2.set_ylabel("Sample Quantiles (normalized)")
    ax2.set_title("Figure 1b: Q-Q Plot vs. Standard Normal")
    ax2.legend(loc="upper left", fontsize=8)

    save_fig(fig, "fig01_distribution_evolution", out_dir)
    return fig


if __name__ == "__main__":
    plot_distribution_evolution()
