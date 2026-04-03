"""Figure 5: HAD vs. Random Rotation — MSE vs. Tensor Dimension N.

Key research question: at what dimension N does cheap structured HAD
achieve the same quantization quality as expensive random rotation?

X-axis: tensor dimension N (64 → 4096, power-of-2 steps).
Y-axis: MSE after quantization.

Plotted curves:
  - HAD + INT4       (structured butterfly, hardware-fixable)
  - TurboQuant + INT4 (random ±1 diagonal, near-free)
  - RandRot + INT4   (dense random orthogonal, expensive ROM)
  - INT4 (no transform, baseline)
  - MXFP4 (hardware-native reference)
"""

import numpy as np
import matplotlib.pyplot as plt

from formats.transforms.hadamard import hadamard_transform
from formats.transforms.random_rotation import RandomRotationTransform, TurboQuantTransform
from formats import build_all_formats
from distributions.generators import channel_outliers, spiky_outliers
from distributions.metrics import mse as compute_mse
from visualization.style import save_fig, PALETTE, LINESTYLES


_DIM_SWEEP = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
_BITS = 4
_SEED = 42


def _quantize_int(x: np.ndarray, bits: int) -> np.ndarray:
    """Fast INT symmetric per-tensor quantize."""
    q_max = 2 ** (bits - 1) - 1
    absmax = np.max(np.abs(x))
    if absmax == 0:
        return x.copy()
    scale = absmax / q_max
    return np.clip(np.round(x / scale), -q_max, q_max).astype(np.float32) * scale


def _mxfp4_simple(x: np.ndarray, block_size: int = 32) -> np.ndarray:
    """Simplified MXFP4 for speed in dimension sweep."""
    from formats.mxfp import MXFPFormat
    fmt = MXFPFormat(element_bits=4, block_size=block_size)
    return fmt.quantize(x)


def run_dim_sweep(
    dist_name: str = "channel_outliers",
    outlier_sigma: float = 50.0,
    bits: int = _BITS,
    seed: int = _SEED,
    dims: list = _DIM_SWEEP,
) -> dict:
    """Run MSE sweep over dimensions for all transform types."""
    results = {k: [] for k in [
        "INT4 (no transform)",
        "TurboQuant+INT4",
        "HAD+INT4",
        "RandRot+INT4",
        "MXFP4",
    ]}

    for n in dims:
        # Generate distribution of size n
        if dist_name == "channel_outliers":
            x, _ = channel_outliers(n=n, outlier_sigma=outlier_sigma, seed=seed)
        else:
            x, _ = spiky_outliers(n=n, spike_multiplier=outlier_sigma, seed=seed)

        # INT4 no transform
        x_q = _quantize_int(x, bits)
        results["INT4 (no transform)"].append(compute_mse(x, x_q))

        # TurboQuant
        turbo = TurboQuantTransform(dim=n, seed=seed)
        x_t = turbo.forward(x)
        x_qt = _quantize_int(x_t, bits)
        x_r = turbo.inverse(x_qt)
        results["TurboQuant+INT4"].append(compute_mse(x, x_r))

        # HAD
        x_h = hadamard_transform(x, normalize=True)
        x_qh = _quantize_int(x_h, bits)
        from formats.transforms.hadamard import inverse_hadamard_transform
        x_rh = inverse_hadamard_transform(x_qh, normalize=True)
        results["HAD+INT4"].append(compute_mse(x, x_rh))

        # RandRot
        rr = RandomRotationTransform(dim=n, seed=seed)
        x_rr = rr.forward(x)
        x_qrr = _quantize_int(x_rr, bits)
        x_rrr = rr.inverse(x_qrr)
        results["RandRot+INT4"].append(compute_mse(x, x_rrr))

        # MXFP4
        try:
            x_mx = _mxfp4_simple(x)
            results["MXFP4"].append(compute_mse(x, x_mx))
        except Exception:
            results["MXFP4"].append(np.nan)

    return results


def plot_had_vs_random(
    out_dir: str = "results/figures", seed: int = 42
):
    """Plot Figure 5: HAD vs. Random Rotation MSE vs. Dimension."""
    dims = _DIM_SWEEP

    # Run sweep on channel outliers (most challenging)
    results = run_dim_sweep("channel_outliers", outlier_sigma=50.0, seed=seed, dims=dims)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    curve_style = {
        "INT4 (no transform)":  {"color": PALETTE["INT4"],          "ls": (0, (5,2,1,2)), "lw": 1.5},
        "TurboQuant+INT4":      {"color": PALETTE["TurboQuant+INT4"], "ls": "-.", "lw": 2.0},
        "HAD+INT4":             {"color": PALETTE["HAD+INT4"],        "ls": "-",  "lw": 2.5},
        "RandRot+INT4":         {"color": PALETTE["RandRot+INT4"],    "ls": "--", "lw": 2.0},
        "MXFP4":                {"color": PALETTE["MXFP4"],           "ls": ":",  "lw": 2.0},
    }

    for ax_idx, (ax, yscale) in enumerate(zip(axes, ["log", "linear"])):
        for label, mse_vals in results.items():
            style = curve_style.get(label, {})
            ax.plot(
                dims, mse_vals,
                label=label,
                color=style.get("color", "gray"),
                linestyle=style.get("ls", "-"),
                linewidth=style.get("lw", 1.5),
                marker="o", markersize=5,
            )

        ax.set_xscale("log", base=2)
        ax.set_yscale(yscale)
        ax.set_xlabel("Tensor Dimension N (log₂ scale)")
        ax.set_ylabel("MSE (Quantization Error)")
        ax.set_xticks(dims)
        ax.set_xticklabels([str(d) for d in dims], rotation=30)

        if ax_idx == 0:
            ax.set_title("Figure 5a: MSE vs. Dimension (log scale)\nChannel Outlier σ=50")
        else:
            ax.set_title("Figure 5b: MSE vs. Dimension (linear scale)\nFocus on large N convergence")

        if ax_idx == 0:
            ax.legend(loc="upper right", fontsize=9)

        # Annotation: HAD convergence point
        had_vals = np.array(results["HAD+INT4"])
        rr_vals = np.array(results["RandRot+INT4"])
        converge_idx = np.argmin(np.abs(had_vals - rr_vals))
        if converge_idx < len(dims):
            ax.axvline(dims[converge_idx], color="green", linewidth=0.8,
                       linestyle="--", alpha=0.5)
            ax.annotate(
                f"HAD≈RandRot\nN={dims[converge_idx]}",
                xy=(dims[converge_idx], had_vals[converge_idx]),
                xytext=(dims[converge_idx] * 1.5, had_vals[converge_idx] * 1.5),
                fontsize=8, color="green",
                arrowprops=dict(arrowstyle="->", color="green", lw=0.8),
            )

    save_fig(fig, "fig05_had_vs_random_rotation", out_dir)
    return fig


if __name__ == "__main__":
    plot_had_vs_random()
