"""Figure 8: Per-Channel Quantization Error Heatmap.

X-axis: Channel index (0 → C).
Y-axis: Format / pipeline.
Color:  Per-channel MSE (log scale).

Key insight: MX's Block Scale is a LOCAL correction (rescales within a block),
while HAD is a GLOBAL Gaussianization (redistributes outlier energy across ALL channels).
This chart directly shows which formats can rescue specific outlier channels.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from formats import build_all_formats
from visualization.style import save_fig, PALETTE


_CHANNEL_FORMATS = [
    "INT4",
    "MXFP4", "MXINT4", "NF4", "NVFP4",
    "SmoothQuant+INT4",
    "SQ-Format",
    "HAD+INT4(C)",
    "HAD+INT4(T)",
    "HAD+SQ",
    "RandRot+INT4",
]

_N_CHANNELS = 128   # simulate 128 channels
_BATCH = 32         # batch dimension (simulate per-channel stats)


def build_channel_mse(
    n_channels: int = _N_CHANNELS,
    batch: int = _BATCH,
    outlier_ratio: float = 0.05,
    outlier_sigma: float = 50.0,
    seed: int = 42,
) -> dict:
    """Compute per-channel MSE for each format.

    Generates a (batch × n_channels) tensor where:
      - 5% of channels have outlier σ=50 magnitudes.
      - Remaining 95% are N(0,1).

    Returns dict: {format_name: array of per-channel MSE, shape (n_channels,)}
    """
    rng = np.random.default_rng(seed)

    # Systematic channel outliers (same channels always)
    n_outlier_ch = max(1, int(np.ceil(outlier_ratio * n_channels)))
    outlier_channels = np.linspace(0, n_channels - 1, n_outlier_ch, dtype=int)

    X = rng.normal(0.0, 1.0, size=(batch, n_channels)).astype(np.float32)
    X[:, outlier_channels] *= outlier_sigma  # inject systematic outliers

    all_formats = build_all_formats(dim=n_channels, seed=seed)
    formats = {k: v for k, v in all_formats.items() if k in _CHANNEL_FORMATS}

    channel_mse = {}
    for fmt_name, fmt in formats.items():
        per_ch_mse = np.zeros(n_channels)
        for ch in range(n_channels):
            x_ch = X[:, ch]
            try:
                x_q = fmt.quantize(x_ch)
                err = x_ch - x_q
                per_ch_mse[ch] = float(np.mean(err ** 2))
            except Exception:
                per_ch_mse[ch] = np.nan
        channel_mse[fmt_name] = per_ch_mse

    return channel_mse, outlier_channels


def plot_channel_heatmap(
    out_dir: str = "results/figures",
    n_channels: int = _N_CHANNELS,
    seed: int = 42,
):
    """Plot Figure 8: Per-Channel Quantization Error Heatmap."""
    channel_mse, outlier_channels = build_channel_mse(
        n_channels=n_channels, seed=seed
    )

    # Build matrix for heatmap (formats × channels)
    fmt_list = [f for f in _CHANNEL_FORMATS if f in channel_mse]
    matrix = np.array([channel_mse[f] for f in fmt_list])
    log_matrix = np.log10(np.maximum(matrix, 1e-12))

    with plt.rc_context({"figure.constrained_layout.use": False}):
        fig, axes = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={"height_ratios": [1, 4]})

    # ── Top panel: channel outlier indicator ─────────────────────────────────
    ax_top = axes[0]
    indicator = np.zeros(n_channels)
    indicator[outlier_channels] = 1.0
    ax_top.bar(range(n_channels), indicator, color="red", alpha=0.7, width=0.8)
    ax_top.set_xlim(-0.5, n_channels - 0.5)
    ax_top.set_ylim(0, 1.3)
    ax_top.set_yticks([])
    ax_top.set_title("Figure 8: Per-Channel Quantization Error Heatmap\n"
                      "(Top: outlier channel indicator in red)", fontsize=12)
    ax_top.set_xticks([])

    # ── Main heatmap ─────────────────────────────────────────────────────────
    ax_main = axes[1]
    im = sns.heatmap(
        log_matrix,
        ax=ax_main,
        cmap="RdYlGn_r",
        xticklabels=max(1, n_channels // 16),
        yticklabels=fmt_list,
        cbar_kws={"label": "log₁₀(Per-Channel MSE)", "shrink": 0.8},
        linewidths=0,
    )

    # Highlight outlier columns
    for ch in outlier_channels:
        ax_main.axvline(ch + 0.5, color="red", linewidth=0.5, alpha=0.6)
        ax_main.axvline(ch - 0.5, color="red", linewidth=0.5, alpha=0.6)

    ax_main.set_xlabel("Channel Index")
    ax_main.set_ylabel("Quantization Format / Pipeline")
    ax_main.tick_params(axis="y", labelsize=9)

    # Add horizontal separators between format families
    family_boundaries = {
        "Hardware-Native →": 0,
        "← Transform-Based": 5,
    }
    for label, pos in family_boundaries.items():
        ax_main.axhline(pos, color="navy", linewidth=1.5, alpha=0.7)

    # Text labels for families
    ax_main.text(-3, 2.5, "HW-Native", fontsize=8, rotation=90,
                 va="center", ha="right", color="navy", fontstyle="italic")
    ax_main.text(-3, 8.0, "Transform\nBased", fontsize=8, rotation=90,
                 va="center", ha="right", color="darkgreen", fontstyle="italic")

    fig.subplots_adjust(left=0.15, right=0.95, top=0.94, bottom=0.08, hspace=0.05)
    save_fig(fig, "fig08_channel_heatmap", out_dir)
    return fig


if __name__ == "__main__":
    plot_channel_heatmap()
