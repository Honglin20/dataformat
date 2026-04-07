"""Figure 8: Per-Channel Quantization Error Heatmap — 4-bit (top) and 8-bit (bottom).

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


_CHANNEL_FORMATS_4BIT = [
    "INT4",
    "MXINT4", "NVFP4", "NF4",
    "SQ-Format",
    "HAD+INT4(C)",
    "HAD+INT4(T)",
    "HAD+SQ",
    "RandRot+INT4",
]

_CHANNEL_FORMATS_8BIT = [
    "INT8",
    "MXINT8",
    "SQ-Format(8b)",
    "HAD+INT8(C)",
    "HAD+INT8(T)",
    "RandRot+INT8",
]

_N_CHANNELS = 128   # simulate 128 channels
_BATCH = 32         # batch dimension (simulate per-channel stats)


def build_channel_mse(
    n_channels: int = _N_CHANNELS,
    batch: int = _BATCH,
    outlier_ratio: float = 0.05,
    outlier_sigma: float = 50.0,
    seed: int = 42,
) -> tuple:
    """Compute per-channel MSE for 4-bit and 8-bit formats separately.

    Generates a (batch × n_channels) tensor where:
      - 5% of channels have outlier σ=50 magnitudes (systematic, same channels).
      - Remaining 95% are N(0,1).

    Returns:
      channel_mse_4 : dict {fmt: array(n_channels,)} for 4-bit formats
      channel_mse_8 : dict {fmt: array(n_channels,)} for 8-bit formats
      outlier_channels : array of outlier channel indices
    """
    rng = np.random.default_rng(seed)

    n_outlier_ch = max(1, int(np.ceil(outlier_ratio * n_channels)))
    outlier_channels = np.linspace(0, n_channels - 1, n_outlier_ch, dtype=int)

    X = rng.normal(0.0, 1.0, size=(batch, n_channels)).astype(np.float32)
    X[:, outlier_channels] *= outlier_sigma

    all_formats = build_all_formats(dim=n_channels, seed=seed)

    def _compute(fmt_list):
        result = {}
        for fmt_name in fmt_list:
            if fmt_name not in all_formats:
                continue
            fmt = all_formats[fmt_name]
            per_ch_mse = np.zeros(n_channels)
            for ch in range(n_channels):
                x_ch = X[:, ch]
                try:
                    x_q = fmt.quantize(x_ch)
                    per_ch_mse[ch] = float(np.mean((x_ch - x_q) ** 2))
                except Exception:
                    per_ch_mse[ch] = np.nan
            result[fmt_name] = per_ch_mse
        return result

    return _compute(_CHANNEL_FORMATS_4BIT), _compute(_CHANNEL_FORMATS_8BIT), outlier_channels


def _draw_channel_panel(ax, channel_mse: dict, fmt_list: list,
                        outlier_channels: np.ndarray, n_channels: int,
                        title: str) -> None:
    """Draw per-channel MSE heatmap for one bit-width group."""
    fmts = [f for f in fmt_list if f in channel_mse]
    if not fmts:
        return
    matrix = np.array([channel_mse[f] for f in fmts])
    log_matrix = np.log10(np.maximum(matrix, 1e-12))

    sns.heatmap(
        log_matrix,
        ax=ax,
        cmap="RdYlGn_r",
        xticklabels=max(1, n_channels // 16),
        yticklabels=fmts,
        cbar_kws={"label": "log₁₀(Per-Ch MSE)", "shrink": 0.85},
        linewidths=0,
    )
    for ch in outlier_channels:
        ax.axvline(ch + 0.5, color="red", linewidth=0.6, alpha=0.5)
        ax.axvline(ch - 0.5, color="red", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("Channel Index", fontsize=9)
    ax.set_ylabel("Format", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")


def plot_channel_heatmap(
    out_dir: str = "results/figures",
    n_channels: int = _N_CHANNELS,
    seed: int = 42,
):
    """Plot Figure 8: Per-Channel MSE Heatmap — 4-bit (top) and 8-bit (bottom)."""
    ch_mse4, ch_mse8, outlier_channels = build_channel_mse(n_channels=n_channels, seed=seed)

    n4 = len([f for f in _CHANNEL_FORMATS_4BIT if f in ch_mse4])
    n8 = len([f for f in _CHANNEL_FORMATS_8BIT if f in ch_mse8])

    with plt.rc_context({"figure.constrained_layout.use": False}):
        fig, axes = plt.subplots(
            3, 1, figsize=(15, 4 + n4 * 0.6 + n8 * 0.6),
            gridspec_kw={"height_ratios": [0.7, n4, n8]},
        )

    # ── Top strip: outlier channel indicator ─────────────────────────────────
    ax_ind = axes[0]
    indicator = np.zeros(n_channels)
    indicator[outlier_channels] = 1.0
    ax_ind.bar(range(n_channels), indicator, color="red", alpha=0.7, width=0.8)
    ax_ind.set_xlim(-0.5, n_channels - 0.5)
    ax_ind.set_ylim(0, 1.3)
    ax_ind.set_yticks([])
    ax_ind.set_xticks([])
    ax_ind.set_title(
        "Figure 8: Per-Channel Quantization Error Heatmap  "
        "(red = outlier channels, 5% of channels × σ=50)",
        fontsize=11,
    )

    _draw_channel_panel(
        axes[1], ch_mse4, _CHANNEL_FORMATS_4BIT, outlier_channels, n_channels,
        "4-bit Formats: MXINT4 vs SQ-Format vs HAD+INT4(C/T)",
    )
    _draw_channel_panel(
        axes[2], ch_mse8, _CHANNEL_FORMATS_8BIT, outlier_channels, n_channels,
        "8-bit Formats: MXINT8 vs SQ-Format(8b) vs HAD+INT8(C/T)",
    )

    fig.subplots_adjust(left=0.15, right=0.95, top=0.94, bottom=0.05, hspace=0.35)
    save_fig(fig, "fig08_channel_heatmap", out_dir)
    return fig


if __name__ == "__main__":
    plot_channel_heatmap()
