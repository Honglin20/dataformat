"""Figure 9: Format Encoding Efficiency — Storage Bits vs. Effective Bits.

For each format, shows two bars:
  - Storage bits per element (what you actually store in memory).
  - Effective bits (information-theoretic equivalent from rate-distortion).

The gap between them reveals "wasted" bits — format complexity that doesn't
translate to proportional quantization quality gains.

Tight gap → format is efficient.
Wide gap → format overhead (metadata, encoding) doesn't pay off in quality.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from distributions.generators import channel_outliers, gaussian
from distributions.metrics import effective_bits
from formats import build_all_formats
from visualization.style import save_fig, PALETTE, get_color


_EFF_FORMATS = [
    ("FP32",        32, 0.0),
    ("BF16",        16, 0.0),
    ("INT8",         8, 0.0),
    ("MXFP8",        8, 0.25),
    ("MXINT8",       8, 0.25),
    ("FP6",          6, 0.0),
    ("INT4",         4, 0.0),
    ("MXFP4",        4, 0.25),
    ("MXINT4",       4, 0.25),
    ("NVFP4",        4, 0.0),
    ("NF4",          4, 0.0),
    ("SQ-Format",    5, 1.0),
    ("SmoothQuant+INT8", 8, 0.0),
    ("SmoothQuant+INT4", 4, 0.0),
    ("HAD+INT8",     8, 0.0),
    ("HAD+INT4",     4, 0.0),
    ("HAD+LUT4",     4, 0.0),
    ("HAD+SQ",       4, 1.0),
    ("TurboQuant+INT4", 4, 0.0),
    ("RandRot+INT4", 4, 0.0),
]


def compute_encoding_efficiency(n: int = 4096, seed: int = 42) -> list:
    """Compute storage bits vs. effective bits for each format."""
    all_formats = build_all_formats(dim=256, seed=seed)

    # Test on both easy (Gaussian) and hard (channel outlier) distributions
    x_easy, _ = gaussian(n=n, sigma=1.0, seed=seed)
    x_hard, _ = channel_outliers(n=n, outlier_sigma=50.0, seed=seed)

    results = []
    for fmt_name, nominal_bits, meta_bpe in _EFF_FORMATS:
        if fmt_name not in all_formats:
            continue
        fmt = all_formats[fmt_name]
        storage_bits = nominal_bits + meta_bpe

        for dist_name, x in [("easy (Gaussian)", x_easy), ("hard (Channel Outlier)", x_hard)]:
            try:
                x_q = fmt.quantize(x)
                eff = effective_bits(x, x_q)
            except Exception:
                eff = np.nan

            efficiency_ratio = eff / max(storage_bits, 0.1) if np.isfinite(eff) else np.nan

            results.append({
                "format": fmt_name,
                "nominal_bits": nominal_bits,
                "metadata_bpe": meta_bpe,
                "storage_bits": storage_bits,
                "effective_bits": eff,
                "efficiency_ratio": efficiency_ratio,
                "distribution": dist_name,
            })

    return results


def plot_encoding_efficiency(out_dir: str = "results/figures", seed: int = 42):
    """Plot Figure 9: Format Encoding Efficiency."""
    results = compute_encoding_efficiency(seed=seed)

    # Separate easy/hard
    easy = {r["format"]: r for r in results if "easy" in r["distribution"]}
    hard = {r["format"]: r for r in results if "hard" in r["distribution"]}

    fmt_names = [f for f, _, _ in _EFF_FORMATS if f in easy]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_idx, (ax, data_dict, title_suffix) in enumerate(zip(
        axes,
        [easy, hard],
        ["Gaussian N(0,1)", "Channel Outlier σ=50"]
    )):
        x_pos = np.arange(len(fmt_names))
        bar_width = 0.35

        storage_vals = [data_dict[f]["storage_bits"] for f in fmt_names]
        eff_vals = [data_dict[f]["effective_bits"] for f in fmt_names]

        colors_storage = [get_color(f) for f in fmt_names]
        colors_eff = [get_color(f) for f in fmt_names]

        bars1 = ax.bar(x_pos - bar_width / 2, storage_vals,
                       bar_width, label="Storage bits/element",
                       alpha=0.5, color=colors_storage, edgecolor="gray", linewidth=0.5)
        bars2 = ax.bar(x_pos + bar_width / 2, eff_vals,
                       bar_width, label="Effective bits (EffBits, rate-distortion)",
                       alpha=0.9, color=colors_eff, edgecolor="black", linewidth=0.5)

        # Efficiency ratio as text on bars
        for i, (s, e) in enumerate(zip(storage_vals, eff_vals)):
            if np.isfinite(e) and s > 0:
                ratio = e / s
                ax.text(x_pos[i] + bar_width / 2, e + 0.1,
                        f"{ratio:.1%}", ha="center", va="bottom", fontsize=6.5, rotation=90)

        # Reference line: perfect efficiency (eff_bits = storage_bits)
        ax.plot([-0.5, len(fmt_names) - 0.5], [0, 0], "k--", linewidth=0, alpha=0)
        for i, (s, _) in enumerate(zip(storage_vals, eff_vals)):
            ax.plot([x_pos[i] - bar_width / 2, x_pos[i] + bar_width / 2],
                    [s, s], "r--", linewidth=1.0, alpha=0.4)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(fmt_names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Bits per Element")
        ax.set_title(f"Figure 9: Encoding Efficiency — {title_suffix}\n"
                     f"(Dark bar = EffBits; gap between bars = wasted encoding budget)")
        ax.legend(fontsize=8, loc="upper right")

        # Family separator vertical lines
        boundary_indices = [1, 7, 11, 13, 15, 18]
        for bi in boundary_indices:
            if bi < len(fmt_names):
                ax.axvline(bi - 0.5, color="navy", linewidth=0.8, alpha=0.4, linestyle=":")

        # Add a red-line annotation for MX overhead
        mx_fmts = ["MXFP4", "MXFP8", "MXINT4", "MXINT8"]
        for mxf in mx_fmts:
            if mxf in fmt_names:
                xi = fmt_names.index(mxf)
                ax.annotate("+0.25\nbits\n(scale)", xy=(x_pos[xi] - bar_width / 2,
                            data_dict[mxf]["storage_bits"] - 0.1),
                            fontsize=5.5, color="darkred", ha="center")

    plt.tight_layout()
    save_fig(fig, "fig09_encoding_efficiency", out_dir)
    return fig


if __name__ == "__main__":
    plot_encoding_efficiency()
