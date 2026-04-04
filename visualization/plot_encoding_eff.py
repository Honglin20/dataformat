"""Figure 9: Format Encoding Efficiency — Storage Bits vs. Effective Bits.

For each focus format, two bars:
  - Storage bits per element (what you actually store, incl. metadata).
  - Effective bits (information-theoretic equivalent, rate-distortion).

Two sub-panels: Gaussian N(0,1) [easy] and Channel-Outlier σ=50 [hard].
Key comparison: MXINT4 (pays +0.25 bpe overhead) vs HAD+INT4(C) (zero overhead).
"""

import numpy as np
import matplotlib.pyplot as plt

from distributions.generators import channel_outliers, gaussian
from distributions.metrics import effective_bits
from formats import build_all_formats
from visualization.style import save_fig, get_color


# Focus formats: (name, nominal_bits, metadata_bpe)
_EFF_FORMATS = [
    ("FP32",          32, 0.00),
    ("INT8",           8, 0.00),
    ("MXINT8",         8, 0.25),
    ("INT4",           4, 0.00),
    ("MXINT4",         4, 0.25),
    ("NVFP4",          4, 0.00),
    ("NF4",            4, 0.00),
    ("SQ-Format",      4, 1.01),
    ("SQ-Format(8b)",  8, 1.01),
    ("HAD+INT4(C)",    4, 0.00),
    ("HAD+INT4(T)",    4, 0.00),
    ("HAD+INT8(C)",    8, 0.00),
    ("HAD+SQ",         4, 1.01),
    ("RandRot+INT4",   4, 0.00),
    ("RandRot+INT8",   8, 0.00),
]


def compute_encoding_efficiency(n: int = 4096, seed: int = 42) -> tuple:
    all_formats = build_all_formats(dim=256, seed=seed)
    x_easy, _ = gaussian(n=n, sigma=1.0, seed=seed)
    x_hard, _ = channel_outliers(n=n, outlier_sigma=50.0, seed=seed)

    easy, hard = {}, {}
    for fmt_name, nominal_bits, meta_bpe in _EFF_FORMATS:
        if fmt_name not in all_formats:
            continue
        fmt = all_formats[fmt_name]
        storage_bits = nominal_bits + meta_bpe
        for store_dict, x in [(easy, x_easy), (hard, x_hard)]:
            try:
                x_q = fmt.quantize(x)
                eff = effective_bits(x, x_q)
                if np.isfinite(eff):
                    eff = min(eff, storage_bits * 1.05)
                else:
                    eff = 0.0
            except Exception:
                eff = 0.0
            store_dict[fmt_name] = {
                "storage_bits": storage_bits,
                "effective_bits": eff,
            }
    return easy, hard


def _draw_panel(ax, data: dict, fmt_names: list, title: str):
    x_pos = np.arange(len(fmt_names))
    bar_w = 0.38

    storage_vals = [data[f]["storage_bits"] for f in fmt_names]
    eff_vals = [data[f]["effective_bits"] for f in fmt_names]
    colors = [get_color(f) for f in fmt_names]

    # Light bars: storage
    ax.bar(x_pos - bar_w / 2, storage_vals, bar_w,
           color=colors, alpha=0.30, edgecolor="gray", linewidth=0.5,
           label="Storage bits/elem")

    # Dark bars: effective bits
    ax.bar(x_pos + bar_w / 2, eff_vals, bar_w,
           color=colors, alpha=0.90, edgecolor="black", linewidth=0.5,
           label="Effective bits (rate-distortion)")

    # Efficiency % label
    for i, (s, e) in enumerate(zip(storage_vals, eff_vals)):
        if s > 0 and e > 0:
            ax.text(x_pos[i] + bar_w / 2, e + 0.15,
                    f"{e/s:.0%}", ha="center", va="bottom",
                    fontsize=7, rotation=90)

    # Red dashed line at storage level (theoretical maximum)
    for i, s in enumerate(storage_vals):
        ax.plot([x_pos[i] - bar_w / 2, x_pos[i] + bar_w / 2],
                [s, s], color="darkred", linewidth=1.0, linestyle="--", alpha=0.4)

    # Annotate MXINT4 +0.25 bpe overhead
    if "MXINT4" in fmt_names:
        xi = fmt_names.index("MXINT4")
        ax.annotate("+0.25\nbpe",
                    xy=(x_pos[xi] - bar_w / 2, storage_vals[xi]),
                    xytext=(x_pos[xi] - 0.7, storage_vals[xi] + 0.5),
                    fontsize=7, color="darkred", ha="center",
                    arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8))

    ax.set_xticks(x_pos)
    ax.set_xticklabels(fmt_names, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Bits per Element", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_ylim(0, max(storage_vals) * 1.4)


def plot_encoding_efficiency(out_dir: str = "results/figures", seed: int = 42):
    """Plot Figure 9: Format Encoding Efficiency."""
    easy, hard = compute_encoding_efficiency(seed=seed)
    fmt_names = [f for f, _, _ in _EFF_FORMATS if f in easy]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    _draw_panel(axes[0], easy, fmt_names,
                "Encoding Efficiency — Gaussian N(0,1)\n"
                "(light=storage bits, dark=effective bits, label=efficiency%)")
    _draw_panel(axes[1], hard, fmt_names,
                "Encoding Efficiency — Channel Outlier σ=50\n"
                "(MXINT4: pays +0.25bpe overhead; HAD+INT4(C): zero overhead)")

    fig.suptitle(
        "Figure 9: Format Encoding Efficiency — Storage Bits vs. Effective Bits",
        fontsize=13, y=1.01,
    )

    save_fig(fig, "fig09_encoding_efficiency", out_dir)
    return fig


if __name__ == "__main__":
    plot_encoding_efficiency()
