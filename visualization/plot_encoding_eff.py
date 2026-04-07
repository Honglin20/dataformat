"""Figure 9: Format Encoding Efficiency — Storage Bits vs. Effective Bits.

Four sub-panels: 4-bit and 8-bit × Gaussian N(0,1) [easy] and Channel-Outlier σ=50 [hard].
For each format, two bars:
  - Storage bits per element (what you actually store, incl. metadata).
  - Effective bits (information-theoretic equivalent, rate-distortion).
Key comparison: MXINT (pays +0.25 bpe overhead) vs HAD+INT (zero overhead).
"""

import numpy as np
import matplotlib.pyplot as plt

from distributions.generators import gaussian
from distributions.metrics import effective_bits
from formats import build_all_formats
from visualization.style import save_fig, get_color


# Focus formats by bit-width: (name, nominal_bits, metadata_bpe)
_EFF_FORMATS_4BIT = [
    ("FP32",         32, 0.00),
    ("INT4",          4, 0.00),
    ("MXINT4",        4, 0.25),
    ("NVFP4",         4, 0.00),
    ("NF4",           4, 0.00),
    ("SQ-Format",     4, 1.01),
    ("HAD+INT4(C)",   4, 0.00),
    ("HAD+INT4(T)",   4, 0.00),
    ("HAD+SQ",        4, 1.01),
    ("RandRot+INT4",  4, 0.00),
]

_EFF_FORMATS_8BIT = [
    ("FP32",          32, 0.00),
    ("INT8",           8, 0.00),
    ("MXINT8",         8, 0.25),
    ("SQ-Format(8b)",  8, 1.01),
    ("HAD+INT8(C)",    8, 0.00),
    ("HAD+INT8(T)",    8, 0.00),
    ("RandRot+INT8",   8, 0.00),
]

# Combined for data computation
_EFF_FORMATS = _EFF_FORMATS_4BIT + [f for f in _EFF_FORMATS_8BIT if f[0] != "FP32"]


def compute_encoding_efficiency(seed: int = 42) -> tuple:
    """Compute encoding efficiency for easy (Gaussian) and hard (2D row-outlier) cases.

    The hard case uses a 2D (16×256) tensor with row 0 as outlier (σ=50),
    so HAD+INT(C) per-row scale and HAD+INT(T) global scale produce distinct results.
    """
    all_formats = build_all_formats(dim=256, seed=seed)
    rng = np.random.default_rng(seed)

    # Easy: 1D Gaussian (C) and (T) are identical here — shows baseline
    x_easy, _ = gaussian(n=4096, sigma=1.0, seed=seed)

    # Hard: 2D row-outlier tensor — row 0 has σ=50, rows 1-15 are N(0,1)
    # HAD+INT(C) adapts per-row; HAD+INT(T) uses global scale dominated by row 0
    x_hard_2d = rng.normal(0, 1, (16, 256)).astype(np.float32)
    x_hard_2d[0, :] = rng.normal(0, 50.0, 256).astype(np.float32)

    easy, hard = {}, {}
    for fmt_name, nominal_bits, meta_bpe in _EFF_FORMATS:
        if fmt_name not in all_formats:
            continue
        fmt = all_formats[fmt_name]
        storage_bits = nominal_bits + meta_bpe

        # Easy: 1D tensor
        try:
            x_q = fmt.quantize(x_easy)
            eff = effective_bits(x_easy, x_q)
            eff = min(eff, storage_bits * 1.05) if np.isfinite(eff) else 0.0
        except Exception:
            eff = 0.0
        easy[fmt_name] = {"storage_bits": storage_bits, "effective_bits": eff}

        # Hard: 2D tensor (distinguishes C from T)
        try:
            x_q2 = fmt.quantize(x_hard_2d)
            eff2 = effective_bits(x_hard_2d.ravel(), x_q2.ravel())
            eff2 = min(eff2, storage_bits * 1.05) if np.isfinite(eff2) else 0.0
        except Exception:
            eff2 = 0.0
        hard[fmt_name] = {"storage_bits": storage_bits, "effective_bits": eff2}

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
    """Plot Figure 9: Format Encoding Efficiency — 4-panel (4-bit/8-bit × Easy/Hard)."""
    easy, hard = compute_encoding_efficiency(seed=seed)

    names4 = [f for f, _, _ in _EFF_FORMATS_4BIT if f in easy]
    names8 = [f for f, _, _ in _EFF_FORMATS_8BIT if f in easy]

    fig, axes = plt.subplots(2, 2, figsize=(20, 12), constrained_layout=False)

    _draw_panel(axes[0, 0], easy, names4,
                "4-bit — Gaussian N(0,1)\n"
                "(light=storage bits, dark=effective bits, label=efficiency%)")
    _draw_panel(axes[0, 1], hard, names4,
                "4-bit — Channel Outlier σ=50\n"
                "(HAD+INT4(C) > (T); MXINT4 pays +0.25 bpe overhead)")
    _draw_panel(axes[1, 0], easy, names8,
                "8-bit — Gaussian N(0,1)\n"
                "(light=storage bits, dark=effective bits, label=efficiency%)")
    _draw_panel(axes[1, 1], hard, names8,
                "8-bit — Channel Outlier σ=50\n"
                "(HAD+INT8(C) >> (T); MXINT8 pays +0.25 bpe overhead)")

    fig.suptitle(
        "Figure 9: Format Encoding Efficiency — Storage Bits vs. Effective Bits\n"
        "(Top row: 4-bit formats  ·  Bottom row: 8-bit formats)",
        fontsize=13,
    )
    fig.subplots_adjust(left=0.06, right=0.97, top=0.93, bottom=0.12, hspace=0.45, wspace=0.30)
    save_fig(fig, "fig09_encoding_efficiency", out_dir)
    return fig


if __name__ == "__main__":
    plot_encoding_efficiency()
