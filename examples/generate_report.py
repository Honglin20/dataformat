# examples/generate_report.py
"""Generate a self-contained HTML report from profiler results.

Usage:
    python examples/generate_report.py [--results-dir results/mnist]

Requires:
    results/mnist/profiler_results.csv      (from profile_mnist.py)
    results/mnist/training_log.json         (from train_mnist.py)
    results/mnist/profiler_histograms.json  (from profile_mnist.py, optional)

Saves:
    results/mnist/report.html  — opens automatically in default browser
"""
from __future__ import annotations
import argparse
import base64
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Constants ──────────────────────────────────────────────────────────────────

_PAGE_COLS = 20   # max columns (layers) per heatmap page
_PAGE_ROWS = 20   # max rows per vertical heatmap page


# ── Low-level helpers ──────────────────────────────────────────────────────────

def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _imgs_html(b64_list: list[str]) -> str:
    """Embed one or more base64 PNGs as <img> tags."""
    if not b64_list:
        return ""
    return "\n".join(
        f'<img src="data:image/png;base64,{b64}" style="max-width:100%;display:block;margin-bottom:8px">'
        for b64 in b64_list
    )


def _section(title: str, content_html: str, note: str = "") -> str:
    note_html = f'<p class="note">{note}</p>' if note else ""
    return f"<h2>{title}</h2>\n{note_html}\n{content_html}\n<hr>\n"


def _paginate_cols(pivot: pd.DataFrame) -> list[pd.DataFrame]:
    """Split pivot into pages of at most _PAGE_COLS columns."""
    cols = list(pivot.columns)
    return [pivot[cols[i: i + _PAGE_COLS]] for i in range(0, len(cols), _PAGE_COLS)]


def _paginate_rows(pivot: pd.DataFrame) -> list[pd.DataFrame]:
    rows = list(pivot.index)
    return [pivot.loc[rows[i: i + _PAGE_ROWS]] for i in range(0, len(rows), _PAGE_ROWS)]


def _render_heatmap(
    pivot: pd.DataFrame,
    cmap: str,
    vmin: float,
    vmax: float,
    colorbar_label: str,
    title: str,
) -> str:
    """Render a single heatmap page; return base64 PNG."""
    n_cols = len(pivot.columns)
    n_rows = len(pivot.index)
    col_w = max(0.45, min(1.1, 20.0 / max(n_cols, 1)))
    label_fs = max(5, 8 - n_cols // 8)

    fig, ax = plt.subplots(figsize=(max(7, n_cols * col_w), max(3, n_rows * 0.5)))
    vals = pivot.values.astype(float)
    im = ax.imshow(vals, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=label_fs)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(pivot.index, fontsize=max(5, 8 - n_rows // 10))
    plt.colorbar(im, ax=ax, label=colorbar_label)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _heatmap_pages(
    pivot: pd.DataFrame,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    colorbar_label: str,
    title_prefix: str,
    paginate_dim: str = "cols",   # "cols" or "rows"
) -> list[str]:
    """Render a pivot as one or more heatmap pages; return list of b64 PNGs."""
    if pivot.empty:
        return []
    vals_finite = pivot.values[np.isfinite(pivot.values.astype(float))]
    vmin = float(np.percentile(vals_finite, 5))  if vmin is None and len(vals_finite) else (vmin or 0)
    vmax = float(np.percentile(vals_finite, 95)) if vmax is None and len(vals_finite) else (vmax or 1)

    pages = _paginate_cols(pivot) if paginate_dim == "cols" else _paginate_rows(pivot)
    n_pages = len(pages)
    result = []
    for i, page in enumerate(pages):
        suffix = f" ({i+1}/{n_pages})" if n_pages > 1 else ""
        result.append(
            _render_heatmap(page, cmap, vmin, vmax, colorbar_label, title_prefix + suffix)
        )
    return result


# ── Section 1: Training curves ─────────────────────────────────────────────────

def _plot_training_curves(log: dict) -> str:
    epochs = log["epoch"]
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(epochs, log["train_loss"], "b-",  label="Train Loss")
    ax1.plot(epochs, log["test_loss"],  "b--", label="Test Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax2 = ax1.twinx()
    ax2.plot(epochs, log["train_acc"], "r-",  label="Train Acc")
    ax2.plot(epochs, log["test_acc"],  "r--", label="Test Acc")
    ax2.set_ylabel("Accuracy (%)", color="r")
    ax2.tick_params(axis="y", labelcolor="r")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right")
    ax1.set_title("Training Curves")
    fig.tight_layout()
    return _fig_to_b64(fig)


# ── Section 2: FP32 pre-quantization distributions ────────────────────────────

def _plot_fp32_distributions(hist_data: dict) -> list[str]:
    if not hist_data or "FP32" not in hist_data:
        return []
    fp32 = hist_data["FP32"]
    layers_with_weight = [l for l in fp32 if "weight" in fp32.get(l, {})]
    selected = (layers_with_weight if layers_with_weight else list(fp32.keys()))

    tensor_types = ["weight", "input", "output"]
    pages, page_layers = [], []
    # Split into pages of _PAGE_ROWS layers
    for i in range(0, len(selected), _PAGE_ROWS):
        page_layers.append(selected[i: i + _PAGE_ROWS])

    for p_idx, layer_group in enumerate(page_layers):
        nrows = len(layer_group)
        ncols = len(tensor_types)
        fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.2 * nrows), squeeze=False)
        for i, layer in enumerate(layer_group):
            for j, tt in enumerate(tensor_types):
                ax = axes[i][j]
                ld = fp32.get(layer, {})
                if tt not in ld:
                    ax.axis("off"); continue
                h = ld[tt]
                edges, counts = h.get("hist_edges", []), h.get("hist_counts", [])
                if not counts:
                    ax.axis("off"); continue
                centers = [(edges[k] + edges[k+1]) / 2 for k in range(len(counts))]
                widths  = [edges[k+1] - edges[k] for k in range(len(counts))]
                ax.bar(centers, counts, width=widths, color="#2196F3", alpha=0.75, linewidth=0)
                or_ = h.get("outlier_ratio", 0.0)
                short = layer.split(".")[-1] if "." in layer else layer
                ax.set_title(f"{short} · {tt}\noutlier={or_:.4f}", fontsize=7.5)
                ax.tick_params(labelsize=6)
                ax.set_ylabel("count" if j == 0 else "", fontsize=7)
        suffix = f" ({p_idx+1}/{len(page_layers)})" if len(page_layers) > 1 else ""
        fig.suptitle(f"FP32 Distributions — Weight / Input / Output{suffix}", fontsize=11, y=1.01)
        fig.tight_layout()
        pages.append(_fig_to_b64(fig))
    return pages


# ── Section 3: Distribution shape (kurtosis / skewness) ───────────────────────

def _plot_distribution_shape(df: pd.DataFrame) -> list[str]:
    """Kurtosis and skewness per layer for FP32 — reveals heavy-tailed tensors."""
    fp32 = df[df["format"] == "FP32"].copy()
    if fp32.empty or "kurtosis" not in fp32.columns:
        return []

    pages = []
    for metric, label, threshold, cmap in [
        ("kurtosis", "Excess Kurtosis (Gaussian=0, Laplace≈3; high→outlier-prone)", 3.0, "YlOrRd"),
        ("skewness", "Skewness (0=symmetric; |skew|>1 → asymmetric tails)", 1.0, "RdBu_r"),
    ]:
        if metric not in fp32.columns:
            continue
        pivot = fp32.pivot_table(
            values=metric, index="tensor_type", columns="layer_name", aggfunc="mean"
        )
        if pivot.empty:
            continue
        vabs = float(np.nanpercentile(np.abs(pivot.values), 95)) or 1.0
        vmin_ = 0.0 if metric == "kurtosis" else -vabs
        vmax_ = max(vabs, threshold + 0.1)

        pages_for = _heatmap_pages(
            pivot, cmap, vmin_, vmax_, label,
            f"FP32 {metric.capitalize()} — Tensor Type × Layer",
            paginate_dim="cols",
        )
        pages.extend(pages_for)

    return pages


# ── Section 4: Outlier analysis ────────────────────────────────────────────────

def _plot_outlier_analysis(df: pd.DataFrame) -> list[str]:
    fp32 = df[df["format"] == "FP32"].copy()
    if fp32.empty:
        return []

    pages = []

    # Left part: FP32 outlier heatmap (layer × tensor_type), paginated over rows
    pivot = fp32.pivot_table(
        values="outlier_ratio", index="layer_name", columns="tensor_type", aggfunc="mean"
    )
    if not pivot.empty:
        pages.extend(_heatmap_pages(
            pivot, "Oranges", 0.0, None,
            "Outlier Ratio",
            "FP32 Outlier Ratio — Layer × Tensor Type",
            paginate_dim="rows",
        ))

    # Right part: mean outlier ratio by format family
    format_families = {
        "INT(tensor)":  ["INT4(TENSOR)",  "INT8(TENSOR)"],
        "INT(channel)": ["INT4(CHANNEL)", "INT8(CHANNEL)"],
        "HAD+INT":      ["HAD+INT4(C)",   "HAD+INT8(C)", "HAD+INT4(T)", "HAD+INT8(T)"],
        "MX":           ["MXINT4",        "MXINT8"],
        "SQ/FP":        ["SQ-FORMAT-INT", "SQ-FORMAT-FP", "FP16"],
    }
    family_means: dict[str, float] = {}
    for family, fmts in format_families.items():
        sub = df[df["format"].isin(fmts)]
        if not sub.empty:
            family_means[family] = float(sub["outlier_ratio"].mean())

    if family_means:
        fig, ax = plt.subplots(figsize=(8, 4))
        families = list(family_means)
        vals = [family_means[g] for g in families]
        pal = ["#F44336", "#FF9800", "#4CAF50", "#2196F3", "#9C27B0"]
        ax.bar(families, vals, color=pal[:len(families)], alpha=0.85)
        ax.set_ylabel("Mean Outlier Ratio (all layers, all tensor types)")
        ax.set_title("Outlier Ratio by Format Family")
        ax.tick_params(axis="x", labelsize=8)
        fig.tight_layout()
        pages.append(_fig_to_b64(fig))

    return pages


# ── Section 5: HAD energy spread ───────────────────────────────────────────────

def _plot_had_energy_spread(df: pd.DataFrame) -> list[str]:
    """Kurtosis before (FP32 domain) vs after (HAD domain) for HAD+INT formats.

    A drop in kurtosis after HAD indicates effective energy spreading — fewer
    outliers in the quantization domain, which directly improves SQNR.
    """
    if "domain_kurtosis" not in df.columns or "kurtosis" not in df.columns:
        return []

    had_fmts = [f for f in df["format"].unique() if "HAD" in str(f)]
    if not had_fmts:
        return []

    # Per-format, per-layer: kurtosis reduction = orig_kurtosis - domain_kurtosis
    fp32 = df[df["format"] == "FP32"][["layer_name", "tensor_type", "kurtosis"]].copy()
    fp32 = fp32.rename(columns={"kurtosis": "orig_kurtosis"})

    records = []
    for fmt in had_fmts:
        sub = df[df["format"] == fmt][["layer_name", "tensor_type", "domain_kurtosis"]].copy()
        merged = sub.merge(fp32, on=["layer_name", "tensor_type"], how="inner")
        merged["kurtosis_reduction"] = merged["orig_kurtosis"] - merged["domain_kurtosis"]
        merged["format"] = fmt
        records.append(merged)

    if not records:
        return []

    combined = pd.concat(records, ignore_index=True)

    # Bar chart: mean kurtosis reduction per format × tensor_type
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = combined.pivot_table(
        values="kurtosis_reduction", index="format", columns="tensor_type", aggfunc="mean"
    )
    x = np.arange(len(pivot.index))
    width = 0.8 / max(len(pivot.columns), 1)
    for i, tt in enumerate(pivot.columns):
        ax.bar(x + i * width, pivot[tt].values, width, label=tt, alpha=0.85)
    ax.set_xticks(x + width * (len(pivot.columns) - 1) / 2)
    ax.set_xticklabels(pivot.index, rotation=30, ha="right", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_ylabel("Kurtosis Reduction (orig − HAD domain)\nhigher → HAD more effective")
    ax.set_title("HAD Transform Energy Spread — Kurtosis Reduction by Format")
    ax.legend(title="Tensor type", fontsize=8)
    fig.tight_layout()
    return [_fig_to_b64(fig)]


# ── Section 6: Saturation analysis ────────────────────────────────────────────

def _plot_saturation_analysis(df: pd.DataFrame) -> list[str]:
    """Saturation rate (fraction of clipped values) per format × layer."""
    if "saturation_rate" not in df.columns:
        return []
    sub = df[df["format"] != "FP32"].copy()
    if sub.empty:
        return []

    pivot = sub.pivot_table(
        values="saturation_rate", index="format", columns="layer_name", aggfunc="mean"
    )
    # Only keep layers/formats where saturation is non-trivially non-NaN
    pivot = pivot.dropna(how="all", axis=1).dropna(how="all", axis=0)
    if pivot.empty:
        return []

    return _heatmap_pages(
        pivot, "Reds", 0.0, None,
        "Saturation Rate (fraction clipped)",
        "Saturation Rate — Format × Layer",
        paginate_dim="cols",
    )


# ── Section 7: Linear vs Non-Linear QSNR gap ──────────────────────────────────

def _plot_linear_vs_nonlinear(df: pd.DataFrame) -> list[str]:
    non_fp32 = df[df["format"] != "FP32"].copy()
    if non_fp32.empty or "layer_type" not in non_fp32.columns:
        return []

    non_fp32["layer_class"] = non_fp32["layer_type"].apply(
        lambda t: "Linear" if t == "Linear" else "Non-Linear"
    )
    out_df = non_fp32[non_fp32["tensor_type"] == "output"]
    if out_df.empty:
        return []

    formats = sorted(non_fp32["format"].unique())
    linear_snr, nonlin_snr = [], []
    for fmt in formats:
        lin = out_df[(out_df["format"] == fmt) & (out_df["layer_class"] == "Linear")]["snr_db"]
        nln = out_df[(out_df["format"] == fmt) & (out_df["layer_class"] == "Non-Linear")]["snr_db"]
        linear_snr.append(lin.mean() if len(lin) > 0 else float("nan"))
        nonlin_snr.append(nln.mean() if len(nln) > 0 else float("nan"))

    x     = np.arange(len(formats))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(10, len(formats) * 0.8), 5))
    ax.bar(x - width / 2, linear_snr,  width, label="Linear",     color="#2196F3", alpha=0.85)
    ax.bar(x + width / 2, nonlin_snr,  width, label="Non-Linear", color="#F44336", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(formats, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean Output Activation SNR (dB)")
    ax.set_title("QSNR Gap: Linear vs Non-Linear Layers by Format")
    ax.legend()
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return [_fig_to_b64(fig)]


# ── Section 8: End-to-end Linear layer SQNR ───────────────────────────────────

def _plot_e2e_snr(df: pd.DataFrame) -> list[str]:
    """End-to-end output SNR for nn.Linear layers after quantized matmul.

    For HAD+INT: y_q = Q(H(x)) @ Q(H(W))ᵀ / N  (true HAD-domain multiply)
    For others:  y_q = Q(x) @ Q(W)ᵀ + b
    This measures the actual output distortion a downstream layer sees.
    """
    if "e2e_snr_db" not in df.columns:
        return []
    sub = df[
        (df["format"] != "FP32")
        & df["e2e_snr_db"].notna()
        & np.isfinite(df["e2e_snr_db"].values.astype(float))
    ].copy()
    if sub.empty:
        return []

    pivot = sub.pivot_table(
        values="e2e_snr_db", index="format", columns="layer_name", aggfunc="mean"
    )
    if pivot.empty:
        return []

    return _heatmap_pages(
        pivot, "RdYlGn", None, None,
        "End-to-End SNR (dB) — higher is better",
        "End-to-End Linear Layer Output SQNR — Format × Layer",
        paginate_dim="cols",
    )


# ── Sections 9–11: Per-tensor SNR heatmaps ────────────────────────────────────

def _plot_sensitivity_heatmap(df: pd.DataFrame, tensor_type: str) -> list[str]:
    sub = df[(df["tensor_type"] == tensor_type) & (df["format"] != "FP32")]
    if sub.empty:
        return []
    pivot = sub.pivot_table(
        values="snr_db", index="format", columns="layer_name", aggfunc="mean"
    )
    return _heatmap_pages(
        pivot, "RdYlGn", None, None,
        "SNR (dB) — higher is better",
        f"Per-Layer {tensor_type.capitalize()} Quantization SNR (FP32 excluded)",
        paginate_dim="cols",
    )


# ── Section 12: MARE analysis ──────────────────────────────────────────────────

def _plot_mare_analysis(df: pd.DataFrame) -> list[str]:
    """Mean Absolute Relative Error by format and tensor type."""
    if "mare" not in df.columns:
        return []
    sub = df[df["format"] != "FP32"].copy()
    if sub.empty:
        return []

    pivot = sub.pivot_table(
        values="mare", index="format", columns="tensor_type", aggfunc="mean"
    )
    if pivot.empty:
        return []

    x = np.arange(len(pivot.index))
    width = 0.8 / max(len(pivot.columns), 1)
    fig, ax = plt.subplots(figsize=(max(10, len(pivot.index) * 0.7), 5))
    for i, tt in enumerate(pivot.columns):
        ax.bar(x + i * width, pivot[tt].values, width, label=tt, alpha=0.85)
    ax.set_xticks(x + width * (len(pivot.columns) - 1) / 2)
    ax.set_xticklabels(pivot.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean Absolute Relative Error (lower is better)")
    ax.set_title("MARE by Format and Tensor Type")
    ax.legend(title="Tensor type", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return [_fig_to_b64(fig)]


# ── Section 13: Format efficiency scatter ─────────────────────────────────────

def _plot_format_efficiency(df: pd.DataFrame) -> list[str]:
    summary = (
        df.groupby("format")
        .agg(bits=("bits", "first"), mean_eff=("eff_bits", "mean"))
        .reset_index()
    )
    summary = summary[np.isfinite(summary["mean_eff"].values)]
    if summary.empty:
        return []

    fig, ax = plt.subplots(figsize=(8, 6))
    ref = np.linspace(0, summary["bits"].max() + 2, 100)
    ax.plot(ref, ref, "k--", alpha=0.25, label="y=x (ideal)")

    bit_colors = {4: "#2196F3", 8: "#4CAF50", 16: "#FF9800", 32: "#9E9E9E"}
    for _, row in summary.iterrows():
        b = int(row["bits"]) if pd.notna(row["bits"]) else 8
        c = bit_colors.get(b, "#666666")
        ax.scatter(row["bits"], row["mean_eff"], color=c, s=80, zorder=5)
        ax.annotate(row["format"], (row["bits"], row["mean_eff"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=7)

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=9, label=f"{b}-bit")
        for b, c in bit_colors.items() if b in summary["bits"].values
    ]
    ax.legend(handles=legend_elems + [ax.lines[0]], fontsize=8)
    ax.set_xlabel("Nominal Bits"); ax.set_ylabel("Mean Effective Bits (EffBits)")
    ax.set_title("Format Efficiency: Nominal vs Effective Bits\n(closer to diagonal = better)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return [_fig_to_b64(fig)]


# ── Section 14: EffBits ranking ────────────────────────────────────────────────

def _plot_effbits_ranking(df: pd.DataFrame) -> list[str]:
    summary  = df.groupby("format")["eff_bits"].mean().sort_values(ascending=True)
    bits_map = df.groupby("format")["bits"].first()
    colors   = ["#2196F3" if bits_map.get(f, 8) <= 4 else "#4CAF50" for f in summary.index]
    fig, ax  = plt.subplots(figsize=(8, max(4, len(summary) * 0.4)))
    ax.barh(summary.index, summary.values, color=colors)
    ax.axvline(x=4, color="gray", linestyle="--", alpha=0.5, label="4-bit target")
    ax.axvline(x=8, color="gray", linestyle=":",  alpha=0.5, label="8-bit target")
    ax.set_xlabel("Mean EffBits")
    ax.set_title("EffBits Ranking  (blue=4-bit formats, green=8-bit+)")
    ax.legend()
    fig.tight_layout()
    return [_fig_to_b64(fig)]


# ── Section 15: SNR comparison ─────────────────────────────────────────────────

def _plot_snr_comparison(df: pd.DataFrame) -> list[str]:
    tensor_types = sorted(df["tensor_type"].dropna().unique())
    formats      = df["format"].unique()
    x     = np.arange(len(formats))
    width = 0.8 / max(len(tensor_types), 1)
    fig, ax = plt.subplots(figsize=(max(10, len(formats) * 0.8), 5))
    for i, tt in enumerate(tensor_types):
        sub  = df[df["tensor_type"] == tt].groupby("format")["snr_db"].mean()
        vals = [sub.get(f, float("nan")) for f in formats]
        ax.bar(x + i * width, vals, width, label=tt)
    ax.set_xticks(x + width * (len(tensor_types) - 1) / 2)
    ax.set_xticklabels(formats, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean SNR (dB)")
    ax.set_title("SNR Comparison by Format and Tensor Type")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    return [_fig_to_b64(fig)]


# ── Section 16: Summary table ──────────────────────────────────────────────────

def _build_summary_table(df: pd.DataFrame) -> str:
    """Per-format aggregate table with per-tensor-type SNR breakdown."""
    # Build aggregation spec — only include columns that actually exist in the CSV
    agg_spec: dict = {
        "bits":        ("bits",    "first"),
        "mean_snr_db": ("snr_db",  "mean"),
        "mean_eff_bits": ("eff_bits", "mean"),
        "mean_mse":    ("mse",     "mean"),
        "mean_max_ae": ("max_ae",  "mean"),
        "outlier_ratio": ("outlier_ratio", "mean"),
    }
    if "mare" in df.columns:
        agg_spec["mean_mare"] = ("mare", "mean")
    if "saturation_rate" in df.columns:
        agg_spec["mean_sat"] = ("saturation_rate", "mean")

    base = (
        df.groupby("format")
        .agg(**agg_spec)
        .sort_values("mean_eff_bits", ascending=False)
        .reset_index()
    )

    # Per-tensor-type SNR
    tt_pivot = (
        df.groupby(["format", "tensor_type"])["snr_db"]
        .mean()
        .unstack(fill_value=float("nan"))
    )

    # E2E SNR per format (mean over layers)
    e2e_by_fmt = {}
    if "e2e_snr_db" in df.columns:
        e2e_by_fmt = (
            df[df["e2e_snr_db"].notna()]
            .groupby("format")["e2e_snr_db"]
            .mean()
            .to_dict()
        )

    def _snr(fmt, tt):
        if fmt not in tt_pivot.index or tt not in tt_pivot.columns:
            return "—"
        v = tt_pivot.at[fmt, tt]
        return f"{v:.1f}" if np.isfinite(v) else "—"

    def _fmt_f(v, fmt=".2f"):
        return f"{v:{fmt}}" if pd.notna(v) and np.isfinite(v) else "—"

    rows = []
    for _, r in base.iterrows():
        fmt = r["format"]
        bits_str = str(int(r["bits"])) if pd.notna(r["bits"]) else "—"
        e2e_v = e2e_by_fmt.get(fmt, float("nan"))
        mare_cell = _fmt_f(r.get("mean_mare", float("nan")), ".4f")
        sat_cell  = _fmt_f(r.get("mean_sat",  float("nan")), ".4f")
        rows.append(
            f"<tr>"
            f"<td>{fmt}</td><td>{bits_str}</td>"
            f"<td>{_fmt_f(r['mean_snr_db'], '.1f')}</td>"
            f"<td>{_snr(fmt, 'weight')}</td>"
            f"<td>{_snr(fmt, 'input')}</td>"
            f"<td>{_snr(fmt, 'output')}</td>"
            f"<td>{'%.1f' % e2e_v if np.isfinite(e2e_v) else '—'}</td>"
            f"<td>{_fmt_f(r['mean_eff_bits'])}</td>"
            f"<td>{_fmt_f(r['mean_mse'], '.2e')}</td>"
            f"<td>{_fmt_f(r['mean_max_ae'], '.2e')}</td>"
            f"<td>{mare_cell}</td>"
            f"<td>{sat_cell}</td>"
            f"<td>{_fmt_f(r['outlier_ratio'], '.4f')}</td>"
            f"</tr>"
        )
    return "\n".join(rows)


# ── Main entry point ───────────────────────────────────────────────────────────

def generate_report(
    csv_path: str,
    log_path: str,
    out_dir: str,
    hist_path: str | None = None,
    open_browser: bool = True,
) -> str:
    """Generate self-contained HTML report. Returns absolute path to report.html."""
    df = pd.read_csv(csv_path)
    with open(log_path) as f:
        log = json.load(f)

    hist_data: dict | None = None
    if hist_path and os.path.exists(hist_path):
        with open(hist_path) as f:
            hist_data = json.load(f)
    elif hist_path is None:
        candidate = os.path.join(out_dir, "profiler_histograms.json")
        if os.path.exists(candidate):
            with open(candidate) as f:
                hist_data = json.load(f)

    final_acc  = log["test_acc"][-1]
    final_loss = log["test_loss"][-1]

    print("  [ 1/16] Training curves ...")
    img_training = _plot_training_curves(log)

    print("  [ 2/16] FP32 distributions ...")
    imgs_fp32 = _plot_fp32_distributions(hist_data) if hist_data else []

    print("  [ 3/16] Distribution shape (kurtosis / skewness) ...")
    imgs_shape = _plot_distribution_shape(df)

    print("  [ 4/16] Outlier analysis ...")
    imgs_outlier = _plot_outlier_analysis(df)

    print("  [ 5/16] HAD energy spread ...")
    imgs_had = _plot_had_energy_spread(df)

    print("  [ 6/16] Saturation analysis ...")
    imgs_sat = _plot_saturation_analysis(df)

    print("  [ 7/16] Linear vs Non-Linear QSNR ...")
    imgs_lin_nl = _plot_linear_vs_nonlinear(df)

    print("  [ 8/16] End-to-end Linear SQNR ...")
    imgs_e2e = _plot_e2e_snr(df)

    print("  [ 9/16] Weight SNR heatmap ...")
    imgs_w = _plot_sensitivity_heatmap(df, "weight")

    print("  [10/16] Input activation SNR heatmap ...")
    imgs_i = _plot_sensitivity_heatmap(df, "input")

    print("  [11/16] Output activation SNR heatmap ...")
    imgs_o = _plot_sensitivity_heatmap(df, "output")

    print("  [12/16] MARE analysis ...")
    imgs_mare = _plot_mare_analysis(df)

    print("  [13/16] Format efficiency scatter ...")
    imgs_eff = _plot_format_efficiency(df)

    print("  [14/16] EffBits ranking ...")
    imgs_effbits = _plot_effbits_ranking(df)

    print("  [15/16] SNR comparison ...")
    imgs_snr = _plot_snr_comparison(df)

    print("  [16/16] Summary table ...")
    table_rows = _build_summary_table(df)

    def _s(title, imgs, note=""):
        if not imgs:
            return _section(title, "<p><em>No data available.</em></p>", note)
        return _section(title, _imgs_html(imgs if isinstance(imgs, list) else [imgs]), note)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Quantization Format Analysis Report</title>
<style>
  body  {{ font-family: Arial, sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; color: #333; }}
  h1   {{ color: #222; }}
  h2   {{ color: #444; border-bottom: 2px solid #ddd; padding-bottom: 6px; margin-top: 36px; }}
  hr   {{ border: none; border-top: 1px solid #eee; margin: 20px 0; }}
  img  {{ max-width: 100%; }}
  .meta  {{ background: #f5f5f5; padding: 14px 18px; border-radius: 6px; margin-bottom: 28px; font-size: 0.95em; }}
  .note  {{ color: #555; font-size: 0.87em; margin-top: 6px; margin-bottom: 8px; }}
  table  {{ border-collapse: collapse; width: 100%; font-size: 0.88em; overflow-x: auto; display: block; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 9px; text-align: right; white-space: nowrap; }}
  th     {{ background: #f0f0f0; font-weight: bold; }}
  td:first-child, th:first-child {{ text-align: left; position: sticky; left: 0; background: #fff; }}
  tr:hover td {{ background: #fafafa; }}
</style>
</head>
<body>
<h1>Quantization Format Analysis Report</h1>
<div class="meta">
  <strong>Final Test Accuracy:</strong> {final_acc:.1f}% &nbsp;|&nbsp;
  <strong>Final Test Loss:</strong> {final_loss:.4f} &nbsp;|&nbsp;
  <strong>Formats analysed:</strong> {df["format"].nunique()} &nbsp;|&nbsp;
  <strong>Layers profiled:</strong> {df["layer_name"].nunique()}
  {"&nbsp;|&nbsp;<strong>Histogram data:</strong> available" if hist_data else ""}
</div>

{_section("1. Training Curves",
    f'<img src="data:image/png;base64,{img_training}" style="max-width:100%">',
    "Loss (left axis, blue) and accuracy (right axis, red) over training epochs.")}

{_s("2. Pre-Quantization Distributions (FP32)", imgs_fp32,
    "Histogram of raw tensor values (before any quantization). "
    "Outlier ratio = fraction of elements outside the range observed in the first batch."
    if imgs_fp32 else "Re-run profile_mnist.py to generate histogram data.")}

{_s("3. Distribution Shape Analysis (FP32)", imgs_shape,
    "Excess kurtosis (0=Gaussian, 3=Laplace; positive→heavy-tailed outlier-prone) and skewness per layer. "
    "High-kurtosis layers are the primary motivation for Hadamard rotation and per-channel scaling.")}

{_s("4. Outlier Analysis", imgs_outlier,
    "Layer heatmap: FP32 outlier ratio (fraction of elements outside first-batch range) per layer and tensor type. "
    "Bar chart: mean outlier ratio by format family — shows how HAD and per-channel scaling reduce effective outlier rates.")}

{_s("5. HAD Energy Spread (Kurtosis Reduction)", imgs_had,
    "For HAD+INT formats: kurtosis of the original tensor (FP32 domain) minus kurtosis after the Hadamard transform (quantization domain). "
    "Positive values confirm that HAD distributes energy more uniformly, reducing the outlier concentration that degrades quantization quality.")}

{_s("6. Saturation Analysis", imgs_sat,
    "Fraction of values clipped to the quantizer boundary per format and layer. "
    "High saturation indicates the scale is too small for the tensor's dynamic range — "
    "a trade-off between clipping distortion and rounding granularity.")}

{_s("7. Linear vs Non-Linear Layer QSNR Gap", imgs_lin_nl,
    "Mean output activation SNR for linear (nn.Linear) vs non-linear (LayerNorm, GELU, …) layers. "
    "Non-linear activations often have non-Gaussian distributions that are mismatched to uniform quantization grids.")}

{_s("8. End-to-End Linear Layer Output SQNR", imgs_e2e,
    "True end-to-end layer output SQNR simulating the actual hardware computation. "
    "For HAD+INT: y_q = Q(H(x)) @ Q(H(W))ᵀ / N (matmul in Hadamard domain). "
    "For others: y_q = Q(x) @ Q(W)ᵀ + b (both weights and activations quantized). "
    "This is the SQNR that downstream layers actually experience — "
    "the most direct measure of format quality for model accuracy.")}

{_s("9. Weight Quantization SNR Heatmap", imgs_w,
    "Per-tensor SQNR for weight matrices. FP32 excluded as baseline. "
    "For HAD+INT this is equivalent to the SNR of Q(H(W)) relative to H(W) (preserved across the H transform). "
    "Green = high fidelity; red = high distortion.")}

{_s("10. Input Activation SNR Heatmap", imgs_i,
    "Per-tensor SQNR for input activations. Activations are dynamic (batch-dependent) "
    "and typically harder to quantise than static weights.")}

{_s("11. Output Activation SNR Heatmap", imgs_o,
    "Per-tensor SQNR for output activations (applied uniformly across all formats for comparability). "
    "Note: for HAD+INT the output is in the original domain; this measures output distribution amenability, "
    "not the actual matmul error (see Section 8 for that).")}

{_s("12. MARE Analysis", imgs_mare,
    "Mean Absolute Relative Error: mean(|error| / (|original| + 1e-8)). "
    "Scale-invariant complement to MSE — penalises errors on small-magnitude values more heavily. "
    "Useful for detecting format failure on near-zero activations.")}

{_s("13. Format Efficiency Scatter", imgs_eff,
    "Each point is one format. The dashed diagonal is perfect efficiency (EffBits = nominal bits). "
    "Points below the diagonal waste bit-budget due to outliers or format mismatch.")}

{_s("14. EffBits Ranking", imgs_effbits,
    "Mean effective bits across all layers and tensor types, sorted ascending. "
    "Blue = 4-bit formats; green = 8-bit and above.")}

{_s("15. SNR Comparison by Format and Tensor Type", imgs_snr,
    "Grouped by tensor type. Shows whether a format degrades weights, activations, or both asymmetrically.")}

<h2>16. Summary Table (sorted by EffBits ↓)</h2>
<p class="note">
  <strong>E2E SNR</strong>: end-to-end layer output SNR (Section 8) — most important for model accuracy.
  <strong>MARE</strong>: mean absolute relative error — scale-invariant distortion measure.
  <strong>Sat.</strong>: mean saturation rate (fraction of clipped values).
  Sorted by mean effective bits descending.
</p>
<div style="overflow-x:auto">
<table>
<thead>
  <tr>
    <th>Format</th><th>Bits</th>
    <th>Mean SNR (dB)</th>
    <th>Weight SNR (dB)</th><th>Input SNR (dB)</th><th>Output SNR (dB)</th>
    <th>E2E SNR (dB)</th>
    <th>Mean EffBits</th><th>Mean MSE</th><th>Mean MaxAE</th>
    <th>MARE</th><th>Sat. Rate</th><th>Outlier Ratio</th>
  </tr>
</thead>
<tbody>
{table_rows}
</tbody>
</table>
</div>
<hr>

</body>
</html>"""

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    abs_path = os.path.abspath(out_path)
    print(f"\n  Report saved → {abs_path}")

    if open_browser:
        import webbrowser, pathlib
        webbrowser.open(pathlib.Path(abs_path).resolve().as_uri())

    return abs_path


def main():
    parser = argparse.ArgumentParser(description="Generate HTML quantization report")
    parser.add_argument("--results-dir", default="results/mnist")
    args = parser.parse_args()

    csv_path  = os.path.join(args.results_dir, "profiler_results.csv")
    log_path  = os.path.join(args.results_dir, "training_log.json")

    for p in [csv_path, log_path]:
        if not os.path.exists(p):
            print(f"ERROR: missing {p}")
            print("Run train_mnist.py and profile_mnist.py first.")
            sys.exit(1)

    print("Generating report ...")
    generate_report(csv_path, log_path, args.results_dir, open_browser=True)


if __name__ == "__main__":
    main()
