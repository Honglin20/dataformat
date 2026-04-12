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


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _section(title: str, b64: str | None, extra_html: str = "") -> str:
    """Return an HTML section block. Skips img tag if b64 is None."""
    img_html = f'<img src="data:image/png;base64,{b64}" style="max-width:100%">' if b64 else ""
    return f"<h2>{title}</h2>\n{img_html}\n{extra_html}\n<hr>\n"


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


# ── Section 2: Pre-quantization distributions (FP32 histograms) ───────────────

def _plot_fp32_distributions(hist_data: dict) -> str | None:
    if not hist_data or "FP32" not in hist_data:
        return None
    fp32 = hist_data["FP32"]
    # Select layers that have at least weight data; take up to 4
    layers = [l for l in fp32 if fp32[l]]
    if not layers:
        return None
    # Prefer layers with weight tensors first (linear layers)
    layers_with_weight = [l for l in layers if "weight" in fp32[l]]
    selected = (layers_with_weight if layers_with_weight else layers)[:4]

    tensor_types = ["weight", "input", "output"]
    nrows = len(selected)
    ncols = len(tensor_types)

    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.2 * nrows),
                              squeeze=False)
    for i, layer in enumerate(selected):
        for j, tt in enumerate(tensor_types):
            ax = axes[i][j]
            if tt not in fp32[layer]:
                ax.axis("off")
                continue
            h = fp32[layer][tt]
            edges  = h["hist_edges"]
            counts = h["hist_counts"]
            if not counts:
                ax.axis("off")
                continue
            centers = [(edges[k] + edges[k + 1]) / 2 for k in range(len(counts))]
            widths  = [edges[k + 1] - edges[k] for k in range(len(counts))]
            ax.bar(centers, counts, width=widths, color="#2196F3", alpha=0.75, linewidth=0)
            or_ = h.get("outlier_ratio", 0.0)
            short = layer.split(".")[-1] if "." in layer else layer
            ax.set_title(f"{short} · {tt}\noutlier={or_:.4f}", fontsize=7.5)
            ax.tick_params(labelsize=6)
            ax.set_ylabel("count" if j == 0 else "", fontsize=7)

    fig.suptitle(
        "Pre-Quantization (FP32) Distributions — Weight / Input Activation / Output Activation",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    return _fig_to_b64(fig)


# ── Section 3: Outlier analysis ────────────────────────────────────────────────

def _plot_outlier_analysis(df: pd.DataFrame) -> str | None:
    fp32 = df[df["format"] == "FP32"].copy()
    if fp32.empty:
        return None

    # Left subplot: FP32 outlier_ratio heatmap per layer × tensor_type
    pivot = fp32.pivot_table(
        values="outlier_ratio", index="layer_name", columns="tensor_type", aggfunc="mean"
    )

    # Right subplot: mean outlier_ratio grouped by format family
    format_families = {
        "INT\n(tensor)":   ["INT4(TENSOR)",  "INT8(TENSOR)"],
        "INT\n(channel)":  ["INT4(CHANNEL)", "INT8(CHANNEL)"],
        "HAD+INT":         ["HAD+INT4(C)",   "HAD+INT8(C)", "HAD+INT4(T)", "HAD+INT8(T)"],
        "MX":              ["MXINT4",        "MXINT8"],
        "SQ/FP":           ["SQ-FORMAT-INT", "SQ-FORMAT-FP", "FP16"],
    }
    family_means: dict[str, float] = {}
    for family, fmts in format_families.items():
        sub = df[df["format"].isin(fmts)]
        if not sub.empty:
            family_means[family] = sub["outlier_ratio"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(pivot) * 0.5 + 1)))

    # Left: heatmap
    if not pivot.empty:
        im = axes[0].imshow(pivot.values, aspect="auto", cmap="Oranges",
                            vmin=0, vmax=max(pivot.values.max(), 1e-9))
        axes[0].set_xticks(range(len(pivot.columns)))
        axes[0].set_xticklabels(pivot.columns, fontsize=8)
        axes[0].set_yticks(range(len(pivot.index)))
        axes[0].set_yticklabels(pivot.index, fontsize=7)
        plt.colorbar(im, ax=axes[0], label="Outlier Ratio")
    axes[0].set_title("FP32 Outlier Ratio per Layer\n(baseline — before any quantization)", fontsize=9)

    # Right: bar chart by family
    if family_means:
        families = list(family_means.keys())
        vals = [family_means[g] for g in families]
        pal = ["#F44336", "#FF9800", "#4CAF50", "#2196F3", "#9C27B0"]
        axes[1].bar(families, vals, color=pal[:len(families)], alpha=0.85)
        axes[1].set_ylabel("Mean Outlier Ratio (all layers, all tensor types)")
        axes[1].set_title("Outlier Ratio by Format Family", fontsize=9)
        axes[1].tick_params(axis="x", labelsize=8)
    else:
        axes[1].axis("off")

    fig.tight_layout()
    return _fig_to_b64(fig)


# ── Section 4: Linear vs Non-Linear QSNR gap ──────────────────────────────────

def _plot_linear_vs_nonlinear(df: pd.DataFrame) -> str | None:
    non_fp32 = df[df["format"] != "FP32"].copy()
    if non_fp32.empty or "layer_type" not in non_fp32.columns:
        return None

    non_fp32["layer_class"] = non_fp32["layer_type"].apply(
        lambda t: "Linear" if t == "Linear" else "Non-Linear"
    )
    # Focus on output activations for fair layer-type comparison
    out_df = non_fp32[non_fp32["tensor_type"] == "output"]
    if out_df.empty:
        return None

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
    ax.bar(x - width / 2, linear_snr,  width, label="Linear layers",     color="#2196F3", alpha=0.85)
    ax.bar(x + width / 2, nonlin_snr,  width, label="Non-Linear layers",  color="#F44336", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(formats, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean Output Activation SNR (dB)")
    ax.set_title("QSNR Gap: Linear vs Non-Linear Layers by Format\n"
                 "(output activations — higher = less quantization distortion)", fontsize=10)
    ax.legend()
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ── Sections 5–7: Per-layer sensitivity heatmaps ──────────────────────────────

def _plot_sensitivity_heatmap(df: pd.DataFrame, tensor_type: str) -> str | None:
    sub = df[(df["tensor_type"] == tensor_type) & (df["format"] != "FP32")]
    if sub.empty:
        return None
    pivot = sub.pivot_table(
        values="snr_db", index="format", columns="layer_name", aggfunc="mean"
    )
    vals = pivot.values.astype(float)
    finite = vals[np.isfinite(vals)]
    vmin = float(np.percentile(finite, 5))  if len(finite) else -10
    vmax = float(np.percentile(finite, 95)) if len(finite) else  40

    fig, ax = plt.subplots(
        figsize=(max(8, len(pivot.columns) * 0.85), max(4, len(pivot) * 0.55))
    )
    im = ax.imshow(vals, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    plt.colorbar(im, ax=ax, label="SNR (dB) — higher is better")
    ax.set_title(
        f"Per-Layer {tensor_type.capitalize()} Quantization SNR Heatmap\n"
        f"(FP32 excluded as baseline; green = high fidelity, red = high distortion)",
        fontsize=10,
    )
    fig.tight_layout()
    return _fig_to_b64(fig)


# ── Section 8: Format efficiency scatter ──────────────────────────────────────

def _plot_format_efficiency(df: pd.DataFrame) -> str | None:
    summary = (
        df.groupby("format")
        .agg(bits=("bits", "first"), mean_eff=("eff_bits", "mean"))
        .reset_index()
    )
    summary = summary[np.isfinite(summary["mean_eff"].values)]
    if summary.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    ref = np.linspace(0, summary["bits"].max() + 2, 100)
    ax.plot(ref, ref, "k--", alpha=0.25, label="y = x  (ideal efficiency)")

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
    ax.set_xlabel("Nominal Bits")
    ax.set_ylabel("Mean Effective Bits (EffBits)")
    ax.set_title(
        "Format Efficiency: Nominal vs Effective Bits\n"
        "(closer to diagonal = better capacity utilisation)",
        fontsize=10,
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ── Section 9: EffBits ranking ─────────────────────────────────────────────────

def _plot_effbits_ranking(df: pd.DataFrame) -> str:
    summary  = df.groupby("format")["eff_bits"].mean().sort_values(ascending=True)
    bits_map = df.groupby("format")["bits"].first()
    colors   = ["#2196F3" if bits_map.get(f, 8) <= 4 else "#4CAF50" for f in summary.index]
    fig, ax  = plt.subplots(figsize=(8, max(4, len(summary) * 0.4)))
    ax.barh(summary.index, summary.values, color=colors)
    ax.axvline(x=4, color="gray", linestyle="--", alpha=0.5, label="4-bit target")
    ax.axvline(x=8, color="gray", linestyle=":",  alpha=0.5, label="8-bit target")
    ax.set_xlabel("Mean EffBits")
    ax.set_title("EffBits Ranking  (blue = 4-bit formats, green = 8-bit+)")
    ax.legend()
    fig.tight_layout()
    return _fig_to_b64(fig)


# ── Section 10: SNR comparison ─────────────────────────────────────────────────

def _plot_snr_comparison(df: pd.DataFrame) -> str:
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
    return _fig_to_b64(fig)


# ── Section 11: Summary table ──────────────────────────────────────────────────

def _build_summary_table(df: pd.DataFrame) -> str:
    summary = (
        df.groupby("format")
        .agg(bits=("bits", "first"), mean_mse=("mse", "mean"),
             mean_snr_db=("snr_db", "mean"), mean_eff_bits=("eff_bits", "mean"),
             outlier_ratio=("outlier_ratio", "mean"))
        .sort_values("mean_eff_bits", ascending=False)
        .reset_index()
    )
    rows = []
    for _, r in summary.iterrows():
        bits_str = str(int(r["bits"])) if pd.notna(r["bits"]) else "-"
        rows.append(
            f"<tr><td>{r['format']}</td><td>{bits_str}</td>"
            f"<td>{r['mean_mse']:.2e}</td><td>{r['mean_snr_db']:.1f}</td>"
            f"<td>{r['mean_eff_bits']:.2f}</td><td>{r['outlier_ratio']:.4f}</td></tr>"
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

    # Auto-detect histogram file
    if hist_path is None:
        candidate = os.path.join(out_dir, "profiler_histograms.json")
        if os.path.exists(candidate):
            hist_path = candidate
    hist_data: dict | None = None
    if hist_path and os.path.exists(hist_path):
        with open(hist_path) as f:
            hist_data = json.load(f)

    final_acc  = log["test_acc"][-1]
    final_loss = log["test_loss"][-1]

    print("  [1/11] Training curves ...")
    img_training  = _plot_training_curves(log)
    print("  [2/11] Pre-quantization distributions ...")
    img_fp32_dist = _plot_fp32_distributions(hist_data)
    print("  [3/11] Outlier analysis ...")
    img_outliers  = _plot_outlier_analysis(df)
    print("  [4/11] Linear vs Non-Linear QSNR ...")
    img_lin_nl    = _plot_linear_vs_nonlinear(df)
    print("  [5/11] Weight sensitivity heatmap ...")
    img_w_heat    = _plot_sensitivity_heatmap(df, "weight")
    print("  [6/11] Input activation heatmap ...")
    img_i_heat    = _plot_sensitivity_heatmap(df, "input")
    print("  [7/11] Output activation heatmap ...")
    img_o_heat    = _plot_sensitivity_heatmap(df, "output")
    print("  [8/11] Format efficiency scatter ...")
    img_eff_scat  = _plot_format_efficiency(df)
    print("  [9/11] EffBits ranking ...")
    img_effbits   = _plot_effbits_ranking(df)
    print("  [10/11] SNR comparison ...")
    img_snr       = _plot_snr_comparison(df)
    print("  [11/11] Summary table ...")
    table_rows    = _build_summary_table(df)

    def _img(b64):
        return f'<img src="data:image/png;base64,{b64}" style="max-width:100%">' if b64 else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MNIST Transformer Quantization Report</title>
<style>
  body  {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; color: #333; }}
  h1   {{ color: #222; }}
  h2   {{ color: #444; border-bottom: 2px solid #ddd; padding-bottom: 6px; margin-top: 36px; }}
  hr   {{ border: none; border-top: 1px solid #eee; margin: 20px 0; }}
  img  {{ max-width: 100%; }}
  .meta  {{ background: #f5f5f5; padding: 14px 18px; border-radius: 6px; margin-bottom: 28px; font-size: 0.95em; }}
  .note  {{ color: #666; font-size: 0.87em; margin-top: 6px; }}
  table  {{ border-collapse: collapse; width: 100%; font-size: 0.9em; }}
  th, td {{ border: 1px solid #ddd; padding: 7px 10px; text-align: right; }}
  th     {{ background: #f0f0f0; font-weight: bold; }}
  td:first-child, th:first-child {{ text-align: left; }}
  tr:hover {{ background: #fafafa; }}
</style>
</head>
<body>
<h1>MNIST Transformer — Quantization Format Analysis Report</h1>
<div class="meta">
  <strong>Model:</strong> MNISTTransformer (2 encoder layers, d_model=128, nhead=4) &nbsp;|&nbsp;
  <strong>Final Test Accuracy:</strong> {final_acc:.1f}% &nbsp;|&nbsp;
  <strong>Final Test Loss:</strong> {final_loss:.4f} &nbsp;|&nbsp;
  <strong>Formats analysed:</strong> {df["format"].nunique()}
  {"&nbsp;|&nbsp;<strong>Distribution data:</strong> available" if hist_data else ""}
</div>

<h2>1. Training Curves</h2>
<p class="note">Loss (left axis, blue) and accuracy (right axis, red) over training epochs.</p>
{_img(img_training)}
<hr>

<h2>2. Pre-Quantization Distributions (FP32 Baseline)</h2>
<p class="note">Histogram of raw tensor values (before any quantization) for selected linear layers.
Outlier ratio = fraction of elements outside the range captured at first batch.
Heavy-tailed or multi-modal distributions motivate outlier-aware formats (channel-wise, HAD+, MX).</p>
{_img(img_fp32_dist) if img_fp32_dist else '<p><em>Histogram data not available. Re-run profile_mnist.py to generate profiler_histograms.json.</em></p>'}
<hr>

<h2>3. Outlier Analysis</h2>
<p class="note">Left: FP32 outlier ratio per layer and tensor type — reveals which layers have heavy tails.
Right: mean outlier ratio grouped by format family — shows how channel-wise scaling and Hadamard transforms reduce outliers.</p>
{_img(img_outliers)}
<hr>

<h2>4. Linear vs Non-Linear Layer QSNR Gap</h2>
<p class="note">Compares output activation SNR between linear layers (nn.Linear) and non-linear layers (LayerNorm, etc.).
A large gap indicates that a format's quantisation grid is mismatched to non-Gaussian activation distributions.</p>
{_img(img_lin_nl) if img_lin_nl else '<p><em>Insufficient layer_type diversity to compute gap.</em></p>'}
<hr>

<h2>5. Per-Layer Weight Quantization Heatmap (SNR)</h2>
<p class="note">SNR (dB) for weight tensors across all formats and layers. FP32 excluded as it is the reference baseline.
Green = high fidelity; red = high distortion. Reveals which layers are most sensitive to weight quantisation.</p>
{_img(img_w_heat) if img_w_heat else '<p><em>No weight data available.</em></p>'}
<hr>

<h2>6. Per-Layer Input Activation Quantization Heatmap (SNR)</h2>
<p class="note">SNR for input activations. Activation distributions are dynamic and batch-dependent,
making them generally harder to quantise than static weights.</p>
{_img(img_i_heat) if img_i_heat else '<p><em>No input activation data available.</em></p>'}
<hr>

<h2>7. Per-Layer Output Activation Quantization Heatmap (SNR)</h2>
<p class="note">SNR for output activations. Captures the cumulative quantisation effect as signal propagates
through the layer (input distortion + weight distortion combined).</p>
{_img(img_o_heat) if img_o_heat else '<p><em>No output activation data available.</em></p>'}
<hr>

<h2>8. Format Efficiency: Nominal vs Effective Bits</h2>
<p class="note">Each point is one format. The dashed diagonal is perfect efficiency (EffBits = nominal bits).
Points below the diagonal waste bit-budget; distance from the diagonal quantifies efficiency loss due to outliers or format mismatch.</p>
{_img(img_eff_scat) if img_eff_scat else '<p><em>Insufficient data.</em></p>'}
<hr>

<h2>9. EffBits Ranking by Format</h2>
<p class="note">Mean effective bits across all layers and tensor types, sorted ascending.
Blue = 4-bit formats; green = 8-bit and above.</p>
{_img(img_effbits)}
<hr>

<h2>10. SNR Comparison by Format and Tensor Type</h2>
<p class="note">Grouped by tensor type (weight / input / output). Shows whether a format degrades
weights, activations, or both — and whether degradation is symmetric.</p>
{_img(img_snr)}
<hr>

<h2>11. Summary Table (sorted by EffBits ↓)</h2>
<p class="note">Aggregate statistics across all layers and tensor types, sorted by mean effective bits descending.</p>
<table>
<thead>
  <tr>
    <th>Format</th><th>Bits</th><th>Mean MSE</th>
    <th>Mean SNR (dB)</th><th>Mean EffBits</th><th>Outlier Ratio</th>
  </tr>
</thead>
<tbody>
{table_rows}
</tbody>
</table>

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
    hist_path = os.path.join(args.results_dir, "profiler_histograms.json")

    for p in [csv_path, log_path]:
        if not os.path.exists(p):
            print(f"ERROR: missing {p}")
            print("Run train_mnist.py and profile_mnist.py first.")
            sys.exit(1)

    print("Generating report ...")
    generate_report(
        csv_path, log_path, args.results_dir,
        hist_path=hist_path if os.path.exists(hist_path) else None,
        open_browser=True,
    )


if __name__ == "__main__":
    main()
