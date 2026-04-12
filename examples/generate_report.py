"""Generate a self-contained HTML report from profiler results.

Usage:
    python examples/generate_report.py [--results-dir results/mnist]

Requires:
    results/mnist/profiler_results.csv  (from profile_mnist.py)
    results/mnist/training_log.json     (from train_mnist.py)

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


# ── Chart helpers ──────────────────────────────────────────────────────────────

def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _plot_training_curves(log: dict) -> str:
    epochs = log["epoch"]
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(epochs, log["train_loss"], "b-",  label="Train Loss")
    ax1.plot(epochs, log["test_loss"],  "b--", label="Test Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="b")
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


def _plot_effbits_ranking(df: pd.DataFrame) -> str:
    summary   = df.groupby("format")["eff_bits"].mean().sort_values(ascending=True)
    bits_map  = df.groupby("format")["bits"].first()
    colors    = ["#2196F3" if bits_map.get(f, 8) <= 4 else "#4CAF50" for f in summary.index]
    fig, ax   = plt.subplots(figsize=(8, max(4, len(summary) * 0.4)))
    ax.barh(summary.index, summary.values, color=colors)
    ax.axvline(x=4, color="gray", linestyle="--", alpha=0.5, label="4-bit target")
    ax.axvline(x=8, color="gray", linestyle=":",  alpha=0.5, label="8-bit target")
    ax.set_xlabel("Mean EffBits")
    ax.set_title("EffBits Ranking  (blue = 4-bit formats, green = 8-bit+)")
    ax.legend()
    fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_layer_mse_heatmap(df: pd.DataFrame) -> str | None:
    weight_df = df[(df["tensor_type"] == "weight") & (df["format"] != "FP32")]
    if weight_df.empty:
        return None
    pivot = weight_df.pivot_table(
        values="mse", index="format", columns="layer_name", aggfunc="mean"
    )
    log_vals = np.log10(np.clip(pivot.values, 1e-12, None))
    fig, ax  = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.2), max(4, len(pivot) * 0.5)))
    im = ax.imshow(log_vals, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    plt.colorbar(im, ax=ax, label="log₁₀(MSE)")
    ax.set_title("Per-Layer Weight MSE Heatmap (log scale, FP32 excluded)")
    fig.tight_layout()
    return _fig_to_b64(fig)


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
    ax.set_title("SNR by Format and Tensor Type")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    return _fig_to_b64(fig)


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
        bits_str = str(int(r["bits"])) if not pd.isna(r["bits"]) else "-"
        rows.append(
            f"<tr><td>{r['format']}</td><td>{bits_str}</td>"
            f"<td>{r['mean_mse']:.2e}</td><td>{r['mean_snr_db']:.1f}</td>"
            f"<td>{r['mean_eff_bits']:.2f}</td><td>{r['outlier_ratio']:.4f}</td></tr>"
        )
    return "\n".join(rows)


# ── Main ───────────────────────────────────────────────────────────────────────

def generate_report(
    csv_path: str,
    log_path: str,
    out_dir: str,
    open_browser: bool = True,
) -> str:
    """Generate self-contained HTML report.

    Returns absolute path to the written report.html.
    """
    df  = pd.read_csv(csv_path)
    with open(log_path) as f:
        log = json.load(f)

    final_acc  = log["test_acc"][-1]
    final_loss = log["test_loss"][-1]

    print("  Rendering training curves ...")
    img_training = _plot_training_curves(log)
    print("  Rendering EffBits ranking ...")
    img_effbits  = _plot_effbits_ranking(df)
    print("  Rendering MSE heatmap ...")
    img_heatmap  = _plot_layer_mse_heatmap(df)
    print("  Rendering SNR comparison ...")
    img_snr      = _plot_snr_comparison(df)
    table_rows   = _build_summary_table(df)

    heatmap_html = (
        f'<h2>3. Per-Layer Weight MSE Heatmap</h2>'
        f'<img src="data:image/png;base64,{img_heatmap}" style="max-width:100%"><hr>'
        if img_heatmap else ""
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MNIST Transformer Quantization Report</title>
<style>
  body  {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; color: #333; }}
  h1   {{ color: #222; }}
  h2   {{ color: #444; border-bottom: 1px solid #ddd; padding-bottom: 6px; margin-top: 30px; }}
  hr   {{ border: none; border-top: 1px solid #eee; margin: 20px 0; }}
  img  {{ max-width: 100%; }}
  .summary {{ background: #f5f5f5; padding: 14px 18px; border-radius: 6px; margin-bottom: 24px; font-size: 0.95em; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.9em; }}
  th, td {{ border: 1px solid #ddd; padding: 7px 10px; text-align: right; }}
  th {{ background: #f0f0f0; font-weight: bold; }}
  td:first-child, th:first-child {{ text-align: left; }}
  tr:hover {{ background: #fafafa; }}
</style>
</head>
<body>
<h1>MNIST Transformer — Quantization Analysis Report</h1>
<div class="summary">
  <strong>Model:</strong> MNISTTransformer (2 layers, d_model=128, nhead=4) &nbsp;|&nbsp;
  <strong>Final Test Accuracy:</strong> {final_acc:.1f}% &nbsp;|&nbsp;
  <strong>Final Test Loss:</strong> {final_loss:.4f} &nbsp;|&nbsp;
  <strong>Formats analysed:</strong> {df['format'].nunique()}
</div>

<h2>1. Training Curves</h2>
<img src="data:image/png;base64,{img_training}" style="max-width:100%">
<hr>

<h2>2. EffBits Ranking by Format</h2>
<img src="data:image/png;base64,{img_effbits}" style="max-width:100%">
<hr>

{heatmap_html}

<h2>4. SNR by Format and Tensor Type</h2>
<img src="data:image/png;base64,{img_snr}" style="max-width:100%">
<hr>

<h2>5. Summary Table (sorted by EffBits ↓)</h2>
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
    print(f"  Report saved → {abs_path}")

    if open_browser:
        import webbrowser, pathlib
        webbrowser.open(pathlib.Path(abs_path).resolve().as_uri())

    return abs_path


def main():
    parser = argparse.ArgumentParser(description="Generate HTML quantization report")
    parser.add_argument("--results-dir", default="results/mnist")
    args = parser.parse_args()

    csv_path = os.path.join(args.results_dir, "profiler_results.csv")
    log_path = os.path.join(args.results_dir, "training_log.json")

    for p in [csv_path, log_path]:
        if not os.path.exists(p):
            print(f"ERROR: missing {p}")
            print("Run train_mnist.py and profile_mnist.py first.")
            sys.exit(1)

    print("Generating report ...")
    generate_report(csv_path, log_path, args.results_dir, open_browser=True)


if __name__ == "__main__":
    main()
