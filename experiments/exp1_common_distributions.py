"""Experiment 1: SQNR and multi-metric analysis across common distributions.

Uses 2D weight matrices [C_OUT × C_IN] to correctly simulate:
  - Per-channel quantization: each row = one output channel, scale per row
  - HAD transform applied per-row: no cross-channel contamination
  - MX block quantization within rows only (no cross-row block boundary)
  - Channel outlier distributions: entire output channels (rows) are outliers

Format groups
-------------
4-bit : FP16(ref), SQ-Format-INT, SQ-Format-FP, INT4(C), INT4(T),
        HAD+INT4(C), HAD+INT4(T), MXINT4, MXFP4
8-bit : FP16(ref), SQ-Format-INT, SQ-Format-FP, INT8(C), INT8(T),
        HAD+INT8(C), HAD+INT8(T), MXINT8, MXFP8

SQ-Format-INT and SQ-Format-FP are identical in both groups
(high_bits=8, low_bits=4, bank_size=128, sparsity=0.5) per the spec.

Outputs
-------
results/exp1/results_4bit.csv   — all metrics per (format × distribution)
results/exp1/results_8bit.csv
results/exp1/report.html        — interactive multi-metric HTML table
results/exp1/fig1_heatmap_{4,8}bit.png  — SQNR heatmaps
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

# Ensure the project root is first on sys.path so that `config`, `formats`,
# and `distributions` resolve to the top-level packages rather than the
# same-named shadows inside experiments/ (which Python prepends at script launch).
_ROOT = Path(__file__).resolve().parent.parent
_EXPERIMENTS = Path(__file__).resolve().parent
# Remove experiments/ dir shadow and place root first
sys.path = [str(_ROOT)] + [p for p in sys.path if p != str(_EXPERIMENTS)]

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

C_OUT    = 64          # output channels (rows)
C_IN     = 64          # input channels per output (cols)
SEED     = 42

OUT_DIR  = Path("results/exp1")

# Formats to run per bit-width group
FORMATS_4BIT = [
    "FP16", "SQ-Format-INT", "SQ-Format-FP",
    "INT4(T)", "INT4(C)", "HAD+INT4(T)", "HAD+INT4(C)",
    "MXINT4", "MXFP4",
]
FORMATS_8BIT = [
    "FP16", "SQ-Format-INT", "SQ-Format-FP",
    "INT8(T)", "INT8(C)", "HAD+INT8(T)", "HAD+INT8(C)",
    "MXINT8", "MXFP8",
]


# ── 2D Distribution generators ────────────────────────────────────────────────

def _make_gaussian(sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, sigma, size=(C_OUT, C_IN)).astype(np.float32)


def _make_laplace(b: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.laplace(0.0, b, size=(C_OUT, C_IN)).astype(np.float32)


def _make_student_t(nu: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_t(nu, size=(C_OUT, C_IN)).astype(np.float32)


def _make_bimodal(mu: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    half = C_IN // 2
    pos = rng.normal( mu, 0.3, size=(C_OUT, half))
    neg = rng.normal(-mu, 0.3, size=(C_OUT, C_IN - half))
    w = np.concatenate([pos, neg], axis=1).astype(np.float32)
    # Shuffle within each row so sign pattern is random
    for i in range(C_OUT):
        rng.shuffle(w[i])
    return w


def _make_channel_outlier(sigma_out: float, seed: int) -> np.ndarray:
    """~10% of output channels have sigma_out× larger magnitude (entire rows)."""
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 1.0, size=(C_OUT, C_IN)).astype(np.float32)
    n_out = max(1, C_OUT // 10)
    idx   = rng.choice(C_OUT, n_out, replace=False)
    w[idx] *= sigma_out
    return w


def _make_spiky(spike_mult: float, seed: int) -> np.ndarray:
    """1% of individual elements have spike_mult× larger magnitude."""
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 1.0, size=(C_OUT, C_IN)).astype(np.float32)
    n_spike = max(1, (C_OUT * C_IN) // 100)
    rows = rng.integers(0, C_OUT, n_spike)
    cols = rng.integers(0, C_IN,  n_spike)
    w[rows, cols] *= spike_mult
    return w


def _make_lognormal(sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.lognormal(0.0, sigma, size=(C_OUT, C_IN)).astype(np.float32)


def _make_uniform(bound: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-bound, bound, size=(C_OUT, C_IN)).astype(np.float32)


# (name, family, generator_fn taking seed)
DISTRIBUTIONS = [
    ("Gauss(σ=0.5)",   "Gaussian",      lambda s: _make_gaussian(0.5,  s)),
    ("Gauss(σ=1.0)",   "Gaussian",      lambda s: _make_gaussian(1.0,  s)),
    ("Gauss(σ=2.0)",   "Gaussian",      lambda s: _make_gaussian(2.0,  s)),
    ("Gauss(σ=5.0)",   "Gaussian",      lambda s: _make_gaussian(5.0,  s)),
    ("Laplace(b=0.5)", "Laplace",       lambda s: _make_laplace(0.5,   s)),
    ("Laplace(b=1.0)", "Laplace",       lambda s: _make_laplace(1.0,   s)),
    ("Laplace(b=2.0)", "Laplace",       lambda s: _make_laplace(2.0,   s)),
    ("Student-t(ν=3)", "Student-t",     lambda s: _make_student_t(3,   s)),
    ("Student-t(ν=5)", "Student-t",     lambda s: _make_student_t(5,   s)),
    ("Student-t(ν=10)","Student-t",     lambda s: _make_student_t(10,  s)),
    ("Bimodal(μ=±2.0)","Bimodal",       lambda s: _make_bimodal(2.0,   s)),
    ("Bimodal(μ=±3.0)","Bimodal",       lambda s: _make_bimodal(3.0,   s)),
    ("Bimodal(μ=±5.0)","Bimodal",       lambda s: _make_bimodal(5.0,   s)),
    ("ChanOut(σ=30)",  "Chan. Outlier", lambda s: _make_channel_outlier(30,  s)),
    ("ChanOut(σ=50)",  "Chan. Outlier", lambda s: _make_channel_outlier(50,  s)),
    ("ChanOut(σ=100)", "Chan. Outlier", lambda s: _make_channel_outlier(100, s)),
    ("Spiky(10×)",     "Spiky Outlier", lambda s: _make_spiky(10,  s)),
    ("Spiky(50×)",     "Spiky Outlier", lambda s: _make_spiky(50,  s)),
    ("Spiky(100×)",    "Spiky Outlier", lambda s: _make_spiky(100, s)),
    ("LogNorm(σ=1.0)", "Log-Normal",    lambda s: _make_lognormal(1.0, s)),
    ("LogNorm(σ=2.0)", "Log-Normal",    lambda s: _make_lognormal(2.0, s)),
    ("Uniform(±1.0)",  "Uniform",       lambda s: _make_uniform(1.0,   s)),
    ("Uniform(±3.0)",  "Uniform",       lambda s: _make_uniform(3.0,   s)),
]


# ── Core experiment ───────────────────────────────────────────────────────────

def _safe_metrics(x_orig: np.ndarray, x_recon: np.ndarray) -> dict:
    """Compute 5 metrics; clip extreme values for CSV sanity."""
    from distributions.metrics import evaluate_all
    m = evaluate_all(x_orig, x_recon)
    return {
        "SQNR_dB":  round(m["snr_db"],   4),
        "MSE":      round(m["mse"],       6),
        "KL_div":   round(m["kl_div"],    6),
        "MaxAE":    round(m["max_ae"],    4),
        "EffBits":  round(m["eff_bits"],  4),
    }


def run_experiment(format_keys: list[str], bits_label: str) -> list[dict]:
    from formats import build_all_formats
    all_fmts = build_all_formats()
    fmts = {k: all_fmts[k] for k in format_keys}

    rows = []
    for dist_name, family, gen_fn in DISTRIBUTIONS:
        W = gen_fn(SEED)
        for fmt_key, fmt in fmts.items():
            W_q = fmt.quantize(W)
            m   = _safe_metrics(W, W_q)
            rows.append({
                "format":       fmt_key,  # use registry key for consistent CSV names
                "bits":         bits_label,
                "family":       family,
                "distribution": dist_name,
                **m,
            })
        print(f"  {dist_name} done", flush=True)
    return rows


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["format", "bits", "family", "distribution",
                  "SQNR_dB", "MSE", "KL_div", "MaxAE", "EffBits"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {path}")


# ── HTML report ───────────────────────────────────────────────────────────────

_DIST_GROUPS = [
    ("Gaussian",      ["Gauss(σ=0.5)", "Gauss(σ=1.0)", "Gauss(σ=2.0)", "Gauss(σ=5.0)"]),
    ("Laplace",       ["Laplace(b=0.5)", "Laplace(b=1.0)", "Laplace(b=2.0)"]),
    ("Student-t",     ["Student-t(ν=3)", "Student-t(ν=5)", "Student-t(ν=10)"]),
    ("Bimodal",       ["Bimodal(μ=±2.0)", "Bimodal(μ=±3.0)", "Bimodal(μ=±5.0)"]),
    ("Chan. Outlier", ["ChanOut(σ=30)", "ChanOut(σ=50)", "ChanOut(σ=100)"]),
    ("Spiky Outlier", ["Spiky(10×)", "Spiky(50×)", "Spiky(100×)"]),
    ("Log-Normal",    ["LogNorm(σ=1.0)", "LogNorm(σ=2.0)"]),
    ("Uniform",       ["Uniform(±1.0)", "Uniform(±3.0)"]),
]
_ALL_DISTS = [d for _, ds in _DIST_GROUPS for d in ds]

_METRICS = [
    ("SQNR_dB", "SQNR (dB)",  "higher better"),
    ("MSE",     "MSE",         "lower better"),
    ("KL_div",  "KL div",      "lower better"),
    ("MaxAE",   "Max AE",      "lower better"),
    ("EffBits", "EffBits",     "higher better"),
]


def _color(val: float, vmin: float, vmax: float, higher_better: bool) -> str:
    if vmax == vmin:
        ratio = 0.5
    else:
        ratio = (val - vmin) / (vmax - vmin)
    ratio = max(0.0, min(1.0, ratio))
    if not higher_better:
        ratio = 1.0 - ratio
    if ratio < 0.5:
        r, g, b = 255, int(ratio * 2 * 220), int(60 * (1 - ratio * 2))
    else:
        r, g, b = int((1 - (ratio - 0.5) * 2) * 220), 200, int(60 * (ratio - 0.5) * 2)
    return f"rgb({r},{g},{b})"


def _build_metric_table(data: dict, formats: list[str], metric: str,
                        higher_better: bool, section_title: str) -> str:
    """Build one colored table for a single metric."""
    # Collect non-FP16 values for color scale
    vals = []
    for fmt in formats:
        if fmt == "FP16":
            continue
        for d in _ALL_DISTS:
            v = data.get(fmt, {}).get(d, {}).get(metric)
            if v is not None:
                vals.append(v)
    if not vals:
        return ""
    vmin, vmax = min(vals), max(vals)

    html = [f'<h3>{section_title} — {metric}</h3>']
    html.append('<div class="tw"><table>')

    # Header row 1: family groups
    html.append('<thead><tr><th rowspan="2" class="fc">Format</th>')
    for grp, dists in _DIST_GROUPS:
        html.append(f'<th colspan="{len(dists)}" class="gh">{grp}</th>')
    html.append('<th rowspan="2" class="ac">Avg</th></tr>')

    # Header row 2: individual distributions
    html.append('<tr>')
    for _, dists in _DIST_GROUPS:
        for d in dists:
            short = (d.replace("Gauss(σ=", "σ=").replace(")", "")
                      .replace("Laplace(b=", "b=").replace("Student-t(ν=", "ν=")
                      .replace("Bimodal(μ=", "μ=").replace("ChanOut(σ=", "σ=")
                      .replace("Spiky(", "").replace("×", "×")
                      .replace("LogNorm(σ=", "σ=").replace("Uniform(±", "±"))
            html.append(f'<th class="dh" title="{d}">{short}</th>')
    html.append('</tr></thead><tbody>')

    for fmt in formats:
        is_ref = fmt == "FP16"
        rc = ' class="rr"' if is_ref else ''
        html.append(f'<tr{rc}><td class="fn">{fmt}</td>')
        cell_vals = []
        for d in _ALL_DISTS:
            v = data.get(fmt, {}).get(d, {}).get(metric)
            if v is not None:
                cell_vals.append(v)
                if is_ref:
                    html.append(f'<td class="rc">{v:.2f}</td>')
                else:
                    c = _color(v, vmin, vmax, higher_better)
                    html.append(f'<td style="background:{c};color:#000">{v:.2f}</td>')
            else:
                html.append('<td class="na">—</td>')
        if cell_vals:
            avg = sum(cell_vals) / len(cell_vals)
            if is_ref:
                html.append(f'<td class="rc ac2">{avg:.2f}</td>')
            else:
                c = _color(avg, vmin, vmax, higher_better)
                html.append(f'<td style="background:{c};color:#000;font-weight:bold">{avg:.2f}</td>')
        html.append('</tr>')
    html.append('</tbody></table></div>')
    return '\n'.join(html)


def build_html_report(rows4: list[dict], rows8: list[dict]) -> str:
    # Build nested dicts: data[bits][fmt][dist][metric] = value
    def _index(rows):
        d: dict = {}
        for r in rows:
            fmt  = r["format"]
            dist = r["distribution"]
            if fmt not in d:
                d[fmt] = {}
            if dist not in d[fmt]:
                d[fmt][dist] = {}
            for m, _, _ in _METRICS:
                d[fmt][dist][m] = r[m]
        return d

    data4 = _index(rows4)
    data8 = _index(rows8)

    # Determine format order from rows
    seen4: list[str] = []
    for r in rows4:
        if r["format"] not in seen4:
            seen4.append(r["format"])
    seen8: list[str] = []
    for r in rows8:
        if r["format"] not in seen8:
            seen8.append(r["format"])

    body_parts = []
    for metric_key, metric_label, direction in _METRICS:
        higher = "higher" in direction
        body_parts.append(
            _build_metric_table(data4, seen4, metric_key, higher,
                                f"4-bit Formats — {metric_label}")
        )
        body_parts.append(
            _build_metric_table(data8, seen8, metric_key, higher,
                                f"8-bit Formats — {metric_label}")
        )

    css = """
body{font-family:'Segoe UI',Arial,sans-serif;background:#f5f7fa;color:#222;margin:0;padding:20px}
h1{text-align:center;font-size:1.6em;color:#1a2a4a;margin-bottom:4px}
h2{font-size:1.2em;color:#1a2a4a;margin:36px 0 8px;border-left:4px solid #3a7bd5;padding-left:10px}
h3{font-size:1em;color:#1a2a4a;margin:24px 0 6px;padding-left:6px}
.tw{overflow-x:auto;border-radius:8px;box-shadow:0 2px 12px rgba(0,0,0,.12);background:#fff;margin-bottom:20px}
table{border-collapse:collapse;width:100%;font-size:.76em}
th,td{padding:4px 6px;text-align:center;border:1px solid #dde2eb;white-space:nowrap}
th.fc{background:#1a2a4a;color:#fff;min-width:110px;text-align:left;padding-left:8px}
th.gh{background:#2c4a7e;color:#fff;font-size:.8em}
th.dh{background:#3a6ab5;color:#fff;font-size:.7em;font-weight:normal;max-width:52px;overflow:hidden;text-overflow:ellipsis}
th.ac{background:#1a2a4a;color:#ffd700;min-width:44px}
td.fn{font-weight:600;background:#f0f4fa;text-align:left;padding-left:8px;color:#1a2a4a;border-right:2px solid #ccd}
td.rc{background:#e8eefc;color:#444;font-style:italic}
tr.rr td.fn{background:#dce6f7}
td.na{color:#bbb;background:#fafafa}
td.ac2{border-left:2px solid #aab}
.legend{display:flex;align-items:center;gap:8px;margin:8px 0 16px;font-size:.82em;color:#555}
.lb{width:140px;height:12px;background:linear-gradient(to right,rgb(255,60,60),rgb(255,220,0),rgb(0,200,100));border-radius:3px;border:1px solid #ccc}
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Exp 1 — Multi-Metric Quantization Analysis</title>
<style>{css}</style>
</head>
<body>
<h1>Experiment 1 — Quantization Format Analysis Across Distributions</h1>
<p style="text-align:center;color:#666;font-size:.9em">
  Weight matrices [{C_OUT}×{C_IN}] · 23 distributions · 5 metrics<br>
  FP16 is reference baseline (italic cells). Color scale excludes FP16.
</p>

<div class="legend">
  <span>Worst</span><div class="lb"></div><span>Best</span>
  <span style="margin-left:16px;color:#999">| SQNR/EffBits: higher=green · MSE/KL/MaxAE: lower=green</span>
</div>

{''.join(f'<div>{p}</div>' for p in body_parts)}

<p style="font-size:.72em;color:#aaa;margin-top:24px;text-align:right">
  Generated from experiments/exp1_common_distributions.py
</p>
</body>
</html>
"""


# ── Figure generation ─────────────────────────────────────────────────────────

def _make_heatmap(rows: list[dict], bits_label: str, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    # Build matrix: formats × distributions
    fmts: list[str] = []
    for r in rows:
        if r["format"] not in fmts:
            fmts.append(r["format"])
    dists = _ALL_DISTS

    idx_fmt  = {f: i for i, f in enumerate(fmts)}
    idx_dist = {d: i for i, d in enumerate(dists)}
    mat = np.full((len(fmts), len(dists)), np.nan)
    for r in rows:
        fi = idx_fmt.get(r["format"])
        di = idx_dist.get(r["distribution"])
        if fi is not None and di is not None:
            mat[fi, di] = r["SQNR_dB"]

    fig, ax = plt.subplots(figsize=(14, max(4, len(fmts) * 0.6)))
    # Clip extreme values for color scale readability
    vmin = np.nanpercentile(mat, 5)
    vmax = np.nanpercentile(mat, 95)
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(dists)))
    ax.set_xticklabels(dists, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(fmts)))
    ax.set_yticklabels(fmts, fontsize=8)
    ax.set_title(f"SQNR (dB) — {bits_label}", fontsize=11, pad=10)
    plt.colorbar(im, ax=ax, label="SQNR (dB)")
    # Annotate cells
    for fi in range(len(fmts)):
        for di in range(len(dists)):
            v = mat[fi, di]
            if not np.isnan(v):
                ax.text(di, fi, f"{v:.0f}", ha="center", va="center",
                        fontsize=5, color="black")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", choices=["4", "8", "both"], default="both",
                        help="Which bit-widths to run (default: both)")
    parser.add_argument("--out-dir", default=str(OUT_DIR),
                        help="Output directory (default: results/exp1)")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows4: list[dict] = []
    rows8: list[dict] = []

    if args.bits in ("4", "both"):
        print("Running 4-bit formats…")
        rows4 = run_experiment(FORMATS_4BIT, "4")
        save_csv(rows4, out / "results_4bit.csv")

    if args.bits in ("8", "both"):
        print("Running 8-bit formats…")
        rows8 = run_experiment(FORMATS_8BIT, "8")
        save_csv(rows8, out / "results_8bit.csv")

    # HTML report (requires both bit-width results)
    if rows4 and rows8:
        html = build_html_report(rows4, rows8)
        rpt  = out / "report.html"
        rpt.write_text(html)
        print(f"Saved {rpt}")
    elif rows4:
        html = build_html_report(rows4, rows4)
        rpt  = out / "report_4bit.html"
        rpt.write_text(html)
        print(f"Saved {rpt}")
    elif rows8:
        html = build_html_report(rows8, rows8)
        rpt  = out / "report_8bit.html"
        rpt.write_text(html)
        print(f"Saved {rpt}")

    # Heatmap figures
    if rows4:
        _make_heatmap(rows4, "4-bit Formats", out / "fig1_heatmap_4bit.png")
    if rows8:
        _make_heatmap(rows8, "8-bit Formats", out / "fig1_heatmap_8bit.png")


if __name__ == "__main__":
    main()
