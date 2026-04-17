"""Experiment 2: SQNR vs Crest Factor sweep.

Theory
------
For a block X ∈ R^k with i.i.d. entries Xi ~ N(0, σ²):
  - Block RMS = σ (by definition of N(0,σ²))
  - Crest factor κ := max(|X|) / σ

For N = C_OUT × C_IN = 64×64 = 4096 i.i.d. N(0,1) samples, the natural
(expected) crest factor is κ_nat ≈ sqrt(2 ln N) ≈ 4.07.  Values below that
correspond to "sub-Gaussian" or truncated distributions; values above represent
outlier-contaminated tensors.

Injection model
---------------
We control κ via a single-outlier injection:
  1. Generate W_bg ~ N(0,1) [C_OUT × C_IN]; normalise so std(W_bg) = 1.
  2. Inject one outlier at a random position with magnitude κ × σ_bg = κ.
  3. This gives max|W| ≈ κ for κ ≥ κ_nat (single spike dominates).
     For κ < κ_nat some background elements exceed κ, so actual CF is set
     by the background; those points characterise the "no outlier" baseline.

Two placement variants are averaged over N_SEEDS seeds:
  - random: outlier placed at a random (row, col) position each seed.
  - per-channel (worst): the outlier is always at the start of row 0, so
    per-channel quantization sees at most one bad row.

Results structure
-----------------
For each (κ, format, seed): compute SQNR, MSE, MaxAE, EffBits.
Report the mean ± std over seeds.

Outputs
-------
results/exp2/results_4bit.csv     — mean SQNR / MSE / MaxAE / EffBits per (format × κ)
results/exp2/results_8bit.csv
results/exp2/report.html          — SQNR-vs-κ line plots + summary tables (self-contained)
results/exp2/fig_sqnr_4bit.png    — SQNR vs κ, 4-bit formats
results/exp2/fig_sqnr_8bit.png    — SQNR vs κ, 8-bit formats
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import sys
from pathlib import Path

# ── Ensure project root is first on sys.path ──────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
_EXPERIMENTS = Path(__file__).resolve().parent
sys.path = [str(_ROOT)] + [p for p in sys.path if p != str(_EXPERIMENTS)]

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

C_OUT    = 64
C_IN     = 64
N        = C_OUT * C_IN          # 4096 elements
KAPPA_NATURAL = float(np.sqrt(2 * np.log(N)))   # ≈ 4.07

# κ sweep: start below natural, go to extreme outlier
KAPPA_VALUES = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0]
N_SEEDS  = 20          # average over 20 realisations
OUT_DIR  = Path("results/exp2")

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


# ── Controlled crest-factor matrix ────────────────────────────────────────────

def make_cf_matrix(kappa: float, seed: int) -> np.ndarray:
    """Generate a [C_OUT × C_IN] matrix with a controlled crest factor.

    Background: W_bg ~ N(0,1), normalised so std(W_bg) = 1 exactly.
    Outlier   : a single element chosen at random is set to exactly κ.

    Actual CF = max|W|/std(W):
      - For κ ≥ κ_nat (≈4.07): the injected outlier is the max, CF ≈ κ.
      - For κ < κ_nat: background elements dominate; CF ≈ κ_nat (baseline).

    The outlier position is random (uniform over all N elements) so that
    averaging over seeds gives fair results for both per-tensor and per-channel
    formats.

    Parameters
    ----------
    kappa : float   Target outlier magnitude in units of background σ.
    seed  : int     RNG seed for reproducibility.

    Returns
    -------
    W : np.ndarray, shape (C_OUT, C_IN), float32
    """
    rng = np.random.default_rng(seed)
    W   = rng.normal(0.0, 1.0, size=(C_OUT, C_IN)).astype(np.float32)

    # Normalise background std to exactly 1.0 (remove seed variance in σ)
    W = (W / float(np.std(W))).astype(np.float32)

    # Place outlier at a random position
    r = int(rng.integers(0, C_OUT))
    c = int(rng.integers(0, C_IN))
    W[r, c] = float(kappa)

    return W


# ── Metrics ───────────────────────────────────────────────────────────────────

def _metrics(orig: np.ndarray, recon: np.ndarray) -> dict:
    from distributions.metrics import evaluate_all
    m = evaluate_all(orig, recon)
    return {
        "SQNR_dB":  m["snr_db"],
        "MSE":      m["mse"],
        "MaxAE":    m["max_ae"],
        "EffBits":  m["eff_bits"],
    }


# ── Core experiment ───────────────────────────────────────────────────────────

def run_experiment(format_keys: list[str], bits_label: str) -> list[dict]:
    """Run κ sweep for a list of format keys.

    Returns a list of row dicts, one per (format, κ), with mean and std of
    each metric over N_SEEDS seeds.
    """
    from formats import build_all_formats
    all_fmts = build_all_formats()
    fmts = {k: all_fmts[k] for k in format_keys}

    rows = []
    for kappa in KAPPA_VALUES:
        seed_results: dict[str, list[dict]] = {k: [] for k in format_keys}

        for seed in range(N_SEEDS):
            W = make_cf_matrix(kappa, seed)
            for fmt_key, fmt in fmts.items():
                W_q = fmt.quantize(W)
                seed_results[fmt_key].append(_metrics(W, W_q))

        # Aggregate over seeds
        for fmt_key in format_keys:
            sr = seed_results[fmt_key]
            row: dict = {
                "format": fmt_key,
                "bits":   bits_label,
                "kappa":  kappa,
            }
            for metric in ("SQNR_dB", "MSE", "MaxAE", "EffBits"):
                vals = [r[metric] for r in sr if np.isfinite(r[metric])]
                row[metric]         = round(float(np.mean(vals)), 4)
                row[metric + "_std"] = round(float(np.std(vals)),  4)
            rows.append(row)

        print(f"  κ={kappa:6.1f}  done", flush=True)
    return rows


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["format", "bits", "kappa",
              "SQNR_dB", "SQNR_dB_std",
              "MSE",     "MSE_std",
              "MaxAE",   "MaxAE_std",
              "EffBits", "EffBits_std"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {path}")


# ── Figure generation ─────────────────────────────────────────────────────────

# Colour palette (distinguishable for up to 9 formats)
_COLOURS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
]

_LINE_STYLE = {
    "FP16":         ("--", "x",  1.5),
    "SQ-Format-INT":(":",  "D",  1.8),
    "SQ-Format-FP": (":",  "P",  1.8),
    "INT4(T)":      ("-",  "v",  1.5),
    "INT4(C)":      ("-",  "^",  1.5),
    "INT8(T)":      ("-",  "v",  1.5),
    "INT8(C)":      ("-",  "^",  1.5),
    "HAD+INT4(T)":  ("-.", "s",  1.8),
    "HAD+INT4(C)":  ("-.", "o",  1.8),
    "HAD+INT8(T)":  ("-.", "s",  1.8),
    "HAD+INT8(C)":  ("-.", "o",  1.8),
    "MXINT4":       ("-",  "D",  1.5),
    "MXFP4":        ("-",  "P",  1.5),
    "MXINT8":       ("-",  "D",  1.5),
    "MXFP8":        ("-",  "P",  1.5),
}


def make_sqnr_figure(rows: list[dict], format_keys: list[str],
                     bits_label: str, out_path: Path) -> str:
    """Draw SQNR vs κ line plot.  Returns base64 PNG string for HTML embedding."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ""

    # Index data
    data: dict[str, dict[float, tuple[float, float]]] = {}
    for r in rows:
        fmt = r["format"]
        if fmt not in data:
            data[fmt] = {}
        data[fmt][r["kappa"]] = (r["SQNR_dB"], r["SQNR_dB_std"])

    fig, ax = plt.subplots(figsize=(11, 6))

    kappas = sorted(KAPPA_VALUES)
    for idx, fmt_key in enumerate(format_keys):
        if fmt_key not in data:
            continue
        ys    = [data[fmt_key].get(k, (np.nan, 0.0))[0] for k in kappas]
        yerrs = [data[fmt_key].get(k, (np.nan, 0.0))[1] for k in kappas]
        ls, mk, lw = _LINE_STYLE.get(fmt_key, ("-", "o", 1.5))
        col = _COLOURS[idx % len(_COLOURS)]

        ax.plot(kappas, ys, linestyle=ls, marker=mk, linewidth=lw,
                color=col, label=fmt_key, markersize=5, alpha=0.9)
        # Light error-band (±1 std)
        ys_arr    = np.array(ys,    dtype=float)
        yerr_arr  = np.array(yerrs, dtype=float)
        ax.fill_between(kappas,
                        ys_arr - yerr_arr,
                        ys_arr + yerr_arr,
                        color=col, alpha=0.10)

    # Mark κ_natural
    ax.axvline(KAPPA_NATURAL, color="gray", linestyle=":", linewidth=1,
               label=f"κ_nat ≈ {KAPPA_NATURAL:.1f}")

    ax.set_xlabel("Crest factor κ = max|X|/σ", fontsize=12)
    ax.set_ylabel("SQNR (dB)", fontsize=12)
    ax.set_title(f"SQNR vs Crest Factor — {bits_label}", fontsize=13)
    ax.set_xscale("log")
    ax.set_xticks(kappas)
    ax.set_xticklabels([str(int(k)) if k == int(k) else str(k) for k in kappas],
                       rotation=30, fontsize=8)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")

    # Also encode as base64 for HTML embedding
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    plt.close()

    print(f"Saved {out_path}")
    return b64


def make_maxae_figure(rows: list[dict], format_keys: list[str],
                      bits_label: str, out_path: Path) -> str:
    """Draw Max Absolute Error vs κ."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ""

    data: dict[str, dict[float, float]] = {}
    for r in rows:
        fmt = r["format"]
        if fmt not in data:
            data[fmt] = {}
        data[fmt][r["kappa"]] = r["MaxAE"]

    fig, ax = plt.subplots(figsize=(11, 6))
    kappas = sorted(KAPPA_VALUES)

    for idx, fmt_key in enumerate(format_keys):
        if fmt_key not in data:
            continue
        ys  = [data[fmt_key].get(k, np.nan) for k in kappas]
        ls, mk, lw = _LINE_STYLE.get(fmt_key, ("-", "o", 1.5))
        col = _COLOURS[idx % len(_COLOURS)]
        ax.plot(kappas, ys, linestyle=ls, marker=mk, linewidth=lw,
                color=col, label=fmt_key, markersize=5, alpha=0.9)

    ax.axvline(KAPPA_NATURAL, color="gray", linestyle=":", linewidth=1,
               label=f"κ_nat ≈ {KAPPA_NATURAL:.1f}")
    ax.set_xlabel("Crest factor κ", fontsize=12)
    ax.set_ylabel("Max Absolute Error", fontsize=12)
    ax.set_title(f"Max Absolute Error vs Crest Factor — {bits_label}", fontsize=13)
    ax.set_xscale("log")
    ax.set_xticks(kappas)
    ax.set_xticklabels([str(int(k)) if k == int(k) else str(k) for k in kappas],
                       rotation=30, fontsize=8)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    plt.close()
    print(f"Saved {out_path}")
    return b64


def make_degradation_figure(rows4: list[dict], rows8: list[dict],
                             out_path: Path) -> str:
    """SQNR degradation rate: ΔSQNR / Δlog(κ) to show sensitivity to CF."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    kappas = sorted(KAPPA_VALUES)
    log_k  = np.log10(kappas)

    def _plot_degradation(ax, rows, fmts, title):
        data: dict[str, list[float]] = {}
        for r in rows:
            fmt = r["format"]
            if fmt not in data:
                data[fmt] = []
            data[fmt].append(r["SQNR_dB"])

        for idx, fmt_key in enumerate(fmts):
            if fmt_key not in data:
                continue
            ys = np.array(data[fmt_key], dtype=float)
            if len(ys) < 2:
                continue
            # Gradient: dSQNR / d(log10 κ)
            grad = np.gradient(ys, log_k)
            ls, mk, lw = _LINE_STYLE.get(fmt_key, ("-", "o", 1.5))
            col = _COLOURS[idx % len(_COLOURS)]
            ax.plot(kappas, grad, linestyle=ls, marker=mk, linewidth=lw,
                    color=col, label=fmt_key, markersize=4, alpha=0.85)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axvline(KAPPA_NATURAL, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("Crest factor κ", fontsize=11)
        ax.set_ylabel("dSQNR / d(log₁₀ κ)  [dB / decade]", fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.set_xscale("log")
        ax.set_xticks(kappas)
        ax.set_xticklabels([str(int(k)) if k == int(k) else str(k) for k in kappas],
                           rotation=30, fontsize=7)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, which="both", alpha=0.3)

    _plot_degradation(axes[0], rows4, FORMATS_4BIT, "SQNR Sensitivity — 4-bit")
    _plot_degradation(axes[1], rows8, FORMATS_8BIT, "SQNR Sensitivity — 8-bit")

    plt.suptitle("SQNR Degradation Rate vs Crest Factor", fontsize=13, y=1.02)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    plt.close()
    print(f"Saved {out_path}")
    return b64


# ── HTML report ───────────────────────────────────────────────────────────────

def _color_sqnr(val: float, vmin: float, vmax: float) -> str:
    if not np.isfinite(val):
        return "background:#eee"
    ratio = max(0.0, min(1.0, (val - vmin) / max(vmax - vmin, 1e-9)))
    if ratio < 0.5:
        r, g, b = 255, int(ratio * 2 * 220), int(60 * (1 - ratio * 2))
    else:
        r, g, b = int((1 - (ratio - 0.5) * 2) * 220), 200, int(60 * (ratio - 0.5) * 2)
    return f"background:rgb({r},{g},{b})"


def _build_sqnr_table(rows: list[dict], formats: list[str], title: str) -> str:
    kappas = sorted({r["kappa"] for r in rows})
    # index
    data: dict[str, dict[float, tuple[float, float]]] = {}
    for r in rows:
        fmt = r["format"]
        if fmt not in data:
            data[fmt] = {}
        data[fmt][r["kappa"]] = (r["SQNR_dB"], r["SQNR_dB_std"])

    # colour scale (exclude FP16)
    non_ref = [
        data[f][k][0]
        for f in formats if f != "FP16"
        for k in kappas
        if f in data and k in data[f] and np.isfinite(data[f][k][0])
    ]
    vmin = min(non_ref) if non_ref else 0.0
    vmax = max(non_ref) if non_ref else 40.0

    html = [f'<h3>{title}</h3>']
    html.append('<div class="tw"><table>')
    html.append('<thead><tr><th class="fc">Format</th>')
    for k in kappas:
        label = f"κ={int(k)}" if k == int(k) else f"κ={k}"
        html.append(f'<th class="kh">{label}</th>')
    html.append('<th class="ac">Avg</th></tr></thead><tbody>')

    for fmt in formats:
        is_ref = fmt == "FP16"
        rc = ' class="rr"' if is_ref else ''
        html.append(f'<tr{rc}><td class="fn">{fmt}</td>')
        vals = []
        for k in kappas:
            v, s = data.get(fmt, {}).get(k, (None, None))
            if v is not None and np.isfinite(v):
                vals.append(v)
                if is_ref:
                    html.append(f'<td class="rc">{v:.1f}<br><span class="sd">±{s:.1f}</span></td>')
                else:
                    c = _color_sqnr(v, vmin, vmax)
                    html.append(f'<td style="{c};color:#000">{v:.1f}<br><span class="sd">±{s:.1f}</span></td>')
            else:
                html.append('<td class="na">—</td>')
        if vals:
            avg = float(np.mean(vals))
            if is_ref:
                html.append(f'<td class="rc ac2">{avg:.1f}</td>')
            else:
                c = _color_sqnr(avg, vmin, vmax)
                html.append(f'<td style="{c};color:#000;font-weight:bold">{avg:.1f}</td>')
        html.append('</tr>')

    html.append('</tbody></table></div>')
    return '\n'.join(html)


def _build_degradation_table(rows: list[dict], formats: list[str], title: str) -> str:
    """Table showing SQNR drop from κ=4 to κ=50 for each format."""
    kappas = sorted({r["kappa"] for r in rows})
    data: dict[str, dict[float, float]] = {}
    for r in rows:
        if r["format"] not in data:
            data[r["format"]] = {}
        data[r["format"]][r["kappa"]] = r["SQNR_dB"]

    # Find reference κ values
    k_low  = min((k for k in kappas if k >= 3.5), default=kappas[0])
    k_mid  = min((k for k in kappas if k >= 9.5), default=kappas[len(kappas)//2])
    k_high = min((k for k in kappas if k >= 49.0), default=kappas[-1])

    html = [f'<h3>{title} — SQNR drop vs κ_ref={k_low}</h3>']
    html.append('<div class="tw"><table>')
    html.append(f'<thead><tr>'
                f'<th class="fc">Format</th>'
                f'<th class="kh">SQNR @ κ={k_low:.0f}</th>'
                f'<th class="kh">SQNR @ κ={k_mid:.0f}</th>'
                f'<th class="kh">SQNR @ κ={k_high:.0f}</th>'
                f'<th class="kh">Drop κ={k_low:.0f}→{k_mid:.0f}</th>'
                f'<th class="kh">Drop κ={k_low:.0f}→{k_high:.0f}</th>'
                f'<th class="kh">Robustness</th>'
                f'</tr></thead><tbody>')

    for fmt in formats:
        is_ref = fmt == "FP16"
        rc = ' class="rr"' if is_ref else ''
        v_low  = data.get(fmt, {}).get(k_low,  float("nan"))
        v_mid  = data.get(fmt, {}).get(k_mid,  float("nan"))
        v_high = data.get(fmt, {}).get(k_high, float("nan"))
        drop_mid  = v_low - v_mid  if np.isfinite(v_low) and np.isfinite(v_mid)  else float("nan")
        drop_high = v_low - v_high if np.isfinite(v_low) and np.isfinite(v_high) else float("nan")

        # Robustness label
        if np.isfinite(drop_high):
            if drop_high < 1:
                robust = "★★★ Robust"
            elif drop_high < 5:
                robust = "★★ Good"
            elif drop_high < 15:
                robust = "★ Moderate"
            else:
                robust = "✗ Sensitive"
        else:
            robust = "—"

        def _fmt_val(v):
            return f"{v:.1f}" if np.isfinite(v) else "—"

        def _fmt_drop(d):
            if not np.isfinite(d):
                return "—"
            color = "red" if d > 10 else ("orange" if d > 3 else "green")
            return f'<span style="color:{color};font-weight:bold">-{d:.1f} dB</span>'

        html.append(f'<tr{rc}>'
                    f'<td class="fn">{fmt}</td>'
                    f'<td class="rc">{_fmt_val(v_low)}</td>'
                    f'<td class="rc">{_fmt_val(v_mid)}</td>'
                    f'<td class="rc">{_fmt_val(v_high)}</td>'
                    f'<td>{_fmt_drop(drop_mid)}</td>'
                    f'<td>{_fmt_drop(drop_high)}</td>'
                    f'<td style="font-size:.85em">{robust}</td>'
                    f'</tr>')

    html.append('</tbody></table></div>')
    return '\n'.join(html)


def build_html_report(rows4: list[dict], rows8: list[dict],
                      b64_sqnr4: str, b64_sqnr8: str,
                      b64_mae4: str, b64_mae8: str,
                      b64_degradation: str) -> str:

    def _img(b64: str, alt: str = "") -> str:
        if not b64:
            return f'<p style="color:#999">[Figure unavailable — matplotlib not installed]</p>'
        return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="max-width:100%;border-radius:6px;box-shadow:0 2px 8px rgba(0,0,0,.15);margin:12px 0">'

    css = """
body{font-family:'Segoe UI',Arial,sans-serif;background:#f5f7fa;color:#222;margin:0;padding:24px}
h1{text-align:center;color:#1a2a4a;font-size:1.65em;margin-bottom:4px}
h2{color:#1a2a4a;font-size:1.2em;border-left:4px solid #3a7bd5;padding-left:10px;margin:32px 0 10px}
h3{color:#1a2a4a;font-size:1em;padding-left:6px;margin:20px 0 6px}
.subtitle{text-align:center;color:#666;font-size:.9em;margin-bottom:24px}
.theory{background:#fff;border-left:4px solid #2ca02c;border-radius:0 6px 6px 0;padding:12px 16px;margin:16px 0;font-size:.88em;color:#333;line-height:1.6}
.theory code{background:#f0f4fa;padding:1px 5px;border-radius:3px;font-family:monospace}
.tw{overflow-x:auto;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,.1);background:#fff;margin-bottom:20px}
table{border-collapse:collapse;width:100%;font-size:.76em}
th,td{padding:4px 7px;text-align:center;border:1px solid #dde2eb;white-space:nowrap}
th.fc{background:#1a2a4a;color:#fff;min-width:100px;text-align:left;padding-left:8px}
th.kh{background:#2c4a7e;color:#fff;font-size:.78em;min-width:56px}
th.ac{background:#1a2a4a;color:#ffd700;min-width:44px}
td.fn{font-weight:600;background:#f0f4fa;text-align:left;padding-left:8px;color:#1a2a4a;border-right:2px solid #ccd}
td.rc{background:#e8eefc;color:#444}
tr.rr td.fn{background:#dce6f7}
td.na{color:#bbb;background:#fafafa}
td.ac2{border-left:2px solid #aab;font-weight:bold}
.sd{font-size:.72em;color:#888}
.legend{display:flex;align-items:center;gap:8px;margin:8px 0 16px;font-size:.82em;color:#555}
.lb{width:130px;height:11px;background:linear-gradient(to right,rgb(255,60,60),rgb(255,220,0),rgb(0,200,100));border-radius:3px;border:1px solid #ccc}
"""

    sqnr_tbl4 = _build_sqnr_table(rows4, FORMATS_4BIT,  "4-bit — SQNR (dB) vs κ  (mean ± std)")
    sqnr_tbl8 = _build_sqnr_table(rows8, FORMATS_8BIT,  "8-bit — SQNR (dB) vs κ  (mean ± std)")
    deg_tbl4  = _build_degradation_table(rows4, FORMATS_4BIT, "4-bit Robustness")
    deg_tbl8  = _build_degradation_table(rows8, FORMATS_8BIT, "8-bit Robustness")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Exp 2 — Crest Factor Analysis</title>
<style>{css}</style>
</head>
<body>
<h1>Experiment 2 — SQNR vs Crest Factor</h1>
<p class="subtitle">
  Single-spike injection model · [{C_OUT}×{C_IN}] Gaussian weight matrices<br>
  {N_SEEDS} seeds averaged · κ_nat ≈ {KAPPA_NATURAL:.2f} for N={N}
</p>

<div class="theory">
  <b>Crest Factor Definition:</b>
  For X ∈ ℝ<sup>k</sup> with i.i.d. Xi ~ N(0, σ²), block RMS = σ, and
  <code>κ := max(|X|) / σ</code>.
  The natural κ of a pure Gaussian block with N={N} elements is
  <code>κ_nat ≈ √(2 ln N) ≈ {KAPPA_NATURAL:.2f}</code>.
  This experiment injects a single outlier of magnitude κ·σ into an otherwise
  N(0,1) background and sweeps κ from {KAPPA_VALUES[0]} to {KAPPA_VALUES[-1]}.
  For κ ≫ κ_nat the injected spike dominates; for κ ≲ κ_nat the background
  peak sets the effective crest factor.
  <br><br>
  <b>Theoretical predictions:</b>
  <ul style="margin:4px 0 0 18px">
    <li><b>INT(T)</b>: scale ∝ κ → step size ∝ κ → SQNR ∝ −20 log₁₀(κ) <em>(steepest drop)</em></li>
    <li><b>INT(C)</b>: only the outlier's row is affected; 63/64 rows remain at full quality</li>
    <li><b>MXINT/MXFP</b>: only the outlier's block (32 elements) is coarsened; rest at full quality</li>
    <li><b>HAD+INT(C)</b>: per-row HAD spreads the spike uniformly; effective κ_had ≈ √(κ²+N)/max_had → slowly growing</li>
    <li><b>SQ-Format</b>: outlier element wins a high-precision slot; SQNR nearly flat</li>
  </ul>
</div>

<h2>SQNR vs κ — Line Plots</h2>
{_img(b64_sqnr4, "SQNR vs CF 4-bit")}
{_img(b64_sqnr8, "SQNR vs CF 8-bit")}

<h2>Max Absolute Error vs κ</h2>
{_img(b64_mae4, "MaxAE vs CF 4-bit")}
{_img(b64_mae8, "MaxAE vs CF 8-bit")}

<h2>SQNR Degradation Rate</h2>
<p style="font-size:.85em;color:#666">dSQNR/d(log₁₀ κ) — negative slope = degradation per decade of κ. Ideal (robust) format: slope ≈ 0.</p>
{_img(b64_degradation, "Degradation rate")}

<h2>SQNR Summary Tables</h2>
<div class="legend">
  <span>Low SQNR</span><div class="lb"></div><span>High SQNR</span>
  <span style="margin-left:16px;color:#999">| Values: mean (±std) over {N_SEEDS} seeds</span>
</div>
{sqnr_tbl4}
{sqnr_tbl8}

<h2>Robustness Analysis</h2>
<p style="font-size:.85em;color:#666">SQNR drop from κ≈4 (natural) to larger values. ★★★ = &lt;1 dB drop, ★★ = &lt;5 dB, ★ = &lt;15 dB, ✗ = &gt;15 dB.</p>
{deg_tbl4}
{deg_tbl8}

<p style="font-size:.72em;color:#aaa;margin-top:28px;text-align:right">
  Generated by experiments/exp2_crest_factor.py
</p>
</body>
</html>
"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", choices=["4", "8", "both"], default="both")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
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

    # ── Figures ───────────────────────────────────────────────────────────────
    b64_sqnr4 = b64_sqnr8 = b64_mae4 = b64_mae8 = b64_deg = ""

    if rows4:
        b64_sqnr4 = make_sqnr_figure(rows4, FORMATS_4BIT, "4-bit Formats",
                                      out / "fig_sqnr_4bit.png")
        b64_mae4  = make_maxae_figure(rows4, FORMATS_4BIT, "4-bit Formats",
                                      out / "fig_maxae_4bit.png")
    if rows8:
        b64_sqnr8 = make_sqnr_figure(rows8, FORMATS_8BIT, "8-bit Formats",
                                      out / "fig_sqnr_8bit.png")
        b64_mae8  = make_maxae_figure(rows8, FORMATS_8BIT, "8-bit Formats",
                                      out / "fig_maxae_8bit.png")
    if rows4 and rows8:
        b64_deg = make_degradation_figure(rows4, rows8,
                                          out / "fig_degradation_rate.png")

    # ── HTML report ───────────────────────────────────────────────────────────
    r4_used = rows4 if rows4 else rows8
    r8_used = rows8 if rows8 else rows4
    html = build_html_report(r4_used, r8_used,
                              b64_sqnr4, b64_sqnr8,
                              b64_mae4,  b64_mae8,
                              b64_deg)
    rpt = out / "report.html"
    rpt.write_text(html)
    print(f"Saved {rpt}")


if __name__ == "__main__":
    main()
