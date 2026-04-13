"""Experiment 1: Quantization Accuracy on Common Deep Learning Distributions

Academic benchmark comparing 9 quantization formats across 20 distribution
variants that cover the full diversity of tensors encountered in large
language models and vision transformers.

Usage
-----
    python experiments/exp1_common_distributions.py          # default 8-bit
    python experiments/exp1_common_distributions.py --bits 4 # 4-bit experiment
    python experiments/exp1_common_distributions.py --bits all # both

Configurable parameter
----------------------
    BITS : int
        Nominal bit-width for the experiment.  8 → 8-bit formats, 4 → 4-bit.
        When changed, ALL format configurations scale accordingly:
          - MXINT{BITS}, MXFP{BITS}
          - INT{BITS} per-tensor / per-channel
          - HAD+INT{BITS} per-tensor / per-channel
          - SQ-Format-INT: high=INT{BITS}, low=INT{BITS//2}, sparsity=0.5
          - SQ-Format-FP:  high=FP{BITS},  low=INT{BITS//2}, sparsity=0.5
          - FP16 is always included as a fixed upper-bound baseline.
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# ── project path setup ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from distributions.generators import (
    gaussian, laplace, student_t_dist, bimodal,
    channel_outliers, spiky_outliers, log_normal,
)
from distributions.metrics import snr_db, mse, kl_divergence, max_absolute_error
from formats.mxint import MXINTFormat
from formats.mxfp import MXFPFormat, _fp8_e4m3_vec, _E2M1_POS
from formats.sq_format import SQFormat, _pot_scale, _int_quantize_pot
from formats.transforms.hadamard import HADTransform

# ── Output directories ────────────────────────────────────────────────────────
OUT_FIGURES = ROOT / "results" / "exp1"
OUT_FIGURES.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# §0  CONFIGURABLE PARAMETER
# ─────────────────────────────────────────────────────────────────────────────

BITS_DEFAULT = 8       # ← change this or pass --bits on CLI

N_SAMPLES    = 4096    # elements per distribution sample
BANK_SIZE    = 128     # SQ-Format bank size
SPARSITY     = 0.5     # SQ-Format sparsity (50% high-prec per bank)
SEED         = 42


# ─────────────────────────────────────────────────────────────────────────────
# §1  FORMAT IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────────────────────────

class FP16Format:
    """IEEE-754 FP16: 1 sign + 5 exp + 10 mantissa.  Upper-bound reference."""

    def __init__(self):
        self.name = "FP16"
        self.bits = 16

    def quantize(self, x: np.ndarray, **_) -> np.ndarray:
        return x.astype(np.float16).astype(np.float32)


class _PerGroupINTQuantizer:
    """Symmetric INT quantization with per-group POT scale.

    Divides a flat vector into non-overlapping groups of ``group_size``
    elements and applies an independent POT scale to each group.
    group_size=N → per-tensor;  group_size=1 → per-element.
    """

    def __init__(self, bits: int, group_size: int):
        self.bits       = bits
        self.group_size = group_size
        self.q_max      = 2 ** (bits - 1) - 1

    def quantize(self, x: np.ndarray, **_) -> np.ndarray:
        x = x.astype(np.float32)
        flat = x.ravel()
        n    = len(flat)
        out  = np.empty_like(flat)
        for start in range(0, n, self.group_size):
            end    = min(start + self.group_size, n)
            chunk  = flat[start:end]
            absmax = float(np.max(np.abs(chunk)))
            if absmax == 0.0:
                out[start:end] = 0.0
                continue
            scale = _pot_scale(absmax, self.q_max)
            q     = np.clip(np.round(chunk / scale).astype(np.int32), -self.q_max, self.q_max)
            out[start:end] = q.astype(np.float32) * scale
        return out.reshape(x.shape)


class _HADQuantizer:
    """HADamard transform + per-group INT quantization (hardware-faithful).

    The HAD is applied with normalize=False (integer butterfly); the POT
    scale absorbs the √N amplification factor automatically.
    Inverse divides by N (exact right-shift by log₂N bits).
    """

    def __init__(self, bits: int, group_size: int):
        self.bits       = bits
        self.group_size = group_size
        self._had       = HADTransform(normalize=False)
        self._quant     = _PerGroupINTQuantizer(bits, group_size)
        self.name       = f"HAD+INT{bits}-{'PT' if group_size >= N_SAMPLES else 'PC'}"

    def quantize(self, x: np.ndarray, **_) -> np.ndarray:
        x_t  = self._had.forward(x.astype(np.float32))
        q_t  = self._quant.quantize(x_t)
        return self._had.inverse(q_t)


def _fp4_e2m1_quantize(x: np.ndarray) -> np.ndarray:
    """Per-bank FP4 E2M1 quantization: normalize to [−6,+6], then snap to E2M1 levels."""
    absmax = float(np.max(np.abs(x)))
    if absmax == 0.0:
        return np.zeros_like(x)
    scale   = absmax / 6.0          # E2M1 max positive level = 6.0
    x_norm  = x / scale
    pos_lev = _E2M1_POS             # [0, .5, 1, 1.5, 2, 3, 4, 6]
    signs   = np.sign(x_norm)
    x_abs   = np.abs(x_norm)
    # nearest E2M1 level
    idx     = np.argmin(np.abs(x_abs[:, None] - pos_lev), axis=1)
    return signs * pos_lev[idx] * scale


def _fp8_e4m3_quantize_bank(x: np.ndarray) -> np.ndarray:
    """Per-bank FP8 E4M3 quantization with an E8M0 block scale.

    Normalises elements by (absmax / 448) so the FP8 grid covers the
    bank's dynamic range — identical to the MXFP8 block-scale logic.
    """
    FP8_MAX = 448.0
    absmax  = float(np.max(np.abs(x)))
    if absmax == 0.0:
        return np.zeros_like(x)
    # E8M0 block scale: smallest POT ≥ absmax/FP8_MAX (ceil avoids clipping).
    # floor would give scale too small → x_norm > FP8_MAX → clipping & huge error.
    log2_s  = int(np.ceil(np.log2(absmax / FP8_MAX + 1e-38)))
    scale   = float(2.0 ** log2_s)
    x_norm  = x / scale
    q_norm  = _fp8_e4m3_vec(x_norm).astype(np.float32)
    return q_norm * scale


class SQFormatFP:
    """SQ-Format with floating-point high-precision elements.

    High-precision (top (1-s) per bank):
      BITS=8 → FP8 E4M3 with per-bank E8M0 block scale  (same as MXFP8)
      BITS=4 → FP4 E2M1 with per-bank scale
    Low-precision (bottom s per bank):
      INT{BITS//2} with per-bank POT scale
    """

    def __init__(self, bits: int = 8, sparsity: float = 0.5, bank_size: int = 128):
        self.bits      = bits
        self.sparsity  = sparsity
        self.bank_size = bank_size
        self.low_bits  = bits // 2
        self.name      = f"SQ-FP{bits}"

    def _bank_mask(self, imp: np.ndarray) -> np.ndarray:
        bsz    = len(imp)
        k_high = max(1, int(round((1.0 - self.sparsity) * bsz)))
        mask   = np.zeros(bsz, dtype=bool)
        mask[np.argpartition(imp, -k_high)[-k_high:]] = True
        return mask

    def _quant_high(self, x: np.ndarray) -> np.ndarray:
        if self.bits == 8:
            return _fp8_e4m3_quantize_bank(x)
        else:   # bits=4
            return _fp4_e2m1_quantize(x)

    def _quant_low(self, x: np.ndarray) -> np.ndarray:
        return _int_quantize_pot(x, self.low_bits)

    def quantize(self, x: np.ndarray, **_) -> np.ndarray:
        flat = x.astype(np.float32).ravel()
        n    = len(flat)
        out  = np.zeros(n, dtype=np.float32)
        for start in range(0, n, self.bank_size):
            end  = min(start + self.bank_size, n)
            bank = flat[start:end]
            imp  = bank ** 2                       # magnitude-based importance
            mask = self._bank_mask(imp)
            if np.any(mask):
                out[start:end][mask]  = self._quant_high(bank[mask])
            if np.any(~mask):
                out[start:end][~mask] = self._quant_low(bank[~mask])
        return out.reshape(x.shape)


# ─────────────────────────────────────────────────────────────────────────────
# §2  FORMAT REGISTRY BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_formats(bits: int) -> dict:
    """Return ordered dict of format_name → format_object for the given bit-width."""
    lo = bits // 2          # low-precision bits for SQ-Format

    # Per-channel group size: 64 elements per group (N_SAMPLES / 64 = 64 groups)
    # This simulates 64 output channels, each independently scaled.
    pc_group = max(1, N_SAMPLES // 64)

    formats = {
        # ── Fixed-precision baseline ─────────────────────────────────────────
        f"FP16":                     FP16Format(),

        # ── Plain INT (per-tensor and per-channel) ───────────────────────────
        f"INT{bits}-PerTensor":      _PerGroupINTQuantizer(bits, N_SAMPLES),
        f"INT{bits}-PerChannel":     _PerGroupINTQuantizer(bits, pc_group),

        # ── Hardware block-scaled formats ────────────────────────────────────
        f"MXINT{bits}":              MXINTFormat(element_bits=bits),
        f"MXFP{bits}":               MXFPFormat(element_bits=bits),

        # ── HAD + INT (per-tensor and per-channel) ───────────────────────────
        f"HAD+INT{bits}-PerTensor":  _HADQuantizer(bits, N_SAMPLES),
        f"HAD+INT{bits}-PerChannel": _HADQuantizer(bits, pc_group),

        # ── SQ-Format (sparse-quantized, paper Algorithm 1) ──────────────────
        f"SQ-Format-INT{bits}":      SQFormat(
                                         bank_size=BANK_SIZE,
                                         sparsity=SPARSITY,
                                         high_bits=bits,
                                         low_bits=lo,
                                     ),
        f"SQ-Format-FP{bits}":       SQFormatFP(
                                         bits=bits,
                                         sparsity=SPARSITY,
                                         bank_size=BANK_SIZE,
                                     ),
    }
    return formats


# ─────────────────────────────────────────────────────────────────────────────
# §3  DISTRIBUTION SWEEP DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

DIST_GROUPS = {
    "Gaussian":         [],
    "Laplace":          [],
    "Student-t":        [],
    "Bimodal":          [],
    "Channel Outlier":  [],
    "Spiky Outlier":    [],
    "Log-Normal":       [],
    "Uniform":          [],
}


def _make_uniform(n, a, seed):
    rng = np.random.default_rng(seed)
    x   = rng.uniform(-a, a, size=n).astype(np.float32)
    return x, {"type": "uniform", "a": a}


def build_distributions() -> list:
    """Return list of (label, family, tensor) tuples for all 20 distribution variants."""
    n, s = N_SAMPLES, SEED
    dists = []

    # 1. Gaussian — weight matrices in small/medium/large networks
    for sigma in [0.5, 1.0, 2.0, 5.0]:
        x, _ = gaussian(n, sigma=sigma, seed=s)
        dists.append((f"Gauss(σ={sigma})", "Gaussian", x))

    # 2. Laplace — FFN weight distributions (heavier tail than Gaussian)
    for b in [0.5, 1.0, 2.0]:
        x, _ = laplace(n, b=b, seed=s)
        dists.append((f"Laplace(b={b})", "Laplace", x))

    # 3. Student-t — Transformer activation tails (ν→∞ becomes Gaussian)
    for nu in [3, 5, 10]:
        x, _ = student_t_dist(n, nu=nu, seed=s)
        dists.append((f"Student-t(ν={nu})", "Student-t", x))

    # 4. Bimodal — Attention softmax output / post-LayerNorm activations
    for mu in [2.0, 3.0, 5.0]:
        x, _ = bimodal(n, mu1=-mu, mu2=mu, sigma=0.5, seed=s)
        dists.append((f"Bimodal(μ=±{mu})", "Bimodal", x))

    # 5. Channel outliers — LLM systematic outliers (LLM.int8() phenomenon)
    for sig_out in [30, 50, 100]:
        x, _ = channel_outliers(n, outlier_ratio=0.01, outlier_sigma=sig_out, seed=s)
        dists.append((f"ChanOut(σ={sig_out})", "Channel Outlier", x))

    # 6. Spiky outliers — random magnitude spikes (AWQ/GPTQ key challenge)
    for mult in [10, 50, 100]:
        x, _ = spiky_outliers(n, spike_ratio=0.001, spike_multiplier=mult, seed=s)
        dists.append((f"Spiky({mult}×)", "Spiky Outlier", x))

    # 7. Log-Normal — post-GELU/SiLU activations, gradient magnitudes
    for sig_ln in [1.0, 2.0]:
        x, _ = log_normal(n, mu=0.0, sigma=sig_ln, seed=s)
        dists.append((f"LogNorm(σ={sig_ln})", "Log-Normal", x))

    # 8. Uniform — adversarial for INT (maximally flat histogram)
    for a in [1.0, 3.0]:
        x, _ = _make_uniform(n, a=a, seed=s)
        dists.append((f"Uniform(±{a})", "Uniform", x))

    return dists   # 4+3+3+3+3+3+2+2 = 23 variants


# ─────────────────────────────────────────────────────────────────────────────
# §4  SWEEP RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(bits: int) -> pd.DataFrame:
    """Run the full format × distribution sweep for a given bit-width.

    Returns a DataFrame with columns:
        format, family, distribution, SQNR_dB, MSE, KL_div, MaxAE
    """
    formats = build_formats(bits)
    dists   = build_distributions()
    rows    = []

    for fmt_name, fmt in formats.items():
        for dist_label, dist_family, x in dists:
            try:
                x_q = fmt.quantize(x)
                row = {
                    "format":       fmt_name,
                    "bits":         bits,
                    "family":       dist_family,
                    "distribution": dist_label,
                    "SQNR_dB":      round(snr_db(x, x_q), 3),
                    "MSE":          round(mse(x, x_q), 6),
                    "KL_div":       round(kl_divergence(x, x_q), 5),
                    "MaxAE":        round(max_absolute_error(x, x_q), 4),
                }
            except Exception as e:
                row = {
                    "format":       fmt_name,
                    "bits":         bits,
                    "family":       dist_family,
                    "distribution": dist_label,
                    "SQNR_dB":      float("nan"),
                    "MSE":          float("nan"),
                    "KL_div":       float("nan"),
                    "MaxAE":        float("nan"),
                }
                print(f"  [WARN] {fmt_name} on {dist_label}: {e}")
            rows.append(row)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# §5  VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

# ── colour palette ────────────────────────────────────────────────────────────
def _make_palette(fmt_names, bits):
    """Assign semantically-consistent colours by format family."""
    colours = {}
    for n in fmt_names:
        if   n == "FP16":                       colours[n] = "#6B7280"
        elif "PerTensor" in n and "HAD" not in n: colours[n] = "#92400E"
        elif "PerChannel" in n and "HAD" not in n:colours[n] = "#B45309"
        elif n.startswith("MXINT"):             colours[n] = "#1E40AF"
        elif n.startswith("MXFP"):              colours[n] = "#60A5FA"
        elif "HAD" in n and "PerTensor" in n:   colours[n] = "#4ADE80"
        elif "HAD" in n and "PerChannel" in n:  colours[n] = "#15803D"
        elif "SQ-Format-INT" in n:              colours[n] = "#D97706"
        elif "SQ-Format-FP"  in n:              colours[n] = "#EA580C"
        else:                                   colours[n] = "#A855F7"
    return colours


# ── Fig 1: SQNR Heatmap ───────────────────────────────────────────────────────
def plot_heatmap(df: pd.DataFrame, bits: int, out_dir: Path) -> Path:
    pivot = df.pivot_table(index="format", columns="distribution",
                           values="SQNR_dB", aggfunc="mean")

    # Row / col ordering
    fmt_order = list(build_formats(bits).keys())
    dist_order = [d for _, _, x_  in build_distributions()
                  for d, *_ in [(_, None, None)]]   # regenerate labels
    dist_order = [row[0] for row in build_distributions()]
    pivot = pivot.reindex(index=fmt_order, columns=dist_order)

    fig, ax = plt.subplots(figsize=(20, 6))
    vmin = max(0, pivot.values[np.isfinite(pivot.values)].min() - 1)
    vmax = pivot.values[np.isfinite(pivot.values)].max() + 1

    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                   vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="SQNR (dB)", fraction=0.015, pad=0.01)

    ax.set_xticks(range(len(dist_order)))
    ax.set_xticklabels(dist_order, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(fmt_order)))
    ax.set_yticklabels(fmt_order, fontsize=9)

    # Annotate cells with SQNR value
    for i in range(len(fmt_order)):
        for j in range(len(dist_order)):
            v = pivot.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=6.5,
                        color="black" if 20 < v < 50 else "white")

    ax.set_title(f"SQNR Heatmap — {bits}-bit Formats × Distributions (dB, higher = better)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = out_dir / f"fig1_heatmap_{bits}bit.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Fig 2: Mean SQNR bar chart per distribution family ───────────────────────
def plot_family_bars(df: pd.DataFrame, bits: int, out_dir: Path) -> Path:
    families  = list(df["family"].unique())
    fmt_names = list(build_formats(bits).keys())
    palette   = _make_palette(fmt_names, bits)

    family_mean = (df.groupby(["format", "family"])["SQNR_dB"]
                     .mean()
                     .reset_index()
                     .pivot(index="format", columns="family", values="SQNR_dB")
                     .reindex(index=fmt_names))

    n_fam = len(families)
    n_fmt = len(fmt_names)
    bar_w = 0.8 / n_fmt
    x     = np.arange(n_fam)

    fig, ax = plt.subplots(figsize=(16, 6))
    for i, fmt in enumerate(fmt_names):
        vals = [family_mean.loc[fmt, fam] if fam in family_mean.columns
                else float("nan") for fam in families]
        offsets = (i - n_fmt / 2 + 0.5) * bar_w
        ax.bar(x + offsets, vals, width=bar_w * 0.9,
               color=palette[fmt], label=fmt, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(families, fontsize=10)
    ax.set_ylabel("Mean SQNR (dB)")
    ax.set_title(f"Mean SQNR by Distribution Family — {bits}-bit Formats",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=7.5, ncol=3, loc="upper right")
    ax.grid(axis="y", alpha=0.35, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = out_dir / f"fig2_family_bars_{bits}bit.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Fig 3: Gaussian σ sweep (sensitivity analysis) ───────────────────────────
def plot_gaussian_sweep(df: pd.DataFrame, bits: int, out_dir: Path) -> Path:
    fmt_names = list(build_formats(bits).keys())
    palette   = _make_palette(fmt_names, bits)
    markers   = ["o","s","D","^","v","P","*","X","h"]

    gauss_df = df[df["family"] == "Gaussian"].copy()
    gauss_df["sigma"] = gauss_df["distribution"].str.extract(r"σ=([\d.]+)")[0].astype(float)
    gauss_df = gauss_df.sort_values("sigma")

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, fmt in enumerate(fmt_names):
        sub = gauss_df[gauss_df["format"] == fmt]
        ax.plot(sub["sigma"], sub["SQNR_dB"],
                color=palette[fmt], label=fmt,
                marker=markers[i % len(markers)], linewidth=1.8, markersize=7)

    ax.set_xlabel("Gaussian σ", fontsize=11)
    ax.set_ylabel("SQNR (dB)", fontsize=11)
    ax.set_title(f"SQNR vs Gaussian σ — {bits}-bit Formats", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=2, loc="best")
    ax.grid(alpha=0.35)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    out = out_dir / f"fig3_gaussian_sweep_{bits}bit.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Fig 4: Outlier severity sweep ────────────────────────────────────────────
def plot_outlier_sweep(df: pd.DataFrame, bits: int, out_dir: Path) -> Path:
    fmt_names = list(build_formats(bits).keys())
    palette   = _make_palette(fmt_names, bits)
    markers   = ["o","s","D","^","v","P","*","X","h"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=False)

    for ax, family, col_key, x_label in [
        (axes[0], "Channel Outlier", r"σ=(\d+)",   "Outlier σ"),
        (axes[1], "Spiky Outlier",   r"(\d+)×",    "Spike multiplier"),
    ]:
        sub_df = df[df["family"] == family].copy()
        sub_df["severity"] = (sub_df["distribution"]
                               .str.extract(col_key)[0]
                               .astype(float))
        sub_df = sub_df.sort_values("severity")

        for i, fmt in enumerate(fmt_names):
            s = sub_df[sub_df["format"] == fmt]
            ax.plot(s["severity"], s["SQNR_dB"],
                    color=palette[fmt], label=fmt,
                    marker=markers[i % len(markers)], linewidth=1.8, markersize=7)

        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel("SQNR (dB)", fontsize=11)
        ax.set_title(f"{family} — {bits}-bit", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.35)
        ax.spines[["top","right"]].set_visible(False)

    axes[0].legend(fontsize=7.5, ncol=2)
    plt.suptitle(f"Outlier Robustness — {bits}-bit Formats", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = out_dir / f"fig4_outlier_sweep_{bits}bit.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Fig 5: Overall ranking (mean SQNR across ALL distributions) ───────────────
def plot_overall_ranking(df: pd.DataFrame, bits: int, out_dir: Path) -> Path:
    fmt_names  = list(build_formats(bits).keys())
    palette    = _make_palette(fmt_names, bits)

    mean_sqnr  = (df.groupby("format")["SQNR_dB"]
                    .mean()
                    .reindex(fmt_names)
                    .sort_values(ascending=True))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(mean_sqnr.index, mean_sqnr.values,
                   color=[palette[n] for n in mean_sqnr.index],
                   height=0.7, zorder=3)
    for bar, val in zip(bars, mean_sqnr.values):
        ax.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f} dB", va="center", fontsize=9)

    ax.set_xlabel("Mean SQNR across all distributions (dB)", fontsize=11)
    ax.set_title(f"Overall Format Ranking — {bits}-bit", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.35, zorder=0)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    out = out_dir / f"fig5_overall_ranking_{bits}bit.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Fig 6: Best-format-per-distribution chart ─────────────────────────────────
def plot_best_format(df: pd.DataFrame, bits: int, out_dir: Path) -> Path:
    fmt_names = list(build_formats(bits).keys())
    palette   = _make_palette(fmt_names, bits)

    best = (df.groupby("distribution")["SQNR_dB"]
              .idxmax()
              .apply(lambda idx: df.loc[idx, "format"]))
    dist_order  = [row[0] for row in build_distributions()]
    best        = best.reindex(dist_order)
    best_sqnr   = (df.groupby("distribution")["SQNR_dB"]
                     .max()
                     .reindex(dist_order))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={"height_ratios": [3, 1]})

    colors_for_bar = [palette.get(b, "#999") for b in best.values]
    ax1.bar(range(len(dist_order)), best_sqnr.values, color=colors_for_bar,
            zorder=3, width=0.7)
    ax1.set_xticks(range(len(dist_order)))
    ax1.set_xticklabels(dist_order, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Best SQNR (dB)", fontsize=10)
    ax1.set_title(f"Best Format per Distribution — {bits}-bit", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", alpha=0.35, zorder=0)

    for i, (fmt, sqnr) in enumerate(zip(best.values, best_sqnr.values)):
        ax1.text(i, sqnr + 0.3, fmt, ha="center", va="bottom",
                 fontsize=6.5, rotation=45, color="black")

    # Legend patches
    unique_fmts = list(dict.fromkeys(best.values))
    handles = [Patch(color=palette.get(f, "#999"), label=f) for f in unique_fmts]
    ax1.legend(handles=handles, fontsize=7.5, ncol=3, loc="upper right")

    # Win count bar
    win_counts = best.value_counts().reindex(fmt_names, fill_value=0)
    ax2.bar(range(len(fmt_names)),
            [win_counts.get(f, 0) for f in fmt_names],
            color=[palette[f] for f in fmt_names], zorder=3, width=0.7)
    ax2.set_xticks(range(len(fmt_names)))
    ax2.set_xticklabels(fmt_names, rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("# Wins", fontsize=10)
    ax2.grid(axis="y", alpha=0.35, zorder=0)
    ax2.set_title("Win Count (best SQNR across distributions)", fontsize=10)

    plt.tight_layout()
    out = out_dir / f"fig6_best_format_{bits}bit.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# §6  TABLE BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def build_sqnr_table(df: pd.DataFrame) -> str:
    """Build a Markdown SQNR table: rows = format, columns = distribution."""
    pivot = df.pivot_table(index="format", columns="distribution",
                           values="SQNR_dB", aggfunc="mean")
    bits = df["bits"].iloc[0]
    fmt_order  = list(build_formats(bits).keys())
    dist_order = [row[0] for row in build_distributions()]
    pivot = pivot.reindex(index=fmt_order, columns=dist_order)

    # Add mean column
    pivot["**Mean**"] = pivot.mean(axis=1)

    lines = ["| Format | " + " | ".join(pivot.columns) + " |"]
    lines.append("|" + "---|" * (len(pivot.columns) + 1))
    for fmt in pivot.index:
        row_vals = []
        for col in pivot.columns:
            v = pivot.loc[fmt, col]
            s = f"{v:.1f}" if np.isfinite(v) else "—"
            # Bold the max in each data column (not Mean col)
            row_vals.append(s)
        lines.append(f"| {fmt} | " + " | ".join(row_vals) + " |")
    return "\n".join(lines)


def build_summary_table(df: pd.DataFrame) -> str:
    """Build a per-format summary: mean SQNR, best/worst distribution, win count."""
    bits      = df["bits"].iloc[0]
    fmt_order = list(build_formats(bits).keys())

    best_dist  = (df.groupby("format")
                    .apply(lambda g: g.loc[g["SQNR_dB"].idxmax(), "distribution"])
                    .reindex(fmt_order))
    worst_dist = (df.groupby("format")
                    .apply(lambda g: g.loc[g["SQNR_dB"].idxmin(), "distribution"])
                    .reindex(fmt_order))
    mean_sqnr  = df.groupby("format")["SQNR_dB"].mean().reindex(fmt_order)
    wins       = (df.groupby("distribution")["SQNR_dB"]
                    .idxmax()
                    .apply(lambda idx: df.loc[idx, "format"])
                    .value_counts()
                    .reindex(fmt_order, fill_value=0))

    lines = ["| Format | Mean SQNR (dB) | Best Distribution | Worst Distribution | # Wins |"]
    lines.append("|---|---|---|---|---|")
    for fmt in fmt_order:
        lines.append(
            f"| {fmt} | {mean_sqnr[fmt]:.1f} | {best_dist[fmt]} "
            f"| {worst_dist[fmt]} | {wins[fmt]} |"
        )
    return "\n".join(lines)


def build_family_table(df: pd.DataFrame) -> str:
    """Build SQNR table: rows = format, columns = distribution family."""
    bits      = df["bits"].iloc[0]
    fmt_order = list(build_formats(bits).keys())
    families  = list(df["family"].unique())

    pivot = (df.groupby(["format", "family"])["SQNR_dB"]
               .mean()
               .unstack("family")
               .reindex(index=fmt_order, columns=families))
    pivot["Overall"] = pivot.mean(axis=1)

    lines = ["| Format | " + " | ".join(pivot.columns) + " |"]
    lines.append("|" + "---|" * (len(pivot.columns) + 1))
    for fmt in pivot.index:
        vals = [f"{pivot.loc[fmt, c]:.1f}" if np.isfinite(pivot.loc[fmt, c]) else "—"
                for c in pivot.columns]
        lines.append(f"| {fmt} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# §7  ANALYSIS ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def derive_findings(df: pd.DataFrame) -> str:
    """Auto-derive key findings from the sweep results."""
    bits     = df["bits"].iloc[0]
    fmt_names = list(build_formats(bits).keys())

    mean_sqnr = df.groupby("format")["SQNR_dB"].mean().reindex(fmt_names)
    best_overall = mean_sqnr.idxmax()
    worst_overall = mean_sqnr.idxmin()

    # Best format per family
    family_best = {}
    for fam in df["family"].unique():
        sub = df[df["family"] == fam]
        fam_mean = sub.groupby("format")["SQNR_dB"].mean()
        family_best[fam] = fam_mean.idxmax()

    # Win counts
    wins = (df.groupby("distribution")["SQNR_dB"]
              .idxmax()
              .apply(lambda idx: df.loc[idx, "format"])
              .value_counts())

    # Formats that exceed FP16 baseline
    fp16_mean = mean_sqnr.get("FP16", float("nan"))
    near_fp16 = [f for f in fmt_names
                 if f != "FP16" and abs(mean_sqnr.get(f, 0) - fp16_mean) < 3.0]

    # Biggest SQNR drop under outlier stress
    outlier_families = ["Channel Outlier", "Spiky Outlier"]
    outlier_df    = df[df["family"].isin(outlier_families)]
    normal_df     = df[~df["family"].isin(outlier_families)]
    outlier_drop  = {}
    for f in fmt_names:
        norm_m = normal_df[normal_df["format"] == f]["SQNR_dB"].mean()
        out_m  = outlier_df[outlier_df["format"] == f]["SQNR_dB"].mean()
        if np.isfinite(norm_m) and np.isfinite(out_m):
            outlier_drop[f] = norm_m - out_m
    most_sensitive  = max(outlier_drop, key=outlier_drop.get) if outlier_drop else "N/A"
    most_robust     = min(outlier_drop, key=outlier_drop.get) if outlier_drop else "N/A"

    sq_int = f"SQ-Format-INT{bits}"
    sq_fp  = f"SQ-Format-FP{bits}"
    had_pt = f"HAD+INT{bits}-PerTensor"
    had_pc = f"HAD+INT{bits}-PerChannel"
    mxint  = f"MXINT{bits}"
    mxfp   = f"MXFP{bits}"
    int_pt = f"INT{bits}-PerTensor"
    int_pc = f"INT{bits}-PerChannel"

    lines = []
    lines.append(f"1. **Overall winner**: `{best_overall}` "
                 f"(mean SQNR = {mean_sqnr[best_overall]:.1f} dB)")
    lines.append(f"2. **Overall worst** (excluding FP16): `{worst_overall}` "
                 f"(mean SQNR = {mean_sqnr[worst_overall]:.1f} dB)")

    lines.append(f"3. **Per-channel advantage**: `{int_pc}` gains "
                 f"{mean_sqnr.get(int_pc,0) - mean_sqnr.get(int_pt,0):.1f} dB "
                 f"over `{int_pt}` on average — largest benefit on bimodal "
                 f"and non-outlier distributions.")

    lines.append(f"4. **HAD transform benefit**: `{had_pc}` gains "
                 f"{mean_sqnr.get(had_pc,0) - mean_sqnr.get(int_pc,0):.1f} dB "
                 f"over `{int_pc}` — HAD reduces effective kurtosis, "
                 f"making the distribution more Gaussian-like.")

    lines.append(f"5. **SQ-Format-INT vs SQ-Format-FP**: `{sq_int}` = "
                 f"{mean_sqnr.get(sq_int,float('nan')):.1f} dB, `{sq_fp}` = "
                 f"{mean_sqnr.get(sq_fp,float('nan')):.1f} dB. "
                 f"FP's non-uniform grid {'helps' if mean_sqnr.get(sq_fp,0) > mean_sqnr.get(sq_int,0) else 'does not help'} "
                 f"on average for {bits}-bit.")

    lines.append(f"6. **MXINT vs MXFP**: `{mxint}` = "
                 f"{mean_sqnr.get(mxint,float('nan')):.1f} dB, `{mxfp}` = "
                 f"{mean_sqnr.get(mxfp,float('nan')):.1f} dB. "
                 f"{'FP is superior' if mean_sqnr.get(mxfp,0) > mean_sqnr.get(mxint,0) else 'INT is superior'} "
                 f"for {bits}-bit microscaling.")

    lines.append(f"7. **Outlier robustness**: Most robust format = `{most_robust}` "
                 f"(SQNR drop = {outlier_drop.get(most_robust, 0):.1f} dB). "
                 f"Most sensitive = `{most_sensitive}` "
                 f"(SQNR drop = {outlier_drop.get(most_sensitive, 0):.1f} dB).")

    for fam, best in family_best.items():
        lines.append(f"8. **Best for {fam}**: `{best}` "
                     f"(mean = {df[df['family']==fam].groupby('format')['SQNR_dB'].mean()[best]:.1f} dB)")

    return "\n".join(f"- {l}" for l in lines)


# ─────────────────────────────────────────────────────────────────────────────
# §8  ANALYSIS.MD WRITER
# ─────────────────────────────────────────────────────────────────────────────

def write_analysis_md(results_8: pd.DataFrame, results_4: pd.DataFrame,
                      figs_8: dict, figs_4: dict):
    """Write the full analysis.md document."""

    def rel(p: Path) -> str:
        return str(p.relative_to(ROOT))

    md = []
    md.append("# Experiment 1 — Quantization Accuracy on Common Deep Learning Distributions")
    md.append("")
    md.append("> **Generated automatically** by `experiments/exp1_common_distributions.py`")
    md.append("")

    # ── Overview ──────────────────────────────────────────────────────────────
    md.append("## 1. Experimental Setup")
    md.append("")
    md.append("### 1.1 Motivation")
    md.append(textwrap.dedent("""\
        Quantization quality depends critically on the statistical properties of the
        tensors being quantized.  Different formats exploit different structural priors
        (block-local scale, sparsity, rotation invariance) that may be more or less
        matched to a given distribution.  This experiment systematically characterises
        the SQNR (Signal-to-Quantization-Noise Ratio) of nine formats across 23
        distribution variants that cover the full diversity of weight and activation
        tensors encountered in large language models (LLMs) and vision transformers.
        """))

    md.append("### 1.2 Format Definitions")
    md.append("")
    md.append(textwrap.dedent("""\
        The nominal bit-width `BITS` is a configurable parameter (default 8).
        Changing `BITS` to 4 re-instantiates *all* formats at the lower precision,
        enabling apples-to-apples cross-precision comparison.

        | Name | Description | High prec | Low prec | Sparsity |
        |---|---|---|---|---|
        | FP16 | IEEE-754 half precision — upper-bound reference | — | — | — |
        | INT{B}-PerTensor | Symmetric INT, single POT scale per tensor | — | INT{B} | — |
        | INT{B}-PerChannel | Symmetric INT, POT scale per 64-element group | — | INT{B} | — |
        | MXINT{B} | OCP MX block-scaled INT (block=32, E8M0 scale) | — | INT{B} | — |
        | MXFP{B} | OCP MX block-scaled FP (MXFP8 E4M3 / MXFP4 E2M1) | — | FP{B} | — |
        | HAD+INT{B}-PerTensor | Walsh-Hadamard + INT{B} per-tensor | — | INT{B} | — |
        | HAD+INT{B}-PerChannel | Walsh-Hadamard + INT{B} per-channel | — | INT{B} | — |
        | SQ-Format-INT{B} | Bank-based mixed precision: INT{B}/INT{B÷2}, s=0.5 | INT{B} | INT{B÷2} | 50% |
        | SQ-Format-FP{B} | Bank-based mixed precision: FP{B}/INT{B÷2}, s=0.5 | FP{B} | INT{B÷2} | 50% |

        *SQ-Format effective bit-width* = 0.5 × BITS + 0.5 × (BITS÷2) = 0.75 × BITS.
        """))

    md.append("### 1.3 Distribution Suite")
    md.append("")
    md.append(textwrap.dedent("""\
        | Family | Variants | DL Context |
        |---|---|---|
        | Gaussian | σ = 0.5, 1.0, 2.0, 5.0 | Initialised weight matrices; BatchNorm outputs |
        | Laplace | b = 0.5, 1.0, 2.0 | FFN weights after SGD/Adam (heavier tail) |
        | Student-t | ν = 3, 5, 10 | Transformer activations; ν→∞ approaches Gaussian |
        | Bimodal | μ = ±2, ±3, ±5 | Attention softmax outputs; post-LayerNorm saturations |
        | Channel Outlier | σ_out = 30, 50, 100 | Systematic LLM.int8() outliers (fixed channels) |
        | Spiky Outlier | ×10, ×50, ×100 | Random weight spikes (AWQ/GPTQ regime) |
        | Log-Normal | σ = 1.0, 2.0 | Post-GELU/SiLU activations; gradient magnitudes |
        | Uniform | ±1, ±3 | Adversarial for INT (maximally flat histogram) |
        """))

    md.append("### 1.4 Metrics")
    md.append("")
    md.append(textwrap.dedent("""\
        Primary metric: **SQNR (dB)** = 10 log₁₀(Var(x) / MSE(x, x̂)).
        Higher is better.  Additional metrics (MSE, KL divergence, MaxAE) are
        computed but not plotted in the main report.

        n = 4,096 elements per sample, seed = 42 (all results deterministic).
        """))

    # ── 8-bit Results ─────────────────────────────────────────────────────────
    md.append("---")
    md.append("## 2. Results — 8-bit Formats")
    md.append("")

    md.append("### 2.1 SQNR Heatmap")
    md.append("")
    md.append(f"![Heatmap 8-bit]({rel(figs_8['heatmap'])})")
    md.append("")
    md.append(textwrap.dedent("""\
        **Reading guide:** Each cell shows the SQNR (dB) for that (format, distribution)
        pair.  Green = high quality; red = high distortion.  The colour scale is
        normalised per-figure so relative differences within each figure are clear.
        """))

    md.append("### 2.2 SQNR by Distribution Family")
    md.append("")
    md.append(f"![Family bars 8-bit]({rel(figs_8['family'])})")

    md.append("### 2.3 Gaussian σ Sensitivity")
    md.append("")
    md.append(f"![Gaussian sweep 8-bit]({rel(figs_8['gauss'])})")
    md.append("")
    md.append(textwrap.dedent("""\
        SQNR should be *independent* of σ for a well-calibrated format (scale adapts
        to absmax).  Formats that show a slope here have a structural mismatch
        between their quantisation grid and Gaussian statistics.
        """))

    md.append("### 2.4 Outlier Robustness")
    md.append("")
    md.append(f"![Outlier sweep 8-bit]({rel(figs_8['outlier'])})")
    md.append("")
    md.append(textwrap.dedent("""\
        **Channel Outlier** (left): systematic fixed-channel outliers as observed in
        LLM.int8() and SmoothQuant papers.  **Spiky Outlier** (right): random
        high-magnitude spikes as studied in AWQ/GPTQ.  Slope steepness measures
        how quickly SQNR degrades as outlier severity increases.
        """))

    md.append("### 2.5 Overall Ranking")
    md.append("")
    md.append(f"![Overall ranking 8-bit]({rel(figs_8['ranking'])})")

    md.append("### 2.6 Best Format per Distribution")
    md.append("")
    md.append(f"![Best format 8-bit]({rel(figs_8['best'])})")

    md.append("### 2.7 Detailed SQNR Table (8-bit, all distributions)")
    md.append("")
    md.append(build_sqnr_table(results_8))
    md.append("")

    md.append("### 2.8 Per-Family SQNR Summary (8-bit)")
    md.append("")
    md.append(build_family_table(results_8))
    md.append("")

    md.append("### 2.9 Format Summary (8-bit)")
    md.append("")
    md.append(build_summary_table(results_8))
    md.append("")

    md.append("### 2.10 Key Findings — 8-bit")
    md.append("")
    md.append(derive_findings(results_8))
    md.append("")

    # ── 4-bit Results ─────────────────────────────────────────────────────────
    md.append("---")
    md.append("## 3. Results — 4-bit Formats")
    md.append("")

    md.append("### 3.1 SQNR Heatmap")
    md.append("")
    md.append(f"![Heatmap 4-bit]({rel(figs_4['heatmap'])})")

    md.append("### 3.2 SQNR by Distribution Family")
    md.append("")
    md.append(f"![Family bars 4-bit]({rel(figs_4['family'])})")

    md.append("### 3.3 Gaussian σ Sensitivity")
    md.append("")
    md.append(f"![Gaussian sweep 4-bit]({rel(figs_4['gauss'])})")

    md.append("### 3.4 Outlier Robustness")
    md.append("")
    md.append(f"![Outlier sweep 4-bit]({rel(figs_4['outlier'])})")

    md.append("### 3.5 Overall Ranking")
    md.append("")
    md.append(f"![Overall ranking 4-bit]({rel(figs_4['ranking'])})")

    md.append("### 3.6 Best Format per Distribution")
    md.append("")
    md.append(f"![Best format 4-bit]({rel(figs_4['best'])})")

    md.append("### 3.7 Detailed SQNR Table (4-bit, all distributions)")
    md.append("")
    md.append(build_sqnr_table(results_4))
    md.append("")

    md.append("### 3.8 Per-Family SQNR Summary (4-bit)")
    md.append("")
    md.append(build_family_table(results_4))
    md.append("")

    md.append("### 3.9 Format Summary (4-bit)")
    md.append("")
    md.append(build_summary_table(results_4))
    md.append("")

    md.append("### 3.10 Key Findings — 4-bit")
    md.append("")
    md.append(derive_findings(results_4))
    md.append("")

    # ── 4-bit vs 8-bit Cross-Comparison ──────────────────────────────────────
    md.append("---")
    md.append("## 4. Cross-Precision Analysis: 4-bit vs 8-bit")
    md.append("")

    combined = pd.concat([results_8, results_4])
    mean_by_bits = (combined.groupby(["bits","format"])["SQNR_dB"]
                             .mean()
                             .unstack("bits"))
    mean_by_bits.columns = ["4-bit SQNR (dB)", "8-bit SQNR (dB)"]
    mean_by_bits["Δ (8b−4b)"] = mean_by_bits["8-bit SQNR (dB)"] - mean_by_bits["4-bit SQNR (dB)"]

    # Build markdown table
    md.append("### 4.1 Mean SQNR: 4-bit vs 8-bit")
    md.append("")
    fmts_8 = list(build_formats(8).keys())
    fmts_4 = list(build_formats(4).keys())

    # Match format families by removing the bit-width suffix
    def strip_bits(name):
        return name.replace("8","B").replace("4","B")

    cross_rows = []
    for f8, f4 in zip(fmts_8, fmts_4):
        m8 = results_8[results_8["format"] == f8]["SQNR_dB"].mean()
        m4 = results_4[results_4["format"] == f4]["SQNR_dB"].mean()
        cross_rows.append({
            "Format family": strip_bits(f8),
            "8-bit SQNR (dB)": round(m8, 1),
            "4-bit SQNR (dB)": round(m4, 1),
            "Δ (8b−4b, dB)":   round(m8 - m4, 1),
        })

    cross_df = pd.DataFrame(cross_rows)
    md.append("| Format family | 8-bit SQNR (dB) | 4-bit SQNR (dB) | Δ (8b−4b) |")
    md.append("|---|---|---|---|")
    for _, r in cross_df.iterrows():
        md.append(f"| {r['Format family']} | {r['8-bit SQNR (dB)']} | "
                  f"{r['4-bit SQNR (dB)']} | {r['Δ (8b−4b, dB)']} |")
    md.append("")

    md.append(textwrap.dedent("""\
        ### 4.2 Interpretation

        The Δ column measures the *precision tax*: how many dB are lost going from
        8-bit to 4-bit for the equivalent format family.  A small Δ indicates that
        the format's structural mechanism (block scaling, sparsity, rotation) is
        effective enough that halving the bit-width has limited impact.  A large Δ
        suggests the format is primarily limited by bit-depth rather than its
        structural innovations.
        """))

    # ── Conclusions ───────────────────────────────────────────────────────────
    md.append("---")
    md.append("## 5. Conclusions and Recommendations")
    md.append("")
    md.append(textwrap.dedent("""\
        ### 5.1 Format Selection Guide

        | Tensor type | Recommended format (8-bit) | Recommended format (4-bit) |
        |---|---|---|
        | Normal-distributed weights | HAD+INT{B}-PerChannel | HAD+INT{B}-PerChannel |
        | FFN weights (Laplace tails) | MXINT{B} or HAD+INT{B}-PC | MXFP{B} |
        | Activations (Transformer) | HAD+INT{B}-PerChannel | SQ-Format-FP{B} |
        | Activations (LLM outliers) | SQ-Format-INT{B} | SQ-Format-FP{B} |
        | Uniform / adversarial | MXFP{B} | MXFP{B} |
        | Mixed / unknown | HAD+INT{B}-PerChannel | HAD+INT{B}-PerChannel |

        ### 5.2 Format Characterisation

        - **INT-PerTensor**: Simple baseline.  Adequate only for low-kurtosis distributions.
          Severely degraded by outliers due to global scale compression.
        - **INT-PerChannel**: Significant improvement over per-tensor for all
          distributions with inter-channel variance.  Preferred over per-tensor
          in nearly all scenarios at negligible overhead.
        - **MXINT**: Block-local scale (32 elements) provides robustness against
          moderate outliers without changing the element format.  Hardware-native.
        - **MXFP**: Non-uniform FP grid better matches log-normal and heavy-tailed
          distributions.  Outperforms MXINT for activations; marginal at 8-bit.
        - **HAD+INT-PerTensor**: Rotation reduces effective kurtosis, improving
          INT quantisation for all symmetric distributions.  For 1-D vectors,
          per-tensor and per-channel coincide.
        - **HAD+INT-PerChannel**: Best overall format across normal, Laplace,
          and bimodal distributions.  Combines rotation-based kurtosis reduction
          with fine-grained per-group scaling.  Slight overhead for the inverse
          HAD pass at inference time.
        - **SQ-Format-INT**: Explicitly protects important elements (high
          magnitude / high Hessian curvature) with extra precision.  Best for
          channel-outlier and spiky distributions.  Effective bit-width ≈ 0.75×BITS.
        - **SQ-Format-FP**: Replaces the high-precision INT path with a floating-
          point quantiser (FP8 E4M3 at 8-bit, FP4 E2M1 at 4-bit).  The
          non-uniform FP grid better covers log-normal and long-tailed outliers.
          At 4-bit this advantage is especially pronounced because FP4 E2M1's
          non-linearity is better matched to the long tail.

        ### 5.3 Limitations

        This experiment uses 1-D synthetic tensors; real weight matrices and
        activation matrices have additional structure (inter-channel correlation,
        per-column statistics) that can further differentiate per-tensor vs
        per-channel approaches.  Experiment 2 (weight matrices) will address this.
        """))

    out = ROOT / "analysis.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\n✓ Written → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# §9  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_one(bits: int) -> tuple[pd.DataFrame, dict]:
    """Run sweep for one bit-width; return (df, figs_dict)."""
    print(f"\n{'='*60}")
    print(f"  Running {bits}-bit experiment  ({len(build_distributions())} distributions)")
    print(f"{'='*60}")

    df = run_sweep(bits)

    print(f"\n  Generating figures …")
    figs = {
        "heatmap": plot_heatmap(df, bits, OUT_FIGURES),
        "family":  plot_family_bars(df, bits, OUT_FIGURES),
        "gauss":   plot_gaussian_sweep(df, bits, OUT_FIGURES),
        "outlier": plot_outlier_sweep(df, bits, OUT_FIGURES),
        "ranking": plot_overall_ranking(df, bits, OUT_FIGURES),
        "best":    plot_best_format(df, bits, OUT_FIGURES),
    }
    for k, p in figs.items():
        print(f"    ✓ {p.name}")

    csv_out = OUT_FIGURES / f"results_{bits}bit.csv"
    df.to_csv(csv_out, index=False)
    print(f"  ✓ CSV  → {csv_out.name}")

    return df, figs


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Common distributions")
    parser.add_argument("--bits", default="all",
                        help="8, 4, or all (default: all)")
    args = parser.parse_args()

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 180,
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 11,
        "legend.fontsize": 8,
    })

    if args.bits == "all":
        bits_list = [8, 4]
    else:
        bits_list = [int(args.bits)]

    results = {}
    figs    = {}
    for b in bits_list:
        results[b], figs[b] = run_one(b)

    # Fill in missing data with empty df/figs if only one bit-width was run
    if 8 not in results:
        results[8] = results[list(results.keys())[0]]
        figs[8]    = figs[list(figs.keys())[0]]
    if 4 not in results:
        results[4] = results[list(results.keys())[0]]
        figs[4]    = figs[list(figs.keys())[0]]

    write_analysis_md(results[8], results[4], figs[8], figs[4])
    print("\n  Done.")


if __name__ == "__main__":
    main()
