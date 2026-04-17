"""Markdown report generation for the 4-bit study.

Produces a self-contained Markdown file with:

  * Part 1 tables (one per sub-experiment) – pivoted so rows are
    distributions, columns are (transform × format) groups.

  * Part 2 figures – scatter plots of crest factor vs QSNR, one PNG per
    transform (base / smooth / had), three per tensor role (W / X / Y).

  * Part 2 Table 1 – per-layer detail grouped into base / smooth / had
    sections.

  * Part 2 Table 2 – per-layer *best transform* choice per format, plus a
    summary of the optimal-combination QSNR achievable for each format.

No external templates are required; the function writes plain GitHub-flavour
Markdown with relative PNG paths.
"""
from __future__ import annotations

import os
from typing import Dict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fourbit.config import FourBitConfig


# ── Helpers ──────────────────────────────────────────────────────────────────

def _fmt_db(v) -> str:
    if pd.isna(v):
        return "–"
    return f"{float(v):.2f}"


def _pivot_qsnr(df: pd.DataFrame, value: str, col: str) -> pd.DataFrame:
    """Pivot ``df`` to (distribution × col) showing ``value`` per format group."""
    return df.pivot_table(
        index="distribution", columns=col, values=value, aggfunc="mean"
    )


def _df_to_md(df: pd.DataFrame, float_cols: list | None = None) -> str:
    """Render a small DataFrame as a GitHub-flavour Markdown table."""
    if float_cols is None:
        float_cols = [c for c in df.columns if df[c].dtype.kind in ("f",)]
    header = ["| " + " | ".join([df.index.name or ""] + list(map(str, df.columns))) + " |"]
    sep = ["|" + "|".join(["---"] * (len(df.columns) + 1)) + "|"]
    body = []
    for idx, row in df.iterrows():
        vals = []
        for c in df.columns:
            v = row[c]
            if c in float_cols:
                vals.append(_fmt_db(v))
            else:
                vals.append(str(v))
        body.append("| " + " | ".join([str(idx)] + vals) + " |")
    return "\n".join(header + sep + body)


def _section_header(title: str, level: int = 2) -> str:
    return "#" * level + " " + title


# ── Part 1 tables ────────────────────────────────────────────────────────────

def _part1_tables(part1: Dict[str, pd.DataFrame]) -> str:
    md = []
    md.append(_section_header("Part 1 — Synthetic Distribution Analysis"))

    # 1.1
    md.append(_section_header("1.1 Direct quantization of common distributions", 3))
    p = _pivot_qsnr(part1["exp11"], "qsnr_db", "format")
    p.columns.name = "format"
    p.index.name = "distribution"
    # Stable column order
    wanted_order = ["INT4", "FP4", "NF4", "NVFP4", "MXINT4", "MXFP4"]
    cols = [c for c in wanted_order if c in p.columns]
    p = p[cols]
    md.append("QSNR (dB) — higher is better.")
    md.append("")
    md.append(_df_to_md(p.round(2)))
    md.append("")

    # 1.2
    md.append(_section_header(
        "1.2 Linear Y = X W^T, base quantization only — Y QSNR (dB)", 3
    ))
    p = _pivot_qsnr(part1["exp12"], "qsnr_y_db", "format")
    p = p[[c for c in wanted_order if c in p.columns]]
    p.index.name = "W/X distribution"
    md.append(_df_to_md(p.round(2)))
    md.append("")

    # 1.3 – three sub-tables (base, smooth, had)
    md.append(_section_header(
        "1.3 Smooth-friendly pairs — base / smooth / had transforms", 3
    ))
    for t in ["base", "smooth", "had"]:
        sub = part1["exp13"][part1["exp13"]["transform"] == t]
        if sub.empty:
            continue
        md.append(f"**Transform: `{t}` — Y QSNR (dB)**")
        md.append("")
        p = sub.pivot_table(
            index="distribution", columns="format", values="qsnr_y_db", aggfunc="mean"
        )
        p = p[[c for c in wanted_order if c in p.columns]]
        p.index.name = "distribution"
        md.append(_df_to_md(p.round(2)))
        md.append("")
    return "\n".join(md)


# ── Part 2 figures ───────────────────────────────────────────────────────────

def _scatter_figure(
    df: pd.DataFrame, crest_col: str, qsnr_col: str,
    title: str, out_path: str,
) -> None:
    """One scatter per format, colour-coded by format."""
    formats = sorted(df["format"].unique())
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, fmt in enumerate(formats):
        sub = df[df["format"] == fmt].dropna(subset=[crest_col, qsnr_col])
        if sub.empty:
            continue
        ax.scatter(sub[crest_col], sub[qsnr_col],
                   label=fmt, alpha=0.8, s=35, color=cmap(i % 10))
    ax.set_xlabel("Crest factor (peak / std)")
    ax.set_ylabel("QSNR (dB)")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _render_all_scatter_plots(
    df: pd.DataFrame, out_dir: str,
) -> dict[str, str]:
    """Emit 9 scatter plots (3 transforms × 3 tensor roles). Return relative paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths: dict[str, str] = {}

    role_map = {
        "input":  ("X_crest", "qsnr_x_db", "Inputs"),
        "output": ("Y_crest", "qsnr_y_db", "Outputs"),
        "weight": ("W_crest", "qsnr_w_db", "Weights"),
    }
    for transform in ["base", "smooth", "had"]:
        tdf = df[df["transform"] == transform]
        for role_key, (xcol, ycol, role_lbl) in role_map.items():
            fname = f"scatter_{role_key}_{transform}.png"
            fpath = os.path.join(out_dir, fname)
            _scatter_figure(
                tdf,
                crest_col=xcol, qsnr_col=ycol,
                title=f"Part 2: {role_lbl} – transform = {transform}",
                out_path=fpath,
            )
            paths[f"{role_key}_{transform}"] = fname
    return paths


# ── Part 2 tables ────────────────────────────────────────────────────────────

def _part2_detail_table(df: pd.DataFrame) -> str:
    """Table 1 – per-layer QSNR grouped by format and transform, rounded to 2 dp."""
    md = []
    md.append("**Table 1 — Per-layer output QSNR (dB) grouped by transform.**")
    md.append("")
    for transform in ["base", "smooth", "had"]:
        md.append(f"*Transform: {transform}*")
        md.append("")
        sub = df[df["transform"] == transform]
        if sub.empty:
            md.append("_(no rows)_")
            continue
        p = sub.pivot_table(
            index="layer", columns="format", values="qsnr_y_db", aggfunc="mean"
        )
        order = ["INT4", "FP4", "NF4", "NVFP4", "MXINT4", "MXFP4"]
        p = p[[c for c in order if c in p.columns]]
        p.index.name = "layer"
        md.append(_df_to_md(p.round(2)))
        md.append("")
    return "\n".join(md)


def _part2_best_transform_table(df: pd.DataFrame) -> str:
    """Table 2 – per-layer best transform + per-format optimal-mix summary."""
    md: list[str] = []
    md.append("**Table 2 — Best transform per layer per format + per-format "
              "optimal-combination summary.**")
    md.append("")

    # Per-layer best transform for every format
    by = (
        df.dropna(subset=["qsnr_y_db"])
          .sort_values("qsnr_y_db", ascending=False)
          .groupby(["layer", "format"], as_index=False)
          .first()
    )
    if by.empty:
        md.append("_(no rows – all NaN)_")
        return "\n".join(md)

    order = ["INT4", "FP4", "NF4", "NVFP4", "MXINT4", "MXFP4"]

    md.append("*Per-layer best transform (best QSNR in dB).*")
    md.append("")
    # Build a multi-column table: rows = layer, columns = (format best_transform / qsnr)
    pivot_lbl = by.pivot_table(
        index="layer", columns="format",
        values="transform", aggfunc="first",
    )
    pivot_val = by.pivot_table(
        index="layer", columns="format", values="qsnr_y_db", aggfunc="first",
    )
    formatted = pd.DataFrame(index=pivot_lbl.index)
    for col in order:
        if col not in pivot_lbl.columns:
            continue
        formatted[col] = [
            f"{pivot_lbl.loc[i, col]} ({_fmt_db(pivot_val.loc[i, col])})"
            if pd.notna(pivot_lbl.loc[i, col]) else "–"
            for i in formatted.index
        ]
    formatted.index.name = "layer"
    md.append(_df_to_md(formatted, float_cols=[]))
    md.append("")

    # Per-format optimal-mix summary: mean of best QSNR across layers
    md.append("*Per-format optimal-combination QSNR summary (mean over layers).*")
    md.append("")
    summary_rows = []
    for fmt in order:
        sub = by[by["format"] == fmt]
        if sub.empty:
            continue
        # Transform frequency
        counts = sub["transform"].value_counts().to_dict()
        total = sum(counts.values())
        freq = " / ".join(f"{t}:{counts.get(t, 0)}/{total}"
                          for t in ["base", "smooth", "had"])
        summary_rows.append({
            "format":          fmt,
            "mean_qsnr_db":    round(float(sub["qsnr_y_db"].mean()), 2),
            "median_qsnr_db":  round(float(sub["qsnr_y_db"].median()), 2),
            "min_qsnr_db":     round(float(sub["qsnr_y_db"].min()), 2),
            "transform_split": freq,
        })
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).set_index("format")
        md.append(_df_to_md(summary_df))
    return "\n".join(md)


# ── Full report ──────────────────────────────────────────────────────────────

def generate_report(
    config: FourBitConfig,
    part1: Dict[str, pd.DataFrame],
    part2_metrics: pd.DataFrame,
    out_path: str,
) -> str:
    """Write the full Markdown report to ``out_path`` and return the written text."""
    report_dir = os.path.dirname(out_path) or "."
    figures_dir = os.path.join(report_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Render scatter plots
    scatter_paths = _render_all_scatter_plots(part2_metrics, figures_dir)

    lines: list[str] = []
    lines.append("# 4-bit Data Format Study — Report")
    lines.append("")
    lines.append(
        "Compares six 4-bit formats (**INT4, FP4, NF4, NVFP4, MXINT4, MXFP4**) "
        "under three transforms (**base, smooth, had**) across synthetic "
        "distributions (Part 1) and a real MNIST Transformer (Part 2)."
    )
    lines.append("")
    lines.append("Formats: `INT4`, `FP4`, `NF4` use per-channel absmax "
                 "(POT scale for INT4 / FP4, float scale for NF4). "
                 "`NVFP4` is per-tensor E2M1. `MXINT4`, `MXFP4` use OCP-MX "
                 "block scaling (block size 32).")
    lines.append("")

    # ── Part 1 ──────────────────────────────────────────────────────────
    lines.append(_part1_tables(part1))
    lines.append("")

    # ── Part 2 ──────────────────────────────────────────────────────────
    lines.append(_section_header("Part 2 — Real Model Analysis (MNIST Transformer)"))
    lines.append("")
    lines.append(
        "Profiled a trained MNISTTransformer on a held-out test subset. "
        "For every `nn.Linear` layer the profiler records the weight matrix, "
        "the batch of inputs, and the FP32 output reference. QSNR is then "
        "computed for every (format × transform) combination, including the "
        "full quantized linear simulation for the output."
    )
    lines.append("")

    # Part 2 figures
    lines.append(_section_header("Figures — Crest Factor vs QSNR", 3))
    for role_key, role_name in [("input", "Input"), ("output", "Output"), ("weight", "Weight")]:
        lines.append(f"**{role_name}**")
        lines.append("")
        for t in ["base", "smooth", "had"]:
            key = f"{role_key}_{t}"
            if key in scatter_paths:
                rel = os.path.join("figures", scatter_paths[key])
                lines.append(f"*{t}*")
                lines.append("")
                lines.append(f"![{role_name} / {t}]({rel})")
                lines.append("")

    # Part 2 Table 1
    lines.append(_section_header("Table 1 — Per-layer QSNR detail", 3))
    lines.append("")
    lines.append(_part2_detail_table(part2_metrics))
    lines.append("")

    # Part 2 Table 2
    lines.append(_section_header("Table 2 — Per-layer best transform + per-format "
                                  "optimal-combination QSNR", 3))
    lines.append("")
    lines.append(_part2_best_transform_table(part2_metrics))
    lines.append("")

    # ── Config appendix ─────────────────────────────────────────────────
    lines.append(_section_header("Appendix — Configuration"))
    lines.append("")
    lines.append("```")
    lines.append(f"n_samples          = {config.n_samples}")
    lines.append(f"batch_size         = {config.batch_size}")
    lines.append(f"in_features        = {config.in_features}  "
                  "(must be power of 2 for HAD)")
    lines.append(f"out_features       = {config.out_features}")
    lines.append(f"smooth_alpha       = {config.smooth_alpha}")
    lines.append(f"profile_samples    = {config.profile_samples}")
    lines.append(f"seed               = {config.seed}")
    lines.append(f"formats            = {[f.display_name for f in config.formats]}")
    lines.append(f"transforms         = {[t.display_name for t in config.transforms]}")
    lines.append("```")
    lines.append("")

    text = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(text)
    return text
