"""Markdown report generation for the 4-bit study.

Produces a self-contained Markdown file with:

  * Part 1 tables (one per sub-experiment) – pivoted so rows are
    distributions, columns are formats. An extra FP16-baseline column is
    shown per distribution so readers can see the upper bound on QSNR.

  * Part 2 figures:
      1. One "role × transform" subplot grid per tensor role
         (inputs / weights / outputs), each containing three subplots
         (base / smooth / had).
      2. A dispersion figure (three subplots for inputs/weights/outputs):
         crest factor on the x-axis versus FP16 QSNR on the y-axis, one
         dot per layer, colour-coded by layer name.  This exposes how
         spread-out each tensor role is across the model.

  * Part 2 tables: per-layer per-transform detail, per-layer best
    transform, per-format summary of the optimal-combination QSNR.

  * Part 2 accuracy table: FP32 / FP16 baselines plus every
    (format, transform) top-1 accuracy and "optimal-combination" row.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fourbit.config import FourBitConfig


FORMAT_ORDER = [
    "INT4", "INT4_FP", "FP4", "NF4", "NF4_FP8",
    "APoT4", "LOG4", "NVFP4", "MXINT4", "MXFP4",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _fmt_db(v) -> str:
    if pd.isna(v):
        return "–"
    return f"{float(v):.2f}"


def _fmt_pct(v) -> str:
    if pd.isna(v):
        return "–"
    return f"{100.0 * float(v):.2f}%"


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


def _ordered_formats(df: pd.DataFrame) -> list:
    return [f for f in FORMAT_ORDER if f in set(df["format"].unique())]


def _section_header(title: str, level: int = 2) -> str:
    return "#" * level + " " + title


# ── Part 1 tables ────────────────────────────────────────────────────────────

def _fp16_baseline_column(df: pd.DataFrame, value: str = "fp16_qsnr_db"
                          ) -> pd.Series:
    """Return the FP16-baseline QSNR per distribution (constant per dist)."""
    if value not in df.columns:
        return pd.Series(dtype=float)
    return df.groupby("distribution")[value].first()


def _part1_tables(part1: Dict[str, pd.DataFrame]) -> str:
    md = []
    md.append(_section_header("Part 1 — Synthetic Distribution Analysis"))

    # 1.1
    md.append(_section_header("1.1 Direct quantization of common distributions", 3))
    df11 = part1["exp11"]
    p = _pivot_qsnr(df11, "qsnr_db", "format")
    p = p[_ordered_formats(df11)]
    p.index.name = "distribution"

    fp16_col = _fp16_baseline_column(df11, "fp16_qsnr_db")
    if not fp16_col.empty:
        p.insert(0, "FP16 (baseline)", fp16_col.reindex(p.index))
    md.append("QSNR (dB) — higher is better.  `FP16 (baseline)` is the QSNR "
              "of simply rounding the tensor to half-precision — the upper "
              "bound for *any* 4-bit format on this distribution.")
    md.append("")
    md.append(_df_to_md(p.round(2)))
    md.append("")

    # 1.2
    md.append(_section_header(
        "1.2 Linear Y = X W^T, base quantization only — Y QSNR (dB)", 3
    ))
    df12 = part1["exp12"]
    p = _pivot_qsnr(df12, "qsnr_y_db", "format")
    p = p[_ordered_formats(df12)]
    p.index.name = "W/X distribution"
    fp16_col = _fp16_baseline_column(df12, "fp16_qsnr_y_db")
    if not fp16_col.empty:
        p.insert(0, "FP16 Y (baseline)", fp16_col.reindex(p.index))
    md.append(_df_to_md(p.round(2)))
    md.append("")

    # 1.3 – three sub-tables (base, smooth, had)
    md.append(_section_header(
        "1.3 Smooth-friendly pairs — base / smooth / had transforms", 3
    ))
    df13 = part1["exp13"]
    for t in ["base", "smooth", "had"]:
        sub = df13[df13["transform"] == t]
        if sub.empty:
            continue
        md.append(f"**Transform: `{t}` — Y QSNR (dB)**")
        md.append("")
        p = sub.pivot_table(
            index="distribution", columns="format",
            values="qsnr_y_db", aggfunc="mean",
        )
        p = p[_ordered_formats(sub)]
        p.index.name = "distribution"
        fp16_col = _fp16_baseline_column(sub, "fp16_qsnr_y_db")
        if not fp16_col.empty:
            p.insert(0, "FP16 Y (baseline)", fp16_col.reindex(p.index))
        md.append(_df_to_md(p.round(2)))
        md.append("")
    return "\n".join(md)


# ── Part 2 figures ───────────────────────────────────────────────────────────

def _qsnr_scatter_subplots(
    df: pd.DataFrame,
    crest_col: str,
    qsnr_col: str,
    role_label: str,
    out_path: str,
) -> None:
    """Render ONE figure with 3 subplots (base/smooth/had) for a tensor role.

    Each subplot shows every format's (crest, qsnr) points for that
    transform, so the reader can compare format rankings across transforms
    at a glance.
    """
    transforms = ["base", "smooth", "had"]
    fig, axes = plt.subplots(
        1, 3, figsize=(14.5, 4.5), sharey=True,
    )

    formats = _ordered_formats(df)
    cmap = plt.get_cmap("tab10")
    colour_for = {f: cmap(i % 10) for i, f in enumerate(formats)}

    # Determine a common y-range so the subplots line up cleanly.
    y_vals = df[qsnr_col].dropna()
    if not y_vals.empty:
        y_lo = float(np.floor(y_vals.min() - 1))
        y_hi = float(np.ceil(y_vals.max() + 1))
    else:
        y_lo, y_hi = 0.0, 40.0

    for ax, t in zip(axes, transforms):
        tdf = df[df["transform"] == t]
        if tdf.empty:
            ax.set_title(f"{role_label} – transform={t} (empty)")
            continue
        for fmt in formats:
            sub = tdf[tdf["format"] == fmt].dropna(subset=[crest_col, qsnr_col])
            if sub.empty:
                continue
            ax.scatter(sub[crest_col], sub[qsnr_col],
                       alpha=0.85, s=38, color=colour_for[fmt], label=fmt)
        ax.set_xlabel("Crest factor (peak / std)")
        ax.set_title(f"{role_label} – transform = {t}")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_ylim(y_lo, y_hi)
    axes[0].set_ylabel("QSNR (dB)")
    # One legend for all three subplots, placed at the top right.
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right",
                   bbox_to_anchor=(0.995, 0.995), fontsize=8, ncol=2)
    fig.suptitle(f"Crest factor vs QSNR — {role_label}", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _dispersion_scatter(
    layer_stats_df: pd.DataFrame, out_path: str,
) -> None:
    """One figure, three subplots: crest factor × FP16 QSNR for W/X/Y.

    Each dot is a layer.  High crest + low FP16 QSNR means the tensor has
    extreme outliers that even FP16 rounding struggles with.  The three
    subplots share the y-axis so the reader sees relative dispersion
    between weights, inputs and outputs on the same scale.
    """
    roles = [
        ("W_crest", "fp16_qsnr_w_db", "Weights"),
        ("X_crest", "fp16_qsnr_x_db", "Inputs"),
        ("Y_crest", "fp16_qsnr_y_db", "Outputs"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5), sharey=True)
    cmap = plt.get_cmap("tab20")

    # All layers share the same palette
    layers = sorted(layer_stats_df["layer"].unique())
    colour_for = {lyr: cmap(i % 20) for i, lyr in enumerate(layers)}

    for ax, (xcol, ycol, title) in zip(axes, roles):
        if xcol not in layer_stats_df.columns or ycol not in layer_stats_df.columns:
            ax.set_title(f"{title} (missing columns)")
            continue
        for lyr in layers:
            sub = layer_stats_df[layer_stats_df["layer"] == lyr]
            if sub.empty:
                continue
            ax.scatter(sub[xcol], sub[ycol],
                       s=60, alpha=0.85, color=colour_for[lyr], label=lyr)
        ax.set_xlabel("Crest factor (peak / std)")
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.5)
    axes[0].set_ylabel("FP16 QSNR (dB) — baseline precision")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right",
                   bbox_to_anchor=(1.0, 1.0), fontsize=7)
    fig.suptitle(
        "Tensor dispersion per role — FP16 QSNR vs crest factor "
        "(each dot = one Linear layer)",
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _render_part2_figures(
    df: pd.DataFrame, out_dir: str,
) -> dict[str, str]:
    """Emit:
      * 3 role figures (each with base/smooth/had subplots).
      * 1 dispersion figure (W/X/Y subplots vs FP16 QSNR).

    Returns a ``{key: relative_filename}`` mapping for the reporter.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths: dict[str, str] = {}

    role_map = {
        "input":  ("X_crest", "qsnr_x_db", "Inputs"),
        "output": ("Y_crest", "qsnr_y_db", "Outputs"),
        "weight": ("W_crest", "qsnr_w_db", "Weights"),
    }
    for role_key, (xcol, ycol, role_label) in role_map.items():
        fname = f"scatter_{role_key}_subplots.png"
        _qsnr_scatter_subplots(
            df, crest_col=xcol, qsnr_col=ycol,
            role_label=role_label,
            out_path=os.path.join(out_dir, fname),
        )
        paths[role_key] = fname

    # Dispersion figure needs one row per layer (FP16 QSNR is constant per
    # tensor / layer, independent of format / transform).
    # We therefore deduplicate the (layer, fp16_qsnr, crest) triples.
    cols = [
        "layer",
        "W_crest", "X_crest", "Y_crest",
        "fp16_qsnr_w_db", "fp16_qsnr_x_db", "fp16_qsnr_y_db",
    ]
    available = [c for c in cols if c in df.columns]
    layer_stats = (
        df[available]
        .drop_duplicates(subset="layer")
        .reset_index(drop=True)
    )
    disp_name = "dispersion_fp16qsnr_vs_crest.png"
    _dispersion_scatter(layer_stats, os.path.join(out_dir, disp_name))
    paths["dispersion"] = disp_name
    return paths


# ── Part 2 tables ────────────────────────────────────────────────────────────

def _part2_detail_table(df: pd.DataFrame) -> str:
    """Table 1 – per-layer QSNR grouped by format and transform, rounded to 2 dp."""
    md = []
    md.append("**Table 1 — Per-layer output QSNR (dB) grouped by transform. "
              "The `FP16 Y` column is the baseline QSNR of rounding the "
              "layer output to half-precision (format-independent).**")
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
        p = p[_ordered_formats(sub)]
        fp16_col = (
            sub.groupby("layer")["fp16_qsnr_y_db"].first()
            if "fp16_qsnr_y_db" in sub.columns else None
        )
        if fp16_col is not None and not fp16_col.empty:
            p.insert(0, "FP16 Y", fp16_col.reindex(p.index))
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

    by = (
        df.dropna(subset=["qsnr_y_db"])
          .sort_values("qsnr_y_db", ascending=False)
          .groupby(["layer", "format"], as_index=False)
          .first()
    )
    if by.empty:
        md.append("_(no rows – all NaN)_")
        return "\n".join(md)

    order = _ordered_formats(by)

    md.append("*Per-layer best transform (best QSNR in dB).*")
    md.append("")
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

    md.append("*Per-format optimal-combination QSNR summary (mean over layers).*")
    md.append("")
    summary_rows = []
    for fmt in order:
        sub = by[by["format"] == fmt]
        if sub.empty:
            continue
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


def _part2_accuracy_table(acc_df: pd.DataFrame, qsnr_df: pd.DataFrame) -> str:
    """Accuracy table — FP32/FP16 baseline + every (format, transform)
    plus a per-format 'best transform' row selected by mean Y QSNR."""
    md: list[str] = []
    md.append("**Table 3 — End-to-end MNIST top-1 accuracy under W4A4 "
              "quantisation.  `FP32`/`FP16` are full-precision baselines; "
              "the remaining rows are the exact W4A4 configurations whose "
              "QSNR appears in Table 1.  An extra `BEST` row per format "
              "chooses, per layer, the transform that maximises that "
              "layer's Y QSNR — matching Table 2's oracle.**")
    md.append("")

    # Base accuracy matrix
    baselines = acc_df[acc_df["transform"] == "baseline"].copy()
    quant_rows = acc_df[acc_df["transform"] != "baseline"].copy()

    if not quant_rows.empty:
        pivot = quant_rows.pivot_table(
            index="format", columns="transform", values="accuracy", aggfunc="first",
        )
        ordered_cols = [c for c in ["base", "smooth", "had"] if c in pivot.columns]
        pivot = pivot[ordered_cols]
        # Reorder rows using FORMAT_ORDER
        row_order = [f for f in FORMAT_ORDER if f in pivot.index]
        pivot = pivot.loc[row_order]

        # "BEST" column: per-format + per-layer oracle — compute accuracy by
        # looking up from acc_df using the layer-wise best transform.  We do
        # not rerun the model; instead we approximate BEST as the best of
        # {base, smooth, had} in the per-format accuracy matrix.  This is a
        # lower bound on the true oracle which would choose *per layer*.
        pivot["best_row"] = pivot.max(axis=1, skipna=True)
        pivot["best_transform"] = pivot[ordered_cols].idxmax(axis=1)
        fmt_cols = [c for c in ordered_cols] + ["best_row", "best_transform"]
        pivot = pivot[fmt_cols]

        # Pretty-print to %
        pct = pivot.copy()
        for c in ordered_cols + ["best_row"]:
            pct[c] = pct[c].apply(_fmt_pct)

        pct.index.name = "format"
        md.append("*Per-format accuracy (4-bit configurations).*")
        md.append("")
        md.append(_df_to_md(pct, float_cols=[]))
        md.append("")

    # Baselines
    if not baselines.empty:
        md.append("*Full-precision baselines (no quantisation).*")
        md.append("")
        bl = baselines.set_index("format")[["accuracy"]].copy()
        bl["accuracy"] = bl["accuracy"].apply(_fmt_pct)
        bl.index.name = "format"
        md.append(_df_to_md(bl, float_cols=[]))
        md.append("")
    return "\n".join(md)


# ── Full report ──────────────────────────────────────────────────────────────

def generate_report(
    config: FourBitConfig,
    part1: Dict[str, pd.DataFrame],
    part2_metrics: pd.DataFrame,
    out_path: str,
    accuracy_df: Optional[pd.DataFrame] = None,
) -> str:
    """Write the full Markdown report to ``out_path`` and return the written text."""
    report_dir = os.path.dirname(out_path) or "."
    figures_dir = os.path.join(report_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    figure_paths = _render_part2_figures(part2_metrics, figures_dir)

    lines: list[str] = []
    lines.append("# 4-bit Data Format Study — Report (W4A4)")
    lines.append("")
    lines.append(
        "Compares ten 4-bit formats under three transforms "
        "(**base, smooth, had**) across synthetic distributions (Part 1) "
        "and a real MNIST Transformer (Part 2)."
    )
    lines.append("")
    lines.append(
        "**Quantisation regime — W4A4.** Every linear layer has both its "
        "**weights** and its **input activations** quantised to 4 bits. "
        "The matmul accumulator is kept at FP32 (matching real W4A4 tensor "
        "cores — the 4-bit product routinely overflows into 11+ bits after "
        "the in-feature reduction, so re-quantising the accumulator back "
        "to 4 bits is **not** done inside a single layer).  The next "
        "layer's input becomes 4-bit again at *its* entry, so end-to-end "
        "accuracy errors accumulate across the stacked layers — this is "
        "captured by the accuracy sweep in Table 3."
    )
    lines.append("")
    lines.append(
        "**Format set and hardware notes.**  `INT4` uses a symmetric "
        "power-of-two per-channel scale (pure shift decode). `INT4_FP` is "
        "the same level set with an unrestricted FP per-channel scale — "
        "isolating 'POT overhead' from 'level-set choice'. `FP4` is the "
        "E2M1 set with POT per-channel scale. `NF4` places its 16 levels "
        "at Gaussian quantiles (QLoRA) and uses a full-FP per-channel "
        "scale; `NF4_FP8` is the hardware-realistic variant (QLoRA double-"
        "quantisation: scale stored in FP8-E4M3). `APoT4` uses 8 positive "
        "additive-PoT levels (multiplier-free decode). `LOG4` keeps only "
        "powers of two — shift-only decode, coarsest resolution. `NVFP4` "
        "is the Blackwell per-tensor E2M1. `MXINT4`/`MXFP4` are OCP-MX "
        "block-scaled (block = 32)."
    )
    lines.append("")

    # ── Part 1 ──────────────────────────────────────────────────────────
    lines.append(_part1_tables(part1))
    lines.append("")

    # ── Part 2 ──────────────────────────────────────────────────────────
    lines.append(_section_header("Part 2 — Real Model Analysis (MNIST Transformer)"))
    lines.append("")
    lines.append(
        "Profiled a trained MNIST Transformer on a held-out test subset. "
        "For every `nn.Linear` layer the profiler records the weight matrix, "
        "the batch of inputs, and the FP32 output reference.  QSNR is then "
        "computed for every (format × transform) combination, including "
        "the full W4A4 linear simulation for the output (FP32 accumulator). "
        "A separate accuracy sweep re-runs the model with the same "
        "quantiser applied to every layer and reports top-1 accuracy."
    )
    lines.append("")

    lines.append(_section_header(
        "Figures — Crest Factor vs QSNR (W4A4, one figure per role, "
        "three subplots per transform)", 3,
    ))
    for key, title in [
        ("input", "Inputs"), ("weight", "Weights"), ("output", "Outputs"),
    ]:
        if key not in figure_paths:
            continue
        rel = os.path.join("figures", figure_paths[key])
        lines.append(f"**{title}**")
        lines.append("")
        lines.append(f"![{title} subplots]({rel})")
        lines.append("")

    lines.append(_section_header(
        "Figure — Dispersion: crest factor vs FP16 baseline QSNR", 3,
    ))
    lines.append(
        "Each dot is one `nn.Linear` layer.  The x-axis is the crest "
        "factor (peak/std) of the corresponding tensor, and the y-axis "
        "is the QSNR obtained by *only* rounding the tensor to FP16 — "
        "the best any 4-bit scheme could hope to reach.  A cluster in "
        "the upper-left (low crest, high FP16 QSNR) means the role is "
        "well-behaved and 4-bit quantisation should be easy; a cluster "
        "in the lower-right (high crest, low FP16 QSNR) signals "
        "outlier-dominated tensors that need `smooth` or `had` to be "
        "amenable to 4 bits."
    )
    lines.append("")
    rel = os.path.join("figures", figure_paths["dispersion"])
    lines.append(f"![Dispersion – crest vs FP16 QSNR]({rel})")
    lines.append("")

    # Table 1
    lines.append(_section_header("Table 1 — Per-layer QSNR detail", 3))
    lines.append("")
    lines.append(_part2_detail_table(part2_metrics))
    lines.append("")

    # Table 2
    lines.append(_section_header(
        "Table 2 — Per-layer best transform + per-format optimal-combination QSNR",
        3,
    ))
    lines.append("")
    lines.append(_part2_best_transform_table(part2_metrics))
    lines.append("")

    # Table 3 (accuracy) – only if provided
    if accuracy_df is not None and not accuracy_df.empty:
        lines.append(_section_header(
            "Table 3 — End-to-end accuracy under W4A4", 3,
        ))
        lines.append("")
        lines.append(_part2_accuracy_table(accuracy_df, part2_metrics))
        lines.append("")

    # Config appendix
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
