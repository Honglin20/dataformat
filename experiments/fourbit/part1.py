"""Part 1 – synthetic-distribution experiments.

Three sub-experiments:

  1.1 ``exp11_direct_quant``
      Direct quantization (base scale only) of single-tensor common
      distributions, per format.  Output is a CSV of QSNR and a compact
      summary Markdown table.

  1.2 ``exp12_linear_wa``
      Simulated linear computation Y = X W^T using a handful of realistic
      (W, X) distributions, base quantization only.  Output: QSNR of Y per
      format + distribution.

  1.3 ``exp13_smooth_transforms``
      Smooth-friendly (W, X) pairs.  For every (format × transform) in
      ``DEFAULT_CONFIG`` we report the linear-output QSNR, showing when
      SmoothQuant or Hadamard beats the base mode.

Each sub-experiment returns a pandas DataFrame.  ``run_all`` writes the
DataFrames to ``results/fourbit/part1/`` and also returns a dict for
programmatic use.
"""
from __future__ import annotations

import os
from typing import Dict

import numpy as np
import pandas as pd

from experiments.fourbit.config import FourBitConfig
from experiments.fourbit.registry import (
    build_formats, build_pipelines, make_fresh_transform,
)
from experiments.fourbit.distribution_sets import (
    COMMON_DISTRIBUTIONS, LINEAR_WEIGHT_ACTIVATION, SMOOTH_FRIENDLY,
)
from distributions.metrics import qsnr_db, crest_factor, tensor_summary, fp16_qsnr_db
from experiments.fourbit.pipeline import fp32_linear, Pipeline


# ── Helpers ──────────────────────────────────────────────────────────────────

def _pad_pow2(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Zero-pad ``x`` along ``axis`` so its length is a power of two."""
    n = x.shape[axis]
    p = 1
    while p < n:
        p <<= 1
    if p == n:
        return x
    pad_spec = [(0, 0)] * x.ndim
    pad_spec[axis] = (0, p - n)
    return np.pad(x, pad_spec, mode="constant")


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


# ── Experiment 1.1 – direct quantization of common distributions ─────────────

def exp11_direct_quant(config: FourBitConfig) -> pd.DataFrame:
    """Per-distribution × per-format QSNR for the base mode only.

    Rationale: distributions are 1-D, so per-channel formats degrade to
    per-tensor absmax quantization.  This matches the study spec: "INT4,
    FP4, NF4 全部按 per-channel 量化" – for a 1-D tensor the two are
    identical, so we intentionally leave the format objects as-is.
    """
    rng_seed = config.seed
    n = config.n_samples
    formats = build_formats(config)

    rows: list[dict] = []
    for dist in COMMON_DISTRIBUTIONS:
        x, dist_meta = dist.generate(n, rng_seed)
        stats = tensor_summary(x)
        fp16_qsnr = fp16_qsnr_db(x)

        for fmt_name, fmt in formats.items():
            x_q = fmt.quantize(x)
            rows.append({
                "distribution": dist.name,
                "format":       fmt_name,
                "qsnr_db":      qsnr_db(x, x_q),
                "fp16_qsnr_db": fp16_qsnr,
                "crest":        stats["crest"],
                "kurtosis":     stats["kurtosis"],
                "abs_max":      stats["max_abs"],
                "std":          stats["std"],
                "tags":         ",".join(dist.tags),
            })
    return pd.DataFrame(rows)


# ── Experiment 1.2 – linear simulation with typical W/A distributions ───────

def exp12_linear_wa(config: FourBitConfig) -> pd.DataFrame:
    """Base-mode quantized linear Y = X W^T, reporting Y's QSNR.

    ``in_features`` must be a power of two so that the same experiment can
    be reused for Part 1.3 with the HAD transform.  For 1.2 we only use the
    base transform, so the power-of-two requirement is only cosmetic.
    """
    formats = build_formats(config)
    pipelines_by_fmt = {
        name: Pipeline(
            transform=make_fresh_transform(config, "base"),
            fmt=fmt,
            name=f"base/{name}",
        ) for name, fmt in formats.items()
    }

    rows: list[dict] = []
    for lin in LINEAR_WEIGHT_ACTIVATION:
        X, W, meta = lin.generate(
            batch=config.batch_size,
            in_features=config.in_features,
            out_features=config.out_features,
            seed=config.seed,
        )
        Y_ref = fp32_linear(X, W)
        x_stats = tensor_summary(X)
        w_stats = tensor_summary(W)
        y_stats = tensor_summary(Y_ref)
        fp16_qsnr_x = fp16_qsnr_db(X)
        fp16_qsnr_w = fp16_qsnr_db(W)
        fp16_qsnr_y = fp16_qsnr_db(Y_ref)

        for fmt_name, pipe in pipelines_by_fmt.items():
            pipe.fit(X, W)   # no-op for Identity, but future-proof
            Y_q = pipe.simulate_linear(X, W)
            rows.append({
                "distribution":  lin.name,
                "format":        fmt_name,
                "qsnr_y_db":     qsnr_db(Y_ref, Y_q),
                "qsnr_w_db":     qsnr_db(W, pipe.quantize_tensor(W, role="weight")),
                "qsnr_x_db":     qsnr_db(X, pipe.quantize_tensor(X, role="activation")),
                "fp16_qsnr_w_db": fp16_qsnr_w,
                "fp16_qsnr_x_db": fp16_qsnr_x,
                "fp16_qsnr_y_db": fp16_qsnr_y,
                "crest_X":       x_stats["crest"],
                "crest_W":       w_stats["crest"],
                "crest_Y":       y_stats["crest"],
                "tags":          ",".join(lin.tags),
            })
    return pd.DataFrame(rows)


# ── Experiment 1.3 – smooth-friendly distributions with all 3 transforms ────

def exp13_smooth_transforms(config: FourBitConfig) -> pd.DataFrame:
    """For each smooth-friendly (X, W) pair and each (format, transform).

    Reports
    -------
    * linear-output QSNR (full matmul)
    * per-tensor W and X QSNR (standalone reconstruction)
    * crest factors of X and W (raw, to see "which side has the outliers?")
    """
    formats = build_formats(config)
    transform_names = [t.display_name for t in config.transforms]

    # HAD needs power-of-2 last dim; assert up front so the user gets a clear
    # error instead of a silent skip.
    if "had" in transform_names and not _is_pow2(config.in_features):
        raise ValueError(
            f"HAD transform requires power-of-2 in_features; got {config.in_features}"
        )

    rows: list[dict] = []
    for lin in SMOOTH_FRIENDLY:
        X, W, meta = lin.generate(
            batch=config.batch_size,
            in_features=config.in_features,
            out_features=config.out_features,
            seed=config.seed,
        )
        Y_ref = fp32_linear(X, W)
        x_stats = tensor_summary(X)
        w_stats = tensor_summary(W)
        fp16_qsnr_x = fp16_qsnr_db(X)
        fp16_qsnr_w = fp16_qsnr_db(W)
        fp16_qsnr_y = fp16_qsnr_db(Y_ref)

        for fmt_spec in config.formats:
            fmt = formats[fmt_spec.display_name]
            for t_name in transform_names:
                t = make_fresh_transform(config, t_name)
                pipe = Pipeline(transform=t, fmt=fmt)
                pipe.fit(X, W)

                Y_q = pipe.simulate_linear(X, W)
                W_q = pipe.quantize_tensor(W, role="weight")
                X_q = pipe.quantize_tensor(X, role="activation")

                rows.append({
                    "distribution": lin.name,
                    "format":       fmt_spec.display_name,
                    "transform":    t_name,
                    "qsnr_y_db":    qsnr_db(Y_ref, Y_q),
                    "qsnr_w_db":    qsnr_db(W, W_q),
                    "qsnr_x_db":    qsnr_db(X, X_q),
                    "fp16_qsnr_w_db": fp16_qsnr_w,
                    "fp16_qsnr_x_db": fp16_qsnr_x,
                    "fp16_qsnr_y_db": fp16_qsnr_y,
                    "crest_X":      x_stats["crest"],
                    "crest_W":      w_stats["crest"],
                    "tags":         ",".join(lin.tags),
                })
    return pd.DataFrame(rows)


# ── Orchestrator ─────────────────────────────────────────────────────────────

def run_all(config: FourBitConfig) -> Dict[str, pd.DataFrame]:
    """Run all three Part-1 experiments, persist CSVs, return DataFrames."""
    out_dir = os.path.join(config.output_dir, "part1")
    os.makedirs(out_dir, exist_ok=True)

    print("[Part 1.1] Direct quantization of common distributions ...")
    df11 = exp11_direct_quant(config)
    df11.to_csv(os.path.join(out_dir, "exp11_direct_quant.csv"), index=False)

    print("[Part 1.2] Linear W&A simulation (base quantization) ...")
    df12 = exp12_linear_wa(config)
    df12.to_csv(os.path.join(out_dir, "exp12_linear_wa.csv"), index=False)

    print("[Part 1.3] Smooth-friendly pairs with base / smooth / had ...")
    df13 = exp13_smooth_transforms(config)
    df13.to_csv(os.path.join(out_dir, "exp13_smooth_transforms.csv"), index=False)

    return {"exp11": df11, "exp12": df12, "exp13": df13}
