"""Collection-first profiler: record tensors once, analyse many times.

The original :class:`profiler.ModelProfiler` runs one forward pass per format
and silently swallows per-tensor exceptions via ``except Exception: pass``.
That design has three defects that show up in the 4-bit study:

  1. *Silent failure masks bugs* – format implementations that raise (e.g.
     HAD on non-power-of-2 inputs) produce NaN rows in the CSV, but no
     diagnostic is emitted.  ``LayerCollector`` removes every ``pass``-style
     suppression; failures are routed through logged warnings instead.

  2. *No SmoothQuant calibration* – the original profiler treats each format
     as a drop-in quantizer.  SmoothQuant needs paired (X, W) calibration,
     which was impossible in the per-format loop.  ``LayerCollector``
     separates collection from analysis; calibration is done per-layer with
     the collected X and W tensors.

  3. *No crest factor, and only post-quant statistics kept* – we need raw
     tensor crest factors to plot them on the x-axis of the scatter figures.

``LayerCollector`` fixes all three by recording the raw activation batches
and weight matrices for each ``nn.Linear`` layer.  Analysis is then an
offline sweep over the (format × transform) grid via ``analyse_layers``.

Outputs (returned as an in-memory structure + written to CSV):

    {
      layer_name: {
          "W":       np.ndarray,                # (out, in)
          "X":       np.ndarray,                # (batch*tok, in), concat of batches
          "Y":       np.ndarray,                # (batch*tok, out), FP32 output
          "bias":    np.ndarray | None,
          "stats":   {"W": {...}, "X": {...}, "Y": {...}},  # mean/std/crest/kurtosis
          "metrics": [                          # one row per (format, transform)
              {
                  "format": ..., "transform": ...,
                  "qsnr_w_db": ..., "qsnr_x_db": ..., "qsnr_y_db": ...,
              }, ...
          ],
      }, ...
    }
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from fourbit.config import FourBitConfig
from fourbit.registry import build_formats, make_fresh_transform
from fourbit.pipeline import Pipeline, fp32_linear
from fourbit.metrics import qsnr_db, tensor_summary, fp16_qsnr_db

logger = logging.getLogger("fourbit.profiler")


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class LayerRecord:
    name: str
    module: nn.Linear
    W: Optional[np.ndarray] = None
    bias: Optional[np.ndarray] = None
    # X batches collected during forward
    X_batches: List[np.ndarray] = field(default_factory=list)
    # Y batches collected from the forward pass (FP32 reference)
    Y_batches: List[np.ndarray] = field(default_factory=list)

    def finalize(self) -> Dict[str, np.ndarray]:
        """Concatenate X/Y batches along the flattened-token axis."""
        def _flatten(b: np.ndarray) -> np.ndarray:
            # (batch, [tok], features) -> (batch*tok, features)
            return b.reshape(-1, b.shape[-1])

        X = np.concatenate([_flatten(b) for b in self.X_batches], axis=0) \
            if self.X_batches else np.zeros((0,), dtype=np.float32)
        Y = np.concatenate([_flatten(b) for b in self.Y_batches], axis=0) \
            if self.Y_batches else np.zeros((0,), dtype=np.float32)
        return {"W": self.W, "X": X, "Y": Y, "bias": self.bias}


# ── Collector ────────────────────────────────────────────────────────────────

class LayerCollector:
    """Collect (W, X, Y) tensors for every ``nn.Linear`` in a model."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.layers: Dict[str, LayerRecord] = {}
        self._hooks: List = []

    def __enter__(self):
        self._install_hooks()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._remove_hooks()
        return False

    def _install_hooks(self):
        for name, mod in self.model.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            rec = LayerRecord(name=name, module=mod)
            rec.W = mod.weight.detach().cpu().float().numpy().copy()
            if mod.bias is not None:
                rec.bias = mod.bias.detach().cpu().float().numpy().copy()
            self.layers[name] = rec
            self._hooks.append(mod.register_forward_hook(self._make_post_hook(name)))
            self._hooks.append(mod.register_forward_pre_hook(self._make_pre_hook(name)))

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _make_pre_hook(self, lname: str):
        def pre_hook(module, args):
            if not args:
                return
            t = args[0]
            if isinstance(t, torch.Tensor):
                self.layers[lname].X_batches.append(
                    t.detach().cpu().float().numpy().copy()
                )
        return pre_hook

    def _make_post_hook(self, lname: str):
        def post_hook(module, args, output):
            if isinstance(output, torch.Tensor):
                self.layers[lname].Y_batches.append(
                    output.detach().cpu().float().numpy().copy()
                )
        return post_hook


# ── Analysis ─────────────────────────────────────────────────────────────────

def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _safe_qsnr(x: np.ndarray, x_hat: np.ndarray) -> float:
    """QSNR that returns NaN instead of raising on shape mismatch."""
    try:
        return qsnr_db(x, x_hat)
    except Exception as exc:   # pragma: no cover - diagnostics only
        logger.warning("QSNR computation failed: %s", exc)
        return float("nan")


def analyse_layer(
    rec_data: dict,
    layer_name: str,
    config: FourBitConfig,
) -> pd.DataFrame:
    """Return one row per (format, transform) for a single layer.

    ``rec_data`` must be the dict produced by :meth:`LayerRecord.finalize`,
    i.e. ``{"W": ..., "X": ..., "Y": ..., "bias": ...}`` with X and Y already
    flattened to (N, features).
    """
    W = rec_data["W"]
    X = rec_data["X"]
    Y_ref = rec_data["Y"]
    bias = rec_data["bias"]

    # Tensor statistics – same irrespective of format/transform
    stats_W = tensor_summary(W)
    stats_X = tensor_summary(X)
    stats_Y = tensor_summary(Y_ref)

    # FP16 baseline QSNRs (per-tensor, format-independent) — these appear as
    # constant columns across every format row of a given layer, so the
    # reporter can render them alongside the 4-bit QSNRs without a join.
    fp16_qsnr_w = fp16_qsnr_db(W)
    fp16_qsnr_x = fp16_qsnr_db(X)
    fp16_qsnr_y = fp16_qsnr_db(Y_ref)

    formats = build_formats(config)
    rows: list[dict] = []

    # HAD can only apply when in_features is a power of two.
    had_applicable = _is_pow2(W.shape[-1]) and _is_pow2(X.shape[-1])

    for fmt_spec in config.formats:
        fmt = formats[fmt_spec.display_name]
        for t_spec in config.transforms:
            t_name = t_spec.display_name

            if t_name == "had" and not had_applicable:
                # Record an explicit NaN row so the reporter can see the gap
                # rather than silently dropping the combination.
                rows.append({
                    "layer":        layer_name,
                    "format":       fmt_spec.display_name,
                    "transform":    t_name,
                    "qsnr_w_db":    float("nan"),
                    "qsnr_x_db":    float("nan"),
                    "qsnr_y_db":    float("nan"),
                    "fp16_qsnr_w_db": fp16_qsnr_w,
                    "fp16_qsnr_x_db": fp16_qsnr_x,
                    "fp16_qsnr_y_db": fp16_qsnr_y,
                    "reason":       "HAD requires power-of-2 in_features",
                    **_prefix(stats_W, "W_"),
                    **_prefix(stats_X, "X_"),
                    **_prefix(stats_Y, "Y_"),
                })
                continue

            t = make_fresh_transform(config, t_name)
            pipe = Pipeline(transform=t, fmt=fmt)
            try:
                pipe.fit(X, W)
                Y_q = pipe.simulate_linear(X, W, bias=bias)
                W_q = pipe.quantize_tensor(W, role="weight")
                X_q = pipe.quantize_tensor(X, role="activation")
            except Exception as exc:
                logger.warning(
                    "Layer %s / %s / %s failed: %s",
                    layer_name, fmt_spec.display_name, t_name, exc,
                )
                rows.append({
                    "layer":        layer_name,
                    "format":       fmt_spec.display_name,
                    "transform":    t_name,
                    "qsnr_w_db":    float("nan"),
                    "qsnr_x_db":    float("nan"),
                    "qsnr_y_db":    float("nan"),
                    "fp16_qsnr_w_db": fp16_qsnr_w,
                    "fp16_qsnr_x_db": fp16_qsnr_x,
                    "fp16_qsnr_y_db": fp16_qsnr_y,
                    "reason":       str(exc),
                    **_prefix(stats_W, "W_"),
                    **_prefix(stats_X, "X_"),
                    **_prefix(stats_Y, "Y_"),
                })
                continue

            # For the Y QSNR we want to compare the bias-adjusted reference
            # with the simulation output.  simulate_linear already adds the
            # bias, and Y_ref was recorded post-bias too.
            rows.append({
                "layer":        layer_name,
                "format":       fmt_spec.display_name,
                "transform":    t_name,
                "qsnr_w_db":    _safe_qsnr(W, W_q),
                "qsnr_x_db":    _safe_qsnr(X, X_q),
                "qsnr_y_db":    _safe_qsnr(Y_ref, Y_q),
                "fp16_qsnr_w_db": fp16_qsnr_w,
                "fp16_qsnr_x_db": fp16_qsnr_x,
                "fp16_qsnr_y_db": fp16_qsnr_y,
                "reason":       "",
                **_prefix(stats_W, "W_"),
                **_prefix(stats_X, "X_"),
                **_prefix(stats_Y, "Y_"),
            })

    return pd.DataFrame(rows)


def _prefix(d: dict, pref: str) -> dict:
    return {f"{pref}{k}": v for k, v in d.items()}


def analyse_all(
    layers: Dict[str, LayerRecord],
    config: FourBitConfig,
) -> pd.DataFrame:
    """Run :func:`analyse_layer` on every layer, concatenate into one DataFrame."""
    frames: list[pd.DataFrame] = []
    for name, rec in layers.items():
        data = rec.finalize()
        if data["X"].size == 0:
            logger.warning("Layer %s has no recorded X batches; skipping.", name)
            continue
        frames.append(analyse_layer(data, name, config))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
