"""End-to-end MNIST accuracy under W4A4 quantisation.

``QuantLinear`` wraps every ``nn.Linear`` in a trained model so the forward
pass simulates the exact W4A4 quantiser from :mod:`fourbit.pipeline`:
inputs and weights are both 4-bit quantised, the matmul accumulates in
FP32, and the output is left in FP32 (as on a real W4A4 tensor core –
see docstring of :class:`fourbit.pipeline.Pipeline`).

For SmoothQuant we calibrate the per-channel scale once on the
``LayerCollector`` tensors captured during the Part-2 profiling pass; for
Hadamard the rotation is deterministic and requires no calibration.  Layers
whose ``in_features`` is not a power of two fall back to their FP32 weights
when the current transform is HAD; this mirrors real deployments that
would simply skip HAD for those layers.

The public entry point :func:`accuracy_sweep` returns one row per
(format, transform) combination plus an explicit ``FP32``/``FP16``
baseline, ready for the reporter.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tdata

from fourbit.config import FourBitConfig
from fourbit.registry import build_formats, make_fresh_transform
from fourbit.pipeline import Pipeline
from fourbit.profiler_v2 import LayerRecord

logger = logging.getLogger("fourbit.accuracy")


# ── Quantised Linear --------------------------------------------------------

class QuantLinear(nn.Linear):
    """A drop-in replacement for ``nn.Linear`` that simulates W4A4.

    Subclassing ``nn.Linear`` is intentional: some PyTorch fast paths
    (notably ``TransformerEncoderLayer``) read ``linear1.weight`` /
    ``linear2.weight`` directly off the module, so we keep the ``weight``
    and ``bias`` parameters in-place and simply *replace their values*
    with the dequantised-in-transformed-domain weight.  Activation
    quantisation happens inside :meth:`forward`.

    The dequantised weight is cached as ``weight.data`` so repeated
    forwards do not repeat the per-row NF4 / MX block computation.
    """

    def __init__(self, linear: nn.Linear, pipe: Pipeline):
        super().__init__(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        self.pipe = pipe

        W = linear.weight.detach().cpu().float().numpy().copy()
        W_t = self.pipe.transform.forward_weight(W)
        W_tq = self.pipe.fmt.quantize(W_t)
        with torch.no_grad():
            # weight stored in the *transformed* domain — forward() undoes it
            # by applying the transform to X (not to W) and then multiplying
            # by ``output_correction`` once.
            self.weight.copy_(
                torch.from_numpy(W_tq.astype(np.float32)).to(self.weight.dtype)
            )
            if linear.bias is not None:
                self.bias.copy_(linear.bias.detach().to(self.bias.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_np = x.detach().cpu().float().numpy()
        orig_shape = x_np.shape
        flat = x_np.reshape(-1, orig_shape[-1])
        flat_t = self.pipe.transform.forward_activation(flat)
        flat_tq = self.pipe.fmt.quantize(flat_t)
        x_tq = torch.from_numpy(flat_tq.reshape(orig_shape)).to(x.device, x.dtype)

        y = (x_tq @ self.weight.T) * self.pipe.transform.output_correction()
        if self.bias is not None:
            y = y + self.bias
        return y


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _build_quantised_model(
    model: nn.Module,
    layers: Dict[str, LayerRecord],
    config: FourBitConfig,
    fmt_name: str,
    transform_name: str,
) -> nn.Module:
    """Return a deep-copy of ``model`` with every eligible Linear replaced.

    "Eligible" means:
      * The layer was captured by :class:`LayerCollector` with non-empty
        input batches (so activation quantisation is well-defined).
      * The parent module does not bypass ``Module.forward`` to reach the
        Linear's ``.weight`` directly.  PyTorch's
        :class:`nn.MultiheadAttention` calls
        ``F.multi_head_attention_forward`` which reads
        ``out_proj.weight``/``out_proj.bias`` as parameters, so replacing
        ``out_proj`` with a ``QuantLinear`` breaks the forward.  We skip
        any Linear whose parent module is a ``MultiheadAttention`` and
        leave it at FP32; Table 3 reports end-to-end accuracy with that
        caveat stated explicitly in the report.
    """
    formats = build_formats(config)
    fmt = formats[fmt_name]

    model_q = copy.deepcopy(model)
    model_q.eval()

    for name, mod in list(model_q.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue

        # Skip Linears that live inside MHA — those are accessed as bare
        # parameters by F.multi_head_attention_forward and can't be wrapped.
        parent_name = name.rpartition(".")[0]
        parent = (
            model_q.get_submodule(parent_name) if parent_name else model_q
        )
        if isinstance(parent, nn.MultiheadAttention):
            continue

        rec = layers.get(name)
        if rec is None:
            logger.warning("No LayerRecord for %s; leaving FP32.", name)
            continue

        X = rec.finalize()["X"]
        if X.size == 0:
            # Layer wasn't reached during the profiling pass.
            logger.warning(
                "Layer %s has no recorded X — leaving FP32 in accuracy sweep.",
                name,
            )
            continue

        W = rec.W

        can_transform = True
        if transform_name == "had":
            can_transform = _is_pow2(W.shape[-1])

        if can_transform:
            t = make_fresh_transform(config, transform_name)
            pipe = Pipeline(transform=t, fmt=fmt)
            try:
                pipe.fit(X, W)
            except Exception as exc:
                logger.warning(
                    "Calibration failed for %s (%s/%s): %s — using base.",
                    name, fmt_name, transform_name, exc,
                )
                t = make_fresh_transform(config, "base")
                pipe = Pipeline(transform=t, fmt=fmt)
                pipe.fit(X, W)
        else:
            t = make_fresh_transform(config, "base")
            pipe = Pipeline(transform=t, fmt=fmt)
            pipe.fit(X, W)

        ql = QuantLinear(mod, pipe)
        attr = name.rpartition(".")[2]
        setattr(parent, attr, ql)
    return model_q


def _cast_model_fp16(model: nn.Module) -> nn.Module:
    """Return a deep copy of ``model`` with every parameter/buffer in FP16."""
    m = copy.deepcopy(model).eval()
    m = m.half()
    # Keep long-term tensors as float32 if they contain integer indices, etc.
    return m


# ── Evaluation --------------------------------------------------------------

def _eval_accuracy(model: nn.Module, loader: tdata.DataLoader,
                   dtype: torch.dtype = torch.float32) -> float:
    """Top-1 accuracy of ``model`` on ``loader``.

    PyTorch's ``TransformerEncoderLayer`` has a C++ fast path that
    bypasses ``Module.forward`` on ``linear1`` / ``linear2`` / attention
    projections when certain conditions are met (eval mode, no forward
    hooks, …).  This fast path reads parameter tensors directly off the
    module and would therefore *ignore our QuantLinear's forward*, giving
    bogus full-precision accuracy for everything that lives inside an
    encoder layer.  We disable the fast path explicitly here so every
    Linear goes through ``Module.forward`` — including QuantLinear.
    """
    model.eval()
    total = 0
    correct = 0
    was_fastpath = True
    try:
        was_fastpath = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
    except Exception:  # pragma: no cover – older PyTorch
        pass
    try:
        with torch.no_grad():
            for x, y in loader:
                if dtype == torch.float16:
                    x = x.half()
                logits = model(x)
                pred = logits.argmax(dim=-1)
                correct += int((pred == y).sum().item())
                total += int(y.numel())
    finally:
        try:
            torch.backends.mha.set_fastpath_enabled(was_fastpath)
        except Exception:  # pragma: no cover
            pass
    return correct / max(total, 1)


@dataclass
class AccuracyRow:
    format: str
    transform: str
    accuracy: float


# ── Entry point ------------------------------------------------------------

def accuracy_sweep(
    model: nn.Module,
    layers: Dict[str, LayerRecord],
    loader: tdata.DataLoader,
    config: FourBitConfig,
) -> "list[dict]":
    """Run FP32 / FP16 baselines + every (format, transform) combination.

    Returns a list of dicts ready for ``pd.DataFrame`` conversion.  Layers
    for which HAD is inapplicable (non power-of-two in_features) are kept
    at FP32 inside the quantised model, matching the Part-2 CSV's semantics.
    """
    rows: list[dict] = []

    # FP32 baseline (no quantisation)
    acc_fp32 = _eval_accuracy(model, loader)
    rows.append({"format": "FP32", "transform": "baseline", "accuracy": acc_fp32})

    # FP16 baseline – cast weights to half precision, keep FP32 math inside
    # the transformer's internal ops via autocast.  Simpler: just use .half().
    try:
        model_fp16 = _cast_model_fp16(model)
        acc_fp16 = _eval_accuracy(model_fp16, loader, dtype=torch.float16)
        rows.append({"format": "FP16", "transform": "baseline",
                     "accuracy": acc_fp16})
    except Exception as exc:  # pragma: no cover - depends on PyTorch build
        logger.warning("FP16 eval failed: %s", exc)
        rows.append({"format": "FP16", "transform": "baseline",
                     "accuracy": float("nan")})

    # Every (format, transform) combination
    for fmt_spec in config.formats:
        for t_spec in config.transforms:
            try:
                mq = _build_quantised_model(
                    model, layers, config,
                    fmt_name=fmt_spec.display_name,
                    transform_name=t_spec.display_name,
                )
                acc = _eval_accuracy(mq, loader)
            except Exception as exc:
                logger.warning(
                    "%s / %s failed: %s",
                    fmt_spec.display_name, t_spec.display_name, exc,
                )
                acc = float("nan")
            rows.append({
                "format":    fmt_spec.display_name,
                "transform": t_spec.display_name,
                "accuracy":  acc,
            })
            print(f"[Acc] {fmt_spec.display_name:10s} / "
                  f"{t_spec.display_name:6s} → {acc*100:.2f}%")
    return rows
