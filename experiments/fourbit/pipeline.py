"""Pipeline: one (transform, format) composition.

A :class:`Pipeline` wraps a :class:`fourbit.transforms.Transform` and a
quantization format.  It provides three operations:

  * ``quantize_tensor(x, role)`` – standalone-tensor quantization with the
    transform applied and inverted.  ``role`` selects which side of the
    transform is used (``"weight"`` or ``"activation"``); this matters for
    SmoothQuant where the forward direction differs.

  * ``simulate_linear(X, W, bias=None)`` – full quantized matmul simulation
    for a linear layer ``Y = X W^T (+ bias)``.  Returns the quantized output
    in the original (un-transformed) domain.

  * ``fit(X, W)`` – delegated to the transform; called once per
    Pipeline use when paired calibration data exists.

W4A4 output semantics
---------------------
``simulate_linear`` implements W4A4 *at the matmul level*: X and W are both
quantised to 4 bits, they are multiplied, and the accumulator output is
returned at full precision (FP32).  This mirrors real W4A4 tensor cores –
the accumulator (INT32 or FP32) is *never* a 4-bit value, because 4-bit
products routinely overflow into 11+ bits after the in-features reduction.
A typical stack then applies a LayerNorm / activation in higher precision
before the **next** layer's input is quantised afresh at that layer's
entry.  The per-layer Y QSNR we report therefore measures the error that
W4A4 introduces **at the current layer only**; end-to-end accuracy effects
come from running the whole model under the same quantiser, which is what
``fourbit.accuracy.quantized_inference`` does.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import numpy as np

from experiments.fourbit.transforms import Transform


def _apply_output_fmt(out_fmt: object, y: np.ndarray) -> np.ndarray:
    """Quantise a GEMM output tensor ``y`` with ``out_fmt``.

    ``y`` has shape ``(batch, out_features)``.  For the generic format
    case we just call ``out_fmt.quantize(y)``.  For SQ-Format variants
    the class banks along the *K* axis (rows) of its 2-D input, which
    for Y maps naturally onto the *out_features* axis (per design
    decision R2 — "one bank per token").  We therefore transpose
    ``y`` to ``(out_features, batch)`` around the call and transpose
    back afterwards so the SQ banking aligns with output channels
    rather than tokens.

    R2 auto-adapt: if ``out_features < out_fmt.bank_size``, we use a
    shallow-copied SQ-Format whose ``bank_size`` is shrunk to
    ``out_features``.  Without this, a format built with the default
    ``bank_size=128`` applied to a small head (e.g. MNIST classifier
    out_features=10) would silently pad the whole row into a single
    bank and effectively disable the sparsity split.

    Imports are done lazily to avoid circular-import with
    ``formats/sq_format.py``.
    """
    from formats.sq_format import SQFormat, SQFormatActivations, SQFormatFP

    if isinstance(out_fmt, (SQFormat, SQFormatActivations, SQFormatFP)):
        y_t = np.ascontiguousarray(y.T)
        out_features = y_t.shape[0]
        if out_features < out_fmt.bank_size:
            fmt = copy.copy(out_fmt)
            fmt.bank_size = out_features
        else:
            fmt = out_fmt
        y_tq = fmt.quantize(y_t)
        return np.ascontiguousarray(y_tq.T)
    return out_fmt.quantize(y)


@dataclass
class Pipeline:
    transform: Transform
    fmt: object   # any object with .quantize(x) -> np.ndarray and .name
    output_fmt: object | None = None  # optional Y-quantiser; default FP32 accumulator
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"{self.transform.name}/{self.fmt.name}"

    # ── Calibration passthrough ──────────────────────────────────────────────

    def fit(self, X: np.ndarray | None = None, W: np.ndarray | None = None) -> None:
        self.transform.fit(X, W)

    # ── Per-tensor SNR computation ───────────────────────────────────────────

    def quantize_tensor(self, x: np.ndarray, role: str = "weight") -> np.ndarray:
        """Return the reconstructed tensor after transform → quantize → inverse.

        ``role`` must be ``"weight"`` or ``"activation"``.  For HAD and
        Identity the choice is irrelevant; for SmoothQuant it selects the
        correct per-channel scale direction.
        """
        if role not in ("weight", "activation"):
            raise ValueError(f"role must be 'weight' or 'activation', got {role!r}")
        x = np.asarray(x, dtype=np.float32)
        x_t  = self.transform.apply(x, role)
        x_tq = self.fmt.quantize(x_t)
        return self.transform.invert(x_tq, role)

    # ── Full linear-layer simulation ─────────────────────────────────────────

    def simulate_linear(
        self,
        X: np.ndarray,
        W: np.ndarray,
        bias: np.ndarray | None = None,
    ) -> np.ndarray:
        """Simulate Y = X W^T with both X and W quantized.

        Handles all three transforms uniformly:

            * base   : Y_q = Q(X) @ Q(W)^T
            * smooth : Y_q = Q(X·s) @ Q(W/s)^T       (correction = 1)
            * had    : Y_q = Q(H X) @ Q(H W)^T / N   (correction = 1/N)

        Returns an array of shape ``X.shape[:-1] + (W.shape[0],)``.
        """
        X = np.asarray(X, dtype=np.float32)
        W = np.asarray(W, dtype=np.float32)

        X_t = self.transform.forward_activation(X)
        W_t = self.transform.forward_weight(W)

        X_q = self.fmt.quantize(X_t)
        W_q = self.fmt.quantize(W_t)

        y = X_q @ W_q.T
        y = y * self.transform.output_correction()

        if self.output_fmt is not None:
            y = _apply_output_fmt(self.output_fmt, y.astype(np.float32))

        if bias is not None:
            y = y + bias
        return y.astype(np.float32)


def fp32_linear(
    X: np.ndarray, W: np.ndarray, bias: np.ndarray | None = None
) -> np.ndarray:
    """Reference FP32 linear op ``Y = X W^T (+ bias)``."""
    y = np.asarray(X, dtype=np.float32) @ np.asarray(W, dtype=np.float32).T
    if bias is not None:
        y = y + np.asarray(bias, dtype=np.float32)
    return y.astype(np.float32)
