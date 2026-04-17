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
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from fourbit.transforms import Transform


@dataclass
class Pipeline:
    transform: Transform
    fmt: object   # any object with .quantize(x) -> np.ndarray and .name
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
