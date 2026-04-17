"""Transforms that can be composed in front of any quantization format.

Three transforms are provided:

  * IdentityTransform  – "base" mode, no pre-processing.
  * SmoothQuant        – per input-channel scale migration from activations
                         to weights (Chen et al., 2022).
  * Hadamard           – Walsh–Hadamard rotation along the last axis
                         (hardware-friendly, power-of-2 length required).

All transforms implement the same small interface so they are interchangeable
inside :class:`fourbit.pipeline.Pipeline`:

    class Transform(Protocol):
        name: str

        def fit(self, X: np.ndarray | None, W: np.ndarray | None) -> None:
            '''Populate any calibration state from paired (activation, weight)
            tensors.  Transforms that need no calibration (Identity, HAD) may
            implement this as a no-op.'''

        def forward_weight(self, W: np.ndarray) -> np.ndarray: ...
        def inverse_weight(self, W_q: np.ndarray) -> np.ndarray: ...
        def forward_activation(self, X: np.ndarray) -> np.ndarray: ...
        def inverse_activation(self, X_q: np.ndarray) -> np.ndarray: ...

        def output_correction(self) -> float:
            '''Scalar multiplier applied to the quantized linear output so
            that   Y_q @ correction ≈ X @ W.T    holds after the matmul.'''
"""
from __future__ import annotations

import numpy as np

from formats.transforms.hadamard import hadamard_transform


# ── Base class ───────────────────────────────────────────────────────────────

class Transform:
    """Default no-op implementation. Subclasses override the relevant methods."""

    name: str = "identity"
    requires_pow2_last_dim: bool = False

    def fit(self, X: np.ndarray | None, W: np.ndarray | None) -> None:
        pass

    # Per-tensor direction (used for standalone tensor QSNR)
    def forward_weight(self, W: np.ndarray) -> np.ndarray:
        return W

    def inverse_weight(self, W_q: np.ndarray) -> np.ndarray:
        return W_q

    def forward_activation(self, X: np.ndarray) -> np.ndarray:
        return X

    def inverse_activation(self, X_q: np.ndarray) -> np.ndarray:
        return X_q

    def output_correction(self) -> float:
        return 1.0

    # Convenience so callers can treat W and X uniformly.
    def apply(self, x: np.ndarray, role: str) -> np.ndarray:
        return self.forward_weight(x) if role == "weight" else self.forward_activation(x)

    def invert(self, x_q: np.ndarray, role: str) -> np.ndarray:
        return self.inverse_weight(x_q) if role == "weight" else self.inverse_activation(x_q)


# ── 1. Identity ──────────────────────────────────────────────────────────────

class IdentityTransform(Transform):
    name = "base"


# ── 2. SmoothQuant ───────────────────────────────────────────────────────────

class SmoothQuantTransform(Transform):
    """Per input-channel scale that migrates outliers from X onto W.

    Following Xiao et al. 2022 (§3.2, equation 4):

        Y = (X · diag(s)^{-1}) · (diag(s) · W_col)
          = X̂ · Ŵ

        s_k = max(|X_k|)^alpha · max(|W_k|)^-(1-alpha)

    where W_col is the input-channel-indexed column of W.  The transform
    therefore divides activations by ``s`` (shrinking outlier channels) and
    multiplies the matching weight columns by ``s``.

    Standalone tensor reconstruction undoes the scaling with the opposite
    operation — the per-tensor SNR reflects what the decoder sees after the
    combined quantizer + rescale stage in inference.
    """

    def __init__(self, alpha: float = 0.5, eps: float = 1e-8):
        self.name = "smooth"
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.scales: np.ndarray | None = None   # shape: (in_features,)

    def fit(self, X: np.ndarray | None, W: np.ndarray | None) -> None:
        if X is None or W is None:
            raise ValueError("SmoothQuantTransform.fit() requires both X and W")
        X = np.asarray(X, dtype=np.float32)
        W = np.asarray(W, dtype=np.float32)

        # Per input-channel maxima.  X has shape (..., in); PyTorch W has
        # shape (out, in) so the per-input-channel max is along axis 0.
        x_max = np.max(np.abs(X.reshape(-1, X.shape[-1])), axis=0)      # (in,)
        if W.ndim == 2:
            w_max = np.max(np.abs(W), axis=0)                           # (in,)
        else:
            w_max = np.maximum(np.abs(W), 0.0)

        x_max = np.maximum(x_max, self.eps)
        w_max = np.maximum(w_max, self.eps)
        self.scales = (x_max ** self.alpha) / (w_max ** (1.0 - self.alpha))
        self.scales = self.scales.astype(np.float32)

    # X' = X / s   (paper eq. 4 — divide to shrink outlier channels)
    def forward_activation(self, X: np.ndarray) -> np.ndarray:
        self._require_scales()
        return (np.asarray(X, dtype=np.float32) / self.scales).astype(np.float32)

    def inverse_activation(self, X_q: np.ndarray) -> np.ndarray:
        self._require_scales()
        return (np.asarray(X_q, dtype=np.float32) * self.scales).astype(np.float32)

    # W' = W · s   (last axis of (out, in) = input channels)
    def forward_weight(self, W: np.ndarray) -> np.ndarray:
        self._require_scales()
        W = np.asarray(W, dtype=np.float32)
        if W.ndim == 2:
            return (W * self.scales[np.newaxis, :]).astype(np.float32)
        return (W * self.scales).astype(np.float32)

    def inverse_weight(self, W_q: np.ndarray) -> np.ndarray:
        self._require_scales()
        W_q = np.asarray(W_q, dtype=np.float32)
        if W_q.ndim == 2:
            return (W_q / self.scales[np.newaxis, :]).astype(np.float32)
        return (W_q / self.scales).astype(np.float32)

    def _require_scales(self):
        if self.scales is None:
            raise RuntimeError("SmoothQuantTransform.fit() must be called first.")


# ── 3. Hadamard ──────────────────────────────────────────────────────────────

def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


class HadamardTransform(Transform):
    """Un-normalised FWHT along the last axis.

    Forward: ``x_h = H x``  (integer-valued multiplies avoided, no √N).
    Inverse: ``x = H x_h / N``  (power-of-two right-shift in hardware).
    For a linear op we want

        Y = X W^T
        X_h = H(X),  W_h = H(W)
        Y ≈ X_h Q(W_h)^T / N         — matmul with dequantised W_h
        Y ≈ Q(X_h) Q(W_h)^T / N      — both quantised, divide by N once

    so ``output_correction = 1/N`` where ``N`` is the contracted dim.
    """

    requires_pow2_last_dim = True

    def __init__(self):
        self.name = "had"
        self._N: int | None = None

    def fit(self, X: np.ndarray | None, W: np.ndarray | None) -> None:
        # Need to know the length of the contracted axis so inverse() and
        # output_correction() can scale by 1/N.  If both provided, their last
        # dim must match.
        if W is not None:
            self._N = int(W.shape[-1])
        elif X is not None:
            self._N = int(X.shape[-1])
        else:
            self._N = None
        if self._N is not None and not _is_pow2(self._N):
            raise ValueError(
                f"HadamardTransform requires power-of-2 last dim; got {self._N}"
            )

    def _ensure_fit(self, x: np.ndarray) -> None:
        if self._N is None:
            self._N = int(x.shape[-1])
            if not _is_pow2(self._N):
                raise ValueError(
                    f"HadamardTransform requires power-of-2 last dim; got {self._N}"
                )

    def forward_activation(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        self._ensure_fit(X)
        return hadamard_transform(X, normalize=False)

    def forward_weight(self, W: np.ndarray) -> np.ndarray:
        W = np.asarray(W, dtype=np.float32)
        self._ensure_fit(W)
        return hadamard_transform(W, normalize=False)

    def inverse_activation(self, X_q: np.ndarray) -> np.ndarray:
        X_q = np.asarray(X_q, dtype=np.float32)
        self._ensure_fit(X_q)
        return hadamard_transform(X_q, normalize=False) / float(self._N)

    def inverse_weight(self, W_q: np.ndarray) -> np.ndarray:
        W_q = np.asarray(W_q, dtype=np.float32)
        self._ensure_fit(W_q)
        return hadamard_transform(W_q, normalize=False) / float(self._N)

    def output_correction(self) -> float:
        if self._N is None:
            return 1.0
        return 1.0 / float(self._N)


# ── Factory registry ─────────────────────────────────────────────────────────

TRANSFORM_FACTORIES = {
    "identity":   IdentityTransform,
    "smoothquant": SmoothQuantTransform,
    "hadamard":   HadamardTransform,
}


def make_transform(factory_name: str, **kwargs) -> Transform:
    if factory_name not in TRANSFORM_FACTORIES:
        raise KeyError(
            f"Unknown transform factory '{factory_name}'. "
            f"Known: {sorted(TRANSFORM_FACTORIES)}"
        )
    return TRANSFORM_FACTORIES[factory_name](**kwargs)
