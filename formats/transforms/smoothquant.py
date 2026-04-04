"""SmoothQuant: Channel-wise algebraic scaling for quantization-friendly activations.

Key idea (Chen et al., 2022): for a linear layer Y = XW,
introduce a per-channel scale vector s such that:
  Y = (X · diag(s)) · (diag(s)^{-1} · W)
    = X_smoothed · W_smoothed

where the scale shifts the quantization difficulty from activations (X) to
weights (W), balancing outlier magnitudes between the two.

Scale computation:
  s_j = max(|X_j|)^α / max(|W_j|)^(1-α)

where α ∈ [0, 1] controls the migration strength (α=0.5 default).

Hardware fixability:
  - s is pre-computed from calibration data → stored as a fixed ROM vector.
  - At inference: X_smoothed = X * s (element-wise, per-channel scale multiply).
  - W_smoothed is absorbed into weights offline — zero runtime overhead on W.
  - Per-channel scale multiply on activations: N scalar multiplications → trivial.
"""

import numpy as np


class SmoothQuantTransform:
    """SmoothQuant channel-wise scale computation and application.

    This transform is applied to activation tensors X.
    The weight transform (W / s) is computed offline and folded into weights.

    Parameters
    ----------
    alpha : float
        Migration strength from activations to weights (0.5 default).
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.scales = None

    def fit(self, X: np.ndarray, W: np.ndarray) -> "SmoothQuantTransform":
        """Compute per-channel scales from calibration data.

        Parameters
        ----------
        X : np.ndarray, shape (..., C)
            Activation tensor. Scales computed over last dimension (channels).
        W : np.ndarray, shape (C, ...)
            Weight matrix. Per input-channel max computed over first dimension.
        """
        X = X.astype(np.float32)
        W = W.astype(np.float32)

        # Per input-channel max of activations
        x_max = np.max(np.abs(X.reshape(-1, X.shape[-1])), axis=0)  # (C,)

        # Per input-channel max of weights
        w_max = np.max(np.abs(W), axis=tuple(range(1, W.ndim)))  # (C,)

        # Avoid division by zero
        x_max = np.maximum(x_max, 1e-8)
        w_max = np.maximum(w_max, 1e-8)

        self.scales = (x_max ** self.alpha) / (w_max ** (1 - self.alpha))
        self.scales = self.scales.astype(np.float32)
        return self

    def fit_from_stats(self, x_max: np.ndarray, w_max: np.ndarray) -> "SmoothQuantTransform":
        """Compute scales from pre-computed per-channel maxima."""
        x_max = np.maximum(np.asarray(x_max, dtype=np.float32), 1e-8)
        w_max = np.maximum(np.asarray(w_max, dtype=np.float32), 1e-8)
        self.scales = (x_max ** self.alpha) / (w_max ** (1 - self.alpha))
        return self

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Scale activations: X_smooth = X * s (per-channel, last dim)."""
        if self.scales is None:
            raise RuntimeError("Call fit() or fit_from_stats() before forward().")
        return X.astype(np.float32) * self.scales

    def inverse(self, X_smooth: np.ndarray) -> np.ndarray:
        """Undo activation scaling: X = X_smooth / s."""
        if self.scales is None:
            raise RuntimeError("Call fit() first.")
        return X_smooth.astype(np.float32) / self.scales

    def transform_weights(self, W: np.ndarray) -> np.ndarray:
        """Offline weight transform: W_smooth = W / s (absorbed into model)."""
        if self.scales is None:
            raise RuntimeError("Call fit() first.")
        return W.astype(np.float32) / self.scales[:, None] if W.ndim == 2 else W / self.scales

    def hardware_ops(self, n: int, n_channels: int) -> dict:
        """Per-token cost: n_channels scale multiplications (one per channel)."""
        return {
            "n_elements": n,
            "n_channels": n_channels,
            "multiplications": n_channels,  # one mul per channel per token
            "additions": 0,
            "total_ops": n_channels,
            "rom_bits": n_channels * 32,    # FP32 scale per channel stored in ROM
        }


class SmoothQuantINTQuantizer:
    """Combines SmoothQuant transform with standard INT4/INT8 quantization.

    This represents the full SmoothQuant + INT pipeline.
    """

    def __init__(
        self,
        bits: int = 4,
        alpha: float = 0.5,
        per_channel: bool = True,
    ):
        self.bits = bits
        self.alpha = alpha
        self.per_channel = per_channel
        self.name = f"SmoothQuant+INT{bits}"
        self._sq = SmoothQuantTransform(alpha=alpha)
        self._q_max = 2 ** (bits - 1) - 1

    def _int_quantize(self, x: np.ndarray) -> np.ndarray:
        # Scale = absmax / q_max — this is NOT a power-of-two scale.
        # Hardware note: q_max = 7 (INT4) or 127 (INT8), neither is a power of 2
        # that allows a simple right-shift. In hardware this requires a FP32
        # reciprocal multiply, making SmoothQuant's quantizer hardware-UNFRIENDLY
        # compared to POT-scale INT variants. The smoothing scales (s_j) are
        # pre-computed and stored as FP32 ROM values — each token requires one
        # FP32 multiply per channel before INT quantization.
        if self.per_channel:
            # Per-channel (last dim) scale
            absmax = np.max(np.abs(x), axis=-1, keepdims=True)
        else:
            absmax = np.max(np.abs(x))
        absmax = np.maximum(absmax, 1e-8)
        scale = absmax / self._q_max
        q = np.round(x / scale).astype(np.int32)
        q = np.clip(q, -self._q_max, self._q_max)
        return q.astype(np.float32) * scale

    def quantize_with_smooth(
        self, X: np.ndarray, x_max: np.ndarray, w_max: np.ndarray
    ) -> np.ndarray:
        """Apply SmoothQuant scaling then INT quantize."""
        self._sq.fit_from_stats(x_max, w_max)
        X_smooth = self._sq.forward(X)
        q = self._int_quantize(X_smooth)
        return self._sq.inverse(q)

    def quantize(self, x: np.ndarray, bits: int = None) -> np.ndarray:
        """Fallback: pure INT quantization without smoothing (for distribution tests)."""
        return self._int_quantize(x.astype(np.float32))

    def encoding_overhead(self) -> dict:
        return {
            "data_bits_per_element": self.bits,
            "metadata_bits_per_element": 0,
            "bandwidth_amplification": 1.0,
        }
