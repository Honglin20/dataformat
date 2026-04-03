"""Fast Walsh-Hadamard Transform (FWHT) for outlier suppression.

HAD rotates the tensor into a basis where energy is spread uniformly,
converting sparse high-magnitude outliers into a dense, near-Gaussian
distribution that is much more amenable to low-bit quantization.

Key properties:
  - Deterministic, hardware-fixable (butterfly network, no random state).
  - O(N log N) complexity.
  - Orthonormal (when divided by sqrt(N)): preserves L2 norm.
  - Self-inverse: H^{-1} = H / N  (or equivalently, apply H twice and divide by N).

Hardware implementation: cascaded butterfly stages, purely combinational
logic with XOR / addition — no multipliers required.
"""

import numpy as np


def _next_power_of_two(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def hadamard_transform(x: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Apply the Fast Walsh-Hadamard Transform along the last axis.

    Parameters
    ----------
    x : np.ndarray
        Input array. The last dimension will be padded to power of 2 if needed.
    normalize : bool
        If True, divides by sqrt(N) to make the transform orthonormal.

    Returns
    -------
    np.ndarray
        Transformed array (same shape after padding trimmed back).
    """
    orig_len = x.shape[-1]
    n = _next_power_of_two(orig_len)

    # Zero-pad along last axis if needed
    if n != orig_len:
        pad_width = [(0, 0)] * (x.ndim - 1) + [(0, n - orig_len)]
        x = np.pad(x, pad_width)

    x = x.astype(np.float32).copy()

    # In-place Cooley-Tukey butterfly
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                a = x[..., j]
                b = x[..., j + h]
                x[..., j] = a + b
                x[..., j + h] = a - b
        h *= 2

    if normalize:
        x = x / np.sqrt(n)

    # Trim back to original length
    return x[..., :orig_len]


def inverse_hadamard_transform(x: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Inverse FWHT. For orthonormal HAD, forward == inverse."""
    return hadamard_transform(x, normalize=normalize)


class HADTransform:
    """Wrapper that applies HAD before quantization and its inverse after.

    Usage:
        had = HADTransform()
        x_had = had.forward(x)
        q_had = some_quantizer.quantize(x_had)
        x_reconstructed = had.inverse(q_had)
    """

    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def forward(self, x: np.ndarray) -> np.ndarray:
        return hadamard_transform(x, normalize=self.normalize)

    def inverse(self, y: np.ndarray) -> np.ndarray:
        return inverse_hadamard_transform(y, normalize=self.normalize)

    def hardware_ops(self, n: int) -> dict:
        """Analytical hardware operation count for FWHT of length n.

        Each butterfly stage: n/2 additions + n/2 subtractions.
        Stages: log2(n).
        Total add/sub = n/2 × log2(n) × 2 = n × log2(n).
        No multiplications required (normalization uses a single scale,
        but this can be absorbed into the subsequent quantization scale).
        """
        n_pad = _next_power_of_two(n)
        stages = int(np.log2(n_pad))
        adds = n_pad * stages          # additions
        subs = n_pad * stages          # subtractions
        muls = 1                        # single normalization factor (optional)
        return {
            "n": n,
            "n_padded": n_pad,
            "stages": stages,
            "additions": adds,
            "subtractions": subs,
            "multiplications": muls,
            "total_ops": adds + subs + muls,
        }
