"""Fast Walsh-Hadamard Transform (FWHT) for outlier suppression.

HAD rotates the tensor into a basis where energy is spread uniformly,
converting sparse high-magnitude outliers into a dense, near-Gaussian
distribution that is much more amenable to low-bit quantization.

Key properties:
  - Deterministic, hardware-fixable (butterfly network, no random state).
  - O(N log N) complexity.
  - Self-inverse up to a factor of N: H² = N·I.
  - No multiplications inside the transform (only additions/subtractions).

Hardware implementation notes:
  - Cascaded butterfly stages, purely combinational logic — no multipliers.
  - Bit-growth: each stage adds 1 bit of word-width to prevent overflow.
    For N=256 (8 stages), INT4 input → INT12 intermediate → INT16 final.
    All sizes fit comfortably in INT16 or INT32.
  - Normalization (÷√N) is FOLDED INTO the downstream quantizer scale.
    The quantizer sets scale = absmax_had / q_max, which automatically
    absorbs the √N amplification. The inverse transform divides by N (exact
    power-of-2 right-shift in hardware), not by √N (irrational float div).
  - This keeps the entire HAD pipeline integer-only.

Invertibility constraint:
  - Only power-of-2 lengths support exact round-trip reconstruction.
    Non-power-of-2 inputs are REJECTED (AssertionError).
    Callers must pad to power of 2 before transforming.
"""

import numpy as np
import warnings


def _next_power_of_two(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def hadamard_transform(x: np.ndarray, normalize: bool = False) -> np.ndarray:
    """Apply the Fast Walsh-Hadamard Transform along the last axis.

    Hardware-faithful integer model:
      - Uses float32 arithmetic that exactly mirrors int32 butterfly ops.
      - No normalization applied inside the transform (normalize=False default).
        Division by √N introduces irrational numbers — hardware-unfriendly.
      - With normalize=False the inverse divides by N (a power-of-2 right-shift).

    Parameters
    ----------
    x : np.ndarray
        Input array. Last dimension must be a power of 2.
    normalize : bool
        If True, divides the result by √N (float, breaks integer semantics).
        Default False — caller should fold the scale into the quantizer.

    Returns
    -------
    np.ndarray, float32, same shape as x.
    """
    n = x.shape[-1]
    assert n == _next_power_of_two(n), (
        f"HAD requires power-of-2 dimension, got {n}. "
        "Pad to the next power of 2 before calling."
    )

    # Simulate integer butterfly in float32 (sufficient mantissa bits for
    # INT4 inputs: max value after log2(N) stages ≤ N * absmax ≤ 4096 * absmax,
    # well within float32 range and integer precision).
    work = x.astype(np.float32)

    h = 1
    while h < n:
        # Reshape into (…, n//(2h), 2, h): pairs of h-element groups
        shape = work.shape
        work = work.reshape(shape[:-1] + (n // (2 * h), 2, h))
        # Explicit .copy() avoids 0-d view aliasing bug in numpy
        a = work[..., 0, :].copy()   # (…, n//(2h), h)
        b = work[..., 1, :].copy()
        work[..., 0, :] = a + b      # butterfly addition
        work[..., 1, :] = a - b      # butterfly subtraction
        work = work.reshape(shape)
        h *= 2

    if normalize:
        # Optional float normalization (breaks integer semantics, avoid in HW)
        work = work / np.sqrt(float(n))

    return work  # float32, integer-valued when normalize=False


def inverse_hadamard_transform(x: np.ndarray, normalize: bool = False) -> np.ndarray:
    """Inverse FWHT.

    With normalize=False (default, hardware model):
        H² = N·I  →  H⁻¹ = H / N
        Apply forward HAD, then divide by N (right-shift by log₂N in hardware).

    With normalize=True (orthonormal mode):
        (H/√N)² = I  →  forward == inverse.
    """
    if normalize:
        return hadamard_transform(x, normalize=True)
    else:
        n = x.shape[-1]
        result = hadamard_transform(x, normalize=False)
        # Divide by N: exact in hardware as arithmetic right-shift log₂(N) bits
        return result / float(n)


class HADTransform:
    """Wrapper that applies HAD before quantization and its inverse after.

    Usage (hardware-faithful, normalize=False):
        had = HADTransform()
        x_had = had.forward(x)          # integer-valued, √N-amplified
        scale = absmax(x_had) / q_max   # automatically absorbs √N factor
        q_had = quantize(x_had, scale)
        x_rec = had.inverse(q_had)      # H(q)/N, exact power-of-2 division
    """

    def __init__(self, normalize: bool = False):
        self.normalize = normalize

    def forward(self, x: np.ndarray) -> np.ndarray:
        return hadamard_transform(x, normalize=self.normalize)

    def inverse(self, y: np.ndarray) -> np.ndarray:
        return inverse_hadamard_transform(y, normalize=self.normalize)

    def hardware_ops(self, n: int) -> dict:
        """Analytical hardware operation count for FWHT of length n.

        Topology: log₂(n) stages, each with n/2 butterfly units.
        Each butterfly unit: 1 addition + 1 subtraction = 2 ops.

        Total additions    = (n/2) × log₂(n)
        Total subtractions = (n/2) × log₂(n)
        Total ops          = n × log₂(n)
        Multiplications    = 0  (normalization folded into quantizer scale)

        Note: Previous (buggy) implementation had `n × stages` for adds AND
        subs, doubling the count. Correct value is `n/2 × stages` each.
        """
        assert n == _next_power_of_two(n), f"n must be power of 2, got {n}"
        stages = int(np.log2(n))
        butterflies_per_stage = n // 2
        adds = butterflies_per_stage * stages
        subs = butterflies_per_stage * stages
        return {
            "n": n,
            "stages": stages,
            "butterflies_per_stage": butterflies_per_stage,
            "additions": adds,
            "subtractions": subs,
            "multiplications": 0,      # normalization folded into quantizer
            "total_ops": adds + subs,  # = n * log2(n)
        }
