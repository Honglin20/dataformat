"""Random Orthogonal Rotation Matrix for outlier suppression.

Used as the "precision upper bound" reference to benchmark how much HAD
sacrifices compared to a fully random (unconstrained) orthogonal rotation.

The key difference from HAD:
  - HAD:  structured butterfly network, O(N log N), hardware-fixable.
  - RandRot: dense N×N matrix multiply, O(N²), fixed ROM weight matrix.
  - TurboQuant: diagonal ±1 random sign matrix (Hadamard preconditioner),
    O(N), trivially hardware-fixable.

All three are deterministic once the matrix is generated (fixed seed).
"""

import numpy as np
from scipy.stats import ortho_group


class RandomRotationTransform:
    """Random orthogonal rotation applied along the last axis.

    The rotation matrix Q is drawn from the Haar distribution over O(N)
    and is fixed at construction time (reproducible via seed).

    Parameters
    ----------
    dim : int
        Dimension of the rotation.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, dim: int, seed: int = 42):
        self.dim = dim
        self.seed = seed
        rng = np.random.default_rng(seed)
        # Draw from Haar measure (uniform distribution over orthogonal group)
        self.Q = ortho_group.rvs(dim, random_state=rng.integers(0, 2**31))
        self.Q = self.Q.astype(np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply rotation: y = Q @ x (last dim must equal self.dim)."""
        orig_shape = x.shape
        flat = x.reshape(-1, self.dim).astype(np.float32)
        rotated = flat @ self.Q.T
        return rotated.reshape(orig_shape)

    def inverse(self, y: np.ndarray) -> np.ndarray:
        """Inverse rotation: x = Q^T @ y (Q orthogonal → Q^{-1} = Q^T)."""
        orig_shape = y.shape
        flat = y.reshape(-1, self.dim).astype(np.float32)
        restored = flat @ self.Q
        return restored.reshape(orig_shape)

    def hardware_ops(self, n: int) -> dict:
        """Dense matrix-vector multiply: N² multiplications + N(N-1) additions."""
        return {
            "n": n,
            "multiplications": n * n,
            "additions": n * (n - 1),
            "total_ops": n * n + n * (n - 1),
            "rom_bits": n * n * 32,  # storing Q as FP32 ROM
        }


class TurboQuantTransform:
    """TurboQuant: diagonal random ±1 scaling (random sign matrix).

    Cheaper alternative to full random rotation.
    Equivalent to multiplying each element by a random ±1 sign,
    which can be viewed as a random Hadamard preconditioner.

    Hardware: XOR with a fixed sign bitmask — trivially implementable.
    Cost: N XOR operations (essentially free).
    """

    def __init__(self, dim: int, seed: int = 42):
        self.dim = dim
        self.seed = seed
        rng = np.random.default_rng(seed)
        # Random ±1 diagonal
        self.signs = rng.choice([-1.0, 1.0], size=dim).astype(np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x * self.signs

    def inverse(self, y: np.ndarray) -> np.ndarray:
        return y * self.signs  # self-inverse since signs² = 1

    def hardware_ops(self, n: int) -> dict:
        return {
            "n": n,
            "multiplications": 0,         # just XOR/sign flip
            "additions": 0,
            "xor_ops": n,
            "total_ops": n,
            "rom_bits": n,                # 1 bit per element for sign storage
        }
