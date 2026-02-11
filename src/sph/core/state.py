from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class ParticleState:
    """
    Unified particle storage for both fluid and boundary particles.

    We keep a boolean flag is_boundary to allow:
    - boundary particles to contribute to density and pressure force sums
    - but not be integrated (static boundary)

    This follows the particle-based boundary handling idea described in:
    "SPH Techniques for the Physics Based Simulation of Fluids and Solids - SPH_Tutorial.pdf", Section 5.1.
    """

    dim: int

    pos: np.ndarray          # (N, dim)
    vel: np.ndarray          # (N, dim)
    acc: np.ndarray          # (N, dim)

    mass: np.ndarray         # (N,)
    rho: np.ndarray          # (N,)
    p: np.ndarray            # (N,)

    is_boundary: np.ndarray  # (N,) bool

    @property
    def n(self) -> int:
        return int(self.pos.shape[0])

    @property
    def fluid_indices(self) -> np.ndarray:
        """Indices of non-boundary (fluid) particles."""
        return np.where(~self.is_boundary)[0]

    @property
    def boundary_indices(self) -> np.ndarray:
        """Indices of boundary particles."""
        return np.where(self.is_boundary)[0]

    def validate(self) -> None:
        n = self.n
        if self.pos.shape != (n, self.dim):
            raise ValueError(f"pos shape {self.pos.shape} != (N, dim) = ({n},{self.dim})")

        for name, arr, shape in [
            ("vel", self.vel, (n, self.dim)),
            ("acc", self.acc, (n, self.dim)),
            ("mass", self.mass, (n,)),
            ("rho", self.rho, (n,)),
            ("p", self.p, (n,)),
            ("is_boundary", self.is_boundary, (n,)),
        ]:
            if arr.shape != shape:
                raise ValueError(f"{name} shape {arr.shape} != {shape}")

        if self.is_boundary.dtype != np.bool_:
            raise ValueError("is_boundary must be a bool array")

        if not np.isfinite(self.pos).all():
            raise ValueError("pos contains NaN/Inf")


