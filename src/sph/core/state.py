from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class ParticleState:
    """Data-oriented particle state (SoA-like but kept minimal for now)."""

    dim: int
    pos: np.ndarray   # (N, dim)
    vel: np.ndarray   # (N, dim)
    acc: np.ndarray   # (N, dim)

    mass: np.ndarray  # (N,)
    rho: np.ndarray   # (N,)
    p: np.ndarray     # (N,)

    @property
    def n(self) -> int:
        return int(self.pos.shape[0])

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
        ]:
            if arr.shape != shape:
                raise ValueError(f"{name} shape {arr.shape} != {shape}")
        if not np.isfinite(self.pos).all():
            raise ValueError("pos contains NaN/Inf")
