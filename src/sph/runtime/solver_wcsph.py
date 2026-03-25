"""WCSPH solver core following the SPH tutorial formulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sph.core.state import ParticleState
from sph.neighbors.spatial_hash import SpatialHash
from sph.sph.density import compute_density_summation
from sph.sph.pressure import (
    pressure_acceleration_symmetric,
    pressure_state_equation_linear,
)


@dataclass(slots=True)
class WCSPHSolver:
    """Encapsulates density, pressure and pressure-force computations.

    The implementation mirrors Algorithm 1 from Ihmsen et al. (SPH Tutorial)
    but exposes the building blocks as methods so the simulation pipeline can
    orchestrate them explicitly, similar in spirit to SPlisHSPlasH solvers.
    """

    rho0: float
    eos_k: float
    support_radius: float
    clamp_negative_pressure: bool = True

    def compute_density(self, state: ParticleState, neighbor_search: SpatialHash) -> np.ndarray:
        """Reconstruct density via SPH summation (Eq. 11)."""

        rho = compute_density_summation(state=state, neighbor_search=neighbor_search, h=float(self.support_radius))
        state.rho[:] = rho
        return rho

    def compute_pressure(self, state: ParticleState) -> np.ndarray:
        """Update pressure using the linear EOS p = k (rho - rho0)."""

        p = pressure_state_equation_linear(state.rho, rho0=float(self.rho0), k=float(self.eos_k))
        if self.clamp_negative_pressure:
            p = np.maximum(p, 0.0)
        state.p[:] = p
        return p

    def compute_pressure_forces(self, state: ParticleState, neighbor_search: SpatialHash) -> np.ndarray:
        """Return symmetric pressure acceleration (Eq. 53) for all particles."""

        return pressure_acceleration_symmetric(
            state=state,
            neighbor_search=neighbor_search,
            h=float(self.support_radius),
        )
