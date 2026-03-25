"""
FluidModel — Holds all particle data as flat NumPy arrays.

Inspired by SPlisHSPlasH FluidModel architecture.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sph.neighbor_pairs import NeighborPairs


class FluidModel:
    """
    Manages all fluid particle data as flat NumPy arrays.

    This replaces the previous ParticleState dataclass with a more
    explicit array-based model that's easier to vectorize.
    """

    def __init__(self, n_particles: int, rho0: float, mass: float, h: float, dim: int = 2):
        """
        Initialize fluid model with particle arrays.

        Args:
            n_particles: Number of fluid particles
            rho0: Rest density
            mass: Particle mass (uniform)
            h: Smoothing length
            dim: Spatial dimension (2 or 3)
        """
        self.n = n_particles
        self.rho0 = float(rho0)
        self.mass = float(mass)
        self.h = float(h)
        self.dim = dim

        # State arrays — shape (n, dim)
        self.positions = np.zeros((n_particles, dim), dtype=np.float64)
        self.velocities = np.zeros((n_particles, dim), dtype=np.float64)
        self.accelerations = np.zeros((n_particles, dim), dtype=np.float64)

        # Scalar arrays — shape (n,)
        self.densities = np.full(n_particles, rho0, dtype=np.float64)
        self.pressures = np.zeros(n_particles, dtype=np.float64)
        self.k_dfsph   = np.zeros(n_particles, dtype=np.float64)  # DFSPH stiffness factor
        self.p_cd_prev = np.zeros(n_particles, dtype=np.float64)  # DFSPH warm-start (const density)
        self.p_df_prev = np.zeros(n_particles, dtype=np.float64)  # DFSPH warm-start (div-free)
        self.rho_self = np.zeros(n_particles, dtype=np.float64)
        self.rho_ff = np.zeros(n_particles, dtype=np.float64)
        self.rho_fb = np.zeros(n_particles, dtype=np.float64)

        # Neighbor data (rebuilt each time step)
        self.neighbor_pairs: NeighborPairs | None = None

    def clear_accelerations(self):
        """Reset accelerations to zero before force computation."""
        self.accelerations[:] = 0.0

    def get_support_radius(self) -> float:
        """Get the compact support radius (2h for cubic spline)."""
        return 2.0 * self.h


class BoundaryModel:
    """
    Manages boundary particle data.

    Boundary particles contribute to density and forces but don't move.
    """

    def __init__(self, n_particles: int, rho0: float, mass: float, dim: int = 2):
        """
        Initialize boundary model.

        Args:
            n_particles: Number of boundary particles
            rho0: Rest density (boundary particles kept at this value)
            mass: Particle mass (uniform)
            dim: Spatial dimension
        """
        self.n = n_particles
        self.rho0 = float(rho0)
        self.mass = float(mass)
        self.dim = dim

        # Boundary particles are static in position, but have mirror velocities
        self.positions = np.zeros((n_particles, dim), dtype=np.float64)
        self.velocities = np.zeros((n_particles, dim), dtype=np.float64)  # Mirror velocities (Adami 2012)
        self.densities = np.full(n_particles, rho0, dtype=np.float64)
        self.pressures = np.zeros(n_particles, dtype=np.float64)

        # Boundary volumes (computed once at initialization)
        # Used for accurate boundary contribution: rho_i += rho0 * volume_b * W_ib
        self.volumes = np.zeros(n_particles, dtype=np.float64)
        self.psi = np.zeros(n_particles, dtype=np.float64)

    def compute_volumes(self, kernel, support_radius: float, fluid_positions: np.ndarray | None = None):
        """
        Compute boundary particle volumes.

        For Adami 2012 boundaries, volume = mass / rho0 (physical volume).

        This is called once at initialization.
        """
        default_volume = self.mass / self.rho0
        self.volumes[:] = default_volume
        self.psi[:] = self.mass  # psi = rho0 * volume = mass
