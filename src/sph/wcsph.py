"""
WCSPH (Weakly Compressible SPH) time integration.

Fully vectorized implementation using neighbor pair arrays.

Reference:
"SPH Techniques for the Physics Based Simulation of Fluids and Solids - SPH_Tutorial.pdf"
Algorithm 1: Standard WCSPH scheme
"""
from __future__ import annotations

import numpy as np

from sph.fluid_model import FluidModel, BoundaryModel
from sph.kernel import CubicSplineKernel
from sph.neighbor_search_kdtree import KDTreeNeighborSearch, NeighborPairs
from sph.time_step import TimeStep
from sph.kernels_nb import (
    cubic_spline_W_batch,
    cubic_spline_gradW_batch,
    accumulate_density,
    accumulate_force,
    scatter_add_1d,
    scatter_add_2d,
)


class WCSPHTimeStep(TimeStep):
    """
    Weakly Compressible SPH time integration.

    Uses:
    - Tait EOS for pressure
    - Symplectic Euler integration
    - Vectorized force computation (no Python loops over particles)
    """

    def __init__(
        self,
        kernel: CubicSplineKernel,
        neighbor_search: KDTreeNeighborSearch,
        eos_k: float = 8000.0,
        nu: float = 0.0,
        gravity: np.ndarray | None = None,
    ):
        """
        Initialize WCSPH time step.

        Args:
            kernel: SPH kernel
            neighbor_search: Neighbor search
            eos_k: Tait EOS stiffness parameter
            nu: Kinematic viscosity
            gravity: Gravity vector (or None)
        """
        self.kernel = kernel
        self.neighbor_search = neighbor_search
        self.eos_k = float(eos_k)
        self.nu = float(nu)
        self.gravity = gravity if gravity is not None else np.zeros(kernel.dim, dtype=np.float64)

    def step(self, fluid: FluidModel, boundary: BoundaryModel | None, dt: float):
        """
        Advance simulation by one time step using WCSPH.

        Stages:
        1. Build neighbor pairs
        2. Density summation
        3. Equation of state
        4. Boundary extrapolation
        5. Non-pressure forces (viscosity + gravity)
        6. Pressure forces
        7. Integrate (Symplectic Euler)
        """
        boundary_pos = boundary.positions if boundary is not None else None
        pairs = self.neighbor_search.build_neighbor_pairs(fluid.positions, boundary_pos)
        fluid.neighbor_pairs = pairs

        self._compute_densities(fluid, boundary, pairs)
        self._compute_pressures(fluid)
        self._update_boundary_state(fluid, boundary, pairs)
        self._compute_non_pressure_forces(fluid, boundary, pairs)
        self._compute_pressure_forces(fluid, boundary, pairs)
        self._integrate(fluid, dt)

    def apply_periodic_bc_x(self, fluid: FluidModel, x_min: float, x_max: float):
        """
        Apply periodic boundary condition in x-direction.

        Wraps particles from right boundary back to left boundary.
        This is used for Poiseuille flow to maintain mass conservation.

        Args:
            fluid: Fluid model
            x_min: Minimum x-coordinate of periodic domain
            x_max: Maximum x-coordinate of periodic domain
        """
        L_x = x_max - x_min
        # Wrap positions: x_new = x_min + mod(x - x_min, L_x)
        fluid.positions[:, 0] = x_min + np.mod(fluid.positions[:, 0] - x_min, L_x)

    def _compute_densities(
        self,
        fluid: FluidModel,
        boundary: BoundaryModel | None,
        pairs: NeighborPairs,
    ):
        """
        Compute fluid densities using SPH summation (fully vectorized).

        rho_i = sum_j m_j W_ij

        Reference: Eq. (11) from SPH Tutorial
        """
        density = fluid.densities
        h = fluid.h

        W_self = self.kernel.W(0.0)
        fluid.rho_self.fill(fluid.mass * W_self)
        fluid.rho_ff.fill(0.0)
        fluid.rho_fb.fill(0.0)
        if boundary is not None:
            boundary.densities.fill(boundary.rho0)

        if pairs.ff_i.size:
            W_ff = cubic_spline_W_batch(pairs.ff_dist, h)
            accumulate_density(fluid.rho_ff, pairs.ff_i, pairs.ff_j, fluid.mass, W_ff)

        if boundary is not None and pairs.fb_i.size:
            W_fb = cubic_spline_W_batch(pairs.fb_dist, h)
            contribution = boundary.psi[pairs.fb_j] * W_fb
            scatter_add_1d(fluid.rho_fb, pairs.fb_i, contribution)

        density[:] = fluid.rho_self + fluid.rho_ff + fluid.rho_fb

        # Diagnostics: warn if boundary dominates
        dominant = (fluid.rho_fb > fluid.rho_ff) & (fluid.rho_fb > 1e-6)
        if np.any(dominant):
            idx = int(np.argmax(fluid.rho_fb - fluid.rho_ff))
            print(
                f"[density-warning] boundary contribution dominates at particle {idx}: "
                f"rho_fb={fluid.rho_fb[idx]:.2f}, rho_ff={fluid.rho_ff[idx]:.2f}"
            )

    def _compute_pressures(self, fluid: FluidModel):
        """
        Compute pressures using Tait equation of state.

        p = k * ((rho / rho0)^7 - 1)

        Reference: Eq. (9) from SPH Tutorial
        """
        fluid.pressures[:] = self._pressure_from_density(fluid.densities, fluid.rho0)

    def _pressure_from_density(self, density: np.ndarray, rho0: float) -> np.ndarray:
        rho_ratio = density / rho0
        pressure = self.eos_k * (rho_ratio ** 7 - 1.0)
        pressure[pressure < 0.0] = 0.0
        return pressure

    def _update_boundary_state(
        self,
        fluid: FluidModel,
        boundary: BoundaryModel | None,
        pairs: NeighborPairs,
    ):
        """
        Update boundary particle state using Adami et al. 2012 approach.

        Extrapolates pressure and velocity from fluid to boundary particles.
        For no-slip walls, boundary velocities are mirrored to enforce zero
        velocity at the wall interface.

        Reference: Adami et al. "A generalized wall boundary condition for
        smoothed particle hydrodynamics" (2012)
        """
        if boundary is None:
            return

        if boundary is None or not pairs.fb_i.size:
            return

        idx_i_b = pairs.fb_i
        j_boundary = pairs.fb_j
        W = cubic_spline_W_batch(pairs.fb_dist, fluid.h)

        # --- Density extrapolation ---
        rho_num = np.zeros(boundary.n, dtype=np.float64)
        rho_den = np.zeros(boundary.n, dtype=np.float64)
        scatter_add_1d(rho_num, j_boundary, fluid.densities[idx_i_b] * W)
        scatter_add_1d(rho_den, j_boundary, W)
        prior = 0.25
        rho_num += prior * boundary.rho0
        rho_den += prior

        valid_rho = rho_den > 1e-6
        boundary.densities[valid_rho] = rho_num[valid_rho] / rho_den[valid_rho]
        boundary.densities[~valid_rho] = boundary.rho0
        boundary.pressures[valid_rho] = self._pressure_from_density(
            boundary.densities[valid_rho], boundary.rho0
        )
        boundary.pressures[~valid_rho] = 0.0

        # --- Velocity extrapolation (no-slip mirror) ---
        # For static walls (v_wall = 0):
        # v_b = -sum_f(v_f * W_fb) / sum_f(W_fb)
        # This creates a mirror velocity such that the average at the wall is zero

        v_numerator = np.zeros((boundary.n, self.kernel.dim), dtype=np.float64)
        v_denominator = np.zeros(boundary.n, dtype=np.float64)

        scatter_add_2d(v_numerator, j_boundary, fluid.velocities[idx_i_b] * W[:, np.newaxis])
        scatter_add_1d(v_denominator, j_boundary, W)
        v_denominator += prior

        # Update boundary velocity (mirror to enforce no-slip)
        valid_mask = v_denominator > 1e-6
        boundary.velocities[valid_mask] = -v_numerator[valid_mask] / v_denominator[valid_mask, np.newaxis]
        boundary.velocities[~valid_mask] = 0.0

    def _compute_pressure_forces(
        self,
        fluid: FluidModel,
        boundary: BoundaryModel | None,
        pairs: NeighborPairs,
    ):
        """
        Compute pressure forces (fully vectorized).

        a_p,i = -sum_j m_j (p_i/rho_i^2 + p_j/rho_j^2) grad W_ij

        Reference: Eq. (10) from SPH Tutorial (symmetric form)
        """
        acc = fluid.accelerations
        h = fluid.h

        if pairs.ff_i.size:
            grad_ff = cubic_spline_gradW_batch(pairs.ff_r, pairs.ff_dist, h)
            p_i = fluid.pressures[pairs.ff_i]
            p_j = fluid.pressures[pairs.ff_j]
            rho_i = fluid.densities[pairs.ff_i]
            rho_j = fluid.densities[pairs.ff_j]
            coeff = -fluid.mass * (p_i / rho_i**2 + p_j / rho_j**2)
            force = coeff[:, np.newaxis] * grad_ff
            accumulate_force(acc, pairs.ff_i, pairs.ff_j, force)

        if boundary is not None and pairs.fb_i.size:
            grad_fb = cubic_spline_gradW_batch(pairs.fb_r, pairs.fb_dist, h)
            p_i = fluid.pressures[pairs.fb_i]
            rho_i = fluid.densities[pairs.fb_i]
            p_j = boundary.pressures[pairs.fb_j]
            rho_j = boundary.densities[pairs.fb_j]
            coeff = -boundary.mass * (p_i / rho_i**2 + p_j / rho_j**2)
            force = coeff[:, np.newaxis] * grad_fb
            scatter_add_2d(acc, pairs.fb_i, force)

    def _compute_non_pressure_forces(
        self,
        fluid: FluidModel,
        boundary: BoundaryModel | None,
        pairs: NeighborPairs,
    ):
        """Clear accelerations and add viscosity + gravity."""
        fluid.accelerations.fill(0.0)

        if self.nu > 0.0:
            self._accumulate_viscosity(fluid, boundary, pairs)

        fluid.accelerations += self.gravity

    def _accumulate_viscosity(
        self,
        fluid: FluidModel,
        boundary: BoundaryModel | None,
        pairs: NeighborPairs,
    ):
        acc = fluid.accelerations
        vel = fluid.velocities
        h = fluid.h
        nu = self.nu

        if pairs.ff_i.size:
            grad_ff = cubic_spline_gradW_batch(pairs.ff_r, pairs.ff_dist, h)
            v_ij = vel[pairs.ff_i] - vel[pairs.ff_j]
            rdotgw = np.sum(pairs.ff_r * grad_ff, axis=1)
            denom = pairs.ff_dist**2 + 0.01 * h**2
            coeff = 2.0 * nu * fluid.mass / fluid.rho0 * rdotgw / denom
            force = coeff[:, np.newaxis] * v_ij
            accumulate_force(acc, pairs.ff_i, pairs.ff_j, force)

        if boundary is not None and pairs.fb_i.size:
            grad_fb = cubic_spline_gradW_batch(pairs.fb_r, pairs.fb_dist, h)
            v_diff = vel[pairs.fb_i] - boundary.velocities[pairs.fb_j]
            rdotgw = np.sum(pairs.fb_r * grad_fb, axis=1)
            denom = pairs.fb_dist**2 + 0.01 * h**2
            coeff = 2.0 * nu * boundary.mass / boundary.rho0 * rdotgw / denom
            force = coeff[:, np.newaxis] * v_diff
            scatter_add_2d(acc, pairs.fb_i, force)

    def _integrate(self, fluid: FluidModel, dt: float):
        fluid.velocities += dt * fluid.accelerations
        fluid.positions += dt * fluid.velocities
