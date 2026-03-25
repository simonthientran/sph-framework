"""
DFSPH (Divergence-Free SPH) time integration.

Fully vectorized implementation following:
  Bender & Koschier (2017) "Divergence-Free SPH for Incompressible
  and Viscous Fluids", IEEE TVCG.

Algorithm 6 from the paper:
  1. Compute non-pressure forces
  2. Predict velocities
  3. Constant Density Solver  (correct density error before position update)
  4. Update positions
  5. Apply periodic BC
  6. Rebuild neighbor search
  7. Recompute densities + k factors
  8. Divergence-Free Solver   (correct velocity divergence)
"""
from __future__ import annotations

import numpy as np

from sph.fluid_model import FluidModel, BoundaryModel
from sph.kernel import CubicSplineKernel
from sph.kernels_nb import (
    accumulate_density,
    accumulate_difference,
    accumulate_force,
    cubic_spline_W_batch,
    cubic_spline_gradW_batch,
    scatter_add_1d,
    scatter_add_2d,
)
from sph.neighbor_search_kdtree import KDTreeNeighborSearch, NeighborPairs
from sph.time_step import TimeStep


class DFSPHTimeStep(TimeStep):
    """
    Divergence-Free SPH time integration (Bender & Koschier 2017).

    Key properties vs WCSPH:
    - Enforces near-incompressibility via two iterative PPE solvers
    - Allows 5–10× larger time steps than WCSPH
    - Density error < 0.5% at steady state
    - Divergence-free velocity field
    """

    def __init__(
        self,
        kernel: CubicSplineKernel,
        neighbor_search: KDTreeNeighborSearch,
        nu: float = 0.0,
        gravity: np.ndarray | None = None,
        eta_cd: float = 0.01,
        eta_df: float = 0.01,
        max_iter_cd: int = 20,
        max_iter_df: int = 20,
        periodic_x: tuple[float, float] | None = None,
        eos_k: float = 8000.0,
        c0: float = 10.0,
        density_diffusion_delta: float = 0.0,
        density_diffusion_enable: bool = False,
    ):
        """
        Initialize DFSPH time step.

        Args:
            kernel: SPH kernel
            neighbor_search: Neighbor search
            nu: Kinematic viscosity
            gravity: Gravity vector (or None)
            eta_cd: Convergence tolerance for constant density solver (relative)
            eta_df: Convergence tolerance for divergence-free solver (relative)
            max_iter_cd: Maximum iterations for constant density solver
            max_iter_df: Maximum iterations for divergence-free solver
            periodic_x: Optional (x_min, x_max) for periodic BC in x.
                        Must be set here (not post-step) because positions are
                        updated mid-step before neighbor rebuild.
        """
        self.kernel = kernel
        self.neighbor_search = neighbor_search
        self.nu = float(nu)
        self.gravity = gravity if gravity is not None else np.zeros(kernel.dim, dtype=np.float64)
        self.eta_cd = float(eta_cd)
        self.eta_df = float(eta_df)
        self.max_iter_cd = int(max_iter_cd)
        self.max_iter_df = int(max_iter_df)
        self.periodic_x = periodic_x  # applied internally after position update
        self.eos_k = float(eos_k)
        self.c0 = float(c0)
        self.delta_density = float(density_diffusion_delta)
        self.enable_density_diffusion = bool(density_diffusion_enable and self.delta_density > 0.0)
        self.last_report: dict[str, float] = {}

    def step(
        self,
        fluid: FluidModel,
        boundary: BoundaryModel | None,
        dt: float,
    ) -> tuple[int, int]:
        """
        Advance simulation by one time step using DFSPH (Algorithm 6).

        Returns:
            (iter_cd, iter_df): iteration counts for both solvers
        """
        # 1. Build neighbors + density field
        pairs = self._build_neighbors(fluid, boundary)
        self._compute_densities(fluid, boundary, pairs)
        self._update_fluid_pressure(fluid)
        self._update_boundary_state(fluid, boundary, pairs)

        # 2. Non-pressure forces and velocity prediction
        fluid.accelerations.fill(0.0)
        if self.nu > 0.0:
            self._compute_viscosity_forces(fluid, boundary, pairs)
        fluid.accelerations += self.gravity
        fluid.velocities += dt * fluid.accelerations

        # 3. Constant-density PPE
        self._compute_k_factor(fluid, boundary, pairs)
        iter_cd, cd_converged, rho_err_mean, rho_err_max = self._solve_constant_density(
            fluid, boundary, pairs, dt
        )

        # 4. Position update + periodic wrap
        fluid.positions += dt * fluid.velocities
        if self.periodic_x is not None:
            self.apply_periodic_bc_x(fluid, *self.periodic_x)

        # 5. Divergence-free PPE on updated configuration
        pairs2 = self._build_neighbors(fluid, boundary)
        self._compute_densities(fluid, boundary, pairs2)
        if self.enable_density_diffusion:
            self._apply_density_diffusion(fluid, pairs2, dt)
        self._update_fluid_pressure(fluid)
        self._compute_k_factor(fluid, boundary, pairs2)
        iter_df, df_converged, div_err_mean = self._solve_divergence_free(
            fluid, boundary, pairs2, dt
        )

        report = {
            "iter_cd": int(iter_cd),
            "iter_df": int(iter_df),
            "rho_error_mean": float(rho_err_mean),
            "rho_error_max": float(rho_err_max),
            "div_error_mean": float(div_err_mean),
            "cd_converged": bool(cd_converged),
            "df_converged": bool(df_converged),
            "converged": bool(cd_converged and df_converged),
        }
        self.last_report = report
        return iter_cd, iter_df

    # ------------------------------------------------------------------ helpers

    def _build_neighbors(
        self,
        fluid: FluidModel,
        boundary: BoundaryModel | None,
    ) -> NeighborPairs:
        """Build neighbor pairs and cache them on the fluid model."""
        boundary_pos = boundary.positions if boundary is not None else None
        pairs = self.neighbor_search.build_neighbor_pairs(fluid.positions, boundary_pos)
        fluid.neighbor_pairs = pairs
        return pairs

    def _compute_densities(
        self,
        fluid: FluidModel,
        boundary: BoundaryModel | None,
        pairs: NeighborPairs,
    ):
        """SPH density summation (same as WCSPH)."""
        densities = fluid.densities
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
            bc = boundary.psi[pairs.fb_j] * W_fb
            scatter_add_1d(fluid.rho_fb, pairs.fb_i, bc)

        densities[:] = fluid.rho_self + fluid.rho_ff + fluid.rho_fb

        dominant = (fluid.rho_fb > fluid.rho_ff) & (fluid.rho_fb > 1e-6)
        if np.any(dominant):
            idx = int(np.argmax(fluid.rho_fb - fluid.rho_ff))
            print(
                f"[density-warning] boundary contribution dominates at particle {idx}: "
                f"rho_fb={fluid.rho_fb[idx]:.2f}, rho_ff={fluid.rho_ff[idx]:.2f}"
            )

    def _update_boundary_state(
        self,
        fluid: FluidModel,
        boundary: BoundaryModel | None,
        pairs: NeighborPairs,
    ):
        """Adami-inspired density/pressure and velocity extrapolation to boundary."""
        if boundary is None or not pairs.fb_i.size:
            return

        idx_i_b = pairs.fb_i
        j_boundary = pairs.fb_j
        W = cubic_spline_W_batch(pairs.fb_dist, fluid.h)

        rho_num = np.zeros(boundary.n)
        rho_den = np.zeros(boundary.n)
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

        v_num = np.zeros((boundary.n, self.kernel.dim))
        v_den = np.zeros(boundary.n)
        scatter_add_2d(v_num, j_boundary, fluid.velocities[idx_i_b] * W[:, np.newaxis])
        scatter_add_1d(v_den, j_boundary, W)
        v_den += prior
        valid = v_den > 1e-6
        boundary.velocities[valid] = -v_num[valid] / v_den[valid, np.newaxis]
        boundary.velocities[~valid] = 0.0

    def _compute_viscosity_forces(
        self,
        fluid: FluidModel,
        boundary: BoundaryModel | None,
        pairs: NeighborPairs,
    ):
        """Morris (1997) Laplacian viscosity (same formulation as WCSPH)."""
        acc = fluid.accelerations
        h = fluid.h

        if pairs.ff_i.size:
            grad_ff = cubic_spline_gradW_batch(pairs.ff_r, pairs.ff_dist, h)
            v_ij = fluid.velocities[pairs.ff_i] - fluid.velocities[pairs.ff_j]
            rdgw = np.sum(pairs.ff_r * grad_ff, axis=1)
            denom = pairs.ff_dist**2 + 0.01 * h**2
            coeff = 2.0 * self.nu * fluid.mass / fluid.rho0 * rdgw / denom
            force = coeff[:, np.newaxis] * v_ij
            accumulate_force(acc, pairs.ff_i, pairs.ff_j, force)

        if boundary is not None and pairs.fb_i.size:
            grad_fb = cubic_spline_gradW_batch(pairs.fb_r, pairs.fb_dist, h)
            v_diff = fluid.velocities[pairs.fb_i] - boundary.velocities[pairs.fb_j]
            rdgw = np.sum(pairs.fb_r * grad_fb, axis=1)
            denom = pairs.fb_dist**2 + 0.01 * h**2
            coeff = 2.0 * self.nu * boundary.mass / boundary.rho0 * rdgw / denom
            force = coeff[:, np.newaxis] * v_diff
            scatter_add_2d(acc, pairs.fb_i, force)

    # ------------------------------------------------------------------ DFSPH core

    def _compute_k_factor(self, fluid: FluidModel, boundary: BoundaryModel | None, pairs: NeighborPairs):
        """
        Precompute per-particle DFSPH stiffness factor k_i using a stable denominator:

            k_i = 1 / (||Σ_j grad W_ij||^2 + Σ_j ||grad W_ij||^2 + eps)
        """
        if fluid.n == 0:
            return

        sum_grad = np.zeros((fluid.n, self.kernel.dim))
        sum_grad_sq = np.zeros(fluid.n)

        if pairs.ff_i.size:
            gw = cubic_spline_gradW_batch(pairs.ff_r, pairs.ff_dist, fluid.h)
            accumulate_force(sum_grad, pairs.ff_i, pairs.ff_j, gw)
            gw_sq = np.sum(gw ** 2, axis=1)
            accumulate_density(sum_grad_sq, pairs.ff_i, pairs.ff_j, 1.0, gw_sq)

        if boundary is not None and pairs.fb_i.size:
            gw_fb = cubic_spline_gradW_batch(pairs.fb_r, pairs.fb_dist, fluid.h)
            scatter_add_2d(sum_grad, pairs.fb_i, gw_fb)
            gw_fb_sq = np.sum(gw_fb ** 2, axis=1)
            scatter_add_1d(sum_grad_sq, pairs.fb_i, gw_fb_sq)

        denom = np.sum(sum_grad ** 2, axis=1) + sum_grad_sq
        denom = np.maximum(denom, 1e-12)

        fluid.k_dfsph[:] = 0.0
        mask = denom > 0.0
        fluid.k_dfsph[mask] = 1.0 / denom[mask]

    def _solve_constant_density(
        self,
        fluid: FluidModel,
        boundary: BoundaryModel | None,
        pairs: NeighborPairs,
        dt: float,
    ) -> tuple[int, bool, float, float]:
        """
        Constant Density Solver (Sec. 3.2 in Bender & Koschier 2017).

        Returns (iterations, converged, mean_rel_error, max_rel_error).
        """
        has_pairs = pairs.ff_i.size or (boundary is not None and pairs.fb_i.size)
        if not has_pairs:
            fluid.p_cd_prev[:] = 0.0
            return 0, True, 0.0, 0.0

        i_u = pairs.ff_i
        j_u = pairs.ff_j
        gw = cubic_spline_gradW_batch(pairs.ff_r, pairs.ff_dist, fluid.h) if pairs.ff_i.size else np.zeros((0, self.kernel.dim))
        boundary_active = boundary is not None and pairs.fb_i.size
        if boundary_active:
            gw_fb = cubic_spline_gradW_batch(pairs.fb_r, pairs.fb_dist, fluid.h)
            psi_fb = boundary.psi[pairs.fb_j]
            boundary_vel = boundary.velocities[pairs.fb_j]
        else:
            gw_fb = None
            psi_fb = None
            boundary_vel = None

        lambda_prev = fluid.p_cd_prev.copy()
        if np.any(lambda_prev) and pairs.ff_i.size:
            pi_prev = lambda_prev[i_u]
            pj_prev = lambda_prev[j_u]
            ri_prev = fluid.densities[i_u]
            rj_prev = fluid.densities[j_u]
            coeff_prev = fluid.mass * (pi_prev / (ri_prev ** 2 + 1e-12) + pj_prev / (rj_prev ** 2 + 1e-12))
            dv_prev = -dt * coeff_prev[:, np.newaxis] * gw
            accumulate_force(fluid.velocities, i_u, j_u, dv_prev)
        if boundary_active and np.any(lambda_prev):
            lam_prev_fb = lambda_prev[pairs.fb_i]
            ri_prev_fb = fluid.densities[pairs.fb_i]
            coeff_fb_prev = psi_fb * (lam_prev_fb / (ri_prev_fb ** 2 + 1e-12))
            dv_prev_fb = -dt * coeff_fb_prev[:, np.newaxis] * gw_fb
            scatter_add_2d(fluid.velocities, pairs.fb_i, dv_prev_fb)

        lambda_cd = np.zeros_like(fluid.p_cd_prev)
        rho_err_mean = 0.0
        rho_err_max = 0.0
        converged = False
        iterations = 0

        for iteration in range(self.max_iter_cd):
            iterations = iteration + 1
            if i_u.size:
                dv = fluid.velocities[i_u] - fluid.velocities[j_u]
                drho = dt * fluid.mass * np.sum(dv * gw, axis=1)
            else:
                drho = np.zeros(0)

            rho_star = fluid.densities.copy()
            if drho.size:
                accumulate_difference(rho_star, i_u, j_u, drho)
            if boundary_active:
                vel_diff_fb = fluid.velocities[pairs.fb_i] - boundary_vel
                contrib_fb = dt * psi_fb * np.sum(vel_diff_fb * gw_fb, axis=1)
                scatter_add_1d(rho_star, pairs.fb_i, contrib_fb)

            rho_err = np.maximum(rho_star - fluid.rho0, 0.0)
            if rho_err.size:
                rel_err = rho_err / fluid.rho0
                rho_err_mean = float(rel_err.mean())
                rho_err_max = float(rel_err.max())
            else:
                rho_err_mean = 0.0
                rho_err_max = 0.0

            if rho_err_mean < self.eta_cd:
                converged = True
                break

            lambda_cd = (rho_err / (dt * dt + 1e-12)) * fluid.k_dfsph

            if i_u.size:
                pi = lambda_cd[i_u]
                pj = lambda_cd[j_u]
                ri = fluid.densities[i_u]
                rj = fluid.densities[j_u]
                coeff = fluid.mass * (pi / (ri ** 2 + 1e-12) + pj / (rj ** 2 + 1e-12))
                dv_p = -dt * coeff[:, np.newaxis] * gw
                accumulate_force(fluid.velocities, i_u, j_u, dv_p)

            if boundary_active:
                lambda_fb = lambda_cd[pairs.fb_i]
                ri_fb = fluid.densities[pairs.fb_i]
                coeff_fb = psi_fb * (lambda_fb / (ri_fb ** 2 + 1e-12))
                dv_fb = -dt * coeff_fb[:, np.newaxis] * gw_fb
                scatter_add_2d(fluid.velocities, pairs.fb_i, dv_fb)

        fluid.p_cd_prev[:] = lambda_cd
        return iterations, converged, rho_err_mean, rho_err_max

    def _solve_divergence_free(
        self,
        fluid: FluidModel,
        boundary: BoundaryModel | None,
        pairs: NeighborPairs,
        dt: float,
    ) -> tuple[int, bool, float]:
        """
        Divergence-Free Solver (Sec. 3.3 in Bender & Koschier 2017).

        Returns (iterations, converged, mean_relative_divergence).
        """
        has_pairs = pairs.ff_i.size or (boundary is not None and pairs.fb_i.size)
        if not has_pairs:
            fluid.p_df_prev[:] = 0.0
            return 0, True, 0.0

        i_u = pairs.ff_i
        j_u = pairs.ff_j
        gw = cubic_spline_gradW_batch(pairs.ff_r, pairs.ff_dist, fluid.h) if pairs.ff_i.size else np.zeros((0, self.kernel.dim))
        boundary_active = boundary is not None and pairs.fb_i.size
        if boundary_active:
            gw_fb = cubic_spline_gradW_batch(pairs.fb_r, pairs.fb_dist, fluid.h)
            psi_fb = boundary.psi[pairs.fb_j]
            boundary_vel = boundary.velocities[pairs.fb_j]
        else:
            gw_fb = None
            psi_fb = None
            boundary_vel = None

        lambda_prev = fluid.p_df_prev.copy()
        if np.any(lambda_prev) and i_u.size:
            pi_prev = lambda_prev[i_u]
            pj_prev = lambda_prev[j_u]
            ri_prev = fluid.densities[i_u]
            rj_prev = fluid.densities[j_u]
            coeff_prev = fluid.mass * (pi_prev / (ri_prev ** 2 + 1e-12) + pj_prev / (rj_prev ** 2 + 1e-12))
            dv_prev = -dt * coeff_prev[:, np.newaxis] * gw
            accumulate_force(fluid.velocities, i_u, j_u, dv_prev)
        if boundary_active and np.any(lambda_prev):
            lam_prev_fb = lambda_prev[pairs.fb_i]
            ri_prev_fb = fluid.densities[pairs.fb_i]
            coeff_fb_prev = psi_fb * (lam_prev_fb / (ri_prev_fb ** 2 + 1e-12))
            dv_prev_fb = -dt * coeff_fb_prev[:, np.newaxis] * gw_fb
            scatter_add_2d(fluid.velocities, pairs.fb_i, dv_prev_fb)

        lambda_df = np.zeros_like(fluid.p_df_prev)
        div_err_mean = 0.0
        converged = False
        iterations = 0

        for iteration in range(self.max_iter_df):
            iterations = iteration + 1
            if i_u.size:
                dv = fluid.velocities[i_u] - fluid.velocities[j_u]
                contrib = fluid.mass * np.sum(dv * gw, axis=1)
            else:
                contrib = np.zeros(0)

            drho_dt = np.zeros(fluid.n)
            if contrib.size:
                accumulate_difference(drho_dt, i_u, j_u, contrib)
            if boundary_active:
                vel_diff_fb = fluid.velocities[pairs.fb_i] - boundary_vel
                contrib_fb = psi_fb * np.sum(vel_diff_fb * gw_fb, axis=1)
                scatter_add_1d(drho_dt, pairs.fb_i, contrib_fb)

            if drho_dt.size:
                div_err_mean = float((np.abs(drho_dt) * dt / fluid.rho0).mean())
            else:
                div_err_mean = 0.0

            if div_err_mean < self.eta_df:
                converged = True
                break

            lambda_df = (drho_dt / (dt + 1e-12)) * fluid.k_dfsph

            if i_u.size:
                pi = lambda_df[i_u]
                pj = lambda_df[j_u]
                ri = fluid.densities[i_u]
                rj = fluid.densities[j_u]
                coeff = fluid.mass * (pi / (ri ** 2 + 1e-12) + pj / (rj ** 2 + 1e-12))
                dv_p = -dt * coeff[:, np.newaxis] * gw
                accumulate_force(fluid.velocities, i_u, j_u, dv_p)

            if boundary_active:
                lambda_fb = lambda_df[pairs.fb_i]
                ri_fb = fluid.densities[pairs.fb_i]
                coeff_fb = psi_fb * (lambda_fb / (ri_fb ** 2 + 1e-12))
                dv_fb = -dt * coeff_fb[:, np.newaxis] * gw_fb
                scatter_add_2d(fluid.velocities, pairs.fb_i, dv_fb)

        fluid.p_df_prev[:] = lambda_df
        return iterations, converged, div_err_mean

    def _apply_density_diffusion(
        self,
        fluid: FluidModel,
        pairs: NeighborPairs,
        dt: float,
    ) -> None:
        """
        δ-SPH density diffusion correction (Fourtakas/Marrone).

        Adds a Laplacian-like diffusion term to smooth out density errors,
        particularly near free surfaces where SPH summation underestimates ρ.

            D_i = δ·h·c0 · Σ_j (ρ_j - ρ_i)(r_ij·∇W_ij)/|r_ij|² · V_j
            ρ_i += dt · D_i
        """
        if not pairs.ff_i.size or self.delta_density <= 0.0:
            return

        h = fluid.h
        rho = fluid.densities
        gw = cubic_spline_gradW_batch(pairs.ff_r, pairs.ff_dist, h)
        r_dot_gw = np.sum(pairs.ff_r * gw, axis=1)
        denom = pairs.ff_dist ** 2 + 0.01 * h * h

        rho_i = rho[pairs.ff_i]
        rho_j = rho[pairs.ff_j]
        V_j = fluid.mass / np.maximum(rho_j, 0.01 * fluid.rho0)

        coeff = self.delta_density * h * self.c0
        contrib = coeff * (rho_j - rho_i) * r_dot_gw / denom * V_j

        diffusion = np.zeros_like(rho)
        accumulate_difference(diffusion, pairs.ff_i, pairs.ff_j, contrib)

        rho += dt * diffusion
        np.maximum(rho, 0.1 * fluid.rho0, out=rho)

    def apply_periodic_bc_x(self, fluid: FluidModel, x_min: float, x_max: float):
        """Wrap particles in x-direction for periodic BC."""
        L_x = x_max - x_min
        fluid.positions[:, 0] = x_min + np.mod(fluid.positions[:, 0] - x_min, L_x)

    def _pressure_from_density(self, density: np.ndarray, rho0: float) -> np.ndarray:
        rho_ratio = density / rho0
        pressure = self.eos_k * (rho_ratio ** 7 - 1.0)
        pressure[pressure < 0.0] = 0.0
        return pressure

    def _update_fluid_pressure(self, fluid: FluidModel) -> None:
        fluid.pressures[:] = self._pressure_from_density(fluid.densities, fluid.rho0)
