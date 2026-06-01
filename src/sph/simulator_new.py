"""
Simulator class for SPH simulations.

Loads scene configuration and runs the simulation loop.
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path

from sph.fluid_model import FluidModel, BoundaryModel
from sph.kernel import CubicSplineKernel
from sph.neighbor_search_kdtree import KDTreeNeighborSearch
from sph.wcsph import WCSPHTimeStep
from sph.kernels_nb import (
    cubic_spline_W_batch,
    scatter_add_2d,
    accumulate_density,
    scatter_add_1d,
)


class Simulator:
    """
    Main simulation orchestrator.

    Loads scene configuration and manages the simulation loop.
    """

    def __init__(self, scene_path: Path | str, solver: str = "dfsph"):
        """
        Initialize simulator from scene JSON file.

        Args:
            scene_path: Path to scene JSON file
        """
        self.scene_path = Path(scene_path)
        self.scene = self._load_scene()
        solver_cfg = self.scene.get("solver", {})
        prefer_iterative = bool(solver_cfg.get("prefer_iterative", True))
        solver_type = solver_cfg.get("type", solver).lower()
        if prefer_iterative and solver_type != "dfsph":
            solver_type = "dfsph"
        self.solver_name = solver_type

        # Extract configuration
        self.dim = int(self.scene.get("meta", {}).get("dimensions", 2))
        self.rho0 = float(self.scene["material"]["rho0"])
        self.spacing = float(self.scene["fluid"]["spacing"])

        neighbors_cfg = self.scene.get("neighbors", {})
        self._configure_kernel_lengths(neighbors_cfg)
        self.kernel = CubicSplineKernel(h=self.h, dim=self.dim)
        self.support_radius = self.kernel.support_radius

        # Create models
        self.fluid, self.boundary = self._create_models()

        # Compute boundary volumes (SPlisHSPlasH approach)
        if self.boundary is not None:
            self.boundary.compute_volumes(self.kernel, self.support_radius, self.fluid.positions)

        # Get domain bounds for periodic BC
        domain_cfg = self.scene.get("domain", {})
        if "min" in domain_cfg and "max" in domain_cfg:
            domain_min = np.array(domain_cfg["min"], dtype=np.float64)
            domain_max = np.array(domain_cfg["max"], dtype=np.float64)
            self.x_min = domain_min[0]
            self.x_max = domain_max[0]
            periodic_x = (self.x_min, self.x_max) if domain_cfg.get("periodic_x", False) else None
        else:
            self.x_min = None
            self.x_max = None
            periodic_x = None

        # z-periodic BC (3D only)
        if self.dim == 3 and domain_cfg.get("periodic_z", False) and "min" in domain_cfg:
            fluid_cfg = self.scene["fluid"]
            z_min_f = float(np.array(fluid_cfg["min"], dtype=np.float64)[2])
            z_max_f = float(np.array(fluid_cfg["max"], dtype=np.float64)[2])
            self.z_min: float | None = z_min_f
            self.z_max: float | None = z_max_f
            periodic_z: tuple[float, float] | None = (z_min_f, z_max_f)
        else:
            self.z_min = None
            self.z_max = None
            periodic_z = None

        # Create neighbor search with periodic BC support
        self.neighbor_search = KDTreeNeighborSearch(
            support_radius=self.support_radius,
            dim=self.dim,
            periodic_x=periodic_x,
            periodic_z=periodic_z,
        )

        # Trigger Numba JIT compilation before first simulation step
        from sph.kernels_nb import warmup_numba
        warmup_numba(self.h)

        # Forces configuration (gravity, viscosity, stabilization)
        self.forces_cfg = self.scene.get("forces", {})
        gravity_raw = self.forces_cfg.get("gravity", [0.0, 0.0])
        self.gravity_vec = np.array(gravity_raw[:self.dim], dtype=np.float64)

        # Solver runtime state
        self._last_solver_stats: dict[str, int] = {}

        # Configure EOS, timestep, viscosity, stabilization
        self._configure_solver_parameters()
        self._configure_stabilization()

        # Create time step integrator
        self.time_step = self._create_time_step()

        self.current_step = 0

    def _load_scene(self) -> dict:
        """Load scene configuration from JSON file."""
        with open(self.scene_path, "r") as f:
            data = json.load(f)
        if "fluids" in data and "fluid" not in data:
            data["fluid"] = data["fluids"][0]
        return data

    def _compute_correct_mass(self) -> float:
        """
        Compute particle mass that satisfies partition of unity.

        For a uniform grid with spacing dx and kernel smoothing length h,
        the correct mass is: m = rho0 / sum_j(W_ij)

        where the sum is over all neighbors of an interior particle.
        """
        # Build a test uniform grid patch
        support = self.support_radius
        n_cells = int(np.ceil(support / self.spacing)) + 1

        # Compute kernel sum for all neighbors within support
        kernel = CubicSplineKernel(h=self.h, dim=self.dim)
        W_sum = 0.0

        if self.dim == 3:
            for ix in range(-n_cells, n_cells + 1):
                for iy in range(-n_cells, n_cells + 1):
                    for iz in range(-n_cells, n_cells + 1):
                        r = np.sqrt((ix * self.spacing)**2 + (iy * self.spacing)**2 + (iz * self.spacing)**2)
                        if r <= support:
                            W_sum += kernel.W(r)
        else:
            for ix in range(-n_cells, n_cells + 1):
                for iy in range(-n_cells, n_cells + 1):
                    x = ix * self.spacing
                    y = iy * self.spacing
                    r = np.sqrt(x**2 + y**2)
                    if r <= support:
                        W_sum += kernel.W(r)

        # Correct mass: m = rho0 / sum(W)
        return self.rho0 / W_sum

    def _create_models(self) -> tuple[FluidModel, BoundaryModel | None]:
        """Create fluid and boundary models from scene configuration."""
        # Parse fluid configuration
        fluid_cfg = self.scene["fluid"]
        fluid_min = np.array(fluid_cfg["min"], dtype=np.float64)
        fluid_max = np.array(fluid_cfg["max"], dtype=np.float64)

        # Generate fluid particles on a regular grid
        if self.dim == 2:
            x = np.arange(fluid_min[0], fluid_max[0], self.spacing)
            y = np.arange(fluid_min[1], fluid_max[1], self.spacing)
            xx, yy = np.meshgrid(x, y, indexing='ij')
            fluid_positions = np.column_stack([xx.ravel(), yy.ravel()])
        else:
            x = np.arange(fluid_min[0], fluid_max[0], self.spacing)
            y = np.arange(fluid_min[1], fluid_max[1], self.spacing)
            z = np.arange(fluid_min[2], fluid_max[2], self.spacing)
            xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
            fluid_positions = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

        n_fluid = len(fluid_positions)

        # Compute particle mass using partition of unity
        # For uniform grid: m = rho0 / sum_j(W_ij)
        # where the sum is over neighbors of an interior particle
        particle_mass = self._compute_correct_mass()

        # Create fluid model
        fluid = FluidModel(n_fluid, self.rho0, particle_mass, self.h, self.dim)
        fluid.positions[:] = fluid_positions

        # Set initial velocity
        initial_velocity = fluid_cfg.get("initial_velocity", [0.0, 0.0])
        fluid.velocities[:] = np.array(initial_velocity[:self.dim], dtype=np.float64)

        # Create boundary model (if specified)
        boundary = None
        domain_cfg = self.scene.get("domain", {})
        if "boundary_layers" in domain_cfg:
            boundary = self._create_boundary_particles(domain_cfg)

        return fluid, boundary

    def _create_boundary_particles(self, domain_cfg: dict | None) -> BoundaryModel | None:
        """
        Create boundary particles adjacent to the fluid region.

        Places boundary layers just outside the fluid bounding box so that
        wall particles are always within support_radius of the fluid.
        For periodic-x flows (Poiseuille), left/right walls are omitted.
        """
        n_layers = int(domain_cfg.get("boundary_layers", 0)) if domain_cfg else 0
        if n_layers <= 0:
            return None

        fluid_cfg = self.scene["fluid"]
        fluid_min = np.array(fluid_cfg["min"], dtype=np.float64)
        fluid_max = np.array(fluid_cfg["max"], dtype=np.float64)

        # Determine whether x is periodic (skip left/right walls if so)
        domain_has_x_periodic = (
            domain_cfg is not None
            and "min" in domain_cfg
            and "max" in domain_cfg
        )

        # Span of fluid region in x; use domain x for wall x-extent so that
        # boundary particles cover the entire fluid width plus a small pad.
        if domain_has_x_periodic:
            x_lo = float(np.array(domain_cfg["min"], dtype=np.float64)[0])
            x_hi = float(np.array(domain_cfg["max"], dtype=np.float64)[0])
        else:
            x_lo = float(fluid_min[0])
            x_hi = float(fluid_max[0])

        # x-periodic: avoid duplicate at x_hi (it wraps to x_lo)
        x_periodic = bool(domain_cfg.get("periodic_x", False)) if domain_cfg else False
        if x_periodic:
            x_vals = np.arange(x_lo, x_hi, self.spacing)
        else:
            x_vals = np.arange(x_lo, x_hi + 0.5 * self.spacing, self.spacing)

        periodic_z = bool(domain_cfg.get("periodic_z", False)) if domain_cfg else False

        positions: list[list[float]] = []
        if self.dim == 3:
            # Last actual particle positions (fluid_max is exclusive)
            y_last = float(fluid_max[1]) - self.spacing
            z_last = float(fluid_max[2]) - self.spacing

            # z span for y-wall particles: span the actual fluid z extent only
            if periodic_z:
                z_vals = np.arange(float(fluid_min[2]), float(fluid_max[2]), self.spacing)
            else:
                z_lo = float(fluid_min[2]) - n_layers * self.spacing
                z_hi = z_last + n_layers * self.spacing
                z_vals = np.arange(z_lo, z_hi + 0.5 * self.spacing, self.spacing)

            for layer in range(n_layers):
                # Full-spacing offset from last/first particle position
                offset = (layer + 1) * self.spacing
                y_bottom = float(fluid_min[1]) - offset   # below first particle
                y_top = y_last + offset                     # above last particle
                for x in x_vals:
                    for z in z_vals:
                        positions.append([x, y_bottom, z])
                        positions.append([x, y_top, z])

            if not periodic_z:
                # z walls only when z is not periodic
                y_lo = float(fluid_min[1]) - n_layers * self.spacing
                y_hi = y_last + n_layers * self.spacing
                y_vals_full = np.arange(y_lo, y_hi + 0.5 * self.spacing, self.spacing)
                for layer in range(n_layers):
                    offset = (layer + 1) * self.spacing
                    z_front = float(fluid_min[2]) - offset
                    z_back = z_last + offset
                    for x in x_vals:
                        for y in y_vals_full:
                            positions.append([x, y, z_front])
                            positions.append([x, y, z_back])
        else:
            y_last_2d = float(fluid_max[1]) - self.spacing
            for layer in range(n_layers):
                # Full-spacing offset from first/last fluid particle positions
                offset = (layer + 1) * self.spacing
                y_bottom = float(fluid_min[1]) - offset
                y_top = y_last_2d + offset
                for x in x_vals:
                    positions.append([x, y_bottom])
                    positions.append([x, y_top])

        if not positions:
            return None

        boundary_positions = np.unique(np.array(positions, dtype=np.float64), axis=0)

        particle_mass = self._compute_correct_mass()
        boundary = BoundaryModel(len(boundary_positions), self.rho0, particle_mass, self.dim)
        boundary.positions[:] = boundary_positions
        return boundary

    def _configure_kernel_lengths(self, neighbors_cfg: dict) -> None:
        support_radius = float(neighbors_cfg.get("support_radius", 0.0))
        smoothing_length = float(neighbors_cfg.get("h", 0.0))
        if support_radius > 0.0 and smoothing_length <= 0.0:
            smoothing_length = 0.5 * support_radius
        elif smoothing_length > 0.0 and support_radius <= 0.0:
            support_radius = 2.0 * smoothing_length
        elif support_radius <= 0.0 and smoothing_length <= 0.0:
            smoothing_length = 3.0 * self.spacing
            support_radius = 2.0 * smoothing_length

        if support_radius < 2.0 * smoothing_length:
            support_radius = 2.0 * smoothing_length

        self.h = float(smoothing_length)
        self.support_radius = float(support_radius)

    def _configure_stabilization(self) -> None:
        xsph_cfg = self.forces_cfg.get("xsph", {})
        self.enable_xsph = bool(xsph_cfg.get("enable", True))
        self.xsph_eps = float(xsph_cfg.get("eps", xsph_cfg.get("epsilon", 0.035)))

    def _configure_solver_parameters(self) -> None:
        solver_cfg = self.scene.get("solver", {})
        time_cfg = self.scene.get("time", {})
        self._nu = float(
            self.forces_cfg.get("viscosity", {}).get("nu", 0.0)
        ) if self.forces_cfg.get("viscosity", {}).get("enable", False) else 0.0

        self._cfl = float(time_cfg.get("cfl", 0.4))
        self._dt_min = float(time_cfg.get("dt_min", 1.0e-5))
        self._dt_max = float(time_cfg.get("dt_max", time_cfg.get("dt_fixed", 1.0e-4)))
        dt_fixed = float(time_cfg.get("dt_fixed", self._dt_max))
        mode = str(time_cfg.get("mode", "cfl")).lower()
        self._use_fixed_dt = mode == "fixed"
        self._dt_fixed = min(dt_fixed, self._dt_max)

        derived_c0 = self._derive_sound_speed(solver_cfg, time_cfg)
        self._c0 = float(solver_cfg.get("c0", derived_c0))
        default_k = self.rho0 * self._c0 * self._c0 / 7.0
        self.eos_k = float(solver_cfg.get("eos_k", default_k))

        dt_visc = 0.125 * self.spacing**2 / self._nu if self._nu > 0.0 else np.inf
        self._dt_max = min(self._dt_max, dt_visc)
        if self._use_fixed_dt:
            self.dt = max(self._dt_min, min(self._dt_fixed, self._dt_max))
        else:
            self.dt = min(self._dt_max, self._dt_fixed)

        diff_cfg = solver_cfg.get("density_diffusion", {})
        self._density_diffusion_enable = bool(diff_cfg.get("enable", True))
        delta_val = diff_cfg.get("delta", diff_cfg.get("alpha", 0.1))
        self._density_diffusion_delta = float(delta_val if delta_val is not None else 0.1)

    def _derive_sound_speed(self, solver_cfg: dict, time_cfg: dict) -> float:
        if "c0" in solver_cfg:
            return float(solver_cfg["c0"])

        target_mach = float(solver_cfg.get("target_mach", 0.1))
        velocity_scale = self._estimate_velocity_scale(time_cfg)
        if velocity_scale < 1e-3:
            velocity_scale = 0.8 * self.spacing / max(float(time_cfg.get("dt_max", 1e-4)), 1e-5)
        c0 = velocity_scale / max(target_mach, 1e-3)
        return max(c0, 10.0)

    def _estimate_velocity_scale(self, time_cfg: dict) -> float:
        fluid_cfg = self.scene["fluid"]
        domain_cfg = self.scene.get("domain", {})
        block_min = np.array(fluid_cfg["min"], dtype=np.float64)
        block_max = np.array(fluid_cfg["max"], dtype=np.float64)
        span = block_max - block_min
        vertical_extent = float(span[1]) if self.dim >= 2 else float(np.max(span))

        if "min" in domain_cfg and "max" in domain_cfg and len(domain_cfg["min"]) >= self.dim:
            domain_min = np.array(domain_cfg["min"], dtype=np.float64)
            domain_max = np.array(domain_cfg["max"], dtype=np.float64)
            if self.dim >= 2:
                vertical_extent = max(vertical_extent, float(domain_max[1] - domain_min[1]))
            else:
                vertical_extent = max(vertical_extent, float(np.max(domain_max - domain_min)))

        g_mag = float(np.linalg.norm(self.gravity_vec))
        vel_gravity = np.sqrt(max(2.0 * g_mag * max(vertical_extent, 1e-6), 0.0))
        initial_velocity = np.array(fluid_cfg.get("initial_velocity", [0.0] * self.dim)[: self.dim], dtype=np.float64)
        vel_init = float(np.linalg.norm(initial_velocity))
        vel_scene = float(self.scene.get("solver", {}).get("velocity_scale", 0.0))

        velocity_scale = max(vel_gravity, vel_init, vel_scene)
        if velocity_scale < 1e-3:
            dt_hint = float(time_cfg.get("dt_max", time_cfg.get("dt_fixed", 1.0e-4)))
            velocity_scale = self.spacing / max(dt_hint, 1e-5)

        return max(velocity_scale, 1e-3)

    def _create_time_step(self):
        """Create time step integrator from scene configuration."""
        if self.solver_name == "dfsph":
            from sph.dfsph import DFSPHTimeStep
            periodic_x = (self.x_min, self.x_max) if self.x_min is not None else None
            return DFSPHTimeStep(
                kernel=self.kernel,
                neighbor_search=self.neighbor_search,
                nu=self._nu,
                gravity=self.gravity_vec,
                periodic_x=periodic_x,
                eos_k=self.eos_k,
                c0=self._c0,
                density_diffusion_delta=self._density_diffusion_delta,
                density_diffusion_enable=self._density_diffusion_enable,
            )

        # Default: WCSPH
        return WCSPHTimeStep(
            kernel=self.kernel,
            neighbor_search=self.neighbor_search,
            eos_k=self.eos_k,
            nu=self._nu,
            gravity=self.gravity_vec,
        )

    def _compute_dt(self) -> float:
        """
        Compute adaptive time step from three stability conditions:
          1. Acoustic CFL:   dt <= cfl * dx / (c0 + vmax)
          2. Viscous limit:  dt <= 0.125 * dx^2 / nu
          3. Body-force:     dt <= cfl * sqrt(dx / f_max)

        Returns the minimum, clamped to dt_max.
        """
        if self._use_fixed_dt:
            return max(self._dt_min, min(self._dt_fixed, self._dt_max))

        cfl = self._cfl
        dx = self.spacing

        v_max = float(np.max(np.linalg.norm(self.fluid.velocities, axis=1))) if self.fluid.n else 0.0
        dt_acoustic = cfl * dx / (self._c0 + v_max + 1e-12)

        dt_visc = 0.125 * dx**2 / self._nu if self._nu > 0.0 else np.inf

        f_max = float(np.max(np.linalg.norm(self.fluid.accelerations, axis=1))) if self.fluid.n else 0.0
        dt_force = cfl * (dx / max(f_max, 1e-12)) ** 0.5 if f_max > 1e-12 else np.inf

        dt = min(dt_acoustic, dt_visc, dt_force, self._dt_max)
        return float(max(self._dt_min, dt))

    def step(self):
        """Advance simulation by one time step."""
        self.dt = self._compute_dt()
        solver_report = self.time_step.step(self.fluid, self.boundary, self.dt)
        self._last_solver_stats = self._normalize_solver_stats(solver_report)
        if isinstance(self._last_solver_stats, dict) and not self._last_solver_stats.get("converged", True):
            self._dt_max = max(self._dt_min, 0.5 * self._dt_max)
            if self._use_fixed_dt:
                self._dt_fixed = max(self._dt_min, 0.5 * self._dt_fixed)

        if self.enable_xsph:
            self._apply_xsph()

        # Apply periodic boundary conditions
        if self.x_min is not None and self.x_max is not None:
            self.time_step.apply_periodic_bc_x(self.fluid, self.x_min, self.x_max)
        if self.z_min is not None and self.z_max is not None:
            self.time_step.apply_periodic_bc_z(self.fluid, self.z_min, self.z_max)

        self.current_step += 1

    def run(self, n_steps: int, callback=None):
        """
        Run simulation for n_steps.

        Args:
            n_steps: Number of steps to run
            callback: Optional callback function called after each step
                     with signature: callback(step, fluid, boundary)
        """
        for i in range(n_steps):
            self.step()

            if callback is not None:
                callback(self.current_step, self.fluid, self.boundary)

    def _normalize_solver_stats(self, stats):
        if isinstance(stats, dict):
            return stats
        if isinstance(stats, tuple) and len(stats) == 2:
            return {"iter_cd": int(stats[0]), "iter_df": int(stats[1])}
        return {}

    @property
    def last_solver_stats(self) -> dict[str, int]:
        return self._last_solver_stats

    def density_breakdown(
        self,
        particle_index: int | None = None,
        mode: str = "max",
    ) -> dict:
        """
        Return detailed rho contributions for the selected particle.

        Args:
            particle_index: optional explicit particle id.
            mode: "max" (default) to inspect the densest particle, "min" for the
                  least dense particle. Ignored if particle_index is provided.
        """
        fluid = self.fluid
        n = fluid.n
        if n == 0:
            return {}

        pairs = fluid.neighbor_pairs
        if pairs is None:
            return {}

        self_contrib = fluid.rho_self.copy()
        ff_contrib = fluid.rho_ff.copy()
        fb_contrib = fluid.rho_fb.copy()
        totals = fluid.densities.copy()
        idx = particle_index
        if idx is None or idx < 0 or idx >= n:
            if mode == "min":
                idx = int(np.argmin(totals))
            else:
                idx = int(np.argmax(totals))

        fluid_counts = np.zeros(n, dtype=np.int32)
        if pairs.ff_i.size:
            fluid_counts += np.bincount(pairs.ff_i, minlength=n)
            fluid_counts += np.bincount(pairs.ff_j, minlength=n)

        boundary_counts = np.zeros(n, dtype=np.int32)
        if self.boundary is not None and pairs.fb_i.size:
            boundary_counts += np.bincount(pairs.fb_i, minlength=n)

        summary = {
            "rho_min": float(totals.min()),
            "rho_mean": float(totals.mean()),
            "rho_max": float(totals.max()),
        }
        particle = {
            "index": idx,
            "position": self.fluid.positions[idx].tolist(),
            "rho_self": float(self_contrib[idx]),
            "rho_ff": float(ff_contrib[idx]),
            "rho_fb": float(fb_contrib[idx]),
            "rho_total": float(totals[idx]),
            "fluid_neighbors": int(fluid_counts[idx]),
            "boundary_neighbors": int(boundary_counts[idx]),
        }
        if self.boundary is not None:
            particle["near_boundary"] = bool(boundary_counts[idx] > 0)

        return {"particle": particle, "summary": summary}

    def _apply_xsph(self) -> None:
        pairs = getattr(self.fluid, "neighbor_pairs", None)
        if pairs is None:
            return

        dv = np.zeros_like(self.fluid.velocities)
        rho = np.maximum(self.fluid.densities, 1e-8)

        if pairs.ff_i.size:
            W_ff = cubic_spline_W_batch(pairs.ff_dist, self.fluid.h)
            diff = self.fluid.velocities[pairs.ff_j] - self.fluid.velocities[pairs.ff_i]
            coeff_i = self.xsph_eps * self.fluid.mass * W_ff / rho[pairs.ff_j]
            scatter_add_2d(dv, pairs.ff_i, coeff_i[:, np.newaxis] * diff)

            coeff_j = self.xsph_eps * self.fluid.mass * W_ff / rho[pairs.ff_i]
            scatter_add_2d(dv, pairs.ff_j, coeff_j[:, np.newaxis] * (-diff))

        if self.boundary is not None and pairs.fb_i.size:
            W_fb = cubic_spline_W_batch(pairs.fb_dist, self.fluid.h)
            diff = self.boundary.velocities[pairs.fb_j] - self.fluid.velocities[pairs.fb_i]
            rho_b = np.maximum(self.boundary.densities[pairs.fb_j], 1e-8)
            psi = getattr(self.boundary, "psi", None)
            mass_term = psi[pairs.fb_j] if psi is not None and psi.size else self.boundary.mass
            coeff = self.xsph_eps * mass_term * W_fb / rho_b
            scatter_add_2d(dv, pairs.fb_i, coeff[:, np.newaxis] * diff)

        self.fluid.velocities += dv
