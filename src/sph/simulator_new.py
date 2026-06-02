"""
Simulator class for SPH simulations.

Loads scene configuration and runs the simulation loop.
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path

from sph.core.state_builder import (
    deduplicate_positions,
    sample_cylinder_x,
    sample_cylinder_y,
    sample_cylinder_z,
)
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


class PressureProbe:
    """
    Measures pressure/velocity at a fixed point by SPH kernel interpolation.
    SPlisHSPlasH equivalent: sampling locations.
    """

    def __init__(self, position: list | np.ndarray, name: str = "probe"):
        self.position = np.asarray(position, dtype=np.float64)
        self.name = name
        self.history: list[dict] = []

    def measure(
        self,
        fluid: FluidModel,
        kernel: CubicSplineKernel,
        step: int,
        t: float,
    ) -> dict:
        pos = fluid.positions
        prs = fluid.pressures
        vel = fluid.velocities
        dists = np.linalg.norm(pos - self.position, axis=1)
        mask = dists < 2.0 * kernel.h
        if mask.sum() == 0:
            result = {
                "step": step,
                "t": t,
                "name": self.name,
                "p": 0.0,
                "vx": 0.0,
                "vy": 0.0,
                "vz": 0.0,
            }
            self.history.append(result)
            return result

        w = cubic_spline_W_batch(dists[mask], kernel.h, fluid.dim)
        w_sum = float(np.sum(w)) + 1e-10
        p_interp = float(np.sum(prs[mask] * w) / w_sum)
        v_interp = np.sum(vel[mask] * w[:, np.newaxis], axis=0) / w_sum
        result = {
            "step": step,
            "t": t,
            "name": self.name,
            "p": p_interp,
            "vx": float(v_interp[0]),
            "vy": float(v_interp[1]) if fluid.dim > 1 else 0.0,
            "vz": float(v_interp[2]) if fluid.dim > 2 else 0.0,
        }
        self.history.append(result)
        return result


def compute_reynolds_number(fluid, nu: float, L_ref: float) -> tuple[float, str]:
    """Re = v_mean * L_ref / nu"""
    v_mean = float(np.linalg.norm(fluid.velocities, axis=1).mean())
    Re = v_mean * L_ref / max(nu, 1e-10)
    if Re < 2300:
        regime = "LAMINAR"
    elif Re < 4000:
        regime = "TRANSITIONAL"
    else:
        regime = "TURBULENT"
    return Re, regime


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

        # Get domain bounds — always store x_min/x_max when domain is defined.
        # self._periodic_x/z store whether wrapping is actually active.
        domain_cfg = self.scene.get("domain", {})
        if "min" in domain_cfg and "max" in domain_cfg:
            domain_min = np.array(domain_cfg["min"], dtype=np.float64)
            domain_max = np.array(domain_cfg["max"], dtype=np.float64)
            self.x_min = float(domain_min[0])
            self.x_max = float(domain_max[0])
            self._periodic_x = bool(domain_cfg.get("periodic_x", False))
            if self._periodic_x:
                periodic_x = (self.x_min, self.x_max)
            else:
                periodic_x = None
        else:
            self.x_min = None
            self.x_max = None
            self._periodic_x = False
            periodic_x = None

        # z-periodic BC (3D only) — use fluid extent so that neighbour pairs
        # match the original design.  apply_periodic_bc_z in dfsph.py keeps
        # positions inside [z_min, z_max) so cKDTree never sees a wrapped
        # value that equals L_z due to floating-point arithmetic.
        if self.dim == 3 and domain_cfg.get("periodic_z", False) and "min" in domain_cfg:
            fluid_cfg = self.scene["fluid"]
            z_min_f = float(np.array(fluid_cfg["min"], dtype=np.float64)[2])
            z_max_f = float(np.array(fluid_cfg["max"], dtype=np.float64)[2])
            self.z_min: float | None = z_min_f
            self.z_max: float | None = z_max_f
            self._periodic_z = True
            periodic_z: tuple[float, float] | None = (z_min_f, z_max_f)
        else:
            self.z_min = None
            self.z_max = None
            self._periodic_z = False
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

        # Inlet / outlet (optional)
        io_cfg = domain_cfg.get("inlet_outlet", {}) if domain_cfg else {}
        if io_cfg.get("enable", False):
            self._outlet_x = float(io_cfg.get("outlet_x", 0.19))
            self._inlet_x = float(io_cfg.get("inlet_x", 0.01))
            self._inlet_velocity = float(io_cfg.get("inlet_velocity", 0.0))  # 0 → use vx_mean
            self._outlet_pressure = float(io_cfg.get("outlet_pressure", 0.0))
            # Snapshot yz cross-section from the first x-slice of the initial fluid
            x_first = float(self.fluid.positions[:, 0].min())
            mask_first = np.abs(self.fluid.positions[:, 0] - x_first) < 0.5 * self.spacing
            self._inlet_yz = self.fluid.positions[mask_first, 1:].copy()
        else:
            self._outlet_x = None
            self._inlet_x = None
            self._inlet_velocity = 0.0
            self._outlet_pressure = 0.0
            self._inlet_yz = None

        self.current_step = 0
        self.t = 0.0

        # Pressure probes (optional scene entries)
        self.probes: list[PressureProbe] = []
        self._probe_csv_path: Path | None = None
        for probe_cfg in self.scene.get("probes", []):
            pos = probe_cfg.get("position")
            if pos is None:
                continue
            name = str(probe_cfg.get("name", f"probe_{len(self.probes)}"))
            self.probes.append(PressureProbe(pos, name=name))
        if self.probes:
            probe_dir = Path("out/probes")
            probe_dir.mkdir(parents=True, exist_ok=True)
            self._probe_csv_path = probe_dir / f"{self.scene_path.stem}.csv"
            self._write_probe_csv_header()

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
        fluid_cfg = self.scene["fluid"]
        fluid_type = fluid_cfg.get("type", "box")

        if fluid_cfg.get("sources"):
            blocks: list[np.ndarray] = []
            for src in fluid_cfg["sources"]:
                src_type = str(src.get("type", "block")).lower()
                center = src.get("center", [0.0, 0.0, 0.0])
                radius = float(src.get("radius", 0.05))
                height = float(src.get("height", 0.5))
                if src_type == "cylinder_x":
                    blocks.append(sample_cylinder_x(center, radius, height, self.spacing))
                elif src_type == "cylinder_y":
                    blocks.append(sample_cylinder_y(center, radius, height, self.spacing))
                elif src_type == "cylinder_z":
                    blocks.append(sample_cylinder_z(center, radius, height, self.spacing))
                else:
                    raise ValueError(f"Unsupported fluid source type '{src_type}'.")
            fluid_positions = (
                deduplicate_positions(np.vstack(blocks), self.spacing)
                if blocks else np.zeros((0, self.dim), dtype=np.float64)
            )
        elif fluid_type == "cylinder_x":
            fluid_positions = self._generate_cylinder_fluid(fluid_cfg)
        elif fluid_type == "cylinder_y":
            center = fluid_cfg.get("center", [0.0, 0.0, 0.0])
            fluid_positions = sample_cylinder_y(
                center,
                float(fluid_cfg["radius"]),
                float(fluid_cfg["height"]),
                self.spacing,
            )
        else:
            fluid_min = np.array(fluid_cfg["min"], dtype=np.float64)
            fluid_max = np.array(fluid_cfg["max"], dtype=np.float64)
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

        particle_mass = self._compute_correct_mass()
        fluid = FluidModel(len(fluid_positions), self.rho0, particle_mass, self.h, self.dim)
        fluid.positions[:] = fluid_positions

        initial_velocity = fluid_cfg.get("initial_velocity", [0.0, 0.0])
        fluid.velocities[:] = np.array(initial_velocity[:self.dim], dtype=np.float64)

        boundary = None
        domain_cfg = self.scene.get("domain", {})
        if "box_container" in domain_cfg:
            boundary = self._create_box_container_particles(domain_cfg)
        elif "boundary_layers" in domain_cfg or "cylinder_wall" in domain_cfg:
            boundary = self._create_boundary_particles(domain_cfg)

        # Load STL/OBJ mesh boundaries if present in scene
        if self.scene.get("boundaries"):
            boundary = self._create_stl_boundary(boundary)

        return fluid, boundary

    def _create_stl_boundary(self, existing: BoundaryModel | None) -> BoundaryModel | None:
        """Sample particle boundary from STL/OBJ mesh entries and merge with any existing boundary."""
        from sph.boundaries.mesh_particles import load_mesh_particle_representation_from_scene

        rep = load_mesh_particle_representation_from_scene(
            scene=self.scene,
            scene_path=self.scene_path,
            spacing=self.spacing,
            dim=self.dim,
        )
        if rep is None or rep.count == 0:
            return existing

        stl_positions = rep.project_particle_positions(
            dim=self.dim,
            spacing=self.spacing,
            deduplicate=False,
        )

        if existing is not None and existing.n > 0:
            combined = np.vstack([existing.positions, stl_positions])
        else:
            combined = stl_positions

        particle_mass = self._compute_correct_mass()
        boundary = BoundaryModel(len(combined), self.rho0, particle_mass, self.dim)
        boundary.positions[:] = combined
        return boundary

    def _generate_cylinder_fluid(self, fluid_cfg: dict) -> np.ndarray:
        """Generate fluid particles inside a cylinder with axis along x."""
        dx = self.spacing
        center = fluid_cfg.get("center", [0.0, 0.0])
        if "height" in fluid_cfg and len(center) >= 3:
            cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
            half = float(fluid_cfg["height"]) * 0.5
            axis_min = cx - half
            axis_max = cx + half
        else:
            axis_min = float(fluid_cfg["axis_min"])
            axis_max = float(fluid_cfg["axis_max"])
            cy, cz = float(center[0]), float(center[1])
        x_vals = np.arange(axis_min, axis_max, dx)
        if "height" in fluid_cfg and len(center) >= 3:
            cy, cz = float(center[1]), float(center[2])
        R = float(fluid_cfg["radius"])

        n_max = int(np.ceil(R / dx)) + 1
        offsets = np.arange(-n_max, n_max + 1) * dx

        positions: list[list[float]] = []
        for x in x_vals:
            for dy in offsets:
                for dz in offsets:
                    if dy * dy + dz * dz < R * R:
                        positions.append([x, cy + dy, cz + dz])
        return np.array(positions, dtype=np.float64)

    def _create_box_container_particles(self, domain_cfg: dict) -> BoundaryModel | None:
        """
        Place boundary particles lining a rectangular box container.

        Uses domain.min / domain.max as the interior extents so the floor and
        side walls cover the *full* container, not just the initial fluid block.
        open_top (default True) leaves the top face open for free surfaces.
        Corners are padded so adjacent faces overlap and leave no gap.
        """
        cfg = domain_cfg.get("box_container", {})
        n_layers = int(cfg.get("boundary_layers", 2))
        open_top  = bool(cfg.get("open_top", True))

        dm  = np.array(domain_cfg["min"], dtype=np.float64)
        dM  = np.array(domain_cfg["max"], dtype=np.float64)
        dx  = self.spacing
        buf = n_layers * dx  # padding so adjacent faces overlap at corners

        # Per-axis coordinate arrays for the interior faces
        def _span(lo, hi, margin=0.0):
            return np.arange(lo - margin, hi + margin + 0.5 * dx, dx)

        x_inner = _span(dm[0], dM[0])          # floor / top
        y_inner = _span(dm[1], dM[1] - dx)     # side walls (interior height)
        z_inner = _span(dm[2], dM[2])          # left / right walls
        # Extended spans for corner-filling on side / front / back walls
        y_ext   = _span(dm[1] - buf, dM[1] - dx)
        z_ext   = _span(dm[2] - buf, dM[2] + buf)
        x_ext   = _span(dm[0] - buf, dM[0] + buf)

        positions: list = []

        for lay in range(1, n_layers + 1):
            off = lay * dx

            # Floor  (–y)
            for x in x_inner:
                for z in z_inner:
                    positions.append([x, dm[1] - off, z])

            # Left wall  (–x)
            for y in y_ext:
                for z in z_ext:
                    positions.append([dm[0] - off, y, z])

            # Right wall  (+x)
            for y in y_ext:
                for z in z_ext:
                    positions.append([dM[0] + off, y, z])

            # Front wall  (–z)
            for x in x_ext:
                for y in y_ext:
                    positions.append([x, y, dm[2] - off])

            # Back wall  (+z)
            for x in x_ext:
                for y in y_ext:
                    positions.append([x, y, dM[2] + off])

            # Top  (+y) — only when not open
            if not open_top:
                for x in x_inner:
                    for z in z_inner:
                        positions.append([x, dM[1] + off, z])

        if not positions:
            return None

        bpos = np.unique(np.array(positions, dtype=np.float64), axis=0)
        particle_mass = self._compute_correct_mass()
        boundary = BoundaryModel(len(bpos), self.rho0, particle_mass, self.dim)
        boundary.positions[:] = bpos
        return boundary

    def _create_boundary_particles(self, domain_cfg: dict | None) -> BoundaryModel | None:
        """
        Create boundary particles adjacent to the fluid region.

        Places boundary layers just outside the fluid bounding box so that
        wall particles are always within support_radius of the fluid.
        For periodic-x flows (Poiseuille), left/right walls are omitted.
        """
        if domain_cfg is not None and "cylinder_wall" in domain_cfg:
            return self._create_cylinder_wall_particles(domain_cfg)

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

    def _create_cylinder_wall_particles(self, domain_cfg: dict) -> BoundaryModel | None:
        """Create boundary particles in a shell outside a cylinder aligned with x-axis."""
        wall_cfg = domain_cfg["cylinder_wall"]
        n_layers = int(wall_cfg.get("boundary_layers", 2))
        cy, cz = float(wall_cfg["center"][0]), float(wall_cfg["center"][1])
        R = float(wall_cfg["radius"])
        dx = self.spacing

        x_lo = float(np.array(domain_cfg["min"], dtype=np.float64)[0])
        x_hi = float(np.array(domain_cfg["max"], dtype=np.float64)[0])
        x_periodic = bool(domain_cfg.get("periodic_x", False))
        x_vals = np.arange(x_lo, x_hi, dx) if x_periodic else np.arange(x_lo, x_hi + 0.5 * dx, dx)

        R_outer = R + n_layers * dx
        n_max = int(np.ceil(R_outer / dx)) + 1
        offsets = np.arange(-n_max, n_max + 1) * dx

        positions: list[list[float]] = []
        for k_dy in offsets:
            for k_dz in offsets:
                r2 = k_dy * k_dy + k_dz * k_dz
                if R * R <= r2 < R_outer * R_outer:
                    for x in x_vals:
                        positions.append([x, cy + k_dy, cz + k_dz])

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
        self._density_floor_ratio = float(diff_cfg.get("floor_ratio", 0.85))

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
        if "min" in fluid_cfg and "max" in fluid_cfg:
            block_min = np.array(fluid_cfg["min"], dtype=np.float64)
            block_max = np.array(fluid_cfg["max"], dtype=np.float64)
            span = block_max - block_min
            vertical_extent = float(span[1]) if self.dim >= 2 else float(np.max(span))
        elif "radius" in fluid_cfg:
            vertical_extent = 2.0 * float(fluid_cfg["radius"])
        else:
            vertical_extent = float(self.spacing) * 10.0

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
            periodic_x = (self.x_min, self.x_max) if self._periodic_x else None
            periodic_z = (self.z_min, self.z_max) if self._periodic_z else None
            solver_cfg = self.scene.get("solver", {})
            dfsph_cfg = solver_cfg.get("dfsph", {})
            visc_factor   = float(dfsph_cfg.get("viscosity_factor", 3.0))
            eta_cd        = float(dfsph_cfg.get("eta_cd",        0.01))
            eta_df        = float(dfsph_cfg.get("eta_df",        0.01))
            max_iter_cd   = int(dfsph_cfg.get("max_iter_cd",    20))
            max_iter_df   = int(dfsph_cfg.get("max_iter_df",    20))
            # free-surface stabilisation sub-block
            fss_cfg       = dfsph_cfg.get("free_surface_stabilization", {})
            fss_enable    = bool(fss_cfg.get("enable", True))
            fss_mode      = str(fss_cfg.get("mode", "density_shift"))
            fss_strength  = float(fss_cfg.get("strength", 0.4))
            fss_target    = float(fss_cfg.get("target_ratio", 0.88))
            fss_max_raise = float(fss_cfg.get("max_raise_ratio", 0.12))
            return DFSPHTimeStep(
                kernel=self.kernel,
                neighbor_search=self.neighbor_search,
                nu=self._nu,
                viscosity_factor=visc_factor,
                gravity=self.gravity_vec,
                eta_cd=eta_cd,
                eta_df=eta_df,
                max_iter_cd=max_iter_cd,
                max_iter_df=max_iter_df,
                periodic_x=periodic_x,
                periodic_z=periodic_z,
                eos_k=self.eos_k,
                c0=self._c0,
                density_diffusion_delta=self._density_diffusion_delta,
                density_diffusion_enable=self._density_diffusion_enable,
                density_floor_ratio=self._density_floor_ratio,
                free_surface_stabilization_enable=fss_enable,
                free_surface_stabilization_mode=fss_mode,
                free_surface_stabilization_strength=fss_strength,
                free_surface_target_ratio=fss_target,
                free_surface_max_raise_ratio=fss_max_raise,
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
        stats = self._normalize_solver_stats(solver_report)
        # Merge full DFSPH report (iter counts, density errors, etc.)
        if hasattr(self.time_step, "last_report") and self.time_step.last_report:
            for k, v in self.time_step.last_report.items():
                stats.setdefault(k, v)
        # Reynolds number diagnostics
        nu = self._nu if self._nu > 0.0 else 0.001
        L_ref = float(self.scene.get("hydraulic_diameter", 0.14))
        if self.fluid.n > 0:
            Re, regime = compute_reynolds_number(self.fluid, nu, L_ref)
            stats["reynolds_number"] = Re
            stats["regime"] = regime
        self._last_solver_stats = stats
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

        # Inlet / outlet (open-domain flow)
        if self._inlet_yz is not None:
            self._apply_inlet_outlet()

        self.t += self.dt
        self.current_step += 1

        if self.probes and self.current_step % 10 == 0:
            for probe in self.probes:
                probe.measure(self.fluid, self.kernel, self.current_step, self.t)
            self._write_probe_csv()

    def _write_probe_csv_header(self) -> None:
        if self._probe_csv_path is None:
            return
        with open(self._probe_csv_path, "w", encoding="utf-8") as f:
            f.write("step,t,name,p,vx,vy,vz\n")

    def _write_probe_csv(self) -> None:
        if self._probe_csv_path is None or not self.probes:
            return
        rows: list[str] = []
        for probe in self.probes:
            if not probe.history:
                continue
            row = probe.history[-1]
            rows.append(
                f"{row['step']},{row['t']:.6f},{row['name']},"
                f"{row['p']:.6f},{row['vx']:.6f},{row['vy']:.6f},{row['vz']:.6f}\n"
            )
        if rows:
            with open(self._probe_csv_path, "a", encoding="utf-8") as f:
                f.writelines(rows)
            for probe in self.probes:
                probe.history.clear()

    @property
    def nu(self) -> float:
        return self._nu

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

    def _apply_inlet_outlet(self) -> None:
        """
        Open-domain inlet/outlet: remove particles past the outlet plane and
        inject a fresh layer at the inlet plane to maintain particle count.

        Outlet: particles at x > outlet_x are removed.
        Inlet:  when particles are removed, one cross-section layer is added at
                inlet_x with the current mean flow velocity in x.
        """
        fl = self.fluid

        # --- Outlet ---
        keep = fl.positions[:, 0] <= self._outlet_x
        n_removed = int(np.sum(~keep))
        if n_removed == 0:
            return

        # Compress all per-particle arrays by the keep mask
        _vec_attrs = ("positions", "velocities", "accelerations")
        _scl_attrs = ("densities", "pressures", "k_dfsph",
                      "p_cd_prev", "p_df_prev", "rho_self", "rho_ff", "rho_fb")
        for attr in _vec_attrs:
            setattr(fl, attr, getattr(fl, attr)[keep])
        for attr in _scl_attrs:
            if hasattr(fl, attr):
                setattr(fl, attr, getattr(fl, attr)[keep])
        fl.n = len(fl.positions)

        # --- Inlet: inject exactly n_removed particles ---
        # Use configured inlet_velocity; fall back to bulk mean if not set.
        vx_in = self._inlet_velocity if self._inlet_velocity != 0.0 else float(fl.velocities[:, 0].mean())
        # Inject the same count that left so particle count is conserved exactly.
        # Cycle through the cross-section template if n_removed differs from layer size.
        n_template = len(self._inlet_yz)
        n_add = n_removed
        template_idx = np.arange(n_add) % n_template  # wrap-around selection

        new_pos = np.empty((n_add, fl.dim), dtype=np.float64)
        new_pos[:, 0] = self._inlet_x
        new_pos[:, 1:] = self._inlet_yz[template_idx]

        new_vel = np.zeros((n_add, fl.dim), dtype=np.float64)
        new_vel[:, 0] = vx_in

        fl.positions = np.vstack([fl.positions, new_pos])
        fl.velocities = np.vstack([fl.velocities, new_vel])
        fl.accelerations = np.vstack(
            [fl.accelerations, np.zeros((n_add, fl.dim), dtype=np.float64)]
        )
        fl.densities = np.concatenate([fl.densities, np.full(n_add, fl.rho0, dtype=np.float64)])
        fl.pressures = np.concatenate([fl.pressures, np.zeros(n_add, dtype=np.float64)])
        for attr in ("k_dfsph", "p_cd_prev", "p_df_prev", "rho_self", "rho_ff", "rho_fb"):
            if hasattr(fl, attr):
                setattr(fl, attr, np.concatenate(
                    [getattr(fl, attr), np.zeros(n_add, dtype=np.float64)]
                ))
        fl.n = len(fl.positions)

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
