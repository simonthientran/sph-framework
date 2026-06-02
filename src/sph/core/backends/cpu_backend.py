"""NumPy/Numba CPU backend implementing the SimulationBackend protocol."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict

import numpy as np

from sph.core.backend import RuntimeStats, SimulationBackend, SimulationStateView
from sph.core.scene import ExportSettings, SceneMetadata, parse_export_settings, parse_scene_metadata
from sph.core.state import ParticleState
from sph.simulator_new import Simulator


class NumbaCPUBackend:
    """Current CPU/Numba backend built on the refactored solver pipeline."""

    def __init__(self, scene_path: Path):
        self.scene_path = Path(scene_path)
        self._scene_data: dict | None = None
        self._scene_metadata: SceneMetadata | None = None
        self._export_settings: ExportSettings | None = None
        self._solver_type = "wcsph"
        self.sim: Simulator | None = None
        self._domain_min = np.zeros(2, dtype=np.float64)
        self._domain_max = np.ones(2, dtype=np.float64)
        self.load_scene(self.scene_path)

    # ------------------------------------------------------------------ properties
    @property
    def solver_name(self) -> str:
        return self._solver_type.upper()

    @property
    def scene_name(self) -> str:
        if self._scene_metadata:
            return self._scene_metadata.name
        return self.scene_path.stem

    @property
    def scene_metadata(self) -> SceneMetadata:
        assert self._scene_metadata is not None
        return self._scene_metadata

    @property
    def export_settings(self) -> ExportSettings:
        assert self._export_settings is not None
        return self._export_settings

    @property
    def domain_min(self) -> np.ndarray:
        return self._domain_min

    @property
    def domain_max(self) -> np.ndarray:
        return self._domain_max

    @property
    def frame_index(self) -> int:
        if self.sim is None:
            return 0
        return int(self.sim.current_step)

    # ------------------------------------------------------------------ lifecycle
    def load_scene(self, scene_path: Path) -> None:
        self.scene_path = Path(scene_path)
        with self.scene_path.open("r", encoding="utf-8") as handle:
            self._scene_data = json.load(handle)
        self._scene_metadata = parse_scene_metadata(self._scene_data, self.scene_path)
        self._export_settings = parse_export_settings(self._scene_data, self.scene_path.parent)
        self._solver_type = self._scene_data.get("solver", {}).get("type", "wcsph").lower()
        self._instantiate_simulator()

    def reset(self) -> None:
        self._instantiate_simulator()

    # ------------------------------------------------------------------ simulation
    def step(self) -> RuntimeStats:
        if self.sim is None:
            raise RuntimeError("Backend not initialized.")
        start = time.perf_counter()
        self.sim.step()
        wall_time_ms = (time.perf_counter() - start) * 1000.0
        return self._build_stats(wall_time_ms)

    def state_view(self) -> SimulationStateView:
        if self.sim is None:
            raise RuntimeError("Backend not initialized.")

        fluid = self.sim.fluid
        boundary = self.sim.boundary
        fluid_pos = fluid.positions.copy()
        fluid_vel = fluid.velocities.copy()
        fluid_rho = fluid.densities.copy()
        fluid_pressure = fluid.pressures.copy()
        fluid_speed = np.linalg.norm(fluid_vel, axis=1) if fluid_vel.size else np.zeros(0, dtype=np.float64)

        if boundary is not None and boundary.n:
            boundary_pos = boundary.positions.copy()
            boundary_vel = boundary.velocities.copy()
        else:
            boundary_pos = np.zeros((0, fluid.dim), dtype=np.float64)
            boundary_vel = np.zeros((0, fluid.dim), dtype=np.float64)

        return SimulationStateView(
            fluid_positions=fluid_pos,
            fluid_velocities=fluid_vel,
            fluid_density=fluid_rho,
            fluid_pressure=fluid_pressure,
            fluid_speed=fluid_speed,
            boundary_positions=boundary_pos,
            boundary_velocities=boundary_vel,
            domain_min=self._domain_min.copy(),
            domain_max=self._domain_max.copy(),
        )

    def particle_state(self) -> ParticleState:
        if self.sim is None:
            raise RuntimeError("Backend not initialized.")

        fluid = self.sim.fluid
        dim = fluid.dim
        boundary = self.sim.boundary

        pos_parts = [fluid.positions.copy()]
        vel_parts = [fluid.velocities.copy()]
        acc_parts = [fluid.accelerations.copy()]
        mass_parts = [np.full(fluid.n, fluid.mass, dtype=np.float64)]
        rho_parts = [fluid.densities.copy()]
        p_parts = [fluid.pressures.copy()]
        is_boundary_parts = [np.zeros(fluid.n, dtype=bool)]

        if boundary is not None and boundary.n:
            pos_parts.append(boundary.positions.copy())
            vel_parts.append(boundary.velocities.copy())
            acc_parts.append(np.zeros((boundary.n, dim), dtype=np.float64))
            mass_parts.append(np.full(boundary.n, boundary.mass, dtype=np.float64))
            rho_parts.append(boundary.densities.copy())
            p_parts.append(boundary.pressures.copy())
            is_boundary_parts.append(np.ones(boundary.n, dtype=bool))

        pos = np.vstack(pos_parts) if pos_parts else np.zeros((0, dim), dtype=np.float64)
        vel = np.vstack(vel_parts) if vel_parts else np.zeros((0, dim), dtype=np.float64)
        acc = np.vstack(acc_parts) if acc_parts else np.zeros((0, dim), dtype=np.float64)
        mass = np.concatenate(mass_parts) if mass_parts else np.zeros(0, dtype=np.float64)
        rho = np.concatenate(rho_parts) if rho_parts else np.zeros(0, dtype=np.float64)
        p = np.concatenate(p_parts) if p_parts else np.zeros(0, dtype=np.float64)
        is_boundary = np.concatenate(is_boundary_parts) if is_boundary_parts else np.zeros(0, dtype=bool)

        return ParticleState(
            dim=dim,
            pos=pos,
            vel=vel,
            acc=acc,
            mass=mass,
            rho=rho,
            p=p,
            is_boundary=is_boundary,
        )

    def export_filename(self, kind: str, step: int) -> str:
        suffix = {"csv": ".csv", "vtk": ".vtk"}.get(kind, ".dat")
        return f"particles_step_{step:04d}{suffix}"

    # ------------------------------------------------------------------ helpers
    def _instantiate_simulator(self) -> None:
        self.sim = Simulator(self.scene_path, solver=self._solver_type)
        self._solver_type = self.sim.solver_name
        self._update_domain_bounds()

    def _update_domain_bounds(self) -> None:
        assert self.sim is not None
        scene_dom = (self._scene_data or {}).get("domain", {})
        if "min" in scene_dom and "max" in scene_dom:
            self._domain_min = np.array(scene_dom["min"][: self.sim.dim], dtype=np.float64)
            self._domain_max = np.array(scene_dom["max"][: self.sim.dim], dtype=np.float64)
            return

        pts = self.sim.fluid.positions
        if self.sim.boundary is not None and self.sim.boundary.n:
            pts = np.vstack([pts, self.sim.boundary.positions])
        if pts.size == 0:
            self._domain_min = np.zeros(self.sim.dim, dtype=np.float64)
            self._domain_max = np.ones(self.sim.dim, dtype=np.float64)
            return
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        pad = 0.05 * (maxs - mins + 1e-6)
        self._domain_min = mins - pad
        self._domain_max = maxs + pad

    def _build_stats(self, wall_time_ms: float) -> RuntimeStats:
        assert self.sim is not None
        fluid = self.sim.fluid
        rho = fluid.densities
        pressure = fluid.pressures
        vel_norm = np.linalg.norm(fluid.velocities, axis=1) if fluid.n else np.zeros(0)

        rho_min = float(rho.min()) if rho.size else 0.0
        rho_max = float(rho.max()) if rho.size else 0.0
        rho_mean = float(rho.mean()) if rho.size else 0.0
        rho_err = np.abs(rho - fluid.rho0) / fluid.rho0 if rho.size else np.zeros(0)
        rho_err_mean = float(rho_err.mean()) if rho_err.size else 0.0

        p_min = float(pressure.min()) if pressure.size else 0.0
        p_max = float(pressure.max()) if pressure.size else 0.0
        p_mean = float(pressure.mean()) if pressure.size else 0.0

        v_max = float(vel_norm.max()) if vel_norm.size else 0.0

        neigh_min, neigh_mean, neigh_max = self._neighbor_stats()

        if rho_err_mean < 0.01:
            stability = "pass"
        elif rho_err_mean < 0.03:
            stability = "warn"
        else:
            stability = "fail"

        solver_label = self.solver_name
        solver_metrics: Dict[str, float] = {}
        stats = getattr(self.sim, "last_solver_stats", {})
        stats_dict = stats if isinstance(stats, dict) else {}
        if stats_dict:
            extras = []
            if "iter_cd" in stats_dict:
                extras.append(f"cd={stats_dict['iter_cd']}")
            if "iter_df" in stats_dict:
                extras.append(f"df={stats_dict['iter_df']}")
            if extras:
                solver_label = f"{solver_label} ({', '.join(extras)})"
            if "iter_cd" in stats_dict:
                solver_metrics["iter_cd"] = float(stats_dict["iter_cd"])
            if "iter_df" in stats_dict:
                solver_metrics["iter_df"] = float(stats_dict["iter_df"])
            for key in ("rho_error_mean", "rho_error_max", "div_error_mean"):
                if key in stats_dict:
                    solver_metrics[key] = float(stats_dict[key])
            solver_metrics["cd_converged"] = float(bool(stats_dict.get("cd_converged", True)))
            solver_metrics["df_converged"] = float(bool(stats_dict.get("df_converged", True)))

        re_value = float(stats_dict.get("reynolds_number", 0.0))
        regime_value = str(stats_dict.get("regime", "LAMINAR"))

        return RuntimeStats(
            step=int(self.sim.current_step),
            dt=float(self.sim.dt),
            solver=solver_label,
            scene_name=self.scene_name,
            velocity_max=v_max,
            rho_min=rho_min,
            rho_mean=rho_mean,
            rho_max=rho_max,
            rho_error_mean=rho_err_mean,
            pressure_min=p_min,
            pressure_mean=p_mean,
            pressure_max=p_max,
            neighbor_min=neigh_min,
            neighbor_mean=neigh_mean,
            neighbor_max=neigh_max,
            stability=stability,
            fluid_count=int(fluid.n),
            boundary_count=int(self.sim.boundary.n if self.sim.boundary is not None else 0),
            wall_time_ms=wall_time_ms,
            stage_timings_ms={"total": wall_time_ms},
            solver_metrics=solver_metrics,
            reynolds_number=re_value,
            regime=regime_value,
        )

    def _neighbor_stats(self) -> tuple[int, float, int]:
        assert self.sim is not None
        fluid = self.sim.fluid
        pairs = fluid.neighbor_pairs
        if pairs is None or fluid.n == 0:
            return 0, 0.0, 0

        counts = np.zeros(fluid.n, dtype=np.int32)
        if pairs.ff_i.size:
            counts += np.bincount(pairs.ff_i, minlength=fluid.n)
            counts += np.bincount(pairs.ff_j, minlength=fluid.n)
        if pairs.fb_i.size:
            counts += np.bincount(pairs.fb_i, minlength=fluid.n)

        return (
            int(counts.min()) if counts.size else 0,
            float(counts.mean()) if counts.size else 0.0,
            int(counts.max()) if counts.size else 0,
        )
