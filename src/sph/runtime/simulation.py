"""Structured SPH simulation driver with a SPlisHSPlasH-inspired pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from sph.core.state import ParticleState
from sph.neighbors.spatial_hash import SpatialHash
from sph.runtime.diagnostics import DiagnosticsConfig, DiagnosticsManager
from sph.runtime.solver_wcsph import WCSPHSolver
from sph.runtime.timestep import TimeStepConstraints, TimeStepController
from sph.sph.viscosity import viscosity_acceleration_laplace_eq23
from sph.sph.xsph import xsph_velocity_correction


@dataclass(slots=True)
class SimulationConfig:
    """Physical and numerical parameters for the WCSPH simulation."""

    support_radius: float
    rho0: float
    eos_k: float
    gravity: np.ndarray
    viscosity_nu: float = 0.0
    enable_xsph: bool = False
    xsph_eps: float = 0.05
    dt_min: float = 1e-5
    dt_max: float = 5e-3
    cfl_factor: float = 0.25
    force_factor: float = 0.3
    viscosity_factor: float = 0.125
    pipe_height: float = 1.0
    velocity_profile_bins: int = 16
    clamp_negative_pressure: bool = True


class SPHSimulation:
    """High-level orchestrator exposing the requested pipeline stages."""

    def __init__(self, state: ParticleState, config: SimulationConfig):
        state.validate()
        self.state = state
        self.config = config

        gravity = np.asarray(config.gravity, dtype=np.float64)
        if gravity.shape != (state.dim,):
            raise ValueError("gravity must match simulation dimension")
        self.gravity = gravity

        self.neighbor_search = SpatialHash(support_radius=float(config.support_radius), dim=state.dim)
        self.solver = WCSPHSolver(
            rho0=float(config.rho0),
            eos_k=float(config.eos_k),
            support_radius=float(config.support_radius),
            clamp_negative_pressure=bool(config.clamp_negative_pressure),
        )

        constraints = TimeStepConstraints(
            support_radius=float(config.support_radius),
            dt_min=float(config.dt_min),
            dt_max=float(config.dt_max),
            cfl_factor=float(config.cfl_factor),
            force_factor=float(config.force_factor),
            viscosity_factor=float(config.viscosity_factor),
        )
        self.time_step_controller = TimeStepController(constraints)
        self.dt = float(self.time_step_controller.dt)

        diag_cfg = DiagnosticsConfig(
            rho0=float(config.rho0),
            pipe_height=float(config.pipe_height),
            velocity_profile_bins=int(config.velocity_profile_bins),
        )
        self.diagnostics = DiagnosticsManager(diag_cfg)

        self._last_non_pressure = np.zeros_like(self.state.vel)
        self._last_pressure = np.zeros_like(self.state.vel)

    # Pipeline -----------------------------------------------------------
    def step(self) -> Dict[str, object]:
        self.update_neighbors()
        self.solver.compute_density(self.state, self.neighbor_search)
        non_pressure = self.compute_non_pressure_forces()
        self.solver.compute_pressure(self.state)
        pressure = self.compute_pressure_forces()
        total_acc = non_pressure + pressure
        self.integrate(total_acc)
        self.apply_xsph()
        self.update_time_step(total_acc)
        diagnostics = self.run_diagnostics()
        return diagnostics

    def update_neighbors(self) -> None:
        self.neighbor_search.build(self.state.pos)

    def compute_non_pressure_forces(self) -> np.ndarray:
        state = self.state
        n, dim = state.n, state.dim
        acc = np.zeros((n, dim), dtype=np.float64)

        fluid = ~state.is_boundary
        acc[fluid] += self.gravity

        if self.config.viscosity_nu > 0.0:
            acc += viscosity_acceleration_laplace_eq23(
                state=state,
                neighbor_search=self.neighbor_search,
                h=float(self.config.support_radius),
                nu=float(self.config.viscosity_nu),
            )

        acc[~fluid] = 0.0
        self._last_non_pressure = acc
        return acc

    def compute_pressure_forces(self) -> np.ndarray:
        acc_p = self.solver.compute_pressure_forces(self.state, self.neighbor_search)
        acc_p[self.state.is_boundary] = 0.0
        self._last_pressure = acc_p
        return acc_p

    def integrate(self, acceleration: np.ndarray) -> None:
        dt = float(self.dt)
        state = self.state
        fluid = ~state.is_boundary

        state.acc[:] = acceleration
        state.vel[fluid] += dt * acceleration[fluid]
        state.pos[fluid] += dt * state.vel[fluid]

    def apply_xsph(self) -> None:
        if not self.config.enable_xsph:
            return

        dv = xsph_velocity_correction(
            state=self.state,
            neighbor_search=self.neighbor_search,
            h=float(self.config.support_radius),
            eps=float(self.config.xsph_eps),
        )
        fluid = ~self.state.is_boundary
        self.state.vel[fluid] += dv[fluid]

    def update_time_step(self, acceleration: np.ndarray) -> None:
        fluid = ~self.state.is_boundary
        vel = self.state.vel[fluid]
        acc = acceleration[fluid]
        dt = self.time_step_controller.estimate(velocities=vel, accelerations=acc, nu=float(self.config.viscosity_nu))
        self.dt = dt

    def run_diagnostics(self) -> Dict[str, object]:
        diag = self.diagnostics.collect(self.state, self.neighbor_search)
        diag.update(
            {
                "dt": self.dt,
                "non_pressure_acc_max": float(np.max(np.linalg.norm(self._last_non_pressure, axis=1))) if self.state.n else 0.0,
                "pressure_acc_max": float(np.max(np.linalg.norm(self._last_pressure, axis=1))) if self.state.n else 0.0,
            }
        )
        return diag
