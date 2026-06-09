"""Simulation backend abstractions and shared dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Protocol

import numpy as np

from sph.core.scene import ExportSettings, SceneMetadata
from sph.core.state import ParticleState


@dataclass(slots=True)
class SimulationStateView:
    """Read-only snapshot of the current particle state for rendering/UI."""

    fluid_positions: np.ndarray
    fluid_velocities: np.ndarray
    fluid_density: np.ndarray
    fluid_pressure: np.ndarray
    fluid_speed: np.ndarray
    boundary_positions: np.ndarray
    boundary_velocities: np.ndarray
    domain_min: np.ndarray
    domain_max: np.ndarray
    fluid_rho0: float = 1000.0

    # --- Derived render fields (computed from the snapshot above) ------------
    @property
    def fluid_density_deviation(self) -> np.ndarray:
        """Relative density deviation (rho - rho0) / rho0 per fluid particle."""
        rho0 = self.fluid_rho0 if self.fluid_rho0 > 0.0 else 1000.0
        return (self.fluid_density - rho0) / rho0

    @property
    def fluid_low_density_mask(self) -> np.ndarray:
        rho0 = self.fluid_rho0 if self.fluid_rho0 > 0.0 else 1000.0
        return (self.fluid_density < 0.95 * rho0).astype(np.float64)

    @property
    def fluid_free_surface_score(self) -> np.ndarray:
        return np.clip(-self.fluid_density_deviation, 0.0, 1.0)

    @property
    def fluid_neighbor_counts(self) -> np.ndarray:
        # Neighbour counts are not carried on the render snapshot; expose a
        # zero-filled array of the correct length so render code can bind it.
        return np.zeros(self.fluid_density.shape[0], dtype=np.float64)

    @property
    def fluid_low_neighbor_mask(self) -> np.ndarray:
        return np.zeros(self.fluid_density.shape[0], dtype=np.float64)


@dataclass(slots=True)
class PerformanceSample:
    """Wall-clock performance information."""

    wall_time_ms: float
    steps_per_second: float
    rolling_wall_time_ms: float
    rolling_steps_per_second: float

    @property
    def fps(self) -> float:
        return self.steps_per_second


@dataclass(slots=True)
class RuntimeStats:
    """Backend-provided runtime statistics suitable for logging/export."""

    step: int
    dt: float
    solver: str
    scene_name: str
    velocity_max: float
    rho_min: float
    rho_mean: float
    rho_max: float
    rho_error_mean: float
    pressure_min: float
    pressure_mean: float
    pressure_max: float
    neighbor_min: int
    neighbor_mean: float
    neighbor_max: int
    stability: str
    fluid_count: int
    boundary_count: int
    wall_time_ms: float
    stage_timings_ms: Dict[str, float] = field(default_factory=dict)
    solver_metrics: Dict[str, float] = field(default_factory=dict)
    reynolds_number: float = 0.0
    regime: str = "LAMINAR"
    density_summary: object = None
    solver_health_summary: str = ""
    solver_health_notes: list = field(default_factory=list)
    interior_rel_err: float = 0.0


@dataclass(slots=True)
class StepDiagnostics:
    """UI-facing diagnostics derived from runtime stats."""

    step: int
    dt: float
    solver: str
    scene_name: str
    velocity_max: float
    rho_min: float
    rho_mean: float
    rho_max: float
    rho_error_mean: float
    pressure_min: float
    pressure_mean: float
    pressure_max: float
    neighbor_min: int
    neighbor_mean: float
    neighbor_max: int
    stability: str
    fluid_count: int
    boundary_count: int
    performance: PerformanceSample


class SimulationBackend(Protocol):
    """Backend interface so alternate (GPU, distributed, etc.) implementations can plug in."""

    scene_path: Path

    @property
    def solver_name(self) -> str: ...

    @property
    def scene_name(self) -> str: ...

    @property
    def scene_metadata(self) -> SceneMetadata: ...

    @property
    def export_settings(self) -> ExportSettings: ...

    @property
    def domain_min(self) -> np.ndarray: ...

    @property
    def domain_max(self) -> np.ndarray: ...

    @property
    def frame_index(self) -> int: ...

    def load_scene(self, scene_path: Path) -> None: ...

    def reset(self) -> None: ...

    def step(self) -> RuntimeStats: ...

    def state_view(self) -> SimulationStateView: ...

    def particle_state(self) -> ParticleState: ...

    def export_filename(self, kind: str, step: int) -> str: ...
