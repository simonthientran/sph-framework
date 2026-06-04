"""High-level simulation runner wiring the app layer to the backend."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import numpy as np

from sph.core.backend import (
    PerformanceSample,
    RuntimeStats,
    SimulationBackend,
    SimulationStateView,
    StepDiagnostics,
)
from sph.core.backends import NumbaCPUBackend
from sph.core.exporter import ExportManager
from sph.core.scene import SceneMetadata


@dataclass(slots=True)
class StepResult:
    """Container returned to UI/diagnostics clients each step."""

    diagnostics: StepDiagnostics
    runtime: RuntimeStats
    state: SimulationStateView


BackendFactory = Callable[[Path], SimulationBackend]


def list_scene_files(root: Path) -> list[Path]:
    """Return all JSON scene files inside ``root``."""

    root = Path(root)
    return sorted(p for p in root.glob("*.json") if p.is_file())


class RollingPerformanceTracker:
    """Computes rolling averages for wall-clock performance."""

    def __init__(self, window: int = 60):
        self.window = deque(maxlen=window)

    def reset(self) -> None:
        self.window.clear()

    def sample(self, value_ms: float) -> tuple[float, float]:
        self.window.append(value_ms)
        avg_ms = sum(self.window) / len(self.window)
        avg_sps = 1000.0 / avg_ms if avg_ms > 0.0 else 0.0
        return avg_ms, avg_sps


class SimulationRunner:
    """Thin wrapper around a backend to provide a stable app-facing API."""

    def __init__(self, scene_path: Path, backend_factory: BackendFactory | None = None,
                 backend_name: str | None = None):
        if backend_factory is None and backend_name is not None:
            if backend_name in ("numba_cuda", "cuda"):
                try:
                    from sph.core.backends.numba_cuda_backend import NumbaCUDABackend
                    backend_factory = NumbaCUDABackend
                except Exception:
                    backend_factory = NumbaCPUBackend
            else:
                backend_factory = NumbaCPUBackend
        self.backend_factory = backend_factory or NumbaCPUBackend
        self._scene_path = Path(scene_path)
        self._perf_tracker = RollingPerformanceTracker()
        self.backend = self._create_backend(self._scene_path)
        self._state_cache = self.backend.state_view()
        self._export_manager = ExportManager(self.backend.export_settings)
        self._export_manager.on_scene_loaded(self.backend)
        self._last_runtime_stats: RuntimeStats | None = None

    # ------------------------------------------------------------------ properties
    @property
    def scene_name(self) -> str:
        return self.backend.scene_name

    @property
    def solver_name(self) -> str:
        return self.backend.solver_name

    @property
    def scene_metadata(self) -> SceneMetadata:
        return self.backend.scene_metadata

    @property
    def domain_min(self) -> np.ndarray:
        return self.backend.domain_min

    @property
    def domain_max(self) -> np.ndarray:
        return self.backend.domain_max

    @property
    def state(self) -> SimulationStateView:
        return self._state_cache

    @property
    def scene_path(self) -> Path:
        return self._scene_path

    # ------------------------------------------------------------------ lifecycle
    def load_scene(self, scene_path: Path) -> None:
        self._scene_path = Path(scene_path)
        self.backend = self._create_backend(self._scene_path)
        self._state_cache = self.backend.state_view()
        self._export_manager = ExportManager(self.backend.export_settings)
        self._export_manager.on_scene_loaded(self.backend)
        self._perf_tracker.reset()

    def reset(self) -> None:
        self.backend = self._create_backend(self._scene_path)
        self._state_cache = self.backend.state_view()
        self._export_manager = ExportManager(self.backend.export_settings)
        self._export_manager.on_reset(self.backend)
        self._perf_tracker.reset()

    # ------------------------------------------------------------------ stepping
    def step(self) -> StepResult:
        stats = self.backend.step()
        perf = self._build_performance_sample(stats)
        diagnostics = self._build_diagnostics(stats, perf)
        self._state_cache = self.backend.state_view()
        self._export_manager.maybe_export(self.backend, stats)
        self._last_runtime_stats = stats
        return StepResult(diagnostics=diagnostics, runtime=stats, state=self._state_cache)

    # ------------------------------------------------------------------ exports / observability
    def export_snapshot(self, csv: bool = True, vtk: bool = True) -> dict[str, Path]:
        kinds: list[str] = []
        if csv:
            kinds.append("csv")
        if vtk:
            kinds.append("vtk")
        if not kinds:
            return {}
        outcome = self._export_manager.export_now(self.backend, kinds, step=self.backend.frame_index)
        return outcome.paths()

    # ------------------------------------------------------------------ helpers
    def _create_backend(self, scene_path: Path) -> SimulationBackend:
        return self.backend_factory(Path(scene_path))

    def _build_performance_sample(self, stats: RuntimeStats) -> PerformanceSample:
        avg_ms, avg_sps = self._perf_tracker.sample(stats.wall_time_ms)
        inst_sps = 1000.0 / stats.wall_time_ms if stats.wall_time_ms > 0.0 else 0.0
        return PerformanceSample(
            wall_time_ms=stats.wall_time_ms,
            steps_per_second=inst_sps,
            rolling_wall_time_ms=avg_ms,
            rolling_steps_per_second=avg_sps,
        )

    def _build_diagnostics(self, stats: RuntimeStats, perf: PerformanceSample) -> StepDiagnostics:
        return StepDiagnostics(
            step=stats.step,
            dt=stats.dt,
            solver=stats.solver,
            scene_name=stats.scene_name,
            velocity_max=stats.velocity_max,
            rho_min=stats.rho_min,
            rho_mean=stats.rho_mean,
            rho_max=stats.rho_max,
            rho_error_mean=stats.rho_error_mean,
            pressure_min=stats.pressure_min,
            pressure_mean=stats.pressure_mean,
            pressure_max=stats.pressure_max,
            neighbor_min=stats.neighbor_min,
            neighbor_mean=stats.neighbor_mean,
            neighbor_max=stats.neighbor_max,
            stability=stats.stability,
            fluid_count=stats.fluid_count,
            boundary_count=stats.boundary_count,
            performance=perf,
        )


def default_backend_factory() -> BackendFactory:
    return NumbaCPUBackend
