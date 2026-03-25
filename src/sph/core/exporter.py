"""Export manager orchestrating CSV/VTK snapshot writing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Literal

from sph.core.backend import RuntimeStats, SimulationBackend
from sph.core.scene import ExportSettings, ExportTarget
from sph.io.csv_export import export_particles_csv
from sph.io.vtk_export import export_particles_vtk_legacy

ExportKind = Literal["csv", "vtk"]


@dataclass(slots=True)
class ExportOutcome:
    """Record of files written for a manual export trigger."""

    csv_path: Path | None = None
    vtk_path: Path | None = None

    def paths(self) -> Dict[str, Path]:
        result: Dict[str, Path] = {}
        if self.csv_path is not None:
            result["csv"] = self.csv_path
        if self.vtk_path is not None:
            result["vtk"] = self.vtk_path
        return result


class ExportManager:
    """Handles scheduled and manual exports without leaking into UI code."""

    def __init__(self, settings: ExportSettings | None = None) -> None:
        self.settings = settings or ExportSettings(
            csv=ExportTarget(False, 10, Path("out/csv")),
            vtk=ExportTarget(False, 10, Path("out/vtk")),
        )
        self._last_export_step: Dict[ExportKind, int] = {"csv": -1, "vtk": -1}
        self._initial_done = False

    # ------------------------------------------------------------------ lifecycle
    def on_scene_loaded(self, backend: SimulationBackend) -> None:
        self._last_export_step = {"csv": -1, "vtk": -1}
        self._initial_done = False
        self._export_initial_snapshot(backend)

    def on_reset(self, backend: SimulationBackend) -> None:
        self._last_export_step = {"csv": -1, "vtk": -1}
        self._initial_done = False
        self._export_initial_snapshot(backend)

    # ------------------------------------------------------------------ scheduled exports
    def maybe_export(self, backend: SimulationBackend, stats: RuntimeStats) -> None:
        if not self.settings.has_exports:
            return
        if self.settings.csv.enabled and stats.step % self.settings.csv.every == 0:
            self._export_csv(backend, stats.step)
        if self.settings.vtk.enabled and stats.step % self.settings.vtk.every == 0:
            self._export_vtk(backend, stats.step)

    # ------------------------------------------------------------------ manual exports
    def export_now(self, backend: SimulationBackend, kinds: Iterable[ExportKind], step: int | None = None) -> ExportOutcome:
        step_label = step if step is not None else backend.frame_index
        outcome = ExportOutcome()
        for kind in kinds:
            if kind == "csv":
                outcome.csv_path = self._export_csv(backend, step_label, force=True)
            elif kind == "vtk":
                outcome.vtk_path = self._export_vtk(backend, step_label, force=True)
        return outcome

    # ------------------------------------------------------------------ helpers
    def _export_initial_snapshot(self, backend: SimulationBackend) -> None:
        if self._initial_done or not self.settings.has_exports:
            return
        self._initial_done = True
        if self.settings.csv.enabled:
            self._export_csv(backend, 0, force=True)
        if self.settings.vtk.enabled:
            self._export_vtk(backend, 0, force=True)

    def _export_csv(self, backend: SimulationBackend, step: int, force: bool = False) -> Path | None:
        if not self.settings.csv.enabled and not force:
            return None
        if not force and self._last_export_step["csv"] == step:
            return None
        state = backend.particle_state()
        filename = backend.export_filename("csv", step)
        path = self.settings.csv.directory / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        export_particles_csv(path, state)
        self._last_export_step["csv"] = step
        return path

    def _export_vtk(self, backend: SimulationBackend, step: int, force: bool = False) -> Path | None:
        if not self.settings.vtk.enabled and not force:
            return None
        if not force and self._last_export_step["vtk"] == step:
            return None
        state = backend.particle_state()
        filename = backend.export_filename("vtk", step)
        path = self.settings.vtk.directory / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        export_particles_vtk_legacy(path, state)
        self._last_export_step["vtk"] = step
        return path
