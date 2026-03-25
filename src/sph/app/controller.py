"""Thin controller mediating between UI controls and the simulation runner."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sph.core.backend import SimulationStateView
from sph.core.scene import SceneMetadata
from sph.core.simulation import SimulationRunner, StepResult


class SimulationController:
    """Owns the SimulationRunner and exposes lifecycle helpers for the UI."""

    def __init__(self, scene_path: Path):
        self._scene_path = Path(scene_path)
        self.runner = SimulationRunner(self._scene_path)

    @property
    def scene_path(self) -> Path:
        return self._scene_path

    @property
    def scene_name(self) -> str:
        return self.runner.scene_name

    @property
    def scene_metadata(self) -> SceneMetadata:
        return self.runner.scene_metadata

    @property
    def solver_name(self) -> str:
        return self.runner.solver_name

    @property
    def domain_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.runner.domain_min, self.runner.domain_max

    @property
    def state(self) -> SimulationStateView:
        return self.runner.state

    def load_scene(self, path: Path) -> None:
        self._scene_path = Path(path)
        self.runner.load_scene(self._scene_path)

    def reset(self) -> None:
        self.runner.reset()

    def step(self) -> StepResult:
        return self.runner.step()

    def export_snapshot(self, csv: bool = True, vtk: bool = True) -> dict[str, Path]:
        return self.runner.export_snapshot(csv=csv, vtk=vtk)
