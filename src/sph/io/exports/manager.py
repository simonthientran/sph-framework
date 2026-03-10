from __future__ import annotations

from pathlib import Path

import numpy as np

from sph.core.state import ParticleState
from sph.io.vtk_export import export_particles_vtk_legacy


class ExportManager:
    """
    Centralized export coordinator for step snapshots.

    Keeping this logic out of bootstrap avoids scattering per-export
    conditionals in the simulation loop and makes it easy to add formats later.
    """

    def __init__(self, *, vtk_enabled: bool, vtk_every: int, vtk_dir: Path) -> None:
        self.vtk_enabled = bool(vtk_enabled)
        self.vtk_every = int(max(1, vtk_every))
        self.vtk_dir = Path(vtk_dir)

    def maybe_export_initial(self, state: ParticleState) -> None:
        if not self.vtk_enabled:
            return
        export_particles_vtk_legacy(self.vtk_dir / "particles_000000.vtk", state)

    def maybe_export(self, *, step: int, state: ParticleState, neighbor_count: np.ndarray | None = None) -> None:
        if not self.vtk_enabled:
            return
        if int(step) % self.vtk_every != 0:
            return
        export_particles_vtk_legacy(
            self.vtk_dir / f"particles_{int(step):06d}.vtk",
            state,
            neighbor_count=neighbor_count,
        )

