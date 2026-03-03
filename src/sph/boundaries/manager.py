from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from sph.boundaries.base import BoundaryBase

if TYPE_CHECKING:
    from sph.core.simulator import SimConfig
    from sph.core.state import ParticleState


class BoundaryManager:
    """
    Coordinates boundary execution order for a simulation step.
    """

    def __init__(self, boundaries: Iterable[BoundaryBase] | None = None):
        self.boundaries: list[BoundaryBase] = list(boundaries or [])

    def add_boundary(self, boundary: BoundaryBase) -> None:
        self.boundaries.append(boundary)

    def pre_step(self, state: ParticleState, dt: float) -> None:
        for boundary in self.boundaries:
            boundary.pre_step(state, dt)

    def apply_walls(self, state: ParticleState, cfg: SimConfig, *, debug: bool = False) -> None:
        for boundary in self.boundaries:
            boundary.apply_walls(state, cfg, debug=debug)

    def post_step(self, state: ParticleState) -> None:
        for boundary in self.boundaries:
            boundary.post_step(state)
