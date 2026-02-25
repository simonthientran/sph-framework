from __future__ import annotations

from typing import TYPE_CHECKING, List
from sph.boundaries.base import BoundaryBase

if TYPE_CHECKING:
    from sph.core.state import ParticleState
    from sph.core.simulator import SimConfig

class BoundaryManager:
    """
    Coordinates execution of all boundaries during the simulation loop.
    Maintains order of boundary operations: pre_step, apply_walls, post_step.
    """
    def __init__(self, boundaries: List[BoundaryBase] = None):
        self.boundaries = boundaries if boundaries is not None else []
        
    def add_boundary(self, boundary: BoundaryBase):
        self.boundaries.append(boundary)

    def pre_step(self, state: ParticleState, dt: float) -> None:
        for b in self.boundaries:
            b.pre_step(state, dt)

    def apply_walls(self, state: ParticleState, cfg: SimConfig, debug: bool = False) -> None:
        for b in self.boundaries:
            b.apply_walls(state, cfg, debug)

    def post_step(self, state: ParticleState) -> None:
        for b in self.boundaries:
            b.post_step(state)
