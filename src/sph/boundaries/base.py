from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sph.core.state import ParticleState
    from sph.core.simulator import SimConfig

class BoundaryBase:
    """
    Base interface for all boundaries in the SPH framework.
    """
    def pre_step(self, state: ParticleState, dt: float) -> None:
        """
        Called before the solver step. 
        Useful for spawning particles, setting initial inflow velocity, etc.
        """
        pass

    def apply_walls(self, state: ParticleState, cfg: SimConfig, debug: bool = False) -> None:
        """
        Called after the integration step to enforce structural boundaries 
        (like wall push-out/reflection).
        """
        pass

    def post_step(self, state: ParticleState) -> None:
        """
        Called at the very end of the time step.
        Useful for cleaning up particles (like outflow teleportation).
        """
        pass
