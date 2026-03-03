from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sph.core.simulator import SimConfig
    from sph.core.state import ParticleState


class BoundaryBase:
    """
    Boundary lifecycle hook interface.

    Boundaries are executed by `BoundaryManager` in this strict order:
    1) `pre_step`     -> prepare inflow / source terms before solver integration.
    2) `apply_walls`  -> apply geometric constraints after integration.
    3) `post_step`    -> cleanup/removal after wall handling.
    """

    def pre_step(self, state: ParticleState, dt: float) -> None:
        return

    def apply_walls(self, state: ParticleState, cfg: SimConfig, *, debug: bool = False) -> None:
        return

    def post_step(self, state: ParticleState) -> None:
        return
