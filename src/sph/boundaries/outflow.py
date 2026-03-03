from __future__ import annotations

import numpy as np

from sph.boundaries.base import BoundaryBase

_INACTIVE_BASE = 1e9


class OutflowBoundary(BoundaryBase):
    """
    Open outflow with light sponge damping + particle removal region.
    """

    def __init__(
        self,
        region_min: list[float],
        region_max: list[float],
        *,
        sponge_strength: float = 0.15,
    ):
        self.region_min = np.asarray(region_min, dtype=np.float64)
        self.region_max = np.asarray(region_max, dtype=np.float64)
        self.sponge_strength = float(max(0.0, min(1.0, sponge_strength)))
        self._removed_total = 0

    @property
    def removed_total(self) -> int:
        return int(self._removed_total)

    def pre_step(self, state, dt: float) -> None:
        # Sponge zone: gently damp pressure for particles already inside outflow box.
        fluid_ids = state.fluid_indices
        if fluid_ids.size == 0:
            return
        pos = state.pos[fluid_ids]
        active = pos[:, 0] < 1e8
        if not np.any(active):
            return
        ids = fluid_ids[active]
        apos = state.pos[ids]
        mask = np.all((apos >= self.region_min) & (apos <= self.region_max), axis=1)
        if np.any(mask):
            sid = ids[mask]
            state.p[sid] *= (1.0 - self.sponge_strength)

    def post_step(self, state) -> None:
        fluid_ids = state.fluid_indices
        if fluid_ids.size == 0:
            return
        pos = state.pos[fluid_ids]
        active = pos[:, 0] < 1e8
        if not np.any(active):
            return
        active_ids = fluid_ids[active]
        apos = state.pos[active_ids]
        in_region = np.all((apos >= self.region_min) & (apos <= self.region_max), axis=1)
        if not np.any(in_region):
            return

        remove_ids = active_ids[in_region]
        n_remove = int(remove_ids.size)
        dim = state.dim
        inactive_pos = np.full((n_remove, dim), _INACTIVE_BASE, dtype=np.float64)
        inactive_pos[:, 0] = _INACTIVE_BASE + np.arange(self._removed_total, self._removed_total + n_remove)
        self._removed_total += n_remove

        state.pos[remove_ids] = inactive_pos
        state.vel[remove_ids] = 0.0
        state.acc[remove_ids] = 0.0
        state.p[remove_ids] = 0.0
        state.rho[remove_ids] = 0.0
