from __future__ import annotations
import numpy as np

from sph.boundaries.base import BoundaryBase

class OutflowBoundary(BoundaryBase):
    """
    Removes particles entering a designated AABB. Removed particles
    are ported outside the domain (e.g., x=1e9) and become inactive pool members.
    """
    def __init__(self, region_min: list[float], region_max: list[float]):
        self.region_min = np.array(region_min, dtype=np.float64)
        self.region_max = np.array(region_max, dtype=np.float64)
        self.teleport_pos = 1e9

    def post_step(self, state) -> None:
        fluid_ids = state.fluid_indices
        if fluid_ids.size == 0:
            return

        pos = state.pos[fluid_ids]
        
        # Find particles in the outflow region
        in_region = (pos >= self.region_min) & (pos <= self.region_max)
        in_region_mask = np.all(in_region, axis=1)
        active_mask = pos[:, 0] < 1e8
        to_remove = in_region_mask & active_mask
        
        if np.any(to_remove):
            remove_ids = fluid_ids[to_remove]
            
            # Teleport outside active area
            # Put them far away and zero velocity
            n_remove = remove_ids.size
            dim = state.dim
            
            # Spread them slightly in the inactive pool to avoid degenerate bounds
            # if someone does an AABB on all particles
            inactive_pos = np.zeros((n_remove, dim), dtype=np.float64)
            inactive_pos[:, 0] = self.teleport_pos + np.arange(n_remove)
            if dim > 1:
                inactive_pos[:, 1] = self.teleport_pos
            if dim > 2:
                inactive_pos[:, 2] = self.teleport_pos

            state.pos[remove_ids] = inactive_pos
            state.vel[remove_ids] = 0.0
            state.acc[remove_ids] = 0.0
            state.p[remove_ids] = 0.0
