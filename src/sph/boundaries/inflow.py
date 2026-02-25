from __future__ import annotations
import numpy as np

from sph.boundaries.base import BoundaryBase

class InflowBoundary(BoundaryBase):
    """
    Injects particles and sets a constant velocity for particles in a specified region.
    Teleports inactive particles (pos > 1e8) into the region when empty spaces occur.
    """
    def __init__(self, region_min: list[float], region_max: list[float], velocity: list[float], spacing: float):
        self.region_min = np.array(region_min, dtype=np.float64)
        self.region_max = np.array(region_max, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.spacing = spacing
        self.grid_points = self._precompute_grid()

    def _precompute_grid(self) -> np.ndarray:
        dim = len(self.region_min)
        axes = []
        for d in range(dim):
            pmin = self.region_min[d]
            pmax = self.region_max[d]
            # +1e-12 to ensure inclusive-ish range
            axes.append(np.arange(pmin, pmax + 1e-12, self.spacing))
            
        if dim == 2:
            X, Y = np.meshgrid(axes[0], axes[1], indexing="xy")
            return np.stack([X.ravel(), Y.ravel()], axis=1)
        elif dim == 3:
            X, Y, Z = np.meshgrid(axes[0], axes[1], axes[2], indexing="xy")
            return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        else:
            return np.zeros((0, dim))

    def pre_step(self, state, dt: float) -> None:
        fluid_ids = state.fluid_indices
        if fluid_ids.size == 0 or self.grid_points.shape[0] == 0:
            return

        pos = state.pos[fluid_ids]
        
        # 1) Force velocity for any active particles inside the inflow region
        in_region = (pos >= self.region_min) & (pos <= self.region_max)
        in_region_mask = np.all(in_region, axis=1)
        # Exclude particles that are far away (inactive)
        active_mask = pos[:, 0] < 1e8
        inside_mask = in_region_mask & active_mask
        
        if np.any(inside_mask):
            inside_ids = fluid_ids[inside_mask]
            state.vel[inside_ids] = self.velocity

        # 2) Find empty grid points to inject new particles
        # A grid point is empty if no active particle is within spacing * 0.5 distance
        # For efficiency, only check active particles near the region
        dist_threshold = self.spacing * 0.5
        
        near_region = (pos >= self.region_min - self.spacing) & (pos <= self.region_max + self.spacing)
        near_mask = np.all(near_region, axis=1) & active_mask
        near_pos = pos[near_mask]
        
        empty_grids = []
        for gp in self.grid_points:
            if near_pos.shape[0] > 0:
                dists = np.linalg.norm(near_pos - gp, axis=1)
                if np.min(dists) > dist_threshold:
                    empty_grids.append(gp)
            else:
                empty_grids.append(gp)
                
        if not empty_grids:
            return
            
        empty_grids = np.array(empty_grids)
        num_needed = empty_grids.shape[0]
        
        # Find inactive particles to teleport
        inactive_local_mask = pos[:, 0] >= 1e8
        inactive_ids = fluid_ids[inactive_local_mask]
        
        if inactive_ids.size == 0:
            # We ran out of inactive particles pooling. 
            return
            
        num_to_spawn = min(num_needed, inactive_ids.size)
        spawn_ids = inactive_ids[:num_to_spawn]
        spawn_pos = empty_grids[:num_to_spawn]
        
        state.pos[spawn_ids] = spawn_pos
        state.vel[spawn_ids] = self.velocity
        state.rho[spawn_ids] = 1000.0  # Will be reset by EOS/density solver, but good default
        state.p[spawn_ids] = 0.0
