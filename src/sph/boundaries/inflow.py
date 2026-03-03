from __future__ import annotations

import numpy as np

from sph.boundaries.base import BoundaryBase

_INACTIVE_X = 1e8


class InflowBoundary(BoundaryBase):
    """
    Inflow region with constant velocity assignment and optional refill.
    """

    def __init__(
        self,
        region_min: list[float],
        region_max: list[float],
        velocity: list[float],
        spacing: float,
        *,
        refill: bool = True,
        density_value: float = 1000.0,
    ):
        self.region_min = np.asarray(region_min, dtype=np.float64)
        self.region_max = np.asarray(region_max, dtype=np.float64)
        self.velocity = np.asarray(velocity, dtype=np.float64)
        self.spacing = float(spacing)
        self.refill = bool(refill)
        self.density_value = float(density_value)
        self.grid_points = self._precompute_grid()

    def _precompute_grid(self) -> np.ndarray:
        dim = int(self.region_min.shape[0])
        if dim not in (2, 3):
            return np.zeros((0, dim), dtype=np.float64)
        axes = [
            np.arange(float(self.region_min[d]), float(self.region_max[d]) + 1e-12, self.spacing, dtype=np.float64)
            for d in range(dim)
        ]
        if dim == 2:
            x, y = np.meshgrid(axes[0], axes[1], indexing="xy")
            return np.stack([x.ravel(), y.ravel()], axis=1)
        x, y, z = np.meshgrid(axes[0], axes[1], axes[2], indexing="xy")
        return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

    def pre_step(self, state, dt: float) -> None:
        fluid_ids = state.fluid_indices
        if fluid_ids.size == 0:
            return

        pos_all = state.pos[fluid_ids]
        active_local = pos_all[:, 0] < _INACTIVE_X
        if np.any(active_local):
            active_ids = fluid_ids[active_local]
            active_pos = state.pos[active_ids]
            in_region = np.all((active_pos >= self.region_min) & (active_pos <= self.region_max), axis=1)
            if np.any(in_region):
                state.vel[active_ids[in_region]] = self.velocity

        if not self.refill or self.grid_points.size == 0:
            return

        # Refill with inactive particles to keep inflow occupancy stable.
        active_pos = pos_all[active_local]
        if active_pos.size:
            dists = np.linalg.norm(self.grid_points[:, None, :] - active_pos[None, :, :], axis=2)
            min_dist = np.min(dists, axis=1)
            empty_mask = min_dist > (0.5 * self.spacing)
        else:
            empty_mask = np.ones((self.grid_points.shape[0],), dtype=bool)

        if not np.any(empty_mask):
            return

        inactive_ids = fluid_ids[~active_local]
        if inactive_ids.size == 0:
            return

        spawn_points = self.grid_points[empty_mask]
        n_spawn = int(min(spawn_points.shape[0], inactive_ids.size))
        ids = inactive_ids[:n_spawn]
        state.pos[ids] = spawn_points[:n_spawn]
        state.vel[ids] = self.velocity
        state.acc[ids] = 0.0
        state.rho[ids] = self.density_value
        state.p[ids] = 0.0
