"""
Neighbor search using scipy.cKDTree (compiled C implementation).

Orders of magnitude faster than pure Python implementations.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from sph.neighbor_pairs import NeighborPairs


class KDTreeNeighborSearch:
    """
    Neighbor search using scipy's compiled cKDTree.

    Much faster than custom Python implementations due to C backend.
    """

    def __init__(
        self,
        support_radius: float,
        dim: int = 2,
        periodic_x: tuple[float, float] | None = None,
        periodic_z: tuple[float, float] | None = None,
    ):
        """
        Initialize neighbor search.

        Args:
            support_radius: Search radius (2h for cubic spline)
            dim: Spatial dimension
            periodic_x: Optional tuple (x_min, x_max) for periodic BC in x-direction
            periodic_z: Optional tuple (z_min, z_max) for periodic BC in z-direction (3D only)
        """
        self.support_radius = float(support_radius)
        self.dim = dim
        self.periodic_x = periodic_x
        self.periodic_z = periodic_z if dim == 3 else None

    def build_neighbor_pairs(
        self,
        fluid_positions: np.ndarray,
        boundary_positions: np.ndarray | None = None
    ) -> NeighborPairs:
        """
        Build unique neighbor pairs using cKDTree.

        When periodic_x is set, uses cKDTree's boxsize parameter so that
        pairs across the periodic x-boundary are found correctly.

        Returns:
            NeighborPairs with separate fluid-fluid and fluid-boundary data.
        """
        n_fluid = len(fluid_positions)
        if n_fluid == 0:
            return NeighborPairs.empty(self.dim)

        # Build cKDTree with periodic boxsize for any periodic directions.
        has_periodic = self.periodic_x is not None or self.periodic_z is not None
        if has_periodic:
            fpos = fluid_positions.copy()
            if self.dim == 3:
                # Build boxsize: [L_x, large, L_z] for x+z periodic (or subset)
                L_x = (self.periodic_x[1] - self.periodic_x[0]) if self.periodic_x is not None else 1e30
                L_z = (self.periodic_z[1] - self.periodic_z[0]) if self.periodic_z is not None else 1e30
                boxsize = np.array([L_x, 1e30, L_z], dtype=np.float64)
                if self.periodic_x is not None:
                    x_min = self.periodic_x[0]
                    fpos[:, 0] = (fpos[:, 0] - x_min) % L_x
                if self.periodic_z is not None:
                    z_min = self.periodic_z[0]
                    fpos[:, 2] = (fpos[:, 2] - z_min) % L_z
            else:
                x_min, x_max = self.periodic_x
                L_x = x_max - x_min
                fpos[:, 0] = (fpos[:, 0] - x_min) % L_x
                boxsize = np.array([L_x, 1e30], dtype=np.float64)
            tree = cKDTree(fpos, boxsize=boxsize)
        else:
            fpos = fluid_positions
            boxsize = None
            tree = cKDTree(fpos)

        # Fluid-fluid pairs: query_pairs already enforces i < j
        raw_pairs = tree.query_pairs(self.support_radius, output_type='ndarray')
        if raw_pairs.size == 0:
            idx_i_ff = np.zeros(0, dtype=np.int32)
            idx_j_ff = np.zeros(0, dtype=np.int32)
        else:
            idx_i_ff = raw_pairs[:, 0].astype(np.int32, copy=False)
            idx_j_ff = raw_pairs[:, 1].astype(np.int32, copy=False)

        # Fluid-boundary pairs (one-sided)
        idx_i_fb = np.zeros(0, dtype=np.int32)
        idx_j_fb = np.zeros(0, dtype=np.int32)
        if boundary_positions is not None and len(boundary_positions) > 0:
            if has_periodic:
                bpos = boundary_positions.copy()
                if self.periodic_x is not None:
                    bpos[:, 0] = (bpos[:, 0] - self.periodic_x[0]) % (self.periodic_x[1] - self.periodic_x[0])
                if self.periodic_z is not None and self.dim == 3:
                    bpos[:, 2] = (bpos[:, 2] - self.periodic_z[0]) % (self.periodic_z[1] - self.periodic_z[0])
                btree = cKDTree(bpos, boxsize=boxsize)
            else:
                bpos = boundary_positions
                btree = cKDTree(bpos)
            fb_neighbors = tree.query_ball_tree(btree, self.support_radius)
            counts = np.fromiter((len(nb) for nb in fb_neighbors), dtype=np.int32, count=n_fluid)
            total = int(counts.sum())
            if total > 0:
                idx_i_fb = np.repeat(np.arange(n_fluid, dtype=np.int32), counts)
                idx_j_fb = np.fromiter(
                    (j for nb in fb_neighbors for j in nb),
                    dtype=np.int32,
                    count=total,
                )

        # Precompute displacement vectors using original (un-normalised) positions
        # then apply minimum-image convention for periodic x.
        if idx_i_ff.size > 0:
            r_ff = fluid_positions[idx_i_ff] - fluid_positions[idx_j_ff]
            if self.periodic_x is not None:
                r_ff[:, 0] -= self._wrap_component(r_ff[:, 0], self.periodic_x)
            if self.periodic_z is not None and self.dim == 3:
                r_ff[:, 2] -= self._wrap_component(r_ff[:, 2], self.periodic_z)
            dist_ff = np.linalg.norm(r_ff, axis=1)
            # Filter out any pairs that were actually beyond support after wrapping
            valid = dist_ff < self.support_radius
            idx_i_ff = idx_i_ff[valid]
            idx_j_ff = idx_j_ff[valid]
            r_ff = r_ff[valid]
            dist_ff = dist_ff[valid]
        else:
            r_ff = np.zeros((0, self.dim), dtype=np.float64)
            dist_ff = np.zeros(0, dtype=np.float64)

        if idx_i_fb.size > 0:
            assert boundary_positions is not None
            r_fb = fluid_positions[idx_i_fb] - boundary_positions[idx_j_fb]
            if self.periodic_x is not None:
                r_fb[:, 0] -= self._wrap_component(r_fb[:, 0], self.periodic_x)
            if self.periodic_z is not None and self.dim == 3:
                r_fb[:, 2] -= self._wrap_component(r_fb[:, 2], self.periodic_z)
            dist_fb = np.linalg.norm(r_fb, axis=1)
            valid = dist_fb < self.support_radius
            idx_i_fb = idx_i_fb[valid]
            idx_j_fb = idx_j_fb[valid]
            r_fb = r_fb[valid]
            dist_fb = dist_fb[valid]
        else:
            r_fb = np.zeros((0, self.dim), dtype=np.float64)
            dist_fb = np.zeros(0, dtype=np.float64)

        return NeighborPairs(
            idx_i_ff,
            idx_j_ff,
            r_ff,
            dist_ff,
            idx_i_fb,
            idx_j_fb,
            r_fb,
            dist_fb,
        )

    def _wrap_component(self, delta: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
        """Apply minimum-image convention for a periodic direction."""
        L = bounds[1] - bounds[0]
        return L * np.round(delta / L)
