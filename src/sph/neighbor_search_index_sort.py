"""
Index Sort neighbor search (SPlisHSPlasH approach).

Fastest known CPU-based SPH neighbor search algorithm.
Uses spatial hashing with sorted particle indices for cache-friendly access.
"""
from __future__ import annotations

import numpy as np


class IndexSortNeighborSearch:
    """
    Index Sort neighbor search algorithm.

    Key idea: Sort particles by cell key for sequential memory access.
    This makes the algorithm cache-friendly and very fast.
    """

    def __init__(self, support_radius: float, dim: int = 2):
        """
        Initialize neighbor search.

        Args:
            support_radius: Search radius (2h for cubic spline)
            dim: Spatial dimension
        """
        self.support_radius = float(support_radius)
        self.dim = dim
        self.inv_sr = 1.0 / support_radius

    def build_neighbor_pairs(
        self,
        fluid_positions: np.ndarray,
        boundary_positions: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build all neighbor pairs using Index Sort algorithm.

        Args:
            fluid_positions: Fluid particle positions, shape (n_fluid, dim)
            boundary_positions: Boundary particle positions, shape (n_boundary, dim) or None

        Returns:
            idx_i: Particle indices i, shape (N_pairs,)
            idx_j: Neighbor indices j, shape (N_pairs,)
            r_ij: Displacement vectors r_i - r_j, shape (N_pairs, dim)
            dist: Distances ||r_ij||, shape (N_pairs,)
        """
        n_fluid = len(fluid_positions)

        # Combine fluid and boundary positions
        if boundary_positions is not None:
            all_positions = np.vstack([fluid_positions, boundary_positions])
        else:
            all_positions = fluid_positions

        n_total = len(all_positions)

        # Step 1: Assign particles to cells
        cells = np.floor(all_positions * self.inv_sr).astype(np.int32)

        # Step 2: Compute cell keys
        # Shift to positive coordinates
        grid_min = cells.min(axis=0)
        cells_shifted = cells - grid_min
        grid_size = cells_shifted.max(axis=0) + 1

        # Linear cell key: key = cx + cy * width (+ cz * width * height for 3D)
        if self.dim == 2:
            cell_keys = cells_shifted[:, 0] + cells_shifted[:, 1] * grid_size[0]
        else:  # dim == 3
            cell_keys = (cells_shifted[:, 0] +
                        cells_shifted[:, 1] * grid_size[0] +
                        cells_shifted[:, 2] * grid_size[0] * grid_size[1])

        # Step 3: Sort particles by cell key
        sort_idx = np.argsort(cell_keys, kind='stable')
        sorted_keys = cell_keys[sort_idx]

        # Step 4: Find cell boundaries
        unique_keys = np.unique(sorted_keys)
        cell_starts = np.searchsorted(sorted_keys, unique_keys, side='left')
        cell_ends = np.searchsorted(sorted_keys, unique_keys, side='right')

        # Build lookup table: cell_key → (start_index, end_index)
        cell_lookup = {}
        for k, s, e in zip(unique_keys, cell_starts, cell_ends):
            cell_lookup[int(k)] = (int(s), int(e))

        # Step 5: Collect pairs by checking neighboring cells
        idx_i_list = []
        idx_j_list = []

        # 3x3 stencil for 2D, 3x3x3 for 3D
        if self.dim == 2:
            offsets = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1), (0, 0), (0, 1),
                (1, -1), (1, 0), (1, 1)
            ]
        else:  # dim == 3
            offsets = [(dx, dy, dz) for dx in [-1, 0, 1]
                      for dy in [-1, 0, 1] for dz in [-1, 0, 1]]

        # Iterate over fluid particles only (i must be fluid)
        for i in range(n_fluid):
            cell_i = cells_shifted[i]
            pos_i = all_positions[i]

            # Check all neighboring cells
            for offset in offsets:
                neighbor_cell = cell_i + np.array(offset, dtype=np.int32)

                # Compute neighbor cell key
                if self.dim == 2:
                    neighbor_key = neighbor_cell[0] + neighbor_cell[1] * grid_size[0]
                else:  # dim == 3
                    neighbor_key = (neighbor_cell[0] +
                                  neighbor_cell[1] * grid_size[0] +
                                  neighbor_cell[2] * grid_size[0] * grid_size[1])

                neighbor_key = int(neighbor_key)

                # Check if this cell exists
                if neighbor_key not in cell_lookup:
                    continue

                # Get particles in this cell
                start_idx, end_idx = cell_lookup[neighbor_key]

                # Check all particles in the cell
                for sorted_j in range(start_idx, end_idx):
                    j = sort_idx[sorted_j]

                    if j == i:  # Skip self
                        continue

                    # Distance check (squared distance first)
                    r_ij = pos_i - all_positions[j]
                    dist_sq = np.sum(r_ij ** 2)

                    if dist_sq <= self.support_radius ** 2:
                        idx_i_list.append(i)
                        idx_j_list.append(j)

        # Convert to arrays
        idx_i = np.array(idx_i_list, dtype=np.int32)
        idx_j = np.array(idx_j_list, dtype=np.int32)

        if len(idx_i) == 0:
            # No neighbors found
            empty = np.array([], dtype=np.int32)
            empty_vec = np.zeros((0, self.dim), dtype=np.float64)
            return empty, empty, empty_vec, np.array([], dtype=np.float64)

        # Compute displacement vectors and distances (vectorized)
        pos_i = all_positions[idx_i]
        pos_j = all_positions[idx_j]
        r_ij = pos_i - pos_j
        dist = np.sqrt(np.sum(r_ij ** 2, axis=1))

        return idx_i, idx_j, r_ij, dist
