"""
Vectorized neighbor search using spatial hashing.

Returns neighbor pairs as flat index arrays for vectorized force computation.
"""
from __future__ import annotations

import numpy as np
from collections import defaultdict


class SpatialHashNeighborSearch:
    """
    Spatial hash grid for efficient neighbor search.

    Returns ALL neighbor pairs (i, j) as flat index arrays suitable for
    vectorized SPH computations.
    """

    def __init__(self, support_radius: float, dim: int = 2):
        """
        Initialize spatial hash.

        Args:
            support_radius: Search radius (2h for cubic spline)
            dim: Spatial dimension
        """
        self.support_radius = float(support_radius)
        self.dim = dim
        self.cell_size = support_radius  # One cell per support radius
        self.grid = defaultdict(list)

    def build_neighbor_pairs(
        self,
        fluid_positions: np.ndarray,
        boundary_positions: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build all neighbor pairs (i, j) where particles interact (fully vectorized).

        This includes:
        - Fluid-fluid pairs
        - Fluid-boundary pairs (if boundary provided)

        Args:
            fluid_positions: Fluid particle positions, shape (n_fluid, dim)
            boundary_positions: Boundary particle positions, shape (n_boundary, dim) or None

        Returns:
            idx_i: Particle indices i, shape (N_pairs,)
            idx_j: Neighbor indices j, shape (N_pairs,)
            r_ij: Displacement vectors r_i - r_j, shape (N_pairs, dim)
            dist: Distances ||r_ij||, shape (N_pairs,)

        Note: j indices are in the combined (fluid + boundary) indexing:
            - j < n_fluid: fluid particle
            - j >= n_fluid: boundary particle (offset by n_fluid)
        """
        n_fluid = len(fluid_positions)

        # Combine fluid and boundary positions
        if boundary_positions is not None:
            all_positions = np.vstack([fluid_positions, boundary_positions])
        else:
            all_positions = fluid_positions

        n_total = len(all_positions)

        # Compute cell indices for all particles
        cells = np.floor(all_positions / self.cell_size).astype(np.int32)

        # Hash cell coordinates to 1D keys
        # Use large primes to avoid collisions
        cell_keys = cells[:, 0] * 73856093
        if self.dim >= 2:
            cell_keys ^= cells[:, 1] * 19349663
        if self.dim >= 3:
            cell_keys ^= cells[:, 2] * 83492791

        # Sort particles by cell key for efficient cell iteration
        sort_idx = np.argsort(cell_keys)
        sorted_keys = cell_keys[sort_idx]
        sorted_cells = cells[sort_idx]

        # Find cell boundaries (where cell key changes)
        cell_changes = np.concatenate([[0], np.where(np.diff(sorted_keys) != 0)[0] + 1, [n_total]])

        # Build pairs by iterating over fluid particles and checking 27 neighboring cells (3^3 stencil)
        idx_i_list = []
        idx_j_list = []

        # For 2D: check 9 cells (3x3 stencil)
        if self.dim == 2:
            offsets = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1), (0, 0), (0, 1),
                (1, -1), (1, 0), (1, 1)
            ]
        else:  # 3D: check 27 cells
            offsets = [(dx, dy, dz) for dx in [-1, 0, 1] for dy in [-1, 0, 1] for dz in [-1, 0, 1]]

        # Iterate over fluid particles only
        for i in range(n_fluid):
            cell_i = cells[i]
            pos_i = all_positions[i]

            # Check all 9 (or 27) neighboring cells
            for offset in offsets:
                neighbor_cell = cell_i + np.array(offset, dtype=np.int32)

                # Compute hash for neighbor cell
                neighbor_key = neighbor_cell[0] * 73856093
                if self.dim >= 2:
                    neighbor_key ^= neighbor_cell[1] * 19349663
                if self.dim >= 3:
                    neighbor_key ^= neighbor_cell[2] * 83492791

                # Find particles in this cell using binary search
                cell_start_idx = np.searchsorted(sorted_keys, neighbor_key, side='left')
                cell_end_idx = np.searchsorted(sorted_keys, neighbor_key, side='right')

                # Check all particles in this cell
                for k in range(cell_start_idx, cell_end_idx):
                    j = sort_idx[k]
                    if j == i:  # Skip self
                        continue

                    # Distance check
                    dist_ij = np.linalg.norm(all_positions[j] - pos_i)
                    if dist_ij <= self.support_radius:
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
        r_ij = pos_i - pos_j  # r_i - r_j
        dist = np.linalg.norm(r_ij, axis=1)

        return idx_i, idx_j, r_ij, dist

    def _build_grid(self, positions: np.ndarray):
        """Build the spatial hash grid from particle positions."""
        self.grid.clear()

        for i, pos in enumerate(positions):
            cell = self._cell_index(pos)
            self.grid[cell].append(i)

    def _cell_index(self, position: np.ndarray) -> tuple:
        """Convert position to grid cell index."""
        return tuple(np.floor(position / self.cell_size).astype(int))

    def _query(self, i: int, positions: np.ndarray, pos_i: np.ndarray) -> list[int]:
        """
        Find all neighbors of particle i within support radius.

        Args:
            i: Particle index
            positions: All particle positions
            pos_i: Position of particle i

        Returns:
            List of neighbor indices (excludes i itself)
        """
        base_cell = self._cell_index(pos_i)
        neighbors = []

        # Search neighboring cells (3^dim stencil)
        if self.dim == 2:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    cell = (base_cell[0] + dx, base_cell[1] + dy)
                    for j in self.grid.get(cell, []):
                        if j != i:
                            dist = np.linalg.norm(positions[j] - pos_i)
                            if dist <= self.support_radius:
                                neighbors.append(j)

        elif self.dim == 3:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        cell = (base_cell[0] + dx, base_cell[1] + dy, base_cell[2] + dz)
                        for j in self.grid.get(cell, []):
                            if j != i:
                                dist = np.linalg.norm(positions[j] - pos_i)
                                if dist <= self.support_radius:
                                    neighbors.append(j)

        return neighbors
