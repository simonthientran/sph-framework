"""
Ultra-fast vectorized neighbor search using NumPy broadcasting.

For small-to-medium sized problems (~3000 particles), computing all pairwise
distances with broadcasting is faster than spatial hashing with Python loops.
"""
from __future__ import annotations

import numpy as np


class FastNeighborSearch:
    """
    Brute-force neighbor search using fully vectorized distance computation.

    For N~3000 particles, this is faster than spatial hashing with Python loops.
    Memory usage: O(N^2) for distance matrix.
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

    def build_neighbor_pairs(
        self,
        fluid_positions: np.ndarray,
        boundary_positions: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build all neighbor pairs using vectorized distance computation.

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

        # Compute ALL pairwise displacement vectors using broadcasting
        # r_ij[i, j] = positions[i] - positions[j]
        # Shape: (n_fluid, n_total, dim)
        r_ij_matrix = fluid_positions[:, np.newaxis, :] - all_positions[np.newaxis, :, :]

        # Compute ALL pairwise SQUARED distances (faster than norm)
        # Shape: (n_fluid, n_total)
        dist_sq_matrix = np.sum(r_ij_matrix ** 2, axis=2)

        # Filter pairs: squared_dist <= support_radius^2 AND not self-pairs
        # Shape: (n_fluid, n_total)
        support_sq = self.support_radius ** 2
        mask = (dist_sq_matrix <= support_sq) & (dist_sq_matrix > 0)

        # Get indices of valid pairs
        idx_i, idx_j = np.where(mask)

        if len(idx_i) == 0:
            # No neighbors found
            empty = np.array([], dtype=np.int32)
            empty_vec = np.zeros((0, self.dim), dtype=np.float64)
            return empty, empty, empty_vec, np.array([], dtype=np.float64)

        # Extract displacement vectors and distances for valid pairs
        r_ij = r_ij_matrix[idx_i, idx_j, :]
        # Only compute sqrt for valid pairs (much fewer!)
        dist = np.sqrt(dist_sq_matrix[idx_i, idx_j])

        return idx_i.astype(np.int32), idx_j.astype(np.int32), r_ij, dist
