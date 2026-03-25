from __future__ import annotations

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple


class SpatialHash:
    """
    Uniform grid spatial hash for neighbor search.
    Deterministic cell iteration.
    """

    def __init__(self, support_radius: float, dim: int):
        self.kernel_radius = float(support_radius)
        self.search_radius = 2.0 * self.kernel_radius
        self.dim = dim
        self.cell_size = self.search_radius
        self.grid: Dict[Tuple[int, ...], List[int]] = defaultdict(list)

    def _cell_index(self, position: np.ndarray) -> Tuple[int, ...]:
        return tuple(np.floor(position / self.cell_size).astype(int))

    def build(self, positions: np.ndarray) -> None:
        self.grid.clear()

        for i, pos in enumerate(positions):
            cell = self._cell_index(pos)
            self.grid[cell].append(i)

    def query(self, i: int, positions: np.ndarray) -> List[int]:
        pos = positions[i]
        base_cell = self._cell_index(pos)

        neighbors: List[int] = []

        # iterate over neighboring cells (3^dim region)
        offsets = [-1, 0, 1]

        if self.dim == 2:
            for dx in offsets:
                for dy in offsets:
                    cell = (base_cell[0] + dx, base_cell[1] + dy)
                    for j in self.grid.get(cell, []):
                        if j == i:
                            continue
                        if np.linalg.norm(positions[j] - pos) <= self.search_radius:
                            neighbors.append(j)

        elif self.dim == 3:
            for dx in offsets:
                for dy in offsets:
                    for dz in offsets:
                        cell = (base_cell[0] + dx,
                                base_cell[1] + dy,
                                base_cell[2] + dz)
                        for j in self.grid.get(cell, []):
                            if j == i:
                                continue
                            if np.linalg.norm(positions[j] - pos) <= self.search_radius:
                                neighbors.append(j)

        return neighbors

    def relative_vector(self, pos_i: np.ndarray, pos_j: np.ndarray) -> np.ndarray:
        """Return displacement from j to i (hook for periodic wrapping)."""

        return np.asarray(pos_i - pos_j, dtype=np.float64)

    def get_all_neighbor_pairs(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Build all neighbor pairs (i, j) where j is a neighbor of i.

        Returns:
            idx_i: array of particle indices i (shape: N_pairs,)
            idx_j: array of neighbor indices j (shape: N_pairs,)

        Note: Does NOT include self-pairs (i, i).
        """
        idx_i_list = []
        idx_j_list = []

        n = len(positions)
        for i in range(n):
            neighbors = self.query(i, positions)
            idx_i_list.extend([i] * len(neighbors))
            idx_j_list.extend(neighbors)

        return np.array(idx_i_list, dtype=np.int32), np.array(idx_j_list, dtype=np.int32)
