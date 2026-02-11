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
        self.h = float(support_radius)
        self.dim = dim
        self.cell_size = self.h
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
                        if np.linalg.norm(positions[j] - pos) <= self.h:
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
                            if np.linalg.norm(positions[j] - pos) <= self.h:
                                neighbors.append(j)

        return neighbors
