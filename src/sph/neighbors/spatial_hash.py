from __future__ import annotations

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple


class SpatialHash:
    """
    Uniform grid spatial hash for neighbor search.
    Deterministic cell iteration.
    """

    def __init__(
        self,
        support_radius: float,
        dim: int,
        *,
        periodic_min: np.ndarray | None = None,
        periodic_max: np.ndarray | None = None,
        periodic_axes: tuple[int, ...] = (),
    ):
        self.h = float(support_radius)
        self.dim = dim
        self.cell_size = self.h
        self.grid: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
        self.periodic_axes = tuple(int(a) for a in periodic_axes)
        self.periodic_min = None if periodic_min is None else np.asarray(periodic_min, dtype=np.float64)
        self.periodic_max = None if periodic_max is None else np.asarray(periodic_max, dtype=np.float64)
        self._periodic_L: np.ndarray | None = None
        self._periodic_ncells: np.ndarray | None = None
        if self.periodic_axes:
            if self.periodic_min is None or self.periodic_max is None:
                raise ValueError("periodic_min/max must be provided when periodic_axes is non-empty")
            if self.periodic_min.shape != (self.dim,) or self.periodic_max.shape != (self.dim,):
                raise ValueError("periodic_min/max must have shape (dim,)")
            L = self.periodic_max - self.periodic_min
            if np.any(L <= 0.0):
                raise ValueError("periodic bounds must have positive extent")
            ncells = np.maximum(1, np.ceil(L / self.cell_size).astype(np.int64))
            self._periodic_L = L
            self._periodic_ncells = ncells

    def _cell_index(self, position: np.ndarray) -> Tuple[int, ...]:
        pos = np.asarray(position, dtype=np.float64)
        idx = np.floor(pos / self.cell_size).astype(np.int64)
        if self.periodic_axes and self.periodic_min is not None and self._periodic_L is not None:
            # Map periodic axes into a finite cell index range [0, ncells).
            for a in self.periodic_axes:
                # Wrap into [min, max) then compute cell index relative to min.
                x = pos[a] - self.periodic_min[a]
                xw = x % float(self._periodic_L[a])
                idx[a] = int(np.floor(xw / self.cell_size)) % int(self._periodic_ncells[a])
        return tuple(int(v) for v in idx)

    def displacement(self, pi: np.ndarray, pj: np.ndarray) -> np.ndarray:
        """
        Return displacement vector r_ij = x_i - x_j using minimum-image convention
        on periodic axes (if configured).

        This is a geometry helper for periodic domains; it does not change any
        SPH equations, only how distances are measured across periodic seams.
        """
        rij = np.asarray(pi, dtype=np.float64) - np.asarray(pj, dtype=np.float64)
        if self.periodic_axes and self._periodic_L is not None:
            for a in self.periodic_axes:
                L = float(self._periodic_L[a])
                if L > 0.0:
                    rij[a] -= np.rint(rij[a] / L) * L
        return rij

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
                    cx = base_cell[0] + dx
                    cy = base_cell[1] + dy
                    if self.periodic_axes and self._periodic_ncells is not None:
                        if 0 in self.periodic_axes:
                            cx = cx % int(self._periodic_ncells[0])
                        if 1 in self.periodic_axes:
                            cy = cy % int(self._periodic_ncells[1])
                    cell = (cx, cy)
                    for j in self.grid.get(cell, []):
                        if j == i:
                            continue
                        if np.linalg.norm(self.displacement(positions[j], pos)) <= self.h:
                            neighbors.append(j)

        elif self.dim == 3:
            for dx in offsets:
                for dy in offsets:
                    for dz in offsets:
                        cx = base_cell[0] + dx
                        cy = base_cell[1] + dy
                        cz = base_cell[2] + dz
                        if self.periodic_axes and self._periodic_ncells is not None:
                            if 0 in self.periodic_axes:
                                cx = cx % int(self._periodic_ncells[0])
                            if 1 in self.periodic_axes:
                                cy = cy % int(self._periodic_ncells[1])
                            if 2 in self.periodic_axes:
                                cz = cz % int(self._periodic_ncells[2])
                        cell = (cx, cy, cz)
                        for j in self.grid.get(cell, []):
                            if j == i:
                                continue
                            if np.linalg.norm(self.displacement(positions[j], pos)) <= self.h:
                                neighbors.append(j)

        return neighbors
