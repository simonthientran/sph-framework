from __future__ import annotations

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


class SpatialHash:
    """
    Uniform grid spatial hash for neighbor search with periodic boundary support.

    Periodic handling strategy:
    - On build(), ghost entries are inserted for particles within one support_radius
      of a periodic boundary. These ghosts map back to the original particle index.
    - On query(), the standard 3^dim stencil finds both real and ghost neighbors.
    - relative_vector(xi, xj) returns the minimum-image displacement vector,
      correctly accounting for periodic wrap-around.

    This ensures that ALL downstream code (density, pressure, viscosity, XSPH, PCISPH)
    gets physically correct neighbor lists and displacement vectors without any changes
    to kernel math or force formulations.

    Convention:
    - cell_size = support_radius (= 2*h for cubic spline)
    - Neighbor distance check: ||relative_vector(xi, xj)|| <= support_radius
    """

    def __init__(
        self,
        support_radius: float,
        dim: int,
        domain_min: Optional[np.ndarray] = None,
        domain_max: Optional[np.ndarray] = None,
        periodic_axes: Optional[Tuple[bool, ...]] = None,
    ):
        self.h = float(support_radius)
        self.dim = dim
        self.cell_size = self.h
        self.grid: Dict[Tuple[int, ...], List[int]] = defaultdict(list)

        # Periodic boundary configuration
        self.domain_min: Optional[np.ndarray] = None
        self.domain_max: Optional[np.ndarray] = None
        self.periodic_axes: Tuple[bool, ...] = tuple(False for _ in range(dim))
        self.domain_length: Optional[np.ndarray] = None  # per-axis domain size
        self._has_periodic = False

        if domain_min is not None and domain_max is not None:
            self.domain_min = np.asarray(domain_min, dtype=np.float64).copy()
            self.domain_max = np.asarray(domain_max, dtype=np.float64).copy()
            self.domain_length = self.domain_max - self.domain_min
            if periodic_axes is not None:
                self.periodic_axes = tuple(
                    bool(periodic_axes[d]) if d < len(periodic_axes) else False
                    for d in range(dim)
                )
                self._has_periodic = any(self.periodic_axes)

    def _cell_index(self, position: np.ndarray) -> Tuple[int, ...]:
        return tuple(np.floor(position / self.cell_size).astype(int))

    def relative_vector(self, xi: np.ndarray, xj: np.ndarray) -> np.ndarray:
        """
        Compute the minimum-image displacement vector r_ij = x_i - x_j,
        accounting for periodic wrap-around on periodic axes.

        For non-periodic axes (or when no periodicity is configured),
        this returns the raw difference.

        THIS IS THE ONLY CORRECT WAY TO COMPUTE PARTICLE DISPLACEMENTS.
        All kernel evaluations must use this method.
        """
        r = xi - xj
        if not self._has_periodic or self.domain_length is None:
            return r
        for d in range(self.dim):
            if self.periodic_axes[d]:
                L = float(self.domain_length[d])
                if L > 0.0:
                    # Minimum image convention
                    r[d] = r[d] - L * np.round(r[d] / L)
        return r

    def build(self, positions: np.ndarray) -> None:
        """
        Build the spatial hash grid from particle positions.

        If periodic axes are configured, ghost entries are created for particles
        within one support_radius of a periodic boundary. Ghost entries store the
        ORIGINAL particle index so that neighbor queries return correct indices.
        """
        self.grid.clear()

        n = int(positions.shape[0])

        # Insert all real particles
        for i in range(n):
            cell = self._cell_index(positions[i])
            self.grid[cell].append(i)

        # Insert ghost entries for periodic axes
        if not self._has_periodic or self.domain_min is None or self.domain_max is None:
            return

        for i in range(n):
            pos_i = positions[i]
            # For each periodic axis, check if particle is near a boundary
            # and insert ghost(s) on the opposite side.
            # We must handle corner cases (particle near boundaries on multiple
            # periodic axes simultaneously) by generating all 2^k ghost offsets
            # where k = number of axes where the particle is near a boundary.
            ghost_offsets: List[np.ndarray] = []

            for d in range(self.dim):
                if not self.periodic_axes[d]:
                    continue
                L = float(self.domain_length[d])
                if L <= 0.0:
                    continue
                lo = float(self.domain_min[d])
                hi = float(self.domain_max[d])
                p = float(pos_i[d])

                new_offsets: List[np.ndarray] = []
                # Near lower boundary -> ghost at upper side
                if p - lo < self.h:
                    offset = np.zeros(self.dim, dtype=np.float64)
                    offset[d] = L
                    new_offsets.append(offset)
                # Near upper boundary -> ghost at lower side
                if hi - p < self.h:
                    offset = np.zeros(self.dim, dtype=np.float64)
                    offset[d] = -L
                    new_offsets.append(offset)

                if not new_offsets:
                    continue

                if not ghost_offsets:
                    ghost_offsets = new_offsets
                else:
                    # Combine with existing offsets (tensor product for corners)
                    combined = []
                    for existing in ghost_offsets:
                        combined.append(existing)  # keep existing without this axis offset
                        for new_off in new_offsets:
                            combined.append(existing + new_off)
                    for new_off in new_offsets:
                        combined.append(new_off)
                    ghost_offsets = combined

            # Insert ghost entries
            for offset in ghost_offsets:
                ghost_pos = pos_i + offset
                cell = self._cell_index(ghost_pos)
                self.grid[cell].append(i)  # same particle index!

    def query(self, i: int, positions: np.ndarray) -> List[int]:
        """
        Find all neighbors of particle i within support_radius.

        Uses minimum-image distance for periodic axes.
        Returns unique neighbor indices (no duplicates, excludes self).

        Note: Ghost entries in the grid mean the same particle index j can appear
        in multiple cells. We deduplicate and also ensure j != i.
        """
        pos = positions[i]
        base_cell = self._cell_index(pos)

        neighbors_set: set[int] = set()

        offsets = [-1, 0, 1]

        if self.dim == 2:
            for dx in offsets:
                for dy in offsets:
                    cell = (base_cell[0] + dx, base_cell[1] + dy)
                    for j in self.grid.get(cell, []):
                        if j == i:
                            continue
                        if j in neighbors_set:
                            continue
                        rij = self.relative_vector(pos, positions[j])
                        if float(np.linalg.norm(rij)) <= self.h:
                            neighbors_set.add(j)

        elif self.dim == 3:
            for dx in offsets:
                for dy in offsets:
                    for dz in offsets:
                        cell = (base_cell[0] + dx, base_cell[1] + dy, base_cell[2] + dz)
                        for j in self.grid.get(cell, []):
                            if j == i:
                                continue
                            if j in neighbors_set:
                                continue
                            rij = self.relative_vector(pos, positions[j])
                            if float(np.linalg.norm(rij)) <= self.h:
                                neighbors_set.add(j)

        return list(neighbors_set)