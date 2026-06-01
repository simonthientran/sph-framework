"""Density-regime classification utilities inspired by SPlisHSPlasH diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sph.fluid_model import BoundaryModel, FluidModel
from sph.neighbor_pairs import NeighborPairs


@dataclass(slots=True)
class DensityRegimeSummary:
    """Aggregated density metrics split by physical regime."""

    rho0: float
    fluid_count: int
    boundary_count: int
    interior_count: int
    wall_count: int
    free_surface_count: int
    splash_count: int
    overcompressed_count: int
    under_supported_count: int
    rho_mean_interior: float
    rho_min_interior: float
    rho_max_interior: float
    rho_mean_wall: float
    rho_min_wall: float
    rho_max_wall: float
    rho_mean_free_surface: float
    rho_min_free_surface: float


@dataclass(slots=True)
class DensityRegimeInfo:
    """Full per-particle classification together with the aggregated summary."""

    total_counts: np.ndarray
    fluid_counts: np.ndarray
    boundary_counts: np.ndarray
    interior_mask: np.ndarray
    wall_mask: np.ndarray
    free_surface_mask: np.ndarray
    splash_mask: np.ndarray
    low_density_mask: np.ndarray
    overcompressed_mask: np.ndarray
    interior_neighbor_threshold: int
    free_surface_neighbor_threshold: int
    splash_neighbor_threshold: int
    free_surface_density_threshold: float
    low_density_threshold: float
    summary: DensityRegimeSummary


def analyze_density_regimes(
    fluid: FluidModel,
    boundary: BoundaryModel | None,
    pairs: NeighborPairs | None,
    *,
    precomputed_fluid_counts: np.ndarray | None = None,
    precomputed_boundary_counts: np.ndarray | None = None,
) -> DensityRegimeInfo:
    """
    Classify particles into interior / wall / free-surface / splash regimes.

    The heuristics follow SPlisHSPlasH's diagnostic intuition:
    - Neighbor-based support thresholds distinguish interior vs. surface.
    - Boundary adjacency is counted separately so wall particles are not
      mistaken for free-surface.
    - Extreme under-support (splashes) is flagged explicitly.

    When ``precomputed_fluid_counts`` / ``precomputed_boundary_counts`` are
    provided (e.g. downloaded from a GPU pair build), they are used directly
    instead of recomputing from *pairs*.
    """

    n = fluid.n
    rho0 = float(fluid.rho0)
    boundary_count = int(boundary.n if boundary is not None else 0)

    if precomputed_fluid_counts is not None:
        fluid_counts = precomputed_fluid_counts.astype(np.int32, copy=False)
        boundary_counts = (
            precomputed_boundary_counts.astype(np.int32, copy=False)
            if precomputed_boundary_counts is not None
            else np.zeros(n, dtype=np.int32)
        )
    else:
        fluid_counts = np.zeros(n, dtype=np.int32)
        boundary_counts = np.zeros(n, dtype=np.int32)
        if pairs is not None and n > 0:
            if pairs.ff_i.size:
                fluid_counts += np.bincount(pairs.ff_i, minlength=n).astype(np.int32, copy=False)
            if pairs.ff_j.size:
                fluid_counts += np.bincount(pairs.ff_j, minlength=n).astype(np.int32, copy=False)
            if pairs.fb_i.size:
                boundary_counts += np.bincount(pairs.fb_i, minlength=n).astype(np.int32, copy=False)

    total_counts = np.zeros(n, dtype=np.int32)
    total_counts[:] = fluid_counts + boundary_counts

    density = fluid.densities
    low_density_threshold = 0.8 * rho0
    free_surface_density_threshold = 0.9 * rho0
    interior_density_threshold = 0.97 * rho0
    overcompressed_threshold = 1.05 * rho0

    positive_fluid = fluid_counts[fluid_counts > 0]
    mean_fluid_neighbors = float(np.mean(positive_fluid)) if positive_fluid.size else 0.0
    interior_neighbor_threshold = int(max(8, round(mean_fluid_neighbors * 0.55))) if mean_fluid_neighbors else 8
    free_surface_neighbor_threshold = int(max(4, round(mean_fluid_neighbors * 0.4))) if mean_fluid_neighbors else 4
    splash_neighbor_threshold = int(max(2, round(mean_fluid_neighbors * 0.2))) if mean_fluid_neighbors else 2

    wall_mask = boundary_counts > 0
    low_density_mask = density < low_density_threshold
    # Overcompressed = interior or free-surface particles above threshold.
    # Wall-adjacent particles (wall_mask) legitimately exceed ρ₀ due to boundary
    # kernel contributions (SPlisHSPlasH behaviour after removing the density cap).
    overcompressed_mask = (~wall_mask) & (density > overcompressed_threshold)

    interior_mask = (
        (~wall_mask)
        & (density >= interior_density_threshold)
        & (fluid_counts >= interior_neighbor_threshold)
    )

    splash_mask = (~wall_mask) & (total_counts <= splash_neighbor_threshold)
    free_surface_mask = (~wall_mask) & ~interior_mask & (
        (density < free_surface_density_threshold)
        | (fluid_counts < free_surface_neighbor_threshold)
    )
    free_surface_mask &= ~splash_mask

    under_supported_mask = (~wall_mask) & (fluid_counts < free_surface_neighbor_threshold)

    interior_count = int(np.count_nonzero(interior_mask))
    wall_count = int(np.count_nonzero(wall_mask))
    free_surface_count = int(np.count_nonzero(free_surface_mask))
    splash_count = int(np.count_nonzero(splash_mask))
    overcompressed_count = int(np.count_nonzero(overcompressed_mask))
    under_supported_count = int(np.count_nonzero(under_supported_mask))

    def _stats(mask: np.ndarray) -> tuple[float, float, float]:
        if not mask.size or not np.any(mask):
            return 0.0, 0.0, 0.0
        vals = density[mask]
        return float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))

    rho_mean_interior, rho_min_interior, rho_max_interior = _stats(interior_mask)
    rho_mean_wall, rho_min_wall, rho_max_wall = _stats(wall_mask)
    rho_mean_free_surface, rho_min_free_surface, _ = _stats(free_surface_mask | splash_mask)

    summary = DensityRegimeSummary(
        rho0=rho0,
        fluid_count=n,
        boundary_count=boundary_count,
        interior_count=interior_count,
        wall_count=wall_count,
        free_surface_count=free_surface_count,
        splash_count=splash_count,
        overcompressed_count=overcompressed_count,
        under_supported_count=under_supported_count,
        rho_mean_interior=rho_mean_interior,
        rho_min_interior=rho_min_interior,
        rho_max_interior=rho_max_interior,
        rho_mean_wall=rho_mean_wall,
        rho_min_wall=rho_min_wall,
        rho_max_wall=rho_max_wall,
        rho_mean_free_surface=rho_mean_free_surface,
        rho_min_free_surface=rho_min_free_surface,
    )

    return DensityRegimeInfo(
        total_counts=total_counts,
        fluid_counts=fluid_counts,
        boundary_counts=boundary_counts,
        interior_mask=interior_mask,
        wall_mask=wall_mask,
        free_surface_mask=free_surface_mask,
        splash_mask=splash_mask,
        low_density_mask=low_density_mask,
        overcompressed_mask=overcompressed_mask,
        interior_neighbor_threshold=interior_neighbor_threshold,
        free_surface_neighbor_threshold=free_surface_neighbor_threshold,
        splash_neighbor_threshold=splash_neighbor_threshold,
        free_surface_density_threshold=free_surface_density_threshold,
        low_density_threshold=low_density_threshold,
        summary=summary,
    )
