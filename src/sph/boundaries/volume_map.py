"""
Volume-map boundary representation (Bender et al. 2019).

Computes a volumetric boundary field from a signed distance field (SDF).
For each point in space, the volume map stores what fraction of the kernel
support is occupied by the boundary solid.  At runtime, a fluid particle
queries this field to get a smooth, mesh-resolution-independent boundary
contribution that replaces the noisy particle-based psi from Akinci 2012.

Reference:
  Jan Bender, Tassilo Kugelstadt, Marcel Weiler, Dan Koschier.
  "Volume Maps: An Implicit Boundary Representation for SPH."
  Proc. ACM SIGGRAPH MIG '19, 2019.

This module provides:
  - ``compute_volume_field``: core Bender 2019 volume integral on a grid
  - ``VolumeMapGrid``: SDF grid extended with a volume field
  - ``VolumeMapBoundaryRepresentation``: boundary representation contract
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sph.boundaries.contracts import BoundaryRepresentation, BoundaryRepresentationSource
from sph.boundaries.mesh_particles import MeshBoundarySourceSpec
from sph.boundaries.mesh_sdf import (
    MeshSDFGrid,
    MeshSDFSourceSpec,
    _build_mesh_sdf_grid,
    parse_mesh_sdf_specs,
)
from sph.io.geometry_loader import load_mesh, transform_mesh


# ---------------------------------------------------------------------------
# Quadrature helpers
# ---------------------------------------------------------------------------

def _gauss_legendre_samples_3d(
    support_radius: float,
    n_per_axis: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre quadrature points and weights inside a sphere.

    Returns points (M, 3) and weights (M,) for numerical integration over
    the ball of radius ``support_radius``.  Points outside the ball are
    excluded; weights are adjusted to account for the cube→sphere mapping.
    """
    nodes_1d, weights_1d = np.polynomial.legendre.leggauss(n_per_axis)

    # Scale from [-1, 1] to [-r, r]
    nodes_1d = nodes_1d * support_radius
    weights_1d = weights_1d * support_radius  # Jacobian of scaling

    # Outer product for 3D tensor grid
    xx, yy, zz = np.meshgrid(nodes_1d, nodes_1d, nodes_1d, indexing="ij")
    wx, wy, wz = np.meshgrid(weights_1d, weights_1d, weights_1d, indexing="ij")

    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    weights = (wx * wy * wz).ravel()

    # Keep only points inside the sphere
    r2 = np.sum(points ** 2, axis=1)
    mask = r2 <= support_radius ** 2
    return points[mask], weights[mask]


# ---------------------------------------------------------------------------
# Core volume integral (Bender et al. 2019)
# ---------------------------------------------------------------------------

def _bender_integrand(
    sdf_values: np.ndarray,
    support_radius: float,
    kernel_w0: float,
    kernel_w_func,
) -> np.ndarray:
    """Evaluate the Bender 2019 volume integrand for an array of SDF values.

    For each SDF value d:
      - d <= 0  (inside boundary):  integrand = 1.0
      - 0 < d < support_radius:     integrand = W(d) / W(0)
      - d >= support_radius:         integrand = 0.0
    """
    result = np.zeros_like(sdf_values, dtype=np.float64)
    inside = sdf_values <= 0.0
    result[inside] = 1.0
    transition = (~inside) & (sdf_values < support_radius)
    if np.any(transition):
        w_vals = kernel_w_func(sdf_values[transition])
        result[transition] = w_vals / kernel_w0
    return result


def compute_volume_field(
    sdf_grid: MeshSDFGrid,
    support_radius: float,
    kernel_w_func=None,
    kernel_w0: float | None = None,
    n_quadrature: int = 8,
    scale_factor: float = 0.8,
) -> np.ndarray:
    """Compute Bender 2019 volume map from a signed distance grid.

    For each cell of the SDF grid, integrates the kernel-weighted boundary
    volume fraction over the support sphere using Gauss-Legendre quadrature.

    Args:
        sdf_grid: Pre-computed SDF grid from mesh geometry.
        support_radius: SPH kernel support radius.
        kernel_w_func: Callable W(r) for the SPH kernel (scalar or vectorized).
                       If None, uses cubic spline kernel.
        kernel_w0: W(0) value.  If None, computed from kernel_w_func.
        n_quadrature: Gauss-Legendre points per axis (8 gives ~300 samples
                      in the sphere; matches SPlisHSPlasH's 30-point Gauss).
        scale_factor: Empirical scaling (0.8 in SPlisHSPlasH).

    Returns:
        volume_values: 3D array same shape as sdf_grid.sdf_values containing
                       the volumetric boundary fraction at each grid cell.
    """
    if kernel_w_func is None:
        from sph.kernel import CubicSplineKernel
        h = support_radius / 2.0
        kernel = CubicSplineKernel(h=h, dim=3)
        kernel_w_func = kernel.W
        kernel_w0 = kernel.W(0.0)
    if kernel_w0 is None:
        kernel_w0 = float(kernel_w_func(0.0))

    sdf = sdf_grid.sdf_values.astype(np.float64)
    res = sdf.shape
    vol = np.zeros(res, dtype=np.float64)

    # Grid geometry
    grid_min = sdf_grid.grid_min.astype(np.float64)
    grid_max = sdf_grid.grid_max.astype(np.float64)
    grid_span = grid_max - grid_min
    cell_size_vec = grid_span / np.array([max(r - 1, 1) for r in res], dtype=np.float64)

    # Generate quadrature points within the support sphere
    quad_pts, quad_wts = _gauss_legendre_samples_3d(support_radius, n_quadrature)
    n_quad = len(quad_pts)

    # Identify grid cells near the boundary (|SDF| < 2 * support_radius)
    # Only these cells need the expensive volume integral.
    sdf_flat = sdf.ravel()
    near_mask = np.abs(sdf_flat) < 2.0 * support_radius
    near_indices = np.nonzero(near_mask)[0]

    if len(near_indices) == 0:
        return vol

    # Grid cell centers for near-boundary cells
    iz, iy, ix = np.unravel_index(near_indices, res)
    cell_centers = grid_min[None, :] + np.column_stack([
        iz.astype(np.float64),
        iy.astype(np.float64),
        ix.astype(np.float64),
    ]) * cell_size_vec[None, :]

    n_cells = len(near_indices)

    # Process in batches to control memory (batch_size cells at a time)
    batch_size = max(1, min(2048, 50_000_000 // max(n_quad, 1)))
    vol_flat = vol.ravel()

    for start in range(0, n_cells, batch_size):
        end = min(start + batch_size, n_cells)
        batch_centers = cell_centers[start:end]  # (B, 3)
        b = batch_centers.shape[0]

        # Query positions: (B, M, 3)
        query_pts = batch_centers[:, None, :] + quad_pts[None, :, :]
        query_pts_flat = query_pts.reshape(-1, 3)  # (B*M, 3)

        # Evaluate SDF at query positions using trilinear interpolation
        sdf_at_queries = sdf_grid.sample_signed_distance(query_pts_flat)
        sdf_at_queries = sdf_at_queries.reshape(b, n_quad)  # (B, M)

        # Handle NaN (outside grid) as very large positive SDF (exterior)
        nan_mask = ~np.isfinite(sdf_at_queries)
        sdf_at_queries[nan_mask] = support_radius + 1.0

        # Evaluate Bender integrand for all query points
        integrand = _bender_integrand(
            sdf_at_queries.ravel(),
            support_radius,
            kernel_w0,
            kernel_w_func,
        ).reshape(b, n_quad)

        # Weighted sum (Gauss quadrature)
        cell_volumes = np.dot(integrand, quad_wts)  # (B,)

        # Normalize: divide by total weight sum to get fraction in [0, 1],
        # then apply empirical scale factor (0.8 in SPlisHSPlasH).
        total_weight = np.sum(quad_wts)
        cell_volumes = scale_factor * cell_volumes / max(total_weight, 1e-30)

        vol_flat[near_indices[start:end]] = cell_volumes

    return vol


# ---------------------------------------------------------------------------
# Volume Map Grid
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class VolumeMapGrid:
    """SDF grid extended with the Bender 2019 volume integral field.

    The ``volume_values`` array has the same shape and grid geometry as the
    underlying SDF grid.  Each cell stores the boundary volume fraction: the
    fraction of the kernel support that is occupied by the boundary solid.
    """
    sdf_grid: MeshSDFGrid
    volume_values: np.ndarray
    support_radius: float

    @property
    def grid_min(self) -> np.ndarray:
        return self.sdf_grid.grid_min

    @property
    def grid_max(self) -> np.ndarray:
        return self.sdf_grid.grid_max

    @property
    def cell_size(self) -> float:
        return self.sdf_grid.cell_size

    @property
    def resolution(self) -> int:
        return self.sdf_grid.resolution

    def sample_volume(self, points: np.ndarray) -> np.ndarray:
        """Trilinearly interpolate the volume map at query points.

        Returns the boundary volume fraction at each point.  Points outside
        the grid domain return 0.0 (no boundary contribution).
        """
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"volume_map queries require shape (N, 3), got {points.shape}")

        n = points.shape[0]
        result = np.zeros(n, dtype=np.float64)
        if n == 0:
            return result

        resolution = self.resolution
        if resolution <= 1:
            result[:] = float(self.volume_values.reshape(-1)[0])
            return result

        grid_span = self.grid_max - self.grid_min
        step = grid_span / float(resolution - 1)
        inside = np.all(
            (points >= self.grid_min[None, :]) & (points <= self.grid_max[None, :]),
            axis=1,
        )
        if not np.any(inside):
            return result

        local = (points[inside] - self.grid_min[None, :]) / step[None, :]
        i0 = np.floor(local).astype(np.int64)
        i0 = np.clip(i0, 0, resolution - 2)
        frac = np.clip(local - i0.astype(np.float64), 0.0, 1.0)
        i1 = i0 + 1

        grid = self.volume_values.astype(np.float64)
        c000 = grid[i0[:, 0], i0[:, 1], i0[:, 2]]
        c001 = grid[i0[:, 0], i0[:, 1], i1[:, 2]]
        c010 = grid[i0[:, 0], i1[:, 1], i0[:, 2]]
        c011 = grid[i0[:, 0], i1[:, 1], i1[:, 2]]
        c100 = grid[i1[:, 0], i0[:, 1], i0[:, 2]]
        c101 = grid[i1[:, 0], i0[:, 1], i1[:, 2]]
        c110 = grid[i1[:, 0], i1[:, 1], i0[:, 2]]
        c111 = grid[i1[:, 0], i1[:, 1], i1[:, 2]]

        tx, ty, tz = frac[:, 0], frac[:, 1], frac[:, 2]
        c00 = c000 * (1.0 - tx) + c100 * tx
        c01 = c001 * (1.0 - tx) + c101 * tx
        c10 = c010 * (1.0 - tx) + c110 * tx
        c11 = c011 * (1.0 - tx) + c111 * tx
        c0 = c00 * (1.0 - ty) + c10 * ty
        c1 = c01 * (1.0 - ty) + c11 * ty
        interpolated = c0 * (1.0 - tz) + c1 * tz

        result[inside] = interpolated
        return result

    def sample_boundary_normal(self, points: np.ndarray) -> np.ndarray:
        """Wall normal at query points (from the underlying SDF gradient)."""
        return self.sdf_grid.sample_wall_normal(points)

    def sample_signed_distance(self, points: np.ndarray) -> np.ndarray:
        """Signed distance at query points (from the underlying SDF)."""
        return self.sdf_grid.sample_signed_distance(points)


# ---------------------------------------------------------------------------
# Boundary Representation
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class VolumeMapBoundaryRepresentation(BoundaryRepresentation):
    """Bender 2019 volume-map boundary representation.

    Unlike particle-based boundaries, this representation does not store
    boundary particles.  Instead it provides a continuous volume field that
    can be queried at any fluid particle position to get the boundary
    contribution without explicit boundary neighbors.
    """
    grids: tuple[VolumeMapGrid, ...]

    @property
    def kind(self) -> str:
        return "volume_map"

    @property
    def sources(self) -> tuple[BoundaryRepresentationSource, ...]:
        return tuple(
            grid.sdf_grid.source.as_representation_source() for grid in self.grids
        )

    def sample_volume(self, points: np.ndarray) -> np.ndarray:
        """Query the aggregate boundary volume at fluid particle positions.

        When multiple mesh sources exist, returns the maximum volume across
        all grids (conservative union).
        """
        points = np.asarray(points, dtype=np.float64)
        if not self.grids:
            return np.zeros(points.shape[0], dtype=np.float64)

        if len(self.grids) == 1:
            return self.grids[0].sample_volume(points)

        volumes = np.vstack([g.sample_volume(points) for g in self.grids])
        return np.max(volumes, axis=0)

    def sample_signed_distance(self, points: np.ndarray) -> np.ndarray:
        """Aggregate signed distance (minimum absolute) from underlying SDFs."""
        points = np.asarray(points, dtype=np.float64)
        if not self.grids:
            return np.full(points.shape[0], np.nan, dtype=np.float64)

        sampled = np.vstack([g.sample_signed_distance(points) for g in self.grids])
        abs_sampled = np.abs(sampled)
        abs_sampled[~np.isfinite(abs_sampled)] = np.inf
        best_idx = np.argmin(abs_sampled, axis=0)
        return sampled[best_idx, np.arange(sampled.shape[1])]

    def sample_boundary_normal(self, points: np.ndarray) -> np.ndarray:
        """Wall normal from closest SDF grid."""
        points = np.asarray(points, dtype=np.float64)
        if not self.grids:
            return np.full(points.shape, np.nan, dtype=np.float64)

        sampled = np.vstack([g.sample_signed_distance(points) for g in self.grids])
        abs_sampled = np.abs(sampled)
        abs_sampled[~np.isfinite(abs_sampled)] = np.inf
        best_idx = np.argmin(abs_sampled, axis=0)
        normals = np.stack(
            [g.sample_boundary_normal(points) for g in self.grids], axis=0
        )
        return normals[best_idx, np.arange(points.shape[0])]

    def compute_boundary_psi(
        self,
        fluid_positions: np.ndarray,
        rho0: float,
        mass: float,
    ) -> np.ndarray:
        """Compute per-fluid-particle boundary psi from the volume map.

        This produces a psi array compatible with the existing particle-based
        boundary density formula:  rho_fb_i = sum_b psi_b * W(r_ib).

        For volume-map boundaries, psi is derived from the continuous volume
        field:  psi_i = rho0 * V(x_i), where V is the boundary volume
        fraction at the fluid particle position.

        This can be used as a drop-in replacement for particle psi in the
        density computation, or as a validation reference.
        """
        vol = self.sample_volume(fluid_positions)
        return rho0 * vol

    def project_particle_positions(
        self,
        dim: int,
        spacing: float,
        deduplicate: bool = False,
    ) -> np.ndarray:
        raise RuntimeError(
            "volume_map boundaries are implicit and do not project to "
            "particle positions.  Use sample_volume() for boundary queries."
        )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_volume_map_grid(
    sdf_grid: MeshSDFGrid,
    support_radius: float,
    n_quadrature: int = 8,
) -> VolumeMapGrid:
    """Build a VolumeMapGrid from a pre-computed SDF grid."""
    volume_values = compute_volume_field(
        sdf_grid,
        support_radius=support_radius,
        n_quadrature=n_quadrature,
    )
    return VolumeMapGrid(
        sdf_grid=sdf_grid,
        volume_values=volume_values,
        support_radius=support_radius,
    )


def load_volume_map_representation_from_scene(
    scene: dict,
    scene_path: Path | None,
    spacing: float,
    support_radius: float,
    n_quadrature: int = 8,
) -> VolumeMapBoundaryRepresentation | None:
    """Build volume-map boundary representation when requested by the scene.

    Requires SDF-enabled boundaries.  The volume map is computed on top of
    the SDF grid using the Bender 2019 volume integral.
    """
    sdf_specs = parse_mesh_sdf_specs(scene, scene_path, spacing=spacing)
    if not sdf_specs:
        return None

    grids: list[VolumeMapGrid] = []
    for spec in sdf_specs:
        sdf_grid = _build_mesh_sdf_grid(spec)
        if not sdf_grid.quality.is_usable:
            errors = "; ".join(sdf_grid.quality.errors)
            raise ValueError(
                f"volume_map for '{spec.source_name}' requires usable SDF: {errors}"
            )
        print(
            f"Building volume map for '{spec.source_name}': "
            f"SDF resolution={sdf_grid.resolution}, "
            f"support_radius={support_radius:.4f}, "
            f"quadrature={n_quadrature}^3"
        )
        vm_grid = build_volume_map_grid(
            sdf_grid,
            support_radius=support_radius,
            n_quadrature=n_quadrature,
        )
        near_boundary = np.sum(vm_grid.volume_values > 0.01)
        print(
            f"  Volume map: {near_boundary} cells with V>0.01 "
            f"(range [{vm_grid.volume_values.min():.4f}, "
            f"{vm_grid.volume_values.max():.4f}])"
        )
        grids.append(vm_grid)

    return VolumeMapBoundaryRepresentation(grids=tuple(grids))
