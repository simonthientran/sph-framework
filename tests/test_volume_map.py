"""Tests for volume-map boundary representation (Bender 2019)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sph.boundaries.volume_map import (
    VolumeMapBoundaryRepresentation,
    VolumeMapGrid,
    _bender_integrand,
    _gauss_legendre_samples_3d,
    build_volume_map_grid,
    compute_volume_field,
)
from sph.boundaries.mesh_sdf import MeshSDFGrid, MeshSDFQualityReport, MeshSDFSourceSpec
from sph.kernel import CubicSplineKernel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sphere_sdf(resolution: int = 32, radius: float = 0.3, center: float = 0.5) -> MeshSDFGrid:
    """Create a synthetic SDF grid for a sphere: SDF(x) = |x - center| - radius."""
    grid_min = np.array([0.0, 0.0, 0.0])
    grid_max = np.array([1.0, 1.0, 1.0])
    step = (grid_max - grid_min) / float(resolution - 1)

    coords = [np.arange(resolution) * step[ax] + grid_min[ax] for ax in range(3)]
    xx, yy, zz = np.meshgrid(coords[0], coords[1], coords[2], indexing="ij")
    dist = np.sqrt((xx - center) ** 2 + (yy - center) ** 2 + (zz - center) ** 2)
    sdf_values = dist - radius

    quality = MeshSDFQualityReport(
        is_watertight=True,
        mesh_bounds_min=grid_min,
        mesh_bounds_max=grid_max,
        cells_per_particle_spacing=2.0,
        sdf_min=float(np.min(sdf_values)),
        sdf_max=float(np.max(sdf_values)),
        warnings=(),
        errors=(),
    )
    source = MeshSDFSourceSpec.__new__(MeshSDFSourceSpec)
    object.__setattr__(source, "source", type("S", (), {
        "source_name": "test_sphere",
        "boundary_type": "synthetic",
        "as_representation_source": lambda self: type("R", (), {
            "boundary_type": "synthetic",
            "source_name": "test_sphere",
        })(),
    })())
    object.__setattr__(source, "policy", type("P", (), {
        "resolution": resolution,
        "resolution_source": "test",
        "padding_ratio": 0.0,
        "target_cell_size": float(step[0]),
        "cells_per_particle_spacing": 2.0,
    })())

    return MeshSDFGrid(
        source=source,
        sdf_values=sdf_values.astype(np.float32),
        grid_min=grid_min,
        grid_max=grid_max,
        cell_size=float(step[0]),
        quality=quality,
    )


def _make_halfspace_sdf(resolution: int = 32) -> MeshSDFGrid:
    """SDF for a half-space: boundary at x=0.5, SDF = x - 0.5 (interior is x < 0.5)."""
    grid_min = np.array([0.0, 0.0, 0.0])
    grid_max = np.array([1.0, 1.0, 1.0])
    step = (grid_max - grid_min) / float(resolution - 1)

    coords_x = np.arange(resolution) * step[0] + grid_min[0]
    sdf_values = np.zeros((resolution, resolution, resolution), dtype=np.float64)
    for i, x in enumerate(coords_x):
        sdf_values[i, :, :] = x - 0.5

    quality = MeshSDFQualityReport(
        is_watertight=True,
        mesh_bounds_min=grid_min,
        mesh_bounds_max=grid_max,
        cells_per_particle_spacing=2.0,
        sdf_min=float(np.min(sdf_values)),
        sdf_max=float(np.max(sdf_values)),
        warnings=(),
        errors=(),
    )
    source = MeshSDFSourceSpec.__new__(MeshSDFSourceSpec)
    object.__setattr__(source, "source", type("S", (), {
        "source_name": "test_halfspace",
        "boundary_type": "synthetic",
        "as_representation_source": lambda self: type("R", (), {
            "boundary_type": "synthetic",
            "source_name": "test_halfspace",
        })(),
    })())
    object.__setattr__(source, "policy", type("P", (), {
        "resolution": resolution,
        "resolution_source": "test",
        "padding_ratio": 0.0,
        "target_cell_size": float(step[0]),
        "cells_per_particle_spacing": 2.0,
    })())

    return MeshSDFGrid(
        source=source,
        sdf_values=sdf_values.astype(np.float32),
        grid_min=grid_min,
        grid_max=grid_max,
        cell_size=float(step[0]),
        quality=quality,
    )


# ---------------------------------------------------------------------------
# Quadrature tests
# ---------------------------------------------------------------------------

class TestGaussLegendreSamples:
    def test_samples_inside_sphere(self):
        r = 0.1
        pts, wts = _gauss_legendre_samples_3d(r, n_per_axis=6)
        distances = np.linalg.norm(pts, axis=1)
        assert np.all(distances <= r + 1e-12)
        assert len(pts) == len(wts)
        assert len(pts) > 10  # should have ~100+ points for n=6

    def test_weights_positive(self):
        pts, wts = _gauss_legendre_samples_3d(0.1, n_per_axis=5)
        assert np.all(wts > 0.0) or len(pts) == 0

    def test_symmetry(self):
        pts, _ = _gauss_legendre_samples_3d(0.2, n_per_axis=6)
        centroid = np.mean(pts, axis=0)
        assert np.allclose(centroid, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Bender integrand tests
# ---------------------------------------------------------------------------

class TestBenderIntegrand:
    def test_inside_boundary_returns_one(self):
        sdf = np.array([-0.1, -1.0, -0.001])
        h = 0.065
        kernel = CubicSplineKernel(h=h, dim=3)
        result = _bender_integrand(sdf, 2 * h, kernel.W(0.0), kernel.W)
        np.testing.assert_allclose(result, 1.0)

    def test_far_outside_returns_zero(self):
        sdf = np.array([0.5, 1.0, 10.0])
        h = 0.065
        kernel = CubicSplineKernel(h=h, dim=3)
        result = _bender_integrand(sdf, 2 * h, kernel.W(0.0), kernel.W)
        np.testing.assert_allclose(result, 0.0)

    def test_transition_zone_between_zero_and_one(self):
        h = 0.065
        support_radius = 2 * h
        sdf = np.array([0.01, 0.05, 0.1])
        kernel = CubicSplineKernel(h=h, dim=3)
        result = _bender_integrand(sdf, support_radius, kernel.W(0.0), kernel.W)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
        assert result[0] > result[1] > result[2]


# ---------------------------------------------------------------------------
# Volume field computation tests
# ---------------------------------------------------------------------------

class TestComputeVolumeField:
    def test_sphere_sdf_produces_nonzero_near_boundary(self):
        sdf_grid = _make_sphere_sdf(resolution=24, radius=0.3)
        h = 0.065
        vol = compute_volume_field(sdf_grid, support_radius=2 * h, n_quadrature=5)
        assert vol.shape == sdf_grid.sdf_values.shape
        assert np.any(vol > 0.01)  # should have volume near the sphere surface
        assert vol.min() >= 0.0

    def test_far_from_boundary_volume_is_zero(self):
        sdf_grid = _make_sphere_sdf(resolution=24, radius=0.3, center=0.5)
        h = 0.065
        vol = compute_volume_field(sdf_grid, support_radius=2 * h, n_quadrature=5)
        # Grid center should be deep inside sphere → volume should be near 0
        # (only the transition zone around the surface has nonzero volume)
        center_idx = sdf_grid.sdf_values.shape[0] // 2
        center_sdf = sdf_grid.sdf_values[center_idx, center_idx, center_idx]
        if center_sdf < -2 * h:
            center_vol = vol[center_idx, center_idx, center_idx]
            assert center_vol < 0.01

    def test_halfspace_volume_monotonic(self):
        sdf_grid = _make_halfspace_sdf(resolution=32)
        h = 0.065
        support_radius = 2 * h
        vol = compute_volume_field(sdf_grid, support_radius=support_radius, n_quadrature=6)
        mid = vol.shape[1] // 2
        vol_slice = vol[:, mid, mid]
        # The volume peak is at the inner edge of the near-boundary band where
        # the kernel support is almost fully covered by the solid.  This is at
        # approximately boundary_idx - 2*support_radius/cell_step.
        boundary_idx = vol.shape[0] // 2
        cell_step = 1.0 / (vol.shape[0] - 1)
        expected_peak = int(boundary_idx - 2.0 * support_radius / cell_step)
        peak_idx = np.argmax(vol_slice)
        assert abs(peak_idx - expected_peak) < 4
        # Volume at boundary surface (x=0.5) should be > 0 (partially covered)
        assert vol_slice[boundary_idx] > 0.01
        # Volume far outside (x > 0.5 + 2h) should be ~0
        far_idx = min(boundary_idx + int(3 * support_radius / cell_step), vol.shape[0] - 1)
        assert vol_slice[far_idx] < 0.01


# ---------------------------------------------------------------------------
# VolumeMapGrid tests
# ---------------------------------------------------------------------------

class TestVolumeMapGrid:
    def test_build_and_query(self):
        sdf_grid = _make_halfspace_sdf(resolution=24)
        h = 0.065
        vm = build_volume_map_grid(sdf_grid, support_radius=2 * h, n_quadrature=5)
        assert isinstance(vm, VolumeMapGrid)
        assert vm.volume_values.shape == sdf_grid.sdf_values.shape

        # Query at the boundary (x=0.5, y=0.5, z=0.5): should have high volume
        pts_near = np.array([[0.5, 0.5, 0.5]])
        vol_near = vm.sample_volume(pts_near)
        assert vol_near[0] > 0.1

        # Query far from boundary (x=0.9, y=0.5, z=0.5): should be ~0
        pts_far = np.array([[0.9, 0.5, 0.5]])
        vol_far = vm.sample_volume(pts_far)
        assert vol_far[0] < 0.05

    def test_sample_volume_outside_grid_returns_zero(self):
        sdf_grid = _make_halfspace_sdf(resolution=16)
        h = 0.065
        vm = build_volume_map_grid(sdf_grid, support_radius=2 * h, n_quadrature=4)
        pts = np.array([[-1.0, 0.5, 0.5], [2.0, 0.5, 0.5]])
        vol = vm.sample_volume(pts)
        np.testing.assert_allclose(vol, 0.0)

    def test_sample_volume_shape(self):
        sdf_grid = _make_halfspace_sdf(resolution=16)
        vm = build_volume_map_grid(sdf_grid, support_radius=0.13, n_quadrature=4)
        pts = np.random.rand(50, 3)
        vol = vm.sample_volume(pts)
        assert vol.shape == (50,)

    def test_sdf_pass_through(self):
        sdf_grid = _make_halfspace_sdf(resolution=16)
        vm = build_volume_map_grid(sdf_grid, support_radius=0.13, n_quadrature=4)
        pts = np.array([[0.3, 0.5, 0.5], [0.7, 0.5, 0.5]])
        sdf = vm.sample_signed_distance(pts)
        assert sdf[0] < 0.0  # inside
        assert sdf[1] > 0.0  # outside


# ---------------------------------------------------------------------------
# VolumeMapBoundaryRepresentation tests
# ---------------------------------------------------------------------------

class TestVolumeMapBoundaryRepresentation:
    def test_kind(self):
        sdf_grid = _make_halfspace_sdf(resolution=16)
        vm = build_volume_map_grid(sdf_grid, support_radius=0.13, n_quadrature=4)
        rep = VolumeMapBoundaryRepresentation(grids=(vm,))
        assert rep.kind == "volume_map"

    def test_sample_volume(self):
        sdf_grid = _make_halfspace_sdf(resolution=24)
        vm = build_volume_map_grid(sdf_grid, support_radius=0.13, n_quadrature=5)
        rep = VolumeMapBoundaryRepresentation(grids=(vm,))
        pts = np.array([[0.5, 0.5, 0.5], [0.9, 0.5, 0.5]])
        vol = rep.sample_volume(pts)
        assert vol[0] > vol[1]

    def test_compute_boundary_psi(self):
        sdf_grid = _make_halfspace_sdf(resolution=24)
        vm = build_volume_map_grid(sdf_grid, support_radius=0.13, n_quadrature=5)
        rep = VolumeMapBoundaryRepresentation(grids=(vm,))
        pts = np.array([[0.5, 0.5, 0.5], [0.9, 0.5, 0.5]])
        psi = rep.compute_boundary_psi(pts, rho0=1000.0, mass=0.001)
        assert psi[0] > 0.0
        assert psi[0] > psi[1]
        # psi = rho0 * volume, so near boundary: psi > 0
        assert psi[0] < 1000.0  # volume < 1, so psi < rho0

    def test_project_particle_positions_raises(self):
        sdf_grid = _make_halfspace_sdf(resolution=16)
        vm = build_volume_map_grid(sdf_grid, support_radius=0.13, n_quadrature=4)
        rep = VolumeMapBoundaryRepresentation(grids=(vm,))
        with pytest.raises(RuntimeError, match="implicit"):
            rep.project_particle_positions(dim=3, spacing=0.05)

    def test_empty_representation(self):
        rep = VolumeMapBoundaryRepresentation(grids=())
        pts = np.array([[0.5, 0.5, 0.5]])
        vol = rep.sample_volume(pts)
        np.testing.assert_allclose(vol, 0.0)

    def test_multiple_grids_max_union(self):
        sdf1 = _make_halfspace_sdf(resolution=16)
        sdf2 = _make_sphere_sdf(resolution=16, radius=0.2, center=0.5)
        vm1 = build_volume_map_grid(sdf1, support_radius=0.13, n_quadrature=4)
        vm2 = build_volume_map_grid(sdf2, support_radius=0.13, n_quadrature=4)
        rep = VolumeMapBoundaryRepresentation(grids=(vm1, vm2))
        pts = np.random.rand(20, 3)
        vol = rep.sample_volume(pts)
        vol1 = vm1.sample_volume(pts)
        vol2 = vm2.sample_volume(pts)
        expected = np.maximum(vol1, vol2)
        np.testing.assert_allclose(vol, expected)


# ---------------------------------------------------------------------------
# Integration with real scene
# ---------------------------------------------------------------------------

class TestVolumeMapWithRealScene:
    @pytest.fixture(autouse=True)
    def _check_scene_exists(self):
        scene_path = Path("scenes/examples/stl_volume_map_3d.json").resolve()
        if not scene_path.exists():
            pytest.skip("stl_volume_map_3d.json not found")
        self.scene_path = scene_path

    def test_volume_map_scene_loads(self):
        scene = json.loads(self.scene_path.read_text())
        boundaries = scene.get("boundaries", [])
        assert len(boundaries) > 0
        reps = {r.lower() for r in boundaries[0].get("representations", [])}
        assert "volume_map" in reps

    def test_volume_map_scene_method_flag(self):
        scene = json.loads(self.scene_path.read_text())
        method = scene["boundaries"][0].get("boundary_handling_method", "particle")
        assert method == "volume_map"
