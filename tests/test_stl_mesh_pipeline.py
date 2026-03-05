from __future__ import annotations

from pathlib import Path

import numpy as np

from sph.boundary.mesh_sampling import compute_fluid_boundary_distance_stats, sample_mesh_surface_uniform
from sph.geometry.stl import load_stl_mesh


def test_load_ascii_stl_fixture_mesh_properties() -> None:
    fixture = Path(__file__).resolve().parent / "fixtures" / "square_ascii.stl"
    mesh = load_stl_mesh(fixture)
    assert mesh.triangle_count == 2
    assert mesh.vertex_count == 4
    assert np.all(np.isfinite(mesh.triangle_normals))
    assert np.allclose(mesh.bbox_min, [0.0, 0.0, 0.0])
    assert np.allclose(mesh.bbox_max, [1.0, 1.0, 0.0])
    assert np.isclose(mesh.surface_area, 1.0)


def test_boundary_sampling_count_scales_with_area() -> None:
    fixture = Path(__file__).resolve().parent / "fixtures" / "square_ascii.stl"
    mesh = load_stl_mesh(fixture)
    pts_coarse, _ = sample_mesh_surface_uniform(mesh, spacing=0.2)
    pts_fine, _ = sample_mesh_surface_uniform(mesh, spacing=0.1)
    assert pts_fine.shape[0] > pts_coarse.shape[0]
    # Halving spacing should increase count materially for 2D surface sampling.
    assert pts_fine.shape[0] >= int(2.5 * pts_coarse.shape[0])


def test_overlap_distance_stats_detects_close_particles() -> None:
    boundary = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
        ],
        dtype=np.float64,
    )
    fluid = np.array(
        [
            [0.0, 0.001],
            [0.1, 0.002],
            [0.2, 0.050],
        ],
        dtype=np.float64,
    )
    stats = compute_fluid_boundary_distance_stats(fluid, boundary, threshold=0.01)
    assert stats.min_distance < 0.01
    assert stats.close_count >= 2
    assert stats.close_fraction > 0.5

