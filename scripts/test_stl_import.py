"""
Test STL import pipeline end-to-end.
Does NOT run a simulation — only tests geometry loading.

Run from the sph-framework root:
    python scripts/test_stl_import.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is on the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from sph.io.geometry_loader import (
    load_mesh,
    sample_surface_poisson,
    compute_sdf_grid,
    fill_volume_from_sdf,
)

ASSETS = Path(__file__).parent.parent / "assets"


def test_boundary_sampling() -> None:
    print("\n── test_boundary_sampling ──")
    mesh = load_mesh(str(ASSETS / "box.stl"))
    dx = 0.05

    positions, normals = sample_surface_poisson(mesh, dx, n_layers=3)

    print(f"  Boundary particles : {len(positions)}")
    print(
        f"  Positions range    : "
        f"x=[{positions[:,0].min():.3f}, {positions[:,0].max():.3f}]  "
        f"y=[{positions[:,1].min():.3f}, {positions[:,1].max():.3f}]  "
        f"z=[{positions[:,2].min():.3f}, {positions[:,2].max():.3f}]"
    )
    print(
        f"  Normals unit check : {np.linalg.norm(normals, axis=1).mean():.4f} (expect ~1.0)"
    )

    assert len(positions) > 100, f"Too few boundary particles: {len(positions)}"
    assert not np.any(np.isnan(positions)), "NaN in positions"
    assert not np.any(np.isnan(normals)), "NaN in normals"
    print("  PASS: boundary sampling")


def test_volume_filling() -> None:
    print("\n── test_volume_filling ──")
    mesh = load_mesh(str(ASSETS / "box.stl"))
    dx = 0.05

    sdf, gmin, gmax, cell = compute_sdf_grid(mesh, resolution=30)
    fluid_pos = fill_volume_from_sdf(sdf, gmin, gmax, dx)

    print(f"  Fluid particles from mesh : {len(fluid_pos)}")
    print(f"  SDF grid cell size        : {cell:.4f}")
    assert len(fluid_pos) > 100, f"Too few fluid particles: {len(fluid_pos)}"
    assert not np.any(np.isnan(fluid_pos)), "NaN in fluid positions"
    print("  PASS: volume filling")


def test_adami_psi() -> None:
    """
    Verify Adami psi computation on STL-sampled particles.
    psi_b = rho0 / sum_k W(|x_b - x_k|)
    """
    print("\n── test_adami_psi ──")
    from scipy.spatial import cKDTree
    from sph.kernel import CubicSplineKernel

    mesh = load_mesh(str(ASSETS / "box.stl"))
    dx = 0.05
    h = 1.3 * dx
    rho0 = 1000.0
    kernel = CubicSplineKernel(h, dim=3)

    positions, _ = sample_surface_poisson(mesh, dx, n_layers=3)

    tree = cKDTree(positions)
    sr = 2.0 * h
    psi = np.zeros(len(positions))

    for b in range(len(positions)):
        neighbors = tree.query_ball_point(positions[b], sr)
        dists = np.linalg.norm(positions[b] - positions[neighbors], axis=1)
        w_sum = float(kernel.W(dists).sum())
        psi[b] = rho0 / w_sum if w_sum > 1e-6 else rho0 * h**3

    print(f"  psi range : {psi.min():.2f} – {psi.max():.2f}")
    print(f"  psi mean  : {psi.mean():.2f}  (expect ~{rho0 * dx**3:.2f})")
    assert not np.any(np.isnan(psi)), "NaN in psi"
    print("  PASS: Adami psi")


if __name__ == "__main__":
    missing = [p for p in ["box.stl", "pipe.stl", "sphere.stl"] if not (ASSETS / p).exists()]
    if missing:
        print(f"Missing assets: {missing}")
        print("Run:  python scripts/create_test_geometry.py  first.")
        sys.exit(1)

    test_boundary_sampling()
    test_volume_filling()
    test_adami_psi()
    print("\nAll tests PASSED")
