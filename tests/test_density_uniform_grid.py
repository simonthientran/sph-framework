"""
Unit tests for density computation on uniform particle grids.

Validates that rho ≈ rho0 when particles are placed on a regular grid
with correct mass and kernel normalization.
"""
from __future__ import annotations

import numpy as np

from sph.core.physics import compute_particle_mass
from sph.core.state import ParticleState
from sph.neighbors.spatial_hash import SpatialHash
from sph.sph.density import compute_density_summation


def _make_uniform_grid_2d(
    xmin: float, xmax: float, ymin: float, ymax: float, dx: float, rho0: float
) -> ParticleState:
    """Create 2D uniform fluid grid."""
    x = np.arange(xmin, xmax + 1e-12, dx, dtype=np.float64)
    y = np.arange(ymin, ymax + 1e-12, dx, dtype=np.float64)
    X, Y = np.meshgrid(x, y, indexing="xy")
    pos = np.stack([X.ravel(), Y.ravel()], axis=1)
    n = pos.shape[0]
    mass = np.full((n,), compute_particle_mass(dx, rho0, 2), dtype=np.float64)
    rho = np.full((n,), rho0, dtype=np.float64)
    p = np.zeros((n,), dtype=np.float64)
    vel = np.zeros((n, 2), dtype=np.float64)
    acc = np.zeros((n, 2), dtype=np.float64)
    is_boundary = np.zeros((n,), dtype=np.bool_)
    return ParticleState(
        dim=2, pos=pos, vel=vel, acc=acc, mass=mass, rho=rho, p=p, is_boundary=is_boundary
    )


def _make_uniform_grid_3d(
    xmin: float, xmax: float, ymin: float, ymax: float, zmin: float, zmax: float,
    dx: float, rho0: float
) -> ParticleState:
    """Create 3D uniform fluid grid."""
    x = np.arange(xmin, xmax + 1e-12, dx, dtype=np.float64)
    y = np.arange(ymin, ymax + 1e-12, dx, dtype=np.float64)
    z = np.arange(zmin, zmax + 1e-12, dx, dtype=np.float64)
    X, Y, Z = np.meshgrid(x, y, z, indexing="xy")
    pos = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    n = pos.shape[0]
    mass = np.full((n,), compute_particle_mass(dx, rho0, 3), dtype=np.float64)
    rho = np.full((n,), rho0, dtype=np.float64)
    p = np.zeros((n,), dtype=np.float64)
    vel = np.zeros((n, 3), dtype=np.float64)
    acc = np.zeros((n, 3), dtype=np.float64)
    is_boundary = np.zeros((n,), dtype=np.bool_)
    return ParticleState(
        dim=3, pos=pos, vel=vel, acc=acc, mass=mass, rho=rho, p=p, is_boundary=is_boundary
    )


def test_density_uniform_grid_2d():
    """2D uniform grid: interior density should be close to rho0."""
    dx = 0.01
    rho0 = 1000.0
    h = 0.015  # h/dx = 1.5, in stable range [1.0, 2.0]
    support_radius = 2.0 * h

    # Use larger domain so interior (full support) dominates
    state = _make_uniform_grid_2d(0.0, 0.3, 0.0, 0.3, dx, rho0)
    ns = SpatialHash(support_radius=support_radius, dim=2)
    ns.build(state.pos)

    rho = compute_density_summation(state, ns, h)
    # Exclude boundary layer (within 2h of domain edge) - particle deficiency there
    margin = support_radius + dx * 0.5
    interior = (
        (state.pos[:, 0] >= margin)
        & (state.pos[:, 0] <= 0.3 - margin)
        & (state.pos[:, 1] >= margin)
        & (state.pos[:, 1] <= 0.3 - margin)
    )
    rho_interior = rho[interior]
    assert rho_interior.size > 0, "Need interior particles"
    rho_avg = float(np.mean(rho_interior))
    rel_err = abs(rho_avg - rho0) / rho0

    assert rel_err < 0.02, (
        f"2D interior density error {rel_err:.4%} exceeds 2%: rho_avg={rho_avg:.2f} rho0={rho0}"
    )


def test_density_uniform_grid_3d():
    """3D uniform grid: interior density should be close to rho0."""
    dx = 0.02
    rho0 = 1000.0
    h = 0.03  # h/dx = 1.5, in stable range [1.0, 2.0]
    support_radius = 2.0 * h

    # Use larger domain so interior (full support) dominates
    state = _make_uniform_grid_3d(0.0, 0.2, 0.0, 0.2, 0.0, 0.2, dx, rho0)
    ns = SpatialHash(support_radius=support_radius, dim=3)
    ns.build(state.pos)

    rho = compute_density_summation(state, ns, h)
    margin = support_radius + dx * 0.5
    interior = (
        (state.pos[:, 0] >= margin)
        & (state.pos[:, 0] <= 0.2 - margin)
        & (state.pos[:, 1] >= margin)
        & (state.pos[:, 1] <= 0.2 - margin)
        & (state.pos[:, 2] >= margin)
        & (state.pos[:, 2] <= 0.2 - margin)
    )
    rho_interior = rho[interior]
    assert rho_interior.size > 0, "Need interior particles"
    rho_avg = float(np.mean(rho_interior))
    rel_err = abs(rho_avg - rho0) / rho0

    assert rel_err < 0.02, (
        f"3D interior density error {rel_err:.4%} exceeds 2%: rho_avg={rho_avg:.2f} rho0={rho0}"
    )
