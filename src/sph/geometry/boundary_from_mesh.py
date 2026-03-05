from __future__ import annotations

import numpy as np

from sph.core.state import ParticleState
from sph.boundary.mesh_sampling import generate_boundary_layers, sample_mesh_surface_uniform
from sph.geometry.mesh import Mesh


def create_boundary_particles_from_mesh(
    triangles: np.ndarray | Mesh,
    spacing: float,
    rho0: float,
    *,
    dim: int = 2,
    layers: int = 1,
    layer_mode: str = "outward",
) -> ParticleState:
    """
    Create boundary-only ParticleState from STL triangles.

    Triangles are sampled in 3D and projected to the requested simulation dim.
    """
    spacing = float(spacing)
    rho0 = float(rho0)
    if dim not in (2, 3):
        raise ValueError(f"dim must be 2 or 3, got {dim}")

    if isinstance(triangles, Mesh):
        mesh = triangles
    else:
        mesh = Mesh.from_triangle_vertices(np.asarray(triangles, dtype=np.float64))
    points3, normals3 = sample_mesh_surface_uniform(mesh, spacing=spacing)
    points3, _ = generate_boundary_layers(
        points3,
        normals3,
        layers=max(1, int(layers)),
        layer_spacing=spacing,
        direction=layer_mode,
    )
    pos = points3[:, :dim].copy()
    n = pos.shape[0]

    vel = np.zeros((n, dim), dtype=np.float64)
    acc = np.zeros((n, dim), dtype=np.float64)
    mass_value = rho0 * (spacing**dim)
    mass = np.full((n,), mass_value, dtype=np.float64)
    rho = np.full((n,), rho0, dtype=np.float64)
    p = np.zeros((n,), dtype=np.float64)
    is_boundary = np.ones((n,), dtype=np.bool_)

    state = ParticleState(
        dim=dim,
        pos=pos,
        vel=vel,
        acc=acc,
        mass=mass,
        rho=rho,
        p=p,
        is_boundary=is_boundary,
    )
    state.validate()
    return state

