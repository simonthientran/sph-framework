from __future__ import annotations

import numpy as np

from sph.core.state import ParticleState
from sph.neighbors.spatial_hash import SpatialHash
from sph.sph.kernels import cubic_spline_gradW


def viscosity_acceleration_laplace_eq23(
    state: ParticleState,
    neighbor_search: SpatialHash,
    h: float,
    nu: float,
) -> np.ndarray:
    """
    Physical viscosity acceleration for weakly compressible SPH.

    This implementation uses a robust pairwise SPH viscosity form that is
    better suited for single-phase viscous internal flow than the older
    norm(gradW)-based approximation.

    We compute

        a_visc_i = Σ_j m_j * ( 4 * nu * (r_ij · ∇W_ij) )
                           / ( (rho_i + rho_j) * (|r_ij|^2 + eps_h) ) * v_ij

    with
        r_ij = x_i - x_j
        v_ij = v_i - v_j

    Important implementation detail:
    - For periodic axes, r_ij must be the minimum-image relative vector.
      Therefore we use neighbor_search.relative_vector(...).

    Notes
    -----
    - Stronger and more robust momentum diffusion than the old simplified form.
    - Especially important for one-phase channel / pipe flow where wall shear
      must diffuse correctly into the fluid core.
    - Public API remains unchanged.
    """
    n = state.n
    dim = state.dim
    a = np.zeros((n, dim), dtype=np.float64)

    nu = float(nu)
    if nu <= 0.0:
        return a

    eps = 1e-12
    eps_h = 0.01 * float(h) * float(h)

    pos = state.pos
    vel = state.vel
    rho = state.rho
    mass = state.mass

    for i in range(n):
        xi = pos[i]
        vi = vel[i]
        rhoi = float(rho[i])

        a_i = np.zeros(dim, dtype=np.float64)

        for j in neighbor_search.query(i, pos):
            if j == i:
                continue

            rij = neighbor_search.relative_vector(xi, pos[j])
            r2 = float(np.dot(rij, rij))
            if r2 <= eps:
                continue

            gradW = cubic_spline_gradW(rij, h=h, dim=dim)
            r_dot_gradW = float(np.dot(rij, gradW))

            rhoj = float(rho[j])
            rho_pair = rhoi + rhoj
            if rho_pair <= eps:
                continue

            vij = vi - vel[j]

            coeff = mass[j] * (4.0 * nu) * r_dot_gradW / (rho_pair * (r2 + eps_h))
            a_i += coeff * vij

        a[i] = a_i

    return a