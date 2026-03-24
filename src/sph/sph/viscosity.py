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
    Explicit viscosity using the discrete Laplace operator in Eq. (23)
    of the SPH tutorial.

    We compute an acceleration a_visc = nu * ∇² v_i where the Laplacian of a
    vector field A is approximated by:

        ∇² A_i = - Σ_j (m_j / ρ_j) * A_ij * 2 ||∇_i W_ij|| / ||r_ij||

    with A_ij = A_i - A_j and r_ij = x_i - x_j.
    """
    n = state.n
    dim = state.dim
    a = np.zeros((n, dim), dtype=np.float64)

    eps = 1e-12

    for i in range(n):
        lap_v = np.zeros((dim,), dtype=np.float64)
        xi = state.pos[i]
        vi = state.vel[i]

        for j in neighbor_search.query(i, state.pos):
            xj = state.pos[j]
            vj = state.vel[j]
            rij = xi - xj
            r = float(np.linalg.norm(rij))
            if r <= eps:
                continue

            gradW = cubic_spline_gradW(rij, h=h, dim=dim)
            gradW_norm = float(np.linalg.norm(gradW))

            vij = vi - vj
            lap_v += -(state.mass[j] / (state.rho[j] + eps)) * vij * (2.0 * gradW_norm / r)

        a[i] = float(nu) * lap_v

    return a


