from __future__ import annotations

import numpy as np

from sph.sph.kernels import cubic_spline_W
from sph.neighbors.spatial_hash import SpatialHash
from sph.core.state import ParticleState


def compute_density_summation(
    state: ParticleState,
    neighbor_search: SpatialHash,
    h: float,
) -> np.ndarray:
    """
    Density reconstruction via SPH summation:

        rho_i = sum_j m_j W_ij

    Source: "SPH Techniques for the Physics Based Simulation of Fluids and Solids"
    (Mass Density Estimation, Eq. (11)).

    Notes:
    - This is purely geometric + mass-based, no continuity equation integration.
    - Near free surfaces, rho is often underestimated due to missing neighbors
      ("particle deficiency").
    """
    n = state.n
    rho = np.zeros((n,), dtype=np.float64)

    for i in range(n):
        # include self contribution (j = i) like in standard density summation
        rho_i = state.mass[i] * cubic_spline_W(np.zeros(state.dim), h, state.dim)

        for j in neighbor_search.query(i, state.pos):
            r = state.pos[i] - state.pos[j]
            rho_i += state.mass[j] * cubic_spline_W(r, h, state.dim)

        rho[i] = rho_i

    return rho
