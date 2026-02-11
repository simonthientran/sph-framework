from __future__ import annotations

import numpy as np

from sph.core.state import ParticleState
from sph.neighbors.spatial_hash import SpatialHash
from sph.sph.kernels import cubic_spline_W


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


def compute_density_with_boundaries_eq83(
    state: ParticleState,
    neighbor_search: SpatialHash,
    h: float,
    rho0: float,
) -> np.ndarray:
    """
    Density with boundary contributions (particle-based boundary handling).

    Reference:
    - Document: "SPH Techniques for the Physics Based Simulation of Fluids and Solids - SPH_Tutorial.pdf"
    - Equation: Eq. (83)
        rho_i = sum_if m_if W_iif + sum_ib m_ib W_iib
      We implement the same idea by summing over all neighbors (fluid + boundary)
      for fluid particles. Boundary particles keep rho = rho0 (static samples).

    Notes:
    - We include self contribution for fluid particles (common in density summation).
    - The summation form is consistent with compute_density_summation; the only
      structural change is that boundary neighbors are included and non-fluid
      particles are kept at rho0.
    """
    n = state.n
    rho = np.full((n,), float(rho0), dtype=np.float64)

    fluid_ids = state.fluid_indices

    W0 = cubic_spline_W(np.zeros(state.dim, dtype=np.float64), h=h, dim=state.dim)

    for i in fluid_ids:
        rho_i = state.mass[i] * W0

        for j in neighbor_search.query(i, state.pos):
            r = state.pos[i] - state.pos[j]
            rho_i += state.mass[j] * cubic_spline_W(r, h=h, dim=state.dim)

        rho[i] = rho_i

    return rho


