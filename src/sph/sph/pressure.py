from __future__ import annotations

import numpy as np

from sph.core.state import ParticleState
from sph.neighbors.spatial_hash import SpatialHash
from sph.sph.kernels import cubic_spline_gradW


def pressure_state_equation_linear(rho: np.ndarray, rho0: float, k: float) -> np.ndarray:
    """
    Example linear state equation (WCSPH / SESP H idea):
        p_i = k (rho_i - rho0)

    Reference:
    - Document: "SPH Techniques for the Physics Based Simulation of Fluids and Solids - SPH_Tutorial.pdf"
    - Section: 4.4 "State Equation SPH (SESPH)" lists examples including:
        p_i = k(rho_i - rho0)
      (Note: shown as an example, not a numbered equation.)
    """
    rho0 = float(rho0)
    k = float(k)
    return k * (rho - rho0)


def pressure_acceleration_symmetric(
    state: ParticleState,
    neighbor_search: SpatialHash,
    h: float,
) -> np.ndarray:
    """
    Symmetric pressure acceleration:
        a^p_i = - sum_j m_j * ( p_i/rho_i^2 + p_j/rho_j^2 ) * ∇W_ij

    Reference:
    - Document: "SPH Techniques for the Physics Based Simulation of Fluids and Solids - SPH_Tutorial.pdf"
    - Equation: Eq. (53) (pressure acceleration in the PCISPH derivation).
    """
    n = state.n
    dim = state.dim

    a_p = np.zeros((n, dim), dtype=np.float64)
    eps = 1e-12  # numerical guard for rho^2 in case rho is accidentally ~0

    for i in range(n):
        pi = state.p[i]
        rhoi = state.rho[i]
        rhoi2 = rhoi * rhoi + eps

        acc = np.zeros((dim,), dtype=np.float64)

        for j in neighbor_search.query(i, state.pos):
            pj = state.p[j]
            rhoj = state.rho[j]
            rhoj2 = rhoj * rhoj + eps

            gradW = cubic_spline_gradW(state.pos[i] - state.pos[j], h=h, dim=dim)

            acc -= state.mass[j] * (pi / rhoi2 + pj / rhoj2) * gradW

        a_p[i] = acc

    return a_p


def pressure_state_equation_linear_section44(rho: np.ndarray, rho0: float, k: float) -> np.ndarray:
    """
    Section 4.4 linear state equation wrapper:
        p_i = k (rho_i - rho0)

    This is numerically identical to pressure_state_equation_linear; it is
    provided only to mirror the tutorial's Section 4.4 naming without
    changing the underlying physics.
    """
    return pressure_state_equation_linear(rho=rho, rho0=rho0, k=k)


def pressure_acceleration_with_boundaries_eq84(
    state: ParticleState,
    neighbor_search: SpatialHash,
    h: float,
    rho0: float,
) -> np.ndarray:
    """
    Pressure acceleration including boundary particles (particle-based boundary handling).

    Reference:
    - Document: "SPH Techniques for the Physics Based Simulation of Fluids and Solids - SPH_Tutorial.pdf"
    - Equation: Eq. (84) (pressure force with fluid neighbors and boundary neighbors)
    - Boundary assumptions described in Section 5.1.1:
        rho_boundary = rho0, p_boundary = p_i (pressure mirroring)

    We compute acceleration a_i = F_i / m_i.
    Only fluid particles receive pressure acceleration; boundary particles remain static.
    """
    n = state.n
    dim = state.dim
    a = np.zeros((n, dim), dtype=np.float64)

    eps = 1e-12
    fluid_ids = state.fluid_indices

    for i in fluid_ids:
        pi = state.p[i]
        rhoi = state.rho[i]
        rhoi2 = rhoi * rhoi + eps

        acc = np.zeros((dim,), dtype=np.float64)

        for j in neighbor_search.query(i, state.pos):
            gradW = cubic_spline_gradW(state.pos[i] - state.pos[j], h=h, dim=dim)

            if state.is_boundary[j]:
                # boundary neighbor: rho_j = rho0, p_j = p_i (mirroring)
                rhoj2 = float(rho0) * float(rho0) + eps
                pj = pi
            else:
                pj = state.p[j]
                rhoj = state.rho[j]
                rhoj2 = rhoj * rhoj + eps

            acc -= state.mass[j] * (pi / rhoi2 + pj / rhoj2) * gradW

        a[i] = acc

    return a


