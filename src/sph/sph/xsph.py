from __future__ import annotations

"""
Optional XSPH velocity correction (velocity smoothing).

This module provides a helper used by the simulator as an optional
stabilization term. It is implemented as a standalone function so that
it does not entangle solver core code with optional features.

References:
- "SPH Techniques for the Physics Based Simulation of Fluids and Solids - SPH_Tutorial.pdf"
  - Algorithm 1: we apply optional stabilization after force evaluation and
    before integration, without changing the solver ordering.

Note:
- The SPH tutorial referenced by this project focuses on density/pressure
  formulations (Eq. (33), Eq. (83), Eq. (84)) and does not assign an equation
  number to XSPH. We therefore document it as an optional technique.
"""

import numpy as np

from sph.core.state import ParticleState
from sph.neighbors.spatial_hash import SpatialHash
from sph.sph.kernels import cubic_spline_W


def xsph_velocity_correction(
    state: ParticleState,
    neighbor_search: SpatialHash,
    h: float,
    eps: float = 0.1,
) -> np.ndarray:
    """
    Compute an XSPH velocity correction dv for each particle.

    Common form:
        dv_i = eps * Σ_j (m_j / ρ_j) * (v_j - v_i) * W_ij

    Conventions in this implementation:
    - Returns an array dv with shape (N, dim).
    - Applies correction to fluid particles; returns dv=0 for boundary particles.
    - Purely computes dv; the caller decides whether/where to apply it.
    - Does not modify `state`.
    """
    n = state.n
    dim = state.dim
    dv = np.zeros((n, dim), dtype=np.float64)

    fluid_ids = state.fluid_indices
    eps = float(eps)
    h = float(h)

    for i in fluid_ids:
        vi = state.vel[i]
        corr = np.zeros((dim,), dtype=np.float64)

        for j in neighbor_search.query(int(i), state.pos):
            # Include both fluid and boundary neighbors in the smoothing sum.
            # Boundary velocities are typically zero (static boundary samples),
            # which is consistent with the boundary handling approach.
            vij = state.vel[j] - vi
            Wij = cubic_spline_W(state.pos[i] - state.pos[j], h=h, dim=dim)
            corr += (state.mass[j] / state.rho[j]) * vij * Wij

        dv[i] = eps * corr

    return dv


