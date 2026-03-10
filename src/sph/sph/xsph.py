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

Important design choice for internal one-phase pipe flow:
- We smooth ONLY with fluid neighbors, not with static boundary particles.
- Reason: including zero-velocity wall particles in XSPH adds artificial
  numerical damping and suppresses the physically desired streamwise velocity.
- Boundary effects should be handled by pressure, viscosity, and boundary
  conditions, not by XSPH wall-drag.

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
    eps: float = 0.05,
) -> np.ndarray:
    """
    Compute an XSPH velocity correction dv for each particle.

    Common form:
        dv_i = eps * Σ_j (m_j / rho_j) * (v_j - v_i) * W_ij

    Conventions in this implementation:
    - Returns an array dv with shape (N, dim).
    - Applies correction only to fluid particles.
    - Uses only FLUID neighbors in the smoothing sum.
    - Returns dv=0 for boundary particles.
    - Does not modify `state`.

    Why only fluid neighbors?
    - Static boundary particles typically have zero velocity.
    - Including them in XSPH for internal pipe flow creates extra artificial
      damping near the wall and flattens the velocity profile too much.
    """
    n = state.n
    dim = state.dim
    dv = np.zeros((n, dim), dtype=np.float64)

    fluid_ids = state.fluid_indices
    is_boundary = state.is_boundary

    eps = float(eps)
    h = float(h)
    tiny = 1e-12

    if fluid_ids.size == 0:
        return dv

    for i in fluid_ids:
        vi = state.vel[i]
        corr = np.zeros((dim,), dtype=np.float64)

        for j in neighbor_search.query(int(i), state.pos):
            # Skip self
            if j == i:
                continue

            # IMPORTANT:
            # Only fluid-fluid XSPH smoothing.
            if is_boundary[j]:
                continue

            rhoj = float(state.rho[j])
            if rhoj <= tiny:
                continue

            vij = state.vel[j] - vi
            Wij = cubic_spline_W(state.pos[i] - state.pos[j], h=h, dim=dim)
            corr += (state.mass[j] / rhoj) * vij * Wij

        dv[i] = eps * corr

    return dv