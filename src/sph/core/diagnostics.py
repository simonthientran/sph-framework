from __future__ import annotations

"""
Observability: compute per-step diagnostics ("vital signs") for SPH runs.

What this module does:
- Defines a structured `StepDiagnostics` snapshot for one simulation step.
- Computes statistics for velocity, density, pressure, and neighbor counts.

How it works:
- Statistics are computed on FLUID particles only (boundary excluded), except
  for the counts `n_fluid` and `n_boundary`.
- Neighbor stats are computed from the provided `neighbor_search` and reflect
  the same compact-support neighborhood concept used by the solver.

Physics / solver constraints:
- This module is strictly read-only: it must not modify the particle state.
- It does not implement any SPH equations; it only reports quantities already
  produced by the solver step.

References:
- "SPH Techniques for the Physics Based Simulation of Fluids and Solids - SPH_Tutorial.pdf"
  - Algorithm 1: motivates per-step monitoring of rho/p/v since these are
    recomputed and used to compute forces and integration each step.
  - Eq. (33): CFL time step restriction (dt is reported, not computed here).
  - Eq. (83): density including boundary contributions (rho is reported here).
  - Eq. (84): pressure forces including boundary contributions (p is reported here).
"""

from dataclasses import dataclass

import numpy as np

from sph.core.state import ParticleState
from sph.neighbors.spatial_hash import SpatialHash


@dataclass(frozen=True)
class StepDiagnostics:
    """
    Structured diagnostics for one simulation step.

    All min/mean/max values are computed on FLUID particles only, except
    for the counts n_fluid/n_boundary.
    """

    step: int
    dt: float
    n_fluid: int
    n_boundary: int

    v_max: float

    rho_min: float
    rho_mean: float
    rho_max: float

    rho_rel_err_min: float
    rho_rel_err_mean: float
    rho_rel_err_max: float

    p_min: float
    p_mean: float
    p_max: float

    neigh_min: int
    neigh_mean: float
    neigh_max: int


def compute_step_diagnostics(
    step: int,
    dt: float,
    state: ParticleState,
    rho0: float,
    neighbor_search: SpatialHash,
) -> StepDiagnostics:
    """
    Compute diagnostics for a given step without mutating the simulation state.

    Args:
        step: 1-based step index for logging.
        dt: time step value used in this step (selected by solver; Eq. (33) if CFL).
        state: particle state after the solver step (Algorithm 1 ordering).
        rho0: reference/rest density.
        neighbor_search: neighbor search built on current positions.

    Returns:
        StepDiagnostics with min/mean/max values (fluid-only) and neighbor stats.
    """
    rho0 = float(rho0)

    is_boundary = state.is_boundary
    fluid_mask = ~is_boundary

    n_fluid = int(np.count_nonzero(fluid_mask))
    n_boundary = int(np.count_nonzero(is_boundary))

    if n_fluid == 0:
        # Degenerate scene: avoid reductions on empty arrays.
        return StepDiagnostics(
            step=int(step),
            dt=float(dt),
            n_fluid=0,
            n_boundary=n_boundary,
            v_max=0.0,
            rho_min=0.0,
            rho_mean=0.0,
            rho_max=0.0,
            rho_rel_err_min=0.0,
            rho_rel_err_mean=0.0,
            rho_rel_err_max=0.0,
            p_min=0.0,
            p_mean=0.0,
            p_max=0.0,
            neigh_min=0,
            neigh_mean=0.0,
            neigh_max=0,
        )

    fluid_ids = np.where(fluid_mask)[0]

    # Velocity max (fluid only)
    vnorm = np.linalg.norm(state.vel[fluid_ids], axis=1)
    v_max = float(np.max(vnorm)) if vnorm.size else 0.0

    # Density stats (fluid only)
    rho = state.rho[fluid_ids]
    rho_min = float(np.min(rho))
    rho_mean = float(np.mean(rho))
    rho_max = float(np.max(rho))

    # Relative density error (fluid only)
    rel_err = (rho - rho0) / rho0
    rho_rel_err_min = float(np.min(rel_err))
    rho_rel_err_mean = float(np.mean(rel_err))
    rho_rel_err_max = float(np.max(rel_err))

    # Pressure stats (fluid only)
    p = state.p[fluid_ids]
    p_min = float(np.min(p))
    p_mean = float(np.mean(p))
    p_max = float(np.max(p))

    # Neighbor counts (fluid only; returned neighbors can include boundary particles)
    neigh_counts = np.empty((n_fluid,), dtype=np.int64)
    for k, i in enumerate(fluid_ids):
        neigh_counts[k] = len(neighbor_search.query(int(i), state.pos))

    neigh_min = int(np.min(neigh_counts))
    neigh_mean = float(np.mean(neigh_counts))
    neigh_max = int(np.max(neigh_counts))

    return StepDiagnostics(
        step=int(step),
        dt=float(dt),
        n_fluid=n_fluid,
        n_boundary=n_boundary,
        v_max=v_max,
        rho_min=rho_min,
        rho_mean=rho_mean,
        rho_max=rho_max,
        rho_rel_err_min=rho_rel_err_min,
        rho_rel_err_mean=rho_rel_err_mean,
        rho_rel_err_max=rho_rel_err_max,
        p_min=p_min,
        p_mean=p_mean,
        p_max=p_max,
        neigh_min=neigh_min,
        neigh_mean=neigh_mean,
        neigh_max=neigh_max,
    )


