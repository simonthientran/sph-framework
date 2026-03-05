from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sph.core.state import ParticleState
from sph.neighbors.spatial_hash import SpatialHash


@dataclass
class StepMetrics:
    step: int
    time: float
    dt: float
    v_max: float
    rho_min: float
    rho_avg: float
    rho_max: float
    p_min: float
    p_avg: float
    p_max: float
    neigh_min: int
    neigh_avg: float
    neigh_max: int
    dt_reason_codes: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


def build_step_metrics(
    *,
    step: int,
    time: float,
    dt: float,
    state: ParticleState,
    rho0: float,
    neighbor_search: SpatialHash,
) -> tuple[StepMetrics, np.ndarray]:
    del rho0  # reserved for future metrics; keep signature stable.

    fluid_ids = state.fluid_indices
    if fluid_ids.size == 0:
        m = StepMetrics(
            step=int(step),
            time=float(time),
            dt=float(dt),
            v_max=0.0,
            rho_min=0.0,
            rho_avg=0.0,
            rho_max=0.0,
            p_min=0.0,
            p_avg=0.0,
            p_max=0.0,
            neigh_min=0,
            neigh_avg=0.0,
            neigh_max=0,
        )
        return m, np.zeros((0,), dtype=np.int64)

    vnorm = np.linalg.norm(state.vel[fluid_ids], axis=1)
    rho = state.rho[fluid_ids]
    p = state.p[fluid_ids]
    neigh_counts = np.array([len(neighbor_search.query(int(i), state.pos)) for i in fluid_ids], dtype=np.int64)

    m = StepMetrics(
        step=int(step),
        time=float(time),
        dt=float(dt),
        v_max=float(np.max(vnorm)) if vnorm.size else 0.0,
        rho_min=float(np.min(rho)),
        rho_avg=float(np.mean(rho)),
        rho_max=float(np.max(rho)),
        p_min=float(np.min(p)),
        p_avg=float(np.mean(p)),
        p_max=float(np.max(p)),
        neigh_min=int(np.min(neigh_counts)),
        neigh_avg=float(np.mean(neigh_counts)),
        neigh_max=int(np.max(neigh_counts)),
    )
    return m, neigh_counts

