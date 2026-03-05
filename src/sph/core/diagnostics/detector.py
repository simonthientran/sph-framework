from __future__ import annotations

import numpy as np

from sph.core.diagnostics.metrics import StepMetrics
from sph.core.state import ParticleState


class InstabilityDetector:
    def __init__(
        self,
        *,
        rho0: float,
        neigh_min_threshold: int = 5,
        rho_min_frac: float = 0.7,
        rho_max_frac: float = 1.5,
        v_limit: float = 100.0,
    ) -> None:
        self.rho0 = float(rho0)
        self.neigh_min_threshold = int(neigh_min_threshold)
        self.rho_min_frac = float(rho_min_frac)
        self.rho_max_frac = float(rho_max_frac)
        self.v_limit = float(v_limit)

    def evaluate(self, metrics: StepMetrics, state: ParticleState) -> list[str]:
        flags: list[str] = []
        if metrics.neigh_min < self.neigh_min_threshold:
            flags.append("LOW_NEIGHBORS")
        if metrics.rho_min < self.rho_min_frac * self.rho0:
            flags.append("LOW_DENSITY")
        if metrics.rho_max > self.rho_max_frac * self.rho0:
            flags.append("HIGH_DENSITY")
        if metrics.v_max > self.v_limit:
            flags.append("HIGH_VELOCITY")
        if (
            not np.all(np.isfinite(state.pos))
            or not np.all(np.isfinite(state.vel))
            or not np.all(np.isfinite(state.rho))
            or not np.all(np.isfinite(state.p))
        ):
            flags.append("NONFINITE_STATE")
        return flags

