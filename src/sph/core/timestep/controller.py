from __future__ import annotations

import numpy as np

from sph.core.diagnostics.metrics import StepMetrics


class TimeStepController:
    def __init__(
        self,
        *,
        use_cfl: bool,
        cfl: float,
        h: float,
        dt_min: float,
        dt_max: float,
        ramp_up_max: float = 1.2,
        eps: float = 1e-12,
    ) -> None:
        self.use_cfl = bool(use_cfl)
        self.cfl = float(cfl)
        self.h = float(h)
        self.dt_min = float(dt_min)
        self.dt_max = float(dt_max)
        self.ramp_up_max = float(max(1.0, ramp_up_max))
        self.eps = float(eps)
        self.prev_dt: float | None = None
        self.last_reason_codes: list[str] = []

    def update(self, metrics: StepMetrics) -> float:
        reasons: list[str] = []
        dt_candidate = float(metrics.dt)
        if self.use_cfl:
            dt_cfl = self.cfl * self.h / (float(metrics.v_max) + self.eps)
            dt_candidate = min(dt_candidate, dt_cfl)
            reasons.append("CFL_V")
        else:
            reasons.append("FIXED_DT")

        if dt_candidate < self.dt_min:
            dt_candidate = self.dt_min
            reasons.append("CLAMP_MIN")
        if dt_candidate > self.dt_max:
            dt_candidate = self.dt_max
            reasons.append("CLAMP_MAX")

        if self.prev_dt is not None and dt_candidate > self.prev_dt * self.ramp_up_max:
            dt_candidate = self.prev_dt * self.ramp_up_max
            reasons.append("RAMP_UP_LIMIT")

        if not np.isfinite(dt_candidate):
            dt_candidate = self.dt_min
            reasons.append("NONFINITE_DT")

        self.prev_dt = float(dt_candidate)
        self.last_reason_codes = reasons
        metrics.dt_reason_codes = list(reasons)
        return float(dt_candidate)

