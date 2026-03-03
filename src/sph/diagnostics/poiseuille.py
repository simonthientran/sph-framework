from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ACTIVE_X_LIMIT = 1e8


@dataclass(frozen=True)
class PoiseuilleConfig:
    bins: int
    axis: int
    component: int
    sample_min: np.ndarray
    sample_max: np.ndarray
    channel_min: float
    channel_max: float
    nu: float
    drive_accel: float


class PoiseuilleDiagnostics:
    """
    Mid-step diagnostics for internal channel flow quality.

    Writes sampled profile + analytic profile + relative L2 error to CSV.
    """

    def __init__(self, out_file: str | Path, cfg: PoiseuilleConfig):
        self.out_file = Path(out_file)
        self.out_file.parent.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg

        headers = ["step", "time", "l2_rel"]
        headers.extend([f"vx_num_bin_{i}" for i in range(self.cfg.bins)])
        headers.extend([f"vx_ana_bin_{i}" for i in range(self.cfg.bins)])
        with self.out_file.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(headers)

    def sample_and_log(self, *, step: int, time_value: float, state) -> float:
        y_centers, vx_num = self._sample_numeric_profile(state)
        vx_ana = self._analytic_profile(y_centers)
        den = max(float(np.linalg.norm(vx_ana)), 1e-12)
        l2_rel = float(np.linalg.norm(vx_num - vx_ana) / den)

        row: list[float | int] = [int(step), float(time_value), float(l2_rel)]
        row.extend([float(v) for v in vx_num])
        row.extend([float(v) for v in vx_ana])
        with self.out_file.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(row)
        return l2_rel

    def _sample_numeric_profile(self, state) -> tuple[np.ndarray, np.ndarray]:
        fluid_ids = state.fluid_indices
        if fluid_ids.size == 0:
            y = np.linspace(self.cfg.channel_min, self.cfg.channel_max, self.cfg.bins, dtype=np.float64)
            return y, np.zeros((self.cfg.bins,), dtype=np.float64)

        pos = state.pos[fluid_ids]
        vel = state.vel[fluid_ids]
        active = pos[:, 0] < _ACTIVE_X_LIMIT
        pos = pos[active]
        vel = vel[active]
        if pos.size == 0:
            y = np.linspace(self.cfg.channel_min, self.cfg.channel_max, self.cfg.bins, dtype=np.float64)
            return y, np.zeros((self.cfg.bins,), dtype=np.float64)

        in_region = np.all((pos >= self.cfg.sample_min) & (pos <= self.cfg.sample_max), axis=1)
        pos = pos[in_region]
        vel = vel[in_region]
        if pos.size == 0:
            y = np.linspace(self.cfg.channel_min, self.cfg.channel_max, self.cfg.bins, dtype=np.float64)
            return y, np.zeros((self.cfg.bins,), dtype=np.float64)

        edges = np.linspace(self.cfg.channel_min, self.cfg.channel_max, self.cfg.bins + 1, dtype=np.float64)
        centers = 0.5 * (edges[:-1] + edges[1:])
        means = np.full((self.cfg.bins,), math.nan, dtype=np.float64)
        for b in range(self.cfg.bins):
            if b + 1 == self.cfg.bins:
                mask = (pos[:, self.cfg.axis] >= edges[b]) & (pos[:, self.cfg.axis] <= edges[b + 1])
            else:
                mask = (pos[:, self.cfg.axis] >= edges[b]) & (pos[:, self.cfg.axis] < edges[b + 1])
            if np.any(mask):
                means[b] = float(np.mean(vel[mask, self.cfg.component]))
        means = np.nan_to_num(means, nan=0.0)
        return centers, means

    def _analytic_profile(self, y: np.ndarray) -> np.ndarray:
        # Steady planar Poiseuille with acceleration a_x:
        #   u(y) = a_x / (2*nu) * (y-y0) * (H-(y-y0))
        y0 = float(self.cfg.channel_min)
        h = float(self.cfg.channel_max - self.cfg.channel_min)
        yy = y - y0
        if self.cfg.nu <= 0.0:
            return np.zeros_like(y, dtype=np.float64)
        return (float(self.cfg.drive_accel) / (2.0 * float(self.cfg.nu))) * yy * (h - yy)


def build_poiseuille_config(scene: dict) -> PoiseuilleConfig | None:
    cfg = scene.get("poiseuille_diagnostics", {})
    if not bool(cfg.get("enable", False)):
        return None

    domain = scene.get("domain", {})
    dmin = np.asarray(domain.get("min", []), dtype=np.float64)
    dmax = np.asarray(domain.get("max", []), dtype=np.float64)
    if dmin.size < 2 or dmax.size < 2:
        return None

    axis = int(cfg.get("axis", 1))
    comp = int(cfg.get("component", 0))
    bins = max(4, int(cfg.get("bins", 16)))
    channel_min = float(cfg.get("channel_min", dmin[axis]))
    channel_max = float(cfg.get("channel_max", dmax[axis]))

    sample_min = np.asarray(cfg.get("sample_min", dmin.tolist()), dtype=np.float64)
    sample_max = np.asarray(cfg.get("sample_max", dmax.tolist()), dtype=np.float64)

    nu = float(cfg.get("nu", scene.get("material", {}).get("viscosity", {}).get("nu", 0.0)))
    drive_mode = str(cfg.get("drive_mode", "body_force")).lower()
    drive_component = int(cfg.get("drive_component", comp))
    if drive_mode == "pressure_gradient":
        rho0 = float(scene.get("material", {}).get("rho0", 1000.0))
        dpdx = float(cfg.get("dpdx", 0.0))
        drive_accel = -dpdx / max(rho0, 1e-12)
    else:
        force = np.asarray(scene.get("forces", {}).get("body_force", [0.0, 0.0]), dtype=np.float64)
        drive_accel = float(cfg.get("drive_accel", force[drive_component] if drive_component < force.size else 0.0))

    return PoiseuilleConfig(
        bins=bins,
        axis=axis,
        component=comp,
        sample_min=sample_min,
        sample_max=sample_max,
        channel_min=channel_min,
        channel_max=channel_max,
        nu=nu,
        drive_accel=drive_accel,
    )

