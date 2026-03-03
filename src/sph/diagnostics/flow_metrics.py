from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ACTIVE_X_LIMIT = 1e8


@dataclass(frozen=True)
class PlaneSpec:
    name: str
    min: np.ndarray
    max: np.ndarray
    normal: np.ndarray


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    min: np.ndarray
    max: np.ndarray
    bins: int
    axis: int
    component: int


class FlowMetrics:
    """
    Runtime flow diagnostics writer for benchmark scenes.
    """

    def __init__(self, output_dir: str | Path, scene_name: str, config: dict):
        self.output_dir = Path(output_dir) / scene_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.output_dir / "metrics.csv"

        self.planes = self._parse_planes(config.get("planes", []))
        self.profiles = self._parse_profiles(config.get("profiles", []))

        self.headers = [
            "step",
            "time",
            "dt",
            "rho_min",
            "rho_mean",
            "rho_max",
            "v_max",
            "active_count",
            "inactive_count",
        ]
        self.headers.extend([f"mass_flow_{p.name}" for p in self.planes])
        for profile in self.profiles:
            self.headers.extend([f"vel_prof_{profile.name}_bin{i}" for i in range(profile.bins)])

        with self.metrics_file.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(self.headers)

    def log_step(self, step: int, time_value: float, dt: float, state) -> None:
        fluid_ids = state.fluid_indices
        if fluid_ids.size == 0:
            return

        fluid_pos = state.pos[fluid_ids]
        active_local = fluid_pos[:, 0] < _ACTIVE_X_LIMIT
        active_ids = fluid_ids[active_local]
        inactive_count = int(fluid_ids.size - active_ids.size)
        if active_ids.size == 0:
            return

        pos = state.pos[active_ids]
        vel = state.vel[active_ids]
        rho = state.rho[active_ids]
        mass = state.mass[active_ids]

        row: list[float | int] = [
            int(step),
            float(time_value),
            float(dt),
            float(np.min(rho)),
            float(np.mean(rho)),
            float(np.max(rho)),
            float(np.max(np.linalg.norm(vel, axis=1))),
            int(active_ids.size),
            inactive_count,
        ]

        for plane in self.planes:
            row.append(self._mass_flow(pos=pos, vel=vel, mass=mass, plane=plane))
        for profile in self.profiles:
            row.extend(self._profile_bins(pos=pos, vel=vel, profile=profile))

        with self.metrics_file.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(row)

    @staticmethod
    def _parse_planes(raw_planes: list[dict]) -> list[PlaneSpec]:
        planes: list[PlaneSpec] = []
        for idx, p in enumerate(raw_planes):
            name = str(p.get("name", f"plane_{idx}"))
            pmin = np.asarray(p.get("min", []), dtype=np.float64)
            pmax = np.asarray(p.get("max", []), dtype=np.float64)
            normal = np.asarray(p.get("normal", []), dtype=np.float64)
            if pmin.shape != pmax.shape or pmin.shape != normal.shape:
                continue
            nrm = float(np.linalg.norm(normal))
            normal_unit = normal / nrm if nrm > 0.0 else normal
            planes.append(PlaneSpec(name=name, min=pmin, max=pmax, normal=normal_unit))
        return planes

    @staticmethod
    def _parse_profiles(raw_profiles: list[dict]) -> list[ProfileSpec]:
        profiles: list[ProfileSpec] = []
        for idx, p in enumerate(raw_profiles):
            name = str(p.get("name", f"profile_{idx}"))
            pmin = np.asarray(p.get("min", []), dtype=np.float64)
            pmax = np.asarray(p.get("max", []), dtype=np.float64)
            if pmin.shape != pmax.shape:
                continue
            bins = max(1, int(p.get("bins", 16)))
            axis = int(p.get("axis", 1))
            component = int(p.get("component", 0))
            profiles.append(ProfileSpec(name=name, min=pmin, max=pmax, bins=bins, axis=axis, component=component))
        return profiles

    @staticmethod
    def _mass_flow(*, pos: np.ndarray, vel: np.ndarray, mass: np.ndarray, plane: PlaneSpec) -> float:
        in_plane = np.all((pos >= plane.min) & (pos <= plane.max), axis=1)
        if not np.any(in_plane):
            return 0.0
        v_n = vel[in_plane] @ plane.normal
        return float(np.sum(mass[in_plane] * v_n))

    @staticmethod
    def _profile_bins(*, pos: np.ndarray, vel: np.ndarray, profile: ProfileSpec) -> list[float]:
        in_region = np.all((pos >= profile.min) & (pos <= profile.max), axis=1)
        if not np.any(in_region):
            return [math.nan for _ in range(profile.bins)]

        rpos = pos[in_region]
        rvel = vel[in_region]
        axis = int(profile.axis)
        comp = int(profile.component)
        y0 = float(profile.min[axis])
        y1 = float(profile.max[axis])
        edges = np.linspace(y0, y1, profile.bins + 1, dtype=np.float64)
        values: list[float] = []
        for b in range(profile.bins):
            if b + 1 == profile.bins:
                mask = (rpos[:, axis] >= edges[b]) & (rpos[:, axis] <= edges[b + 1])
            else:
                mask = (rpos[:, axis] >= edges[b]) & (rpos[:, axis] < edges[b + 1])
            if np.any(mask):
                values.append(float(np.mean(rvel[mask, comp])))
            else:
                values.append(math.nan)
        return values
