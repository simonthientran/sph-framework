"""Diagnostics helpers for the modular SPH simulation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from sph.core.state import ParticleState
from sph.neighbors.spatial_hash import SpatialHash


@dataclass(slots=True)
class DiagnosticsConfig:
    rho0: float
    pipe_height: float
    velocity_profile_bins: int = 16


def compute_velocity_profile_y(state: ParticleState, pipe_height: float, bins: int) -> Dict[str, np.ndarray]:
    """Bin particles along the y-axis and average the x-velocity per bin."""

    if state.n == 0 or state.dim < 2:
        centers = np.linspace(0.0, float(pipe_height), bins, endpoint=False)
        zeros = np.zeros_like(centers)
        return {
            "y_centers": centers,
            "vx_avg": zeros,
            "expected_profile": zeros,
        }

    mask = ~state.is_boundary
    if not np.any(mask):
        mask = np.ones(state.n, dtype=bool)

    y = state.pos[mask, 1]
    vx = state.vel[mask, 0]

    if pipe_height <= 0.0:
        pipe_height = float(np.max(y) - np.min(y) + 1e-6)

    hist, edges = np.histogram(y, bins=bins, range=(0.0, float(pipe_height)))
    sums, _ = np.histogram(y, bins=bins, range=(0.0, float(pipe_height)), weights=vx)

    avg = np.zeros_like(hist, dtype=np.float64)
    nonzero = hist > 0
    avg[nonzero] = sums[nonzero] / hist[nonzero]

    centers = 0.5 * (edges[:-1] + edges[1:])
    vmax = float(np.max(np.abs(avg))) if avg.size else 0.0
    expected = vmax * (1.0 - ((centers / float(pipe_height)) - 0.5) ** 2) if pipe_height > 0 else np.zeros_like(centers)

    return {
        "y_centers": centers,
        "vx_avg": avg,
        "expected_profile": expected,
    }


class DiagnosticsManager:
    """Aggregates scalar diagnostics for logging/validation."""

    def __init__(self, config: DiagnosticsConfig):
        self.config = config

    def collect(self, state: ParticleState, neighbor_search: SpatialHash) -> Dict[str, object]:
        rho_err = self._density_error(state)
        pressure_min = float(np.min(state.p)) if state.p.size else 0.0
        pressure_max = float(np.max(state.p)) if state.p.size else 0.0
        vel_mag = np.linalg.norm(state.vel, axis=1)
        vel_max = float(np.max(vel_mag)) if vel_mag.size else 0.0

        neighbor_counts = self._neighbor_counts(state, neighbor_search)
        neighbor_min = int(np.min(neighbor_counts)) if neighbor_counts.size else 0
        neighbor_max = int(np.max(neighbor_counts)) if neighbor_counts.size else 0
        neighbor_avg = float(np.mean(neighbor_counts)) if neighbor_counts.size else 0.0

        profile = compute_velocity_profile_y(
            state=state,
            pipe_height=float(self.config.pipe_height),
            bins=int(self.config.velocity_profile_bins),
        )

        return {
            "density_error": rho_err,
            "pressure_min": pressure_min,
            "pressure_max": pressure_max,
            "velocity_max": vel_max,
            "neighbors_min": neighbor_min,
            "neighbors_max": neighbor_max,
            "neighbors_avg": neighbor_avg,
            "neighbor_counts": neighbor_counts,
            "velocity_profile": profile,
        }

    def _density_error(self, state: ParticleState) -> float:
        rho0 = float(self.config.rho0)
        mask = ~state.is_boundary if state.rho.size else np.array([], dtype=bool)
        rho = state.rho[mask] if np.any(mask) else state.rho
        if rho.size == 0 or rho0 <= 0.0:
            return 0.0
        return float(np.max(np.abs(rho - rho0)) / rho0)

    def _neighbor_counts(self, state: ParticleState, neighbor_search: SpatialHash) -> np.ndarray:
        counts = np.zeros(state.n, dtype=np.int32)
        for i in range(state.n):
            counts[i] = len(neighbor_search.query(i, state.pos))
        return counts
