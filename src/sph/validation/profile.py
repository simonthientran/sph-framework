"""Velocity profile utilities for the pipe-flow benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sph.core.state import ParticleState


@dataclass
class VelocityProfile:
    y_centers: np.ndarray
    vx_avg: np.ndarray
    pipe_height: float

    @property
    def centerline_velocity(self) -> float:
        if self.y_centers.size == 0:
            return 0.0
        centerline_y = 0.5 * float(self.pipe_height)
        idx = int(np.argmin(np.abs(self.y_centers - centerline_y)))
        return float(self.vx_avg[idx])

    @property
    def vmax(self) -> float:
        return float(np.max(self.vx_avg)) if self.vx_avg.size else 0.0


def compute_velocity_profile(state: ParticleState, pipe_height: float, bins: int) -> VelocityProfile:
    pipe_height = float(pipe_height)
    if state.n == 0:
        centers = np.linspace(0.0, pipe_height, bins, endpoint=False)
        return VelocityProfile(y_centers=centers, vx_avg=np.zeros_like(centers), pipe_height=pipe_height)

    fluid_mask = ~state.is_boundary
    y = state.pos[fluid_mask, 1]
    vx = state.vel[fluid_mask, 0]

    hist, edges = np.histogram(y, bins=bins, range=(0.0, pipe_height))
    sums, _ = np.histogram(y, bins=bins, range=(0.0, pipe_height), weights=vx)

    avg = np.zeros_like(hist, dtype=np.float64)
    nz = hist > 0
    avg[nz] = sums[nz] / hist[nz]
    centers = 0.5 * (edges[:-1] + edges[1:])

    return VelocityProfile(y_centers=centers, vx_avg=avg, pipe_height=pipe_height)
