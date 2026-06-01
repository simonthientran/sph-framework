"""GPU-oriented backend skeleton using host fallback execution.

This backend is intentionally structural:
- it implements the SimulationBackend contract
- it introduces explicit backend-owned static/dynamic buffers
- it preserves NeighborPairs as the canonical interaction contract

What it does today:
- executes the current solver through the CPU backend
- mirrors state into backend-owned "device" buffers represented by NumPy arrays
- reports itself as ``gpu_stub`` in RuntimeStats

What moves first in a real device backend:
- neighbor / pair generation
- pairwise density / pressure / DFSPH kernels operating on NeighborPairs
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from sph.core.backend import RuntimeStats, SimulationBackend, SimulationStateView
from sph.core.scene import ExportSettings, SceneMetadata
from sph.core.state import ParticleState
from sph.neighbor_pairs import NeighborPairs

from .cpu_backend import NumbaCPUBackend


@dataclass(slots=True)
class GPUStaticBuffers:
    """Immutable or rarely changing backend-owned data."""

    boundary_positions: np.ndarray
    domain_min: np.ndarray
    domain_max: np.ndarray


@dataclass(slots=True)
class GPUDynamicBuffers:
    """Dynamic particle state mirrored by the backend each step."""

    fluid_positions: np.ndarray
    fluid_velocities: np.ndarray
    fluid_density: np.ndarray
    fluid_density_deviation: np.ndarray
    fluid_pressure: np.ndarray
    fluid_speed: np.ndarray
    fluid_neighbor_counts: np.ndarray
    fluid_low_density_mask: np.ndarray
    fluid_low_neighbor_mask: np.ndarray
    fluid_free_surface_score: np.ndarray
    fluid_rest_density: float
    boundary_velocities: np.ndarray


class GPUStubBackend:
    """
    GPU-oriented backend skeleton with host fallback execution.

    This is not a GPU solver yet. It is the insertion point for a future device
    backend with explicit ownership of:
    - static geometry/boundary data
    - dynamic particle state
    - canonical NeighborPairs that would later live on device memory
    """

    BACKEND_NAME = "gpu_stub"

    def __init__(self, scene_path: Path):
        self.scene_path = Path(scene_path)
        self._host_backend = NumbaCPUBackend(self.scene_path)
        self._static_buffers: GPUStaticBuffers | None = None
        self._dynamic_buffers: GPUDynamicBuffers | None = None
        self._device_pairs: NeighborPairs | None = None
        self._refresh_backend_owned_state()

    @property
    def backend_name(self) -> str:
        return self.BACKEND_NAME

    @property
    def solver_name(self) -> str:
        return self._host_backend.solver_name

    @property
    def scene_name(self) -> str:
        return self._host_backend.scene_name

    @property
    def scene_metadata(self) -> SceneMetadata:
        return self._host_backend.scene_metadata

    @property
    def export_settings(self) -> ExportSettings:
        return self._host_backend.export_settings

    @property
    def domain_min(self) -> np.ndarray:
        assert self._static_buffers is not None
        return self._static_buffers.domain_min

    @property
    def domain_max(self) -> np.ndarray:
        assert self._static_buffers is not None
        return self._static_buffers.domain_max

    @property
    def frame_index(self) -> int:
        return self._host_backend.frame_index

    @property
    def supports_device_execution(self) -> bool:
        """Advertise that this is still a host-fallback skeleton."""
        return False

    def load_scene(self, scene_path: Path) -> None:
        self.scene_path = Path(scene_path)
        self._host_backend.load_scene(self.scene_path)
        self._refresh_backend_owned_state()

    def reset(self) -> None:
        self._host_backend.reset()
        self._refresh_backend_owned_state()

    def step(self) -> RuntimeStats:
        stats = self._host_backend.step()
        self._refresh_backend_owned_state()
        stage_timings = dict(stats.stage_timings_ms)
        stage_timings["gpu_sync"] = 0.0
        return replace(
            stats,
            backend_name=self.backend_name,
            stage_timings_ms=stage_timings,
        )

    def state_view(self) -> SimulationStateView:
        assert self._static_buffers is not None
        assert self._dynamic_buffers is not None
        return SimulationStateView(
            fluid_positions=self._dynamic_buffers.fluid_positions.copy(),
            fluid_velocities=self._dynamic_buffers.fluid_velocities.copy(),
            fluid_density=self._dynamic_buffers.fluid_density.copy(),
            fluid_density_deviation=self._dynamic_buffers.fluid_density_deviation.copy(),
            fluid_pressure=self._dynamic_buffers.fluid_pressure.copy(),
            fluid_speed=self._dynamic_buffers.fluid_speed.copy(),
            fluid_neighbor_counts=self._dynamic_buffers.fluid_neighbor_counts.copy(),
            fluid_low_density_mask=self._dynamic_buffers.fluid_low_density_mask.copy(),
            fluid_low_neighbor_mask=self._dynamic_buffers.fluid_low_neighbor_mask.copy(),
            fluid_free_surface_score=self._dynamic_buffers.fluid_free_surface_score.copy(),
            fluid_rest_density=float(self._dynamic_buffers.fluid_rest_density),
            boundary_positions=self._static_buffers.boundary_positions.copy(),
            boundary_velocities=self._dynamic_buffers.boundary_velocities.copy(),
            domain_min=self._static_buffers.domain_min.copy(),
            domain_max=self._static_buffers.domain_max.copy(),
        )

    def particle_state(self) -> ParticleState:
        return self._host_backend.particle_state()

    def export_filename(self, kind: str, step: int) -> str:
        return self._host_backend.export_filename(kind, step)

    def _refresh_backend_owned_state(self) -> None:
        """
        Synchronize host results into backend-owned buffers.

        In a real GPU backend these buffers would be device arrays, and sync to
        SimulationStateView would become an explicit download path.
        """
        state = self._host_backend.state_view()
        self._static_buffers = GPUStaticBuffers(
            boundary_positions=state.boundary_positions.copy(),
            domain_min=state.domain_min.copy(),
            domain_max=state.domain_max.copy(),
        )
        self._dynamic_buffers = GPUDynamicBuffers(
            fluid_positions=state.fluid_positions.copy(),
            fluid_velocities=state.fluid_velocities.copy(),
            fluid_density=state.fluid_density.copy(),
            fluid_density_deviation=state.fluid_density_deviation.copy(),
            fluid_pressure=state.fluid_pressure.copy(),
            fluid_speed=state.fluid_speed.copy(),
            fluid_neighbor_counts=state.fluid_neighbor_counts.copy(),
            fluid_low_density_mask=state.fluid_low_density_mask.copy(),
            fluid_low_neighbor_mask=state.fluid_low_neighbor_mask.copy(),
            fluid_free_surface_score=state.fluid_free_surface_score.copy(),
            fluid_rest_density=float(state.fluid_rest_density),
            boundary_velocities=state.boundary_velocities.copy(),
        )
        sim = self._host_backend.sim
        if sim is not None:
            self._device_pairs = getattr(sim.fluid, "neighbor_pairs", None)
        else:
            self._device_pairs = None
