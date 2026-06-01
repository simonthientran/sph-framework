"""Numba-CUDA backend coordination layer for single-phase flows."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Literal

import numpy as np
from numba import cuda

from sph.core.backend import RuntimeStats, SimulationStateView
from sph.core.scene import ExportSettings, SceneMetadata
from sph.core.state import ParticleState
from sph.neighbor_pairs import NeighborPairs
from sph.validation.contracts import CUDAValidationStageSnapshot

from .cpu_backend import NumbaCPUBackend
from .cuda_neighbor_search import CUDANeighborSearch
from .cuda_runtime import (
    CUDAAuthoritativeDeviceState,
    CUDAAuthoritativeExecutionContext,
    CUDAAuthoritativeStageRequest,
    CUDADynamicBuffers,
    CUDAExecutionResources,
    CUDAExecutionResult,
    CUDAReplayExecutionContext,
    CUDARuntime,
    CUDAStaticBuffers,
    CUDAStaticSceneResources,
)


class NumbaCUDABackend:
    """
    Coordination-focused CUDA backend over a dedicated internal runtime layer.

    The runtime layer owns CUDA resource state, replay execution, and
    authoritative stage execution.  The backend owns scene/backend lifecycle,
    top-level mode selection, host stats merging, and the public backend API.
    """

    BACKEND_NAME = "numba_cuda"
    _log = logging.getLogger("sph.cuda_backend")

    def __init__(self, scene_path: Path):
        self.scene_path = Path(scene_path)
        self._host_backend = NumbaCPUBackend(self.scene_path)
        self._runtime = CUDARuntime(self._host_backend)
        self._last_state_view: SimulationStateView | None = None
        self.debug_mode: bool = False
        self.diagnostic_interval: int = 50
        self._step_count: int = 0
        self._free_surface_warned: bool = False
        self._cuda_mode = self._detect_cuda_mode()
        self._refresh_static_buffers()
        self._refresh_host_state()
        self._check_free_surface_scene()

    @property
    def _scene_resources(self) -> CUDAStaticSceneResources:
        return self._runtime.scene_resources

    @_scene_resources.setter
    def _scene_resources(self, value: CUDAStaticSceneResources) -> None:
        self._runtime.scene_resources = value

    @property
    def _authoritative_state(self) -> CUDAAuthoritativeDeviceState:
        return self._runtime.authoritative_state

    @_authoritative_state.setter
    def _authoritative_state(self, value: CUDAAuthoritativeDeviceState) -> None:
        self._runtime.authoritative_state = value

    @property
    def _static_buffers(self) -> CUDAStaticBuffers | None:
        return self._runtime.static_buffers

    @_static_buffers.setter
    def _static_buffers(self, value: CUDAStaticBuffers | None) -> None:
        self._runtime.static_buffers = value

    @property
    def _cuda_ns(self) -> CUDANeighborSearch | None:
        return self._runtime.cuda_ns

    @_cuda_ns.setter
    def _cuda_ns(self, value: CUDANeighborSearch | None) -> None:
        self._runtime.cuda_ns = value

    @property
    def _dynamic_buffers(self) -> CUDADynamicBuffers | None:
        return self._runtime.dynamic_buffers

    @_dynamic_buffers.setter
    def _dynamic_buffers(self, value: CUDADynamicBuffers | None) -> None:
        self._runtime.dynamic_buffers = value

    @property
    def _pair_indices(self):
        return self._runtime.pair_indices

    @_pair_indices.setter
    def _pair_indices(self, value) -> None:
        self._runtime.pair_indices = value

    @property
    def _pair_geometry(self):
        return self._runtime.pair_geometry

    @_pair_geometry.setter
    def _pair_geometry(self, value) -> None:
        self._runtime.pair_geometry = value

    @property
    def _fast_step_device_state_valid(self) -> bool:
        return self._runtime.fast_step_device_state_valid

    @_fast_step_device_state_valid.setter
    def _fast_step_device_state_valid(self, value: bool) -> None:
        self._runtime.fast_step_device_state_valid = value

    @property
    def backend_name(self) -> str:
        return self.BACKEND_NAME

    @property
    def solver_name(self) -> str:
        return self._host_backend.solver_name

    @property
    def sim(self):
        """Compatibility alias for UI code that reads the host simulator directly."""
        return self._host_backend.sim

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
        buffers = self._static_buffers
        assert buffers is not None
        return buffers.domain_min

    @property
    def domain_max(self) -> np.ndarray:
        buffers = self._static_buffers
        assert buffers is not None
        return buffers.domain_max

    @property
    def frame_index(self) -> int:
        return self._host_backend.frame_index

    @property
    def supports_device_execution(self) -> bool:
        return True

    def enable_pair_build_diagnostics(self, enabled: bool = True) -> None:
        """Enable or disable sub-phase GPU event timing in the pair-build stage.

        Disabled by default for performance (eliminates ~4 GPU sync points per
        build call).  Enable only for profiling / diagnostics runs.
        """
        if self._cuda_ns is not None:
            self._cuda_ns.enable_pair_build_diagnostics(enabled)

    def load_scene(self, scene_path: Path) -> None:
        self.scene_path = Path(scene_path)
        self._host_backend.load_scene(self.scene_path)
        self._reset_authoritative_device_state()
        self._refresh_static_buffers()
        self._refresh_host_state()
        self._free_surface_warned = False
        self._check_free_surface_scene()

    def reset(self) -> None:
        self._host_backend.reset()
        self._reset_authoritative_device_state()
        self._refresh_static_buffers()
        self._refresh_host_state()

    def step(self) -> RuntimeStats:
        self._step_count += 1
        total_start = time.perf_counter()

        if self.debug_mode:
            execution = self._execute_debug_replay_step()
        else:
            execution = self._execute_authoritative_step()

        total_wall_time_ms = (time.perf_counter() - total_start) * 1000.0
        return self._merge_execution_result(execution, total_wall_time_ms)

    def _host_time_step(self) -> object | None:
        sim = self._host_backend.sim
        if sim is None:
            raise RuntimeError("CUDA backend host simulator is not initialized.")
        return getattr(sim, "time_step", None)

    def _configure_step_mode(self, mode: Literal["authoritative", "debug_replay"]) -> object | None:
        time_step = self._host_time_step()
        if time_step is None:
            return None
        if mode == "authoritative":
            time_step._cuda_solve_callback = self._cuda_solve_stage
            time_step.debug_mode = False
        else:
            time_step._cuda_solve_callback = None
            time_step.debug_mode = True
        return time_step

    def _execute_authoritative_step(self) -> CUDAExecutionResult:
        time_step = self._configure_step_mode("authoritative")
        host_stats = self._host_backend.step()
        self._refresh_host_state()

        cuda_timings = (
            dict(getattr(time_step, "_last_cuda_stage_timings_ms", {}))
            if time_step is not None
            else {}
        )
        cuda_metrics = (
            dict(getattr(time_step, "_last_cuda_stage_extra_metrics", {}))
            if time_step is not None
            else {}
        )
        cuda_metrics["host_step_ms"] = float(host_stats.wall_time_ms)
        return CUDAExecutionResult(
            mode="authoritative",
            host_stats=host_stats,
            cuda_timings=cuda_timings,
            cuda_metrics=cuda_metrics,
        )

    def _execute_debug_replay_step(self) -> CUDAExecutionResult:
        time_step = self._configure_step_mode("debug_replay")
        host_stats = self._host_backend.step()
        self._refresh_host_state()

        stage_snapshots = getattr(time_step, "last_cuda_stage_snapshots", None) if time_step is not None else None
        if not stage_snapshots:
            raise RuntimeError("CUDA backend requires DFSPH stage snapshots from the CPU reference path.")

        cuda_timings, cuda_metrics = self._run_cuda_replay(stage_snapshots)
        cuda_metrics["host_step_ms"] = float(host_stats.wall_time_ms)
        return CUDAExecutionResult(
            mode="debug_replay",
            host_stats=host_stats,
            cuda_timings=cuda_timings,
            cuda_metrics=cuda_metrics,
        )

    def _execution_scope_note(self, mode: Literal["authoritative", "debug_replay"]) -> str:
        if mode == "authoritative":
            return (
                "CUDA scope: device hash-grid pair build + pair geometry + density "
                "+ boundary state + viscosity + velocity predict + k-factor "
                "+ CD/DF PPE solve + position integration + periodic wrap + XSPH "
                "(fully authoritative, device-resident step). "
                "Host-side: CFL dt control, diagnostics."
            )
        return (
            "CUDA scope: host neighbor search + host outer iteration logic, device pair geometry "
            "+ density + k-factor + CD/DF pairwise kernels (debug/validation)."
        )

    def _execution_device_note(self, mode: Literal["authoritative", "debug_replay"]) -> str:
        if self._cuda_mode == "simulator":
            return "CUDA path executed in Numba CUDA simulator mode."
        if mode == "authoritative":
            return (
                "CUDA path executed on a real CUDA device for pair build, pair geometry, density, "
                "boundary state, viscosity, velocity predict, k-factor, CD/DF solve, "
                "position integration, periodic wrap, and XSPH."
            )
        return "CUDA path executed on a real CUDA device for pair geometry, density, k-factor, CD, and DF."

    def _runtime_notes_for_execution(self, execution: CUDAExecutionResult) -> tuple[str, ...]:
        notes = list(execution.host_stats.solver_health_notes)
        notes.append(self._execution_device_note(execution.mode))
        if self._cuda_ns is not None:
            notes.append(f"CUDA neighbor search ordering path: {self._cuda_ns.sort_backend}.")
        notes.append(self._execution_scope_note(execution.mode))
        return tuple(notes)

    def _merge_execution_result(
        self,
        execution: CUDAExecutionResult,
        total_wall_time_ms: float,
    ) -> RuntimeStats:
        merged_timings = dict(execution.host_stats.stage_timings_ms)
        merged_timings.update(execution.cuda_timings)
        merged_timings["total"] = total_wall_time_ms

        merged_metrics = dict(execution.host_stats.solver_metrics)
        merged_metrics.update(execution.cuda_metrics)

        return replace(
            execution.host_stats,
            backend_name=self.backend_name,
            wall_time_ms=total_wall_time_ms,
            stage_timings_ms=merged_timings,
            solver_metrics=merged_metrics,
            solver_health_notes=self._runtime_notes_for_execution(execution),
        )

    def state_view(self) -> SimulationStateView:
        if self._last_state_view is None:
            self._refresh_host_state()
        assert self._last_state_view is not None
        return self._last_state_view

    def particle_state(self) -> ParticleState:
        return self._host_backend.particle_state()

    def export_filename(self, kind: str, step: int) -> str:
        return self._host_backend.export_filename(kind, step)

    def _detect_cuda_mode(self) -> str:
        if os.environ.get("NUMBA_ENABLE_CUDASIM", "").strip() == "1":
            return "simulator"
        if cuda.is_available():
            return "device"
        raise RuntimeError(
            "Numba CUDA backend requested, but no CUDA device is available. "
            "Use NUMBA_ENABLE_CUDASIM=1 for simulator-backed validation on this machine."
        )

    def _refresh_static_buffers(self) -> None:
        self._runtime.refresh_static_buffers()

    def _refresh_host_state(self) -> None:
        self._last_state_view = self._host_backend.state_view()

    def _invalidate_fast_device_state(self) -> None:
        self._fast_step_device_state_valid = False

    def _reset_authoritative_device_state(self) -> None:
        self._runtime.reset_authoritative_device_state()

    def _decode_cuda_authoritative_stage_request(
        self,
        stage_name: str,
        callback_input: dict[str, object],
    ) -> CUDAAuthoritativeStageRequest:
        return self._runtime.decode_authoritative_stage_request(stage_name, callback_input)

    def _build_cuda_execution_resources(self) -> CUDAExecutionResources:
        return self._runtime.build_execution_resources()

    def _build_cuda_authoritative_execution_context(
        self,
        request: CUDAAuthoritativeStageRequest,
        fluid: object,
    ) -> CUDAAuthoritativeExecutionContext:
        return self._runtime.build_authoritative_execution_context(request, fluid)

    def _build_cuda_replay_execution_context(
        self,
        stage_snapshot: CUDAValidationStageSnapshot,
    ) -> CUDAReplayExecutionContext:
        return self._runtime.build_replay_execution_context(stage_snapshot)

    def _run_cuda_replay(
        self,
        stage_snapshots: dict[str, CUDAValidationStageSnapshot],
    ) -> tuple[dict[str, float], dict[str, float]]:
        return self._runtime.run_cuda_replay(stage_snapshots)

    def _cuda_solve_stage(
        self,
        stage_name: str,
        callback_input: dict[str, object],
        fluid: object,
        dt: float,
    ) -> tuple[dict, dict]:
        return self._runtime.cuda_solve_stage(stage_name, callback_input, fluid, dt)

    def _check_free_surface_scene(self) -> None:
        sim = self._host_backend.sim
        if sim is None:
            return
        ts = getattr(sim, "time_step", None)
        if ts is None:
            return
        if getattr(ts, "free_surface_stabilization_enable", False):
            self._log.warning(
                "CUDA authoritative path uses raw SPH density without "
                "free-surface stabilization.  Scene '%s' has stabilization "
                "enabled — DF-stage density will differ from CPU reference.  "
                "Use debug_mode=True for full CPU/CUDA comparison.",
                self.scene_name,
            )
            self._free_surface_warned = True

    def _periodic_config(self) -> tuple[float, int]:
        return self._runtime.periodic_config()

    @staticmethod
    def _density_support_counts(pairs: NeighborPairs, n_particles: int) -> tuple[np.ndarray, np.ndarray]:
        return CUDARuntime.density_support_counts(pairs, n_particles)

    @staticmethod
    def _to_device_or_none(arr: np.ndarray | None):
        return CUDARuntime.to_device_or_none(arr)
