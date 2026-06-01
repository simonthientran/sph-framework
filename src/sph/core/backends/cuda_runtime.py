"""Internal CUDA runtime/execution layer for the Numba CUDA backend."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numba import cuda

from sph.core.backend import RuntimeStats
from sph.cuda_pair_ops import (
    DeviceDensityBuffers,
    DeviceKFactorBuffers,
    DevicePairGeometryBuffers,
    DevicePairIndexBuffers,
    _kernel_alpha_2d,
    _kernel_alpha_3d,
    apply_pressure_fb_kernel,
    apply_pressure_ff_kernel,
    boundary_vel_finalize_kernel,
    boundary_vel_scatter_kernel,
    cd_lambda_reduce_kernel,
    cd_residual_fb_kernel,
    cd_residual_ff_kernel,
    clip_upper_1d_kernel,
    combine_density_kernel,
    density_fb_kernel,
    density_ff_kernel,
    df_lambda_reduce_kernel,
    df_residual_fb_kernel,
    df_residual_ff_kernel,
    fill_1d_kernel,
    fill_2d_kernel,
    finalize_k_factor_kernel,
    k_factor_fb_kernel,
    k_factor_ff_kernel,
    launch_config,
    neighbor_count_kernel,
    pair_geometry_fb_kernel,
    pair_geometry_ff_kernel,
    periodic_wrap_x_kernel,
    position_integrate_kernel,
    add_velocity_delta_kernel,
    velocity_predict_kernel,
    viscosity_fb_kernel,
    viscosity_ff_kernel,
    xsph_kernel,
    zero_scalar_kernel,
)
from sph.neighbor_pairs import NeighborPairs
from sph.validation.contracts import (
    CUDAValidationStageInput,
    CUDAValidationStageOutput,
    CUDAValidationStageSnapshot,
)

from .cpu_backend import NumbaCPUBackend
from .cuda_neighbor_search import CUDANeighborSearch


@dataclass(slots=True)
class CUDAStaticBuffers:
    """Immutable or rarely changing device-owned data."""

    boundary_positions: object | None
    boundary_psi: object | None
    domain_min: np.ndarray
    domain_max: np.ndarray


@dataclass(slots=True)
class CUDAStaticSceneResources:
    """Scene-static CUDA resources rebuilt only when the scene/static state changes."""

    static_buffers: CUDAStaticBuffers | None = None
    neighbor_search: CUDANeighborSearch | None = None


@dataclass(slots=True)
class CUDADynamicBuffers:
    """Dynamic device-owned stage state."""

    fluid_positions: object
    fluid_velocities: object
    boundary_velocities: object | None
    density_ref: object
    k_factor_ref: object
    lambda_values: object
    lambda_next: object
    residual: object
    metric_sum: object
    density: DeviceDensityBuffers
    k_factor: DeviceKFactorBuffers


@dataclass(slots=True)
class CUDAAuthoritativeDeviceState:
    """Persistent device-owned authoritative state shared across CUDA stages."""

    dynamic_buffers: CUDADynamicBuffers | None = None
    pair_indices: DevicePairIndexBuffers | None = None
    pair_geometry: DevicePairGeometryBuffers | None = None
    fast_step_device_state_valid: bool = False


@dataclass(slots=True)
class CUDAExecutionResources:
    """Resolved CUDA resource bundle shared by authoritative and replay execution."""

    scene: CUDAStaticSceneResources
    device: CUDAAuthoritativeDeviceState

    def require_static_buffers(self) -> CUDAStaticBuffers:
        static_buffers = self.scene.static_buffers
        if static_buffers is None:
            raise RuntimeError("CUDA static scene resources are not initialized.")
        return static_buffers

    def require_dynamic_buffers(self) -> CUDADynamicBuffers:
        dynamic_buffers = self.device.dynamic_buffers
        if dynamic_buffers is None:
            raise RuntimeError("CUDA dynamic device buffers are not initialized.")
        return dynamic_buffers

    def require_pair_indices(self) -> DevicePairIndexBuffers:
        pair_indices = self.device.pair_indices
        if pair_indices is None:
            raise RuntimeError("CUDA pair index buffers are not initialized.")
        return pair_indices

    def require_pair_geometry(self) -> DevicePairGeometryBuffers:
        pair_geometry = self.device.pair_geometry
        if pair_geometry is None:
            raise RuntimeError("CUDA pair geometry buffers are not initialized.")
        return pair_geometry


@dataclass(slots=True)
class CUDAExecutionResult:
    """Result bundle for one backend step execution mode."""

    mode: Literal["authoritative", "debug_replay"]
    host_stats: RuntimeStats
    cuda_timings: dict[str, float]
    cuda_metrics: dict[str, float]


@dataclass(slots=True)
class CUDAPairPipelineResult:
    """Resolved pair-pipeline state for one authoritative CUDA stage."""

    gpu_ff_pairs: int
    gpu_fb_pairs: int
    cpu_ff_pairs: int
    cpu_fb_pairs: int
    fluid_counts: np.ndarray | None = None
    boundary_counts: np.ndarray | None = None


@dataclass(slots=True)
class CUDASolverExecutionResult:
    """Solver outputs needed for authoritative CPU writeback."""

    solve_metrics: dict[str, float]
    velocities_gpu: np.ndarray
    lambda_gpu: np.ndarray


@dataclass(slots=True)
class CUDAAuthoritativeStageRequest:
    """Internal authoritative-stage request after decoding the callback payload."""

    stage: Literal["cd", "df"]
    stage_input: CUDAValidationStageInput
    gpu_pre_cd: bool
    nu: float = 0.0
    gravity: np.ndarray | None = None
    boundary_rho0: float = 0.0
    boundary_mass: float = 0.0


@dataclass(slots=True)
class CUDAAuthoritativeWritebackTarget:
    """Host-visible writeback targets for one authoritative CUDA stage."""

    fluid: object
    boundary: object | None


@dataclass(slots=True)
class CUDAAuthoritativeExecutionContext:
    """Per-stage authoritative execution context with explicit ownership boundaries."""

    request: CUDAAuthoritativeStageRequest
    resources: CUDAExecutionResources
    writeback: CUDAAuthoritativeWritebackTarget


@dataclass(slots=True)
class CUDAReplayStageDownloads:
    """Downloaded replay-only arrays used for CPU/GPU comparison metrics."""

    rho_self: np.ndarray
    rho_ff: np.ndarray
    rho_fb: np.ndarray
    density: np.ndarray
    k_factor: np.ndarray


@dataclass(slots=True)
class CUDAReplayStageExecutionResult:
    """Execution result for one CUDA replay stage before CPU comparison."""

    timings: dict[str, float]
    solve_metrics: dict[str, float]
    velocities_gpu: np.ndarray
    lambda_gpu: np.ndarray
    downloads: CUDAReplayStageDownloads


@dataclass(slots=True)
class CUDAReplayExecutionContext:
    """Replay execution borrows scene-static and dynamic device resources explicitly."""

    stage_snapshot: CUDAValidationStageSnapshot
    resources: CUDAExecutionResources


@dataclass(slots=True)
class CUDAReplayStageComparison:
    """Typed comparison result for one replayed CUDA stage."""

    rho_self_linf: float
    rho_self_l1_mean: float
    rho_ff_linf: float
    rho_ff_l1_mean: float
    rho_fb_linf: float
    rho_fb_l1_mean: float
    density_linf: float
    density_l1_mean: float
    density_worst_idx: float
    density_worst_has_boundary: float
    density_worst_fluid_neighbors: float
    density_worst_boundary_neighbors: float
    density_worst_is_sparse: float
    k_linf: float
    k_l1_mean: float
    ff_pairs: float
    fb_pairs: float
    iterations: float
    converged_flag: float
    final_metric: float
    velocity_linf: float
    velocity_l1_mean: float
    lambda_linf: float
    lambda_l1_mean: float
    iterations_match: float
    converged_match: float
    metric_diff: float

    def as_metrics_dict(self) -> dict[str, float]:
        return {
            "rho_self_linf": self.rho_self_linf,
            "rho_self_l1_mean": self.rho_self_l1_mean,
            "rho_ff_linf": self.rho_ff_linf,
            "rho_ff_l1_mean": self.rho_ff_l1_mean,
            "rho_fb_linf": self.rho_fb_linf,
            "rho_fb_l1_mean": self.rho_fb_l1_mean,
            "density_linf": self.density_linf,
            "density_l1_mean": self.density_l1_mean,
            "density_worst_idx": self.density_worst_idx,
            "density_worst_has_boundary": self.density_worst_has_boundary,
            "density_worst_fluid_neighbors": self.density_worst_fluid_neighbors,
            "density_worst_boundary_neighbors": self.density_worst_boundary_neighbors,
            "density_worst_is_sparse": self.density_worst_is_sparse,
            "k_linf": self.k_linf,
            "k_l1_mean": self.k_l1_mean,
            "ff_pairs": self.ff_pairs,
            "fb_pairs": self.fb_pairs,
            "iterations": self.iterations,
            "converged_flag": self.converged_flag,
            "final_metric": self.final_metric,
            "velocity_linf": self.velocity_linf,
            "velocity_l1_mean": self.velocity_l1_mean,
            "lambda_linf": self.lambda_linf,
            "lambda_l1_mean": self.lambda_l1_mean,
            "iterations_match": self.iterations_match,
            "converged_match": self.converged_match,
            "metric_diff": self.metric_diff,
        }


class CUDARuntime:
    """Canonical internal owner for CUDA resources and execution logic."""

    def __init__(self, host_backend: NumbaCPUBackend):
        self._host_backend = host_backend
        self.scene_resources = CUDAStaticSceneResources()
        self.authoritative_state = CUDAAuthoritativeDeviceState()

    @property
    def static_buffers(self) -> CUDAStaticBuffers | None:
        return self.scene_resources.static_buffers

    @static_buffers.setter
    def static_buffers(self, value: CUDAStaticBuffers | None) -> None:
        self.scene_resources.static_buffers = value

    @property
    def cuda_ns(self) -> CUDANeighborSearch | None:
        return self.scene_resources.neighbor_search

    @cuda_ns.setter
    def cuda_ns(self, value: CUDANeighborSearch | None) -> None:
        self.scene_resources.neighbor_search = value

    @property
    def dynamic_buffers(self) -> CUDADynamicBuffers | None:
        return self.authoritative_state.dynamic_buffers

    @dynamic_buffers.setter
    def dynamic_buffers(self, value: CUDADynamicBuffers | None) -> None:
        self.authoritative_state.dynamic_buffers = value

    @property
    def pair_indices(self) -> DevicePairIndexBuffers | None:
        return self.authoritative_state.pair_indices

    @pair_indices.setter
    def pair_indices(self, value: DevicePairIndexBuffers | None) -> None:
        self.authoritative_state.pair_indices = value

    @property
    def pair_geometry(self) -> DevicePairGeometryBuffers | None:
        return self.authoritative_state.pair_geometry

    @pair_geometry.setter
    def pair_geometry(self, value: DevicePairGeometryBuffers | None) -> None:
        self.authoritative_state.pair_geometry = value

    @property
    def fast_step_device_state_valid(self) -> bool:
        return self.authoritative_state.fast_step_device_state_valid

    @fast_step_device_state_valid.setter
    def fast_step_device_state_valid(self, value: bool) -> None:
        self.authoritative_state.fast_step_device_state_valid = bool(value)

    # ── Scene-level helpers ──────────────────────────────────────────────────

    def _scene_dim(self) -> int:
        sim = self._host_backend.sim
        return int(getattr(sim, "dim", 2)) if sim else 2

    def _kernel_alpha(self, h: float) -> float:
        if self._scene_dim() == 3:
            return _kernel_alpha_3d(h)
        return _kernel_alpha_2d(h)

    def _gravity_array(self) -> np.ndarray:
        sim = self._host_backend.sim
        if sim is None:
            return np.zeros(2, dtype=np.float64)
        dim = self._scene_dim()
        g = getattr(sim, "gravity_vec", np.zeros(dim, dtype=np.float64))
        return np.asarray(g, dtype=np.float64)

    def _xsph_config(self) -> tuple[bool, float]:
        sim = self._host_backend.sim
        if sim is None:
            return False, 0.0
        return bool(getattr(sim, "enable_xsph", False)), float(getattr(sim, "xsph_eps", 0.0))

    def refresh_static_buffers(self) -> None:
        state = self._host_backend.state_view()
        boundary_positions = state.boundary_positions
        sim = self._host_backend.sim
        boundary_psi = None
        n_boundary = 0
        if sim is not None and sim.boundary is not None and sim.boundary.n:
            boundary_psi = sim.boundary.psi
            n_boundary = int(sim.boundary.n)

        static_buffers = CUDAStaticBuffers(
            boundary_positions=self.to_device_or_none(boundary_positions),
            boundary_psi=self.to_device_or_none(boundary_psi),
            domain_min=state.domain_min.copy(),
            domain_max=state.domain_max.copy(),
        )

        neighbor_search = None
        if sim is not None and sim.fluid is not None:
            periodic_x = None
            if getattr(sim, "x_min", None) is not None and getattr(sim, "x_max", None) is not None:
                periodic_x = (float(sim.x_min), float(sim.x_max))
            neighbor_search = CUDANeighborSearch(
                support_radius=float(sim.support_radius),
                domain_min=state.domain_min,
                domain_max=state.domain_max,
                n_fluid_max=int(sim.fluid.n),
                n_boundary=n_boundary,
                max_pairs=500_000,
                periodic_x=periodic_x,
                dim=self._scene_dim(),
            )
        self.scene_resources = CUDAStaticSceneResources(
            static_buffers=static_buffers,
            neighbor_search=neighbor_search,
        )

    def reset_authoritative_device_state(self) -> None:
        self.authoritative_state = CUDAAuthoritativeDeviceState()

    def decode_authoritative_stage_request(
        self,
        stage_name: str,
        callback_input: dict[str, object],
    ) -> CUDAAuthoritativeStageRequest:
        if stage_name not in {"cd", "df"}:
            raise ValueError(f"Unknown CUDA stage '{stage_name}'.")

        stage_input = CUDAValidationStageInput(
            dt=float(callback_input["dt"]),
            h=float(callback_input["h"]),
            rho0=float(callback_input["rho0"]),
            mass=float(callback_input["mass"]),
            max_iter=int(callback_input["max_iter"]),
            eta=float(callback_input["eta"]),
            positions=np.asarray(callback_input["positions"]),
            velocities=np.asarray(callback_input["velocities"]),
            densities=np.asarray(callback_input["densities"]),
            rho_self=np.asarray(callback_input["rho_self"]),
            rho_ff=np.asarray(callback_input["rho_ff"]),
            rho_fb=np.asarray(callback_input["rho_fb"]),
            k_factor=np.asarray(callback_input["k_factor"]),
            lambda_prev=np.asarray(callback_input["lambda_prev"]),
            boundary_velocities=np.asarray(callback_input["boundary_velocities"]),
            pairs=callback_input["pairs"],  # type: ignore[arg-type]
        )
        gpu_pre_cd = bool(callback_input.get("gpu_pre_cd", False)) and stage_name == "cd"

        # Build gravity vector from callback or scene
        gravity_raw = callback_input.get("gravity", None)
        if gravity_raw is not None:
            gravity = np.asarray(gravity_raw, dtype=np.float64)
        else:
            gravity = self._gravity_array()

        return CUDAAuthoritativeStageRequest(
            stage=stage_name,
            stage_input=stage_input,
            gpu_pre_cd=gpu_pre_cd,
            nu=float(callback_input.get("nu", 0.0)),
            gravity=gravity,
            boundary_rho0=float(callback_input.get("boundary_rho0", 0.0)),
            boundary_mass=float(callback_input.get("boundary_mass", 0.0)),
        )

    def build_execution_resources(self) -> CUDAExecutionResources:
        return CUDAExecutionResources(
            scene=self.scene_resources,
            device=self.authoritative_state,
        )

    def build_authoritative_execution_context(
        self,
        request: CUDAAuthoritativeStageRequest,
        fluid: object,
    ) -> CUDAAuthoritativeExecutionContext:
        sim = self._host_backend.sim
        boundary = sim.boundary if sim is not None else None
        return CUDAAuthoritativeExecutionContext(
            request=request,
            resources=self.build_execution_resources(),
            writeback=CUDAAuthoritativeWritebackTarget(
                fluid=fluid,
                boundary=boundary,
            ),
        )

    def build_replay_execution_context(
        self,
        stage_snapshot: CUDAValidationStageSnapshot,
    ) -> CUDAReplayExecutionContext:
        return CUDAReplayExecutionContext(
            stage_snapshot=stage_snapshot,
            resources=self.build_execution_resources(),
        )

    def run_cuda_replay(
        self,
        stage_snapshots: dict[str, CUDAValidationStageSnapshot],
    ) -> tuple[dict[str, float], dict[str, float]]:
        timings: dict[str, float] = {}
        metrics: dict[str, float] = {}

        for stage_name in ("cd", "df"):
            context = self.build_replay_execution_context(stage_snapshots[stage_name])
            stage_snapshot = context.stage_snapshot
            stage_execution = self._execute_cuda_replay_stage(context)
            stage_comparison = self._build_cuda_replay_stage_comparison(
                stage_snapshot,
                stage_execution,
            )
            for key, value in stage_execution.timings.items():
                timings[f"cuda_{stage_snapshot.stage}_{key}"] = value
            for key, value in stage_comparison.as_metrics_dict().items():
                metrics[f"cuda_{stage_snapshot.stage}_{key}"] = value

        metrics["cuda_density_linf"] = max(
            metrics.get("cuda_cd_density_linf", 0.0),
            metrics.get("cuda_df_density_linf", 0.0),
        )
        metrics["cuda_density_l1_mean"] = max(
            metrics.get("cuda_cd_density_l1_mean", 0.0),
            metrics.get("cuda_df_density_l1_mean", 0.0),
        )
        metrics["cuda_k_linf"] = max(
            metrics.get("cuda_cd_k_linf", 0.0),
            metrics.get("cuda_df_k_linf", 0.0),
        )
        metrics["cuda_k_l1_mean"] = max(
            metrics.get("cuda_cd_k_l1_mean", 0.0),
            metrics.get("cuda_df_k_l1_mean", 0.0),
        )
        return timings, metrics

    def cuda_solve_stage(
        self,
        stage_name: str,
        callback_input: dict[str, object],
        fluid: object,
        dt: float,
    ) -> tuple[dict, dict]:
        """Authoritative CUDA solve called from DFSPHTimeStep via callback."""
        _ = dt
        request = self.decode_authoritative_stage_request(stage_name, callback_input)
        context = self.build_authoritative_execution_context(request, fluid)

        timings: dict[str, float] = {
            "upload": 0.0,
            "pair_build": 0.0,
            "pair_build_hash_assign": 0.0,
            "pair_build_count_scan_scatter": 0.0,
            "pair_build_boundary_grid": 0.0,
            "pair_build_ff_emit": 0.0,
            "pair_build_fb_emit": 0.0,
            "pair_build_count_read": 0.0,
            "pair_build_materialize": 0.0,
            "neighbor_count": 0.0,
            "pair_geometry": 0.0,
            "density": 0.0,
            "k_factor": 0.0,
            "solve": 0.0,
            "metric_sync": 0.0,
            "position_integrate": 0.0,
            "download": 0.0,
        }
        if context.request.gpu_pre_cd:
            timings["boundary_state"] = 0.0
            timings["viscosity"] = 0.0
            timings["velocity_predict"] = 0.0
        if context.request.stage == "df":
            timings["xsph"] = 0.0

        self._prepare_cuda_device_state_phase(context, timings=timings)
        pair_pipeline = self._prepare_cuda_pair_pipeline_phase(context, timings=timings)
        self._run_cuda_pre_cd_phase(context, timings=timings)
        solver_execution = self._run_cuda_solver_execution_phase(context, timings=timings)
        self._writeback_cuda_stage_results(
            context,
            execution=solver_execution,
            timings=timings,
        )

        meta: dict = {
            "iters": int(solver_execution.solve_metrics["iterations"]),
            "converged": bool(solver_execution.solve_metrics["converged_flag"]),
            "metric": float(solver_execution.solve_metrics["final_metric"]),
            "gpu_ff_pairs": pair_pipeline.gpu_ff_pairs,
            "gpu_fb_pairs": pair_pipeline.gpu_fb_pairs,
            "cpu_ff_pairs": pair_pipeline.cpu_ff_pairs,
            "cpu_fb_pairs": pair_pipeline.cpu_fb_pairs,
            "pair_count_match": int(
                (
                    pair_pipeline.cpu_ff_pairs == 0
                    and pair_pipeline.cpu_fb_pairs == 0
                )
                or (
                    pair_pipeline.gpu_ff_pairs == pair_pipeline.cpu_ff_pairs
                    and pair_pipeline.gpu_fb_pairs == pair_pipeline.cpu_fb_pairs
                )
            ),
        }
        if pair_pipeline.fluid_counts is not None:
            meta["fluid_counts"] = pair_pipeline.fluid_counts
        if pair_pipeline.boundary_counts is not None:
            meta["boundary_counts"] = pair_pipeline.boundary_counts
        timings_out = {f"cuda_{stage_name}_{k}": v for k, v in timings.items()}
        return meta, timings_out

    def periodic_config(self) -> tuple[float, int]:
        sim = self._host_backend.sim
        if sim is not None and sim.x_min is not None and sim.x_max is not None:
            return float(sim.x_max - sim.x_min), 1
        return 0.0, 0

    @staticmethod
    def density_support_counts(pairs: NeighborPairs, n_particles: int) -> tuple[np.ndarray, np.ndarray]:
        fluid_neighbors = np.zeros(n_particles, dtype=np.int32)
        boundary_neighbors = np.zeros(n_particles, dtype=np.int32)
        if pairs.ff_i.size:
            fluid_neighbors += np.bincount(pairs.ff_i, minlength=n_particles).astype(np.int32, copy=False)
            fluid_neighbors += np.bincount(pairs.ff_j, minlength=n_particles).astype(np.int32, copy=False)
        if pairs.fb_i.size:
            boundary_neighbors += np.bincount(pairs.fb_i, minlength=n_particles).astype(np.int32, copy=False)
        return fluid_neighbors, boundary_neighbors

    @staticmethod
    def to_device_or_none(arr: np.ndarray | None):
        if arr is None:
            return None
        array = np.asarray(arr)
        if array.size == 0:
            return None
        return cuda.to_device(np.ascontiguousarray(array))

    def _ensure_dynamic_buffers(self, n: int, dim: int) -> bool:
        current = self.dynamic_buffers
        if current is not None:
            if (
                current.fluid_positions.shape == (n, dim)
                and current.fluid_velocities.shape == (n, dim)
                and current.density_ref.size == n
                and current.k_factor_ref.size == n
                and current.lambda_values.size == n
                and current.lambda_next.size == n
                and current.residual.size == n
                and current.density.rho_self.size == n
                and current.density.rho_ff.size == n
                and current.density.rho_fb.size == n
                and current.density.density.size == n
                and current.k_factor.sum_grad_x.size == n
                and current.k_factor.sum_grad_y.size == n
                and current.k_factor.sum_grad_z.size == n
                and current.k_factor.sum_grad_sq.size == n
                and current.k_factor.k_factor.size == n
            ):
                return False

        self.dynamic_buffers = CUDADynamicBuffers(
            fluid_positions=cuda.device_array((n, dim), dtype=np.float64),
            fluid_velocities=cuda.device_array((n, dim), dtype=np.float64),
            boundary_velocities=None,
            density_ref=cuda.device_array(n, dtype=np.float64),
            k_factor_ref=cuda.device_array(n, dtype=np.float64),
            lambda_values=cuda.device_array(n, dtype=np.float64),
            lambda_next=cuda.device_array(n, dtype=np.float64),
            residual=cuda.device_array(n, dtype=np.float64),
            metric_sum=cuda.device_array(1, dtype=np.float64),
            density=DeviceDensityBuffers(
                rho_self=cuda.device_array(n, dtype=np.float64),
                rho_ff=cuda.device_array(n, dtype=np.float64),
                rho_fb=cuda.device_array(n, dtype=np.float64),
                density=cuda.device_array(n, dtype=np.float64),
            ),
            k_factor=DeviceKFactorBuffers(
                sum_grad_x=cuda.device_array(n, dtype=np.float64),
                sum_grad_y=cuda.device_array(n, dtype=np.float64),
                sum_grad_z=cuda.device_array(n, dtype=np.float64),
                sum_grad_sq=cuda.device_array(n, dtype=np.float64),
                k_factor=cuda.device_array(n, dtype=np.float64),
            ),
        )
        return True

    def _ensure_boundary_velocity_buffer(
        self,
        resources: CUDAExecutionResources,
        shape: tuple[int, int],
    ) -> object:
        dynamic_buffers = resources.require_dynamic_buffers()
        current = dynamic_buffers.boundary_velocities
        if current is None or current.shape != shape:
            current = cuda.device_array(shape, dtype=np.float64)
            dynamic_buffers.boundary_velocities = current
        return current

    @staticmethod
    def _cuda_replay_stage_timings_template() -> dict[str, float]:
        return {
            "pair_upload": 0.0,
            "pair_geometry": 0.0,
            "density": 0.0,
            "k_factor": 0.0,
            "solve": 0.0,
            "metric_sync": 0.0,
            "download": 0.0,
        }

    def _download_cuda_replay_stage_outputs(
        self,
        resources: CUDAExecutionResources,
        timings: dict[str, float],
    ) -> CUDAReplayStageDownloads:
        dynamic_buffers = resources.require_dynamic_buffers()

        t0 = time.perf_counter()
        downloads = CUDAReplayStageDownloads(
            rho_self=dynamic_buffers.density.rho_self.copy_to_host(),
            rho_ff=dynamic_buffers.density.rho_ff.copy_to_host(),
            rho_fb=dynamic_buffers.density.rho_fb.copy_to_host(),
            density=dynamic_buffers.density.density.copy_to_host(),
            k_factor=dynamic_buffers.k_factor.k_factor.copy_to_host(),
        )
        cuda.synchronize()
        timings["download"] = (time.perf_counter() - t0) * 1000.0
        return downloads

    def _execute_cuda_replay_stage(
        self,
        context: CUDAReplayExecutionContext,
    ) -> CUDAReplayStageExecutionResult:
        stage_snapshot = context.stage_snapshot
        stage_input = stage_snapshot.stage_input
        resources = context.resources

        timings = self._cuda_replay_stage_timings_template()

        t0 = time.perf_counter()
        self._sync_dynamic_stage_state(resources, stage_input)
        self._sync_pair_indices(resources, stage_input.pairs)
        cuda.synchronize()
        timings["pair_upload"] = (time.perf_counter() - t0) * 1000.0

        periodic_length, use_periodic = self.periodic_config()
        ndim = self._scene_dim()

        t0 = time.perf_counter()
        self._prepare_pair_geometry(resources, periodic_length, use_periodic, ndim)
        cuda.synchronize()
        timings["pair_geometry"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        self._run_density_kernels(resources, stage_input)
        cuda.synchronize()
        timings["density"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        self._run_kfactor_kernels(resources, stage_input)
        cuda.synchronize()
        timings["k_factor"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        solve_metrics, metric_sync_ms, velocities_gpu, lambda_gpu = self._run_solver_stage(
            resources,
            stage_snapshot.stage,
            stage_input,
        )
        cuda.synchronize()
        timings["solve"] = (time.perf_counter() - t0) * 1000.0
        timings["metric_sync"] = metric_sync_ms

        downloads = self._download_cuda_replay_stage_outputs(resources, timings)
        return CUDAReplayStageExecutionResult(
            timings=timings,
            solve_metrics=solve_metrics,
            velocities_gpu=velocities_gpu,
            lambda_gpu=lambda_gpu,
            downloads=downloads,
        )

    @staticmethod
    def _build_cuda_replay_solver_metrics(
        stage_output: CUDAValidationStageOutput,
        solve_metrics: dict[str, float],
        velocities_gpu: np.ndarray,
        lambda_gpu: np.ndarray,
    ) -> tuple[float, float, float, float, float, float, float, float, float, float]:
        velocity_diff = np.abs(velocities_gpu - stage_output.velocities)
        lambda_diff = np.abs(lambda_gpu - stage_output.lambda_final)
        return (
            float(solve_metrics["iterations"]),
            float(solve_metrics["converged_flag"]),
            float(solve_metrics["final_metric"]),
            float(velocity_diff.max()) if velocity_diff.size else 0.0,
            float(velocity_diff.mean()) if velocity_diff.size else 0.0,
            float(lambda_diff.max()) if lambda_diff.size else 0.0,
            float(lambda_diff.mean()) if lambda_diff.size else 0.0,
            float(solve_metrics["iterations"] == stage_output.iterations),
            float(bool(solve_metrics["converged_flag"]) == stage_output.converged),
            abs(float(solve_metrics["final_metric"]) - stage_output.metric),
        )

    def _build_cuda_replay_stage_comparison(
        self,
        stage_snapshot: CUDAValidationStageSnapshot,
        execution: CUDAReplayStageExecutionResult,
    ) -> CUDAReplayStageComparison:
        stage_input = stage_snapshot.stage_input
        stage_output = stage_snapshot.stage_output
        if stage_output is None:
            raise RuntimeError(
                f"Replay snapshot for stage '{stage_snapshot.stage}' is missing its reference output."
            )
        rho_self_diff = np.abs(execution.downloads.rho_self - stage_input.rho_self)
        rho_ff_diff = np.abs(execution.downloads.rho_ff - stage_input.rho_ff)
        rho_fb_diff = np.abs(execution.downloads.rho_fb - stage_input.rho_fb)
        density_diff = np.abs(execution.downloads.density - stage_input.densities)
        k_diff = np.abs(execution.downloads.k_factor - stage_input.k_factor)
        worst_idx = int(np.argmax(density_diff)) if density_diff.size else -1
        fluid_neighbors, boundary_neighbors = self.density_support_counts(
            stage_input.pairs,
            stage_input.densities.size,
        )
        (
            iterations,
            converged_flag,
            final_metric,
            velocity_linf,
            velocity_l1_mean,
            lambda_linf,
            lambda_l1_mean,
            iterations_match,
            converged_match,
            metric_diff,
        ) = self._build_cuda_replay_solver_metrics(
            stage_output,
            execution.solve_metrics,
            execution.velocities_gpu,
            execution.lambda_gpu,
        )

        return CUDAReplayStageComparison(
            rho_self_linf=float(rho_self_diff.max()) if rho_self_diff.size else 0.0,
            rho_self_l1_mean=float(rho_self_diff.mean()) if rho_self_diff.size else 0.0,
            rho_ff_linf=float(rho_ff_diff.max()) if rho_ff_diff.size else 0.0,
            rho_ff_l1_mean=float(rho_ff_diff.mean()) if rho_ff_diff.size else 0.0,
            rho_fb_linf=float(rho_fb_diff.max()) if rho_fb_diff.size else 0.0,
            rho_fb_l1_mean=float(rho_fb_diff.mean()) if rho_fb_diff.size else 0.0,
            density_linf=float(density_diff.max()) if density_diff.size else 0.0,
            density_l1_mean=float(density_diff.mean()) if density_diff.size else 0.0,
            density_worst_idx=float(worst_idx),
            density_worst_has_boundary=float(boundary_neighbors[worst_idx] > 0) if worst_idx >= 0 else 0.0,
            density_worst_fluid_neighbors=float(fluid_neighbors[worst_idx]) if worst_idx >= 0 else 0.0,
            density_worst_boundary_neighbors=float(boundary_neighbors[worst_idx]) if worst_idx >= 0 else 0.0,
            density_worst_is_sparse=float(
                fluid_neighbors[worst_idx] < float(np.mean(fluid_neighbors))
            ) if worst_idx >= 0 and fluid_neighbors.size else 0.0,
            k_linf=float(k_diff.max()) if k_diff.size else 0.0,
            k_l1_mean=float(k_diff.mean()) if k_diff.size else 0.0,
            ff_pairs=float(stage_input.pairs.ff_i.size),
            fb_pairs=float(stage_input.pairs.fb_i.size),
            iterations=iterations,
            converged_flag=converged_flag,
            final_metric=final_metric,
            velocity_linf=velocity_linf,
            velocity_l1_mean=velocity_l1_mean,
            lambda_linf=lambda_linf,
            lambda_l1_mean=lambda_l1_mean,
            iterations_match=iterations_match,
            converged_match=converged_match,
            metric_diff=metric_diff,
        )

    def _run_solver_stage(
        self,
        resources: CUDAExecutionResources,
        stage_name: str,
        stage_input: CUDAValidationStageInput,
    ) -> tuple[dict[str, float], float, np.ndarray, np.ndarray]:
        dynamic_buffers = resources.require_dynamic_buffers()
        ndim = self._scene_dim()
        alpha = self._kernel_alpha(float(stage_input.h))

        lambda_prev_host = np.asarray(stage_input.lambda_prev).copy()
        dt = float(stage_input.dt)
        h = float(stage_input.h)
        rho0 = float(stage_input.rho0)
        mass = float(stage_input.mass)
        max_iter = int(stage_input.max_iter)
        eta = float(stage_input.eta)
        n = lambda_prev_host.size

        cfg_particles = launch_config(n)
        dynamic_buffers.lambda_values.copy_to_device(np.ascontiguousarray(lambda_prev_host))
        self._zero_device_lambda_next(resources, cfg_particles)

        if np.any(lambda_prev_host):
            self._apply_pressure_update(resources, mass, dt, h, alpha, ndim, use_lambda_device=True)
        fill_1d_kernel[cfg_particles](dynamic_buffers.lambda_values, 0.0)

        iterations = 0
        converged = False
        final_metric = 0.0
        metric_sync_ms = 0.0
        for iteration in range(max_iter):
            iterations = iteration + 1
            fill_1d_kernel[cfg_particles](dynamic_buffers.residual, 0.0)
            if stage_name == "cd":
                self._run_cd_residual_kernels(resources, mass, h, alpha, dt, ndim)
            else:
                self._run_df_residual_kernels(resources, mass, h, alpha, ndim)
            cuda.synchronize()

            zero_scalar_kernel[1, 1](dynamic_buffers.metric_sum)
            if stage_name == "cd":
                cd_lambda_reduce_kernel[cfg_particles](
                    dynamic_buffers.residual,
                    dynamic_buffers.density_ref,
                    rho0, dt,
                    dynamic_buffers.k_factor_ref,
                    dynamic_buffers.lambda_next,
                    dynamic_buffers.metric_sum,
                )
            else:
                df_lambda_reduce_kernel[cfg_particles](
                    dynamic_buffers.residual,
                    rho0, dt,
                    dynamic_buffers.k_factor_ref,
                    dynamic_buffers.lambda_next,
                    dynamic_buffers.metric_sum,
                )
            cuda.synchronize()

            t_sync = time.perf_counter()
            metric_sum = float(dynamic_buffers.metric_sum.copy_to_host()[0])
            metric_sync_ms += (time.perf_counter() - t_sync) * 1000.0

            final_metric = metric_sum / max(n, 1)
            if final_metric < eta:
                converged = True
                break

            self._swap_lambda_buffers(resources)
            self._apply_pressure_update(resources, mass, dt, h, alpha, ndim, use_lambda_device=True)

        velocities_gpu = dynamic_buffers.fluid_velocities.copy_to_host()
        lambda_gpu = dynamic_buffers.lambda_values.copy_to_host()

        metrics: dict[str, float] = {
            "iterations": float(iterations),
            "converged_flag": float(converged),
            "final_metric": float(final_metric),
        }
        return metrics, metric_sync_ms, velocities_gpu, lambda_gpu

    def _sync_dynamic_stage_state(
        self,
        resources: CUDAExecutionResources,
        stage_input: CUDAValidationStageInput,
        *,
        skip_density_kfactor_upload: bool = False,
        reuse_device_state: bool = False,
        upload_boundary_velocities: bool = True,
    ) -> None:
        positions_view = np.asarray(stage_input.positions)
        velocities_view = np.asarray(stage_input.velocities)
        densities_view = np.asarray(stage_input.densities)
        k_factor_view = np.asarray(stage_input.k_factor)
        boundary_velocities_view = np.asarray(stage_input.boundary_velocities)
        n = int(positions_view.shape[0])
        dim = int(positions_view.shape[1]) if positions_view.ndim == 2 else 0

        reallocated = self._ensure_dynamic_buffers(n, dim)
        dynamic_buffers = resources.require_dynamic_buffers()

        if reuse_device_state and reallocated:
            reuse_device_state = False

        if not reuse_device_state:
            dynamic_buffers.fluid_positions.copy_to_device(np.ascontiguousarray(positions_view))
            dynamic_buffers.fluid_velocities.copy_to_device(np.ascontiguousarray(velocities_view))

        if not skip_density_kfactor_upload:
            dynamic_buffers.density_ref.copy_to_device(np.ascontiguousarray(densities_view))
            dynamic_buffers.k_factor_ref.copy_to_device(np.ascontiguousarray(k_factor_view))

        if upload_boundary_velocities:
            if boundary_velocities_view.size == 0:
                dynamic_buffers.boundary_velocities = None
            else:
                d_boundary_velocities = self._ensure_boundary_velocity_buffer(
                    resources,
                    tuple(boundary_velocities_view.shape),
                )
                d_boundary_velocities.copy_to_device(np.ascontiguousarray(boundary_velocities_view))
        elif not reuse_device_state:
            dynamic_buffers.boundary_velocities = None

    def _sync_pair_indices(self, resources: CUDAExecutionResources, pairs: NeighborPairs) -> None:
        resources.device.pair_indices = DevicePairIndexBuffers(
            ff_i=cuda.to_device(np.ascontiguousarray(pairs.ff_i)),
            ff_j=cuda.to_device(np.ascontiguousarray(pairs.ff_j)),
            fb_i=cuda.to_device(np.ascontiguousarray(pairs.fb_i)),
            fb_j=cuda.to_device(np.ascontiguousarray(pairs.fb_j)),
        )
        self._allocate_pair_geometry(resources, int(pairs.ff_i.size), int(pairs.fb_i.size))

    def _allocate_pair_geometry(self, resources: CUDAExecutionResources, ff_count: int, fb_count: int) -> None:
        ff_capacity = max(ff_count, 1)
        fb_capacity = max(fb_count, 1)
        current = resources.device.pair_geometry
        if current is not None:
            if (
                int(current.ff_dx.size) >= ff_capacity
                and int(current.ff_dy.size) >= ff_capacity
                and int(current.ff_dz.size) >= ff_capacity
                and int(current.ff_dist.size) >= ff_capacity
                and int(current.fb_dx.size) >= fb_capacity
                and int(current.fb_dy.size) >= fb_capacity
                and int(current.fb_dz.size) >= fb_capacity
                and int(current.fb_dist.size) >= fb_capacity
            ):
                return

        resources.device.pair_geometry = DevicePairGeometryBuffers(
            ff_dx=cuda.device_array(ff_capacity, dtype=np.float64),
            ff_dy=cuda.device_array(ff_capacity, dtype=np.float64),
            ff_dz=cuda.device_array(ff_capacity, dtype=np.float64),
            ff_dist=cuda.device_array(ff_capacity, dtype=np.float64),
            fb_dx=cuda.device_array(fb_capacity, dtype=np.float64),
            fb_dy=cuda.device_array(fb_capacity, dtype=np.float64),
            fb_dz=cuda.device_array(fb_capacity, dtype=np.float64),
            fb_dist=cuda.device_array(fb_capacity, dtype=np.float64),
        )

    def _prepare_pair_geometry(
        self,
        resources: CUDAExecutionResources,
        periodic_length: float,
        use_periodic: int,
        ndim: int,
    ) -> None:
        dynamic_buffers = resources.require_dynamic_buffers()
        pair_indices = resources.require_pair_indices()
        pair_geometry = resources.require_pair_geometry()

        if pair_indices.ff_count > 0:
            cfg = launch_config(pair_indices.ff_count)
            pair_geometry_ff_kernel[cfg](
                dynamic_buffers.fluid_positions,
                pair_indices.ff_i, pair_indices.ff_j,
                periodic_length, use_periodic,
                pair_geometry.ff_dx, pair_geometry.ff_dy, pair_geometry.ff_dz,
                pair_geometry.ff_dist, ndim,
            )

        if pair_indices.fb_count > 0:
            static_buffers = resources.require_static_buffers()
            cfg = launch_config(pair_indices.fb_count)
            pair_geometry_fb_kernel[cfg](
                dynamic_buffers.fluid_positions,
                static_buffers.boundary_positions,
                pair_indices.fb_i, pair_indices.fb_j,
                periodic_length, use_periodic,
                pair_geometry.fb_dx, pair_geometry.fb_dy, pair_geometry.fb_dz,
                pair_geometry.fb_dist, ndim,
            )

    def _run_density_kernels(
        self,
        resources: CUDAExecutionResources,
        stage_input: CUDAValidationStageInput,
    ) -> None:
        dynamic_buffers = resources.require_dynamic_buffers()
        pair_indices = resources.require_pair_indices()
        pair_geometry = resources.require_pair_geometry()

        n = int(np.asarray(stage_input.densities).size)
        mass = float(stage_input.mass)
        h = float(stage_input.h)
        rho0 = float(stage_input.rho0)
        alpha = self._kernel_alpha(h)
        cfg_particles = launch_config(n)
        w_self = mass * alpha  # W(0) = alpha * 1.0

        fill_1d_kernel[cfg_particles](dynamic_buffers.density.rho_self, w_self)
        fill_1d_kernel[cfg_particles](dynamic_buffers.density.rho_ff, 0.0)
        fill_1d_kernel[cfg_particles](dynamic_buffers.density.rho_fb, 0.0)

        if pair_indices.ff_count > 0:
            cfg = launch_config(pair_indices.ff_count)
            density_ff_kernel[cfg](
                pair_indices.ff_i, pair_indices.ff_j,
                pair_geometry.ff_dist,
                mass, h, alpha,
                dynamic_buffers.density.rho_ff,
            )

        if pair_indices.fb_count > 0:
            static_buffers = resources.require_static_buffers()
            cfg = launch_config(pair_indices.fb_count)
            density_fb_kernel[cfg](
                pair_indices.fb_i, pair_indices.fb_j,
                pair_geometry.fb_dist,
                static_buffers.boundary_psi,
                h, alpha,
                dynamic_buffers.density.rho_fb,
            )

        combine_density_kernel[cfg_particles](
            dynamic_buffers.density.rho_self,
            dynamic_buffers.density.rho_ff,
            dynamic_buffers.density.rho_fb,
            dynamic_buffers.density.density,
        )
        clip_upper_1d_kernel[cfg_particles](dynamic_buffers.density.density, float(rho0 * 1.02))

    def _run_kfactor_kernels(
        self,
        resources: CUDAExecutionResources,
        stage_input: CUDAValidationStageInput,
    ) -> None:
        dynamic_buffers = resources.require_dynamic_buffers()
        pair_indices = resources.require_pair_indices()
        pair_geometry = resources.require_pair_geometry()

        n = int(np.asarray(stage_input.k_factor).size)
        h = float(stage_input.h)
        alpha = self._kernel_alpha(h)
        cfg_particles = launch_config(n)

        fill_1d_kernel[cfg_particles](dynamic_buffers.k_factor.sum_grad_x, 0.0)
        fill_1d_kernel[cfg_particles](dynamic_buffers.k_factor.sum_grad_y, 0.0)
        fill_1d_kernel[cfg_particles](dynamic_buffers.k_factor.sum_grad_z, 0.0)
        fill_1d_kernel[cfg_particles](dynamic_buffers.k_factor.sum_grad_sq, 0.0)

        if pair_indices.ff_count > 0:
            cfg = launch_config(pair_indices.ff_count)
            k_factor_ff_kernel[cfg](
                pair_indices.ff_i, pair_indices.ff_j,
                pair_geometry.ff_dx, pair_geometry.ff_dy, pair_geometry.ff_dz,
                pair_geometry.ff_dist,
                h, alpha,
                dynamic_buffers.k_factor.sum_grad_x,
                dynamic_buffers.k_factor.sum_grad_y,
                dynamic_buffers.k_factor.sum_grad_z,
                dynamic_buffers.k_factor.sum_grad_sq,
            )

        if pair_indices.fb_count > 0:
            cfg = launch_config(pair_indices.fb_count)
            k_factor_fb_kernel[cfg](
                pair_indices.fb_i,
                pair_geometry.fb_dx, pair_geometry.fb_dy, pair_geometry.fb_dz,
                pair_geometry.fb_dist,
                h, alpha,
                dynamic_buffers.k_factor.sum_grad_x,
                dynamic_buffers.k_factor.sum_grad_y,
                dynamic_buffers.k_factor.sum_grad_z,
                dynamic_buffers.k_factor.sum_grad_sq,
            )

        finalize_k_factor_kernel[cfg_particles](
            dynamic_buffers.k_factor.sum_grad_x,
            dynamic_buffers.k_factor.sum_grad_y,
            dynamic_buffers.k_factor.sum_grad_z,
            dynamic_buffers.k_factor.sum_grad_sq,
            dynamic_buffers.k_factor.k_factor,
        )

    def _run_cd_residual_kernels(
        self,
        resources: CUDAExecutionResources,
        mass: float,
        h: float,
        alpha: float,
        dt: float,
        ndim: int,
    ) -> None:
        dynamic_buffers = resources.require_dynamic_buffers()
        pair_indices = resources.require_pair_indices()
        pair_geometry = resources.require_pair_geometry()

        if pair_indices.ff_count > 0:
            cfg = launch_config(pair_indices.ff_count)
            cd_residual_ff_kernel[cfg](
                pair_indices.ff_i, pair_indices.ff_j,
                pair_geometry.ff_dx, pair_geometry.ff_dy, pair_geometry.ff_dz,
                pair_geometry.ff_dist,
                dynamic_buffers.fluid_velocities,
                mass, h, alpha, dt,
                dynamic_buffers.residual, ndim,
            )

        if pair_indices.fb_count > 0 and dynamic_buffers.boundary_velocities is not None:
            static_buffers = resources.require_static_buffers()
            cfg = launch_config(pair_indices.fb_count)
            cd_residual_fb_kernel[cfg](
                pair_indices.fb_i, pair_indices.fb_j,
                pair_geometry.fb_dx, pair_geometry.fb_dy, pair_geometry.fb_dz,
                pair_geometry.fb_dist,
                dynamic_buffers.fluid_velocities,
                dynamic_buffers.boundary_velocities,
                static_buffers.boundary_psi,
                h, alpha, dt,
                dynamic_buffers.residual, ndim,
            )

    def _run_df_residual_kernels(
        self,
        resources: CUDAExecutionResources,
        mass: float,
        h: float,
        alpha: float,
        ndim: int,
    ) -> None:
        dynamic_buffers = resources.require_dynamic_buffers()
        pair_indices = resources.require_pair_indices()
        pair_geometry = resources.require_pair_geometry()

        if pair_indices.ff_count > 0:
            cfg = launch_config(pair_indices.ff_count)
            df_residual_ff_kernel[cfg](
                pair_indices.ff_i, pair_indices.ff_j,
                pair_geometry.ff_dx, pair_geometry.ff_dy, pair_geometry.ff_dz,
                pair_geometry.ff_dist,
                dynamic_buffers.fluid_velocities,
                mass, h, alpha,
                dynamic_buffers.residual, ndim,
            )

        if pair_indices.fb_count > 0 and dynamic_buffers.boundary_velocities is not None:
            static_buffers = resources.require_static_buffers()
            cfg = launch_config(pair_indices.fb_count)
            df_residual_fb_kernel[cfg](
                pair_indices.fb_i, pair_indices.fb_j,
                pair_geometry.fb_dx, pair_geometry.fb_dy, pair_geometry.fb_dz,
                pair_geometry.fb_dist,
                dynamic_buffers.fluid_velocities,
                dynamic_buffers.boundary_velocities,
                static_buffers.boundary_psi,
                h, alpha,
                dynamic_buffers.residual, ndim,
            )

    def _apply_pressure_update(
        self,
        resources: CUDAExecutionResources,
        mass: float,
        dt: float,
        h: float,
        alpha: float,
        ndim: int,
        use_lambda_device: bool,
    ) -> None:
        dynamic_buffers = resources.require_dynamic_buffers()
        pair_indices = resources.require_pair_indices()
        pair_geometry = resources.require_pair_geometry()

        lambda_values = dynamic_buffers.lambda_values if use_lambda_device else None
        if lambda_values is None:
            raise RuntimeError("CUDA pressure update requires device lambda values.")

        if pair_indices.ff_count > 0:
            cfg = launch_config(pair_indices.ff_count)
            apply_pressure_ff_kernel[cfg](
                pair_indices.ff_i, pair_indices.ff_j,
                pair_geometry.ff_dx, pair_geometry.ff_dy, pair_geometry.ff_dz,
                pair_geometry.ff_dist,
                dynamic_buffers.density_ref,
                lambda_values,
                mass, h, alpha, dt,
                dynamic_buffers.fluid_velocities, ndim,
            )

        if pair_indices.fb_count > 0:
            static_buffers = resources.require_static_buffers()
            cfg = launch_config(pair_indices.fb_count)
            apply_pressure_fb_kernel[cfg](
                pair_indices.fb_i, pair_indices.fb_j,
                pair_geometry.fb_dx, pair_geometry.fb_dy, pair_geometry.fb_dz,
                pair_geometry.fb_dist,
                dynamic_buffers.density_ref,
                lambda_values,
                static_buffers.boundary_psi,
                h, alpha, dt,
                dynamic_buffers.fluid_velocities, ndim,
            )

    def _zero_device_lambda_next(
        self,
        resources: CUDAExecutionResources,
        cfg_particles: tuple[int, int],
    ) -> None:
        dynamic_buffers = resources.require_dynamic_buffers()
        fill_1d_kernel[cfg_particles](dynamic_buffers.lambda_next, 0.0)

    def _swap_lambda_buffers(self, resources: CUDAExecutionResources) -> None:
        dynamic_buffers = resources.require_dynamic_buffers()
        dynamic_buffers.lambda_values, dynamic_buffers.lambda_next = (
            dynamic_buffers.lambda_next,
            dynamic_buffers.lambda_values,
        )

    def _run_boundary_state_kernels(
        self,
        resources: CUDAExecutionResources,
        n_boundary: int,
        h: float,
        ndim: int,
    ) -> None:
        dynamic_buffers = resources.require_dynamic_buffers()
        pair_indices = resources.require_pair_indices()
        pair_geometry = resources.require_pair_geometry()
        alpha = self._kernel_alpha(h)

        n_bnd = max(n_boundary, 1)
        cfg_bnd = launch_config(n_bnd)

        bnd_v_num_x = cuda.device_array(n_bnd, dtype=np.float64)
        bnd_v_num_y = cuda.device_array(n_bnd, dtype=np.float64)
        bnd_v_num_z = cuda.device_array(n_bnd, dtype=np.float64)
        bnd_v_den = cuda.device_array(n_bnd, dtype=np.float64)

        fill_1d_kernel[cfg_bnd](bnd_v_num_x, 0.0)
        fill_1d_kernel[cfg_bnd](bnd_v_num_y, 0.0)
        fill_1d_kernel[cfg_bnd](bnd_v_num_z, 0.0)
        fill_1d_kernel[cfg_bnd](bnd_v_den, 0.0)

        if pair_indices.fb_count > 0:
            cfg = launch_config(pair_indices.fb_count)
            boundary_vel_scatter_kernel[cfg](
                pair_indices.fb_i, pair_indices.fb_j,
                pair_geometry.fb_dist,
                dynamic_buffers.fluid_velocities,
                h, alpha,
                bnd_v_num_x, bnd_v_num_y, bnd_v_num_z, bnd_v_den, ndim,
            )

        d_bnd_vel = cuda.device_array((n_bnd, ndim), dtype=np.float64)
        boundary_vel_finalize_kernel[cfg_bnd](
            bnd_v_num_x, bnd_v_num_y, bnd_v_num_z, bnd_v_den,
            0.25, d_bnd_vel, ndim,
        )
        dynamic_buffers.boundary_velocities = d_bnd_vel if n_boundary > 0 else None

    def _run_viscosity_kernels(
        self,
        resources: CUDAExecutionResources,
        n_fluid: int,
        mass: float,
        nu: float,
        h: float,
        rho0: float,
        boundary_mass: float,
        boundary_rho0: float,
        ndim: int,
    ) -> object:
        dynamic_buffers = resources.require_dynamic_buffers()
        pair_indices = resources.require_pair_indices()
        pair_geometry = resources.require_pair_geometry()
        alpha = self._kernel_alpha(h)

        d_acc = cuda.device_array((max(n_fluid, 1), ndim), dtype=np.float64)
        cfg_particles = launch_config(max(n_fluid, 1))
        fill_2d_kernel[cfg_particles](d_acc, 0.0)

        if nu > 0.0:
            if pair_indices.ff_count > 0:
                cfg = launch_config(pair_indices.ff_count)
                viscosity_ff_kernel[cfg](
                    pair_indices.ff_i, pair_indices.ff_j,
                    pair_geometry.ff_dx, pair_geometry.ff_dy, pair_geometry.ff_dz,
                    pair_geometry.ff_dist,
                    dynamic_buffers.fluid_velocities,
                    mass, nu, h, alpha, rho0,
                    d_acc, ndim,
                )

            if (
                pair_indices.fb_count > 0
                and dynamic_buffers.boundary_velocities is not None
                and boundary_rho0 > 0.0
            ):
                cfg = launch_config(pair_indices.fb_count)
                viscosity_fb_kernel[cfg](
                    pair_indices.fb_i, pair_indices.fb_j,
                    pair_geometry.fb_dx, pair_geometry.fb_dy, pair_geometry.fb_dz,
                    pair_geometry.fb_dist,
                    dynamic_buffers.fluid_velocities,
                    dynamic_buffers.boundary_velocities,
                    boundary_mass, nu, h, alpha, boundary_rho0,
                    d_acc, ndim,
                )

        return d_acc

    def _run_velocity_prediction(
        self,
        resources: CUDAExecutionResources,
        d_acc: object,
        gravity: np.ndarray,
        dt: float,
        ndim: int,
    ) -> None:
        dynamic_buffers = resources.require_dynamic_buffers()
        n = int(dynamic_buffers.fluid_velocities.shape[0])
        cfg = launch_config(n)
        d_gravity = cuda.to_device(np.ascontiguousarray(gravity[:ndim].astype(np.float64)))
        velocity_predict_kernel[cfg](
            dynamic_buffers.fluid_velocities,
            d_acc,
            d_gravity,
            dt, ndim,
        )

    def _run_xsph_kernels(
        self,
        resources: CUDAExecutionResources,
        n_fluid: int,
        mass: float,
        h: float,
        xsph_eps: float,
        ndim: int,
    ) -> None:
        if xsph_eps <= 0.0:
            return

        dynamic_buffers = resources.require_dynamic_buffers()
        pair_indices = resources.require_pair_indices()
        pair_geometry = resources.require_pair_geometry()
        if pair_indices.ff_count <= 0:
            return

        alpha = self._kernel_alpha(h)
        d_xsph = cuda.device_array((max(n_fluid, 1), ndim), dtype=np.float64)
        cfg_particles = launch_config(max(n_fluid, 1))
        fill_2d_kernel[cfg_particles](d_xsph, 0.0)

        cfg_pairs = launch_config(pair_indices.ff_count)
        xsph_kernel[cfg_pairs](
            dynamic_buffers.fluid_velocities,
            dynamic_buffers.density_ref,
            pair_indices.ff_i,
            pair_indices.ff_j,
            pair_geometry.ff_dist,
            mass,
            h,
            alpha,
            xsph_eps,
            d_xsph,
            pair_indices.ff_count,
        )
        add_velocity_delta_kernel[cfg_particles](
            dynamic_buffers.fluid_velocities,
            d_xsph,
            ndim,
        )

    def _prepare_cuda_device_state_phase(
        self,
        context: CUDAAuthoritativeExecutionContext,
        timings: dict[str, float],
    ) -> None:
        resources = context.resources
        t0 = time.perf_counter()
        reuse_fast_state = (
            (not context.request.gpu_pre_cd)
            and resources.device.fast_step_device_state_valid
            and resources.device.dynamic_buffers is not None
        )
        self._sync_dynamic_stage_state(
            resources,
            context.request.stage_input,
            skip_density_kfactor_upload=True,
            reuse_device_state=reuse_fast_state,
            upload_boundary_velocities=not context.request.gpu_pre_cd and not reuse_fast_state,
        )
        cuda.synchronize()
        timings["upload"] = (time.perf_counter() - t0) * 1000.0

    def _prepare_cuda_pair_pipeline_phase(
        self,
        context: CUDAAuthoritativeExecutionContext,
        timings: dict[str, float],
    ) -> CUDAPairPipelineResult:
        resources = context.resources
        dynamic_buffers = resources.require_dynamic_buffers()
        ndim = self._scene_dim()

        t0 = time.perf_counter()
        if resources.scene.neighbor_search is not None:
            bnd_pos_gpu = resources.require_static_buffers().boundary_positions
            ff_i, ff_j, n_ff, fb_i, fb_j, n_fb = resources.scene.neighbor_search.build(
                dynamic_buffers.fluid_positions,
                bnd_pos_gpu,
            )
            for key, value in resources.scene.neighbor_search.last_build_timings_ms.items():
                timings[f"pair_build_{key}"] = float(value)
            t_materialize = time.perf_counter()
            resources.device.pair_indices = DevicePairIndexBuffers(
                ff_i=ff_i, ff_j=ff_j, fb_i=fb_i, fb_j=fb_j,
            )
            self._allocate_pair_geometry(resources, n_ff, n_fb)
            cuda.synchronize()
            timings["pair_build_materialize"] = (time.perf_counter() - t_materialize) * 1000.0
        else:
            self._sync_pair_indices(resources, context.request.stage_input.pairs)
            pair_indices = resources.require_pair_indices()
            n_ff = pair_indices.ff_count
            n_fb = pair_indices.fb_count
        cuda.synchronize()
        timings["pair_build"] = (time.perf_counter() - t0) * 1000.0

        cpu_n_ff = int(np.asarray(context.request.stage_input.pairs.ff_i).size)
        cpu_n_fb = int(np.asarray(context.request.stage_input.pairs.fb_i).size)

        host_fluid_counts: np.ndarray | None = None
        host_boundary_counts: np.ndarray | None = None
        if context.request.gpu_pre_cd and resources.device.pair_indices is not None:
            t_nc = time.perf_counter()
            n_fluid = int(dynamic_buffers.fluid_positions.shape[0])
            d_fluid_counts = cuda.device_array(n_fluid, dtype=np.int32)
            d_boundary_counts = cuda.device_array(n_fluid, dtype=np.int32)
            cfg_n = launch_config(n_fluid)
            fill_1d_kernel[cfg_n](d_fluid_counts, 0)
            fill_1d_kernel[cfg_n](d_boundary_counts, 0)
            pi = resources.require_pair_indices()
            if n_ff > 0:
                cfg_ff = launch_config(n_ff)
                neighbor_count_kernel[cfg_ff](pi.ff_i, d_fluid_counts)
                neighbor_count_kernel[cfg_ff](pi.ff_j, d_fluid_counts)
            if n_fb > 0:
                cfg_fb = launch_config(n_fb)
                neighbor_count_kernel[cfg_fb](pi.fb_i, d_boundary_counts)
            cuda.synchronize()
            host_fluid_counts = d_fluid_counts.copy_to_host()
            host_boundary_counts = d_boundary_counts.copy_to_host()
            timings["neighbor_count"] = (time.perf_counter() - t_nc) * 1000.0

        periodic_length, use_periodic = self.periodic_config()

        t0 = time.perf_counter()
        self._prepare_pair_geometry(resources, periodic_length, use_periodic, ndim)
        cuda.synchronize()
        timings["pair_geometry"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        self._run_density_kernels(resources, context.request.stage_input)
        cuda.synchronize()
        timings["density"] = (time.perf_counter() - t0) * 1000.0

        return CUDAPairPipelineResult(
            gpu_ff_pairs=n_ff,
            gpu_fb_pairs=n_fb,
            cpu_ff_pairs=cpu_n_ff,
            cpu_fb_pairs=cpu_n_fb,
            fluid_counts=host_fluid_counts,
            boundary_counts=host_boundary_counts,
        )

    def _run_cuda_pre_cd_phase(
        self,
        context: CUDAAuthoritativeExecutionContext,
        timings: dict[str, float],
    ) -> None:
        if not context.request.gpu_pre_cd:
            return

        resources = context.resources
        ndim = self._scene_dim()
        h = float(context.request.stage_input.h)
        n_boundary = 0
        sim = self._host_backend.sim
        if sim is not None and sim.boundary is not None:
            n_boundary = int(sim.boundary.n)

        t0 = time.perf_counter()
        self._run_boundary_state_kernels(resources, n_boundary, h, ndim)
        cuda.synchronize()
        timings["boundary_state"] = (time.perf_counter() - t0) * 1000.0

        n_fluid = int(np.asarray(context.request.stage_input.positions).shape[0])
        mass = float(context.request.stage_input.mass)
        rho0 = float(context.request.stage_input.rho0)

        t0 = time.perf_counter()
        d_acc = self._run_viscosity_kernels(
            resources, n_fluid, mass, context.request.nu, h, rho0,
            context.request.boundary_mass, context.request.boundary_rho0, ndim,
        )
        cuda.synchronize()
        timings["viscosity"] = (time.perf_counter() - t0) * 1000.0

        gravity = context.request.gravity
        if gravity is None:
            gravity = np.zeros(ndim, dtype=np.float64)

        t0 = time.perf_counter()
        self._run_velocity_prediction(
            resources, d_acc, gravity,
            float(context.request.stage_input.dt), ndim,
        )
        cuda.synchronize()
        timings["velocity_predict"] = (time.perf_counter() - t0) * 1000.0

    def _run_cuda_solver_execution_phase(
        self,
        context: CUDAAuthoritativeExecutionContext,
        timings: dict[str, float],
    ) -> CUDASolverExecutionResult:
        resources = context.resources
        dynamic_buffers = resources.require_dynamic_buffers()
        ndim = self._scene_dim()

        t0 = time.perf_counter()
        self._run_kfactor_kernels(resources, context.request.stage_input)
        cuda.synchronize()
        timings["k_factor"] = (time.perf_counter() - t0) * 1000.0

        dynamic_buffers.density_ref = dynamic_buffers.density.density
        dynamic_buffers.k_factor_ref = dynamic_buffers.k_factor.k_factor

        t0 = time.perf_counter()
        solve_metrics, metric_sync_ms, velocities_gpu, lambda_gpu = self._run_solver_stage(
            resources,
            context.request.stage,
            context.request.stage_input,
        )
        cuda.synchronize()
        timings["solve"] = (time.perf_counter() - t0) * 1000.0
        timings["metric_sync"] = metric_sync_ms

        if context.request.gpu_pre_cd:
            t0 = time.perf_counter()
            n_pos = int(dynamic_buffers.fluid_positions.shape[0])
            cfg_pos = launch_config(n_pos)
            position_integrate_kernel[cfg_pos](
                dynamic_buffers.fluid_positions,
                dynamic_buffers.fluid_velocities,
                float(context.request.stage_input.dt),
                ndim,
            )
            sim = self._host_backend.sim
            if sim is not None and sim.x_min is not None and sim.x_max is not None:
                periodic_wrap_x_kernel[cfg_pos](
                    dynamic_buffers.fluid_positions,
                    float(sim.x_min),
                    float(sim.x_max),
                )
            cuda.synchronize()
            timings["position_integrate"] = (time.perf_counter() - t0) * 1000.0

        if context.request.stage == "df":
            xsph_enabled, xsph_eps = self._xsph_config()
            if xsph_enabled and xsph_eps > 0.0:
                t0 = time.perf_counter()
                self._run_xsph_kernels(
                    resources,
                    int(dynamic_buffers.fluid_velocities.shape[0]),
                    float(context.request.stage_input.mass),
                    float(context.request.stage_input.h),
                    xsph_eps,
                    ndim,
                )
                cuda.synchronize()
                timings["xsph"] = (time.perf_counter() - t0) * 1000.0

        return CUDASolverExecutionResult(
            solve_metrics=solve_metrics,
            velocities_gpu=dynamic_buffers.fluid_velocities.copy_to_host(),
            lambda_gpu=lambda_gpu,
        )

    def _writeback_cuda_stage_results(
        self,
        context: CUDAAuthoritativeExecutionContext,
        execution: CUDASolverExecutionResult,
        timings: dict[str, float],
    ) -> None:
        dynamic_buffers = context.resources.require_dynamic_buffers()
        fluid = context.writeback.fluid
        t0 = time.perf_counter()
        fluid.velocities[:] = execution.velocities_gpu  # type: ignore[union-attr]
        if context.request.stage == "cd":
            fluid.p_cd_prev[:] = execution.lambda_gpu  # type: ignore[union-attr]
        else:
            fluid.p_df_prev[:] = execution.lambda_gpu  # type: ignore[union-attr]

        if context.request.gpu_pre_cd:
            fluid.positions[:] = dynamic_buffers.fluid_positions.copy_to_host()  # type: ignore[union-attr]
            fluid.densities[:] = dynamic_buffers.density.density.copy_to_host()  # type: ignore[union-attr]
            boundary = context.writeback.boundary
            if boundary is not None and boundary.n > 0 and dynamic_buffers.boundary_velocities is not None:
                boundary.velocities[:] = dynamic_buffers.boundary_velocities.copy_to_host()[:boundary.n]

        cuda.synchronize()
        timings["download"] = (time.perf_counter() - t0) * 1000.0

        if context.request.gpu_pre_cd:
            context.resources.device.fast_step_device_state_valid = True
        else:
            context.resources.device.fast_step_device_state_valid = False
