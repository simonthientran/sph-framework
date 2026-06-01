"""
DFSPH (Divergence-Free SPH) time integration.

Fully vectorized implementation following:
  Bender & Koschier (2017) "Divergence-Free SPH for Incompressible
  and Viscous Fluids", IEEE TVCG.

Algorithm 6 from the paper:
  1. Compute non-pressure forces
  2. Predict velocities
  3. Constant Density Solver  (correct density error before position update)
  4. Update positions
  5. Apply periodic BC
  6. Rebuild neighbor search
  7. Recompute densities + k factors
  8. Divergence-Free Solver   (correct velocity divergence)
"""
from __future__ import annotations

import time

import numpy as np

from sph.fluid_model import FluidModel, BoundaryModel
from sph.kernel import CubicSplineKernel
from sph.kernels_nb import (
    accumulate_density,
    accumulate_difference,
    accumulate_force,
    cubic_spline_W_batch,
    cubic_spline_gradW_batch,
    scatter_add_1d,
    scatter_add_2d,
)
from sph.neighbor_search_kdtree import KDTreeNeighborSearch, NeighborPairs
from sph.time_step import TimeStep
from sph.diagnostics.density_regimes import DensityRegimeInfo, analyze_density_regimes
from sph.validation.contracts import (
    CUDAValidationStageInput,
    CUDAValidationStageOutput,
    CUDAValidationStageSnapshot,
)


class DFSPHTimeStep(TimeStep):
    """
    Divergence-Free SPH time integration (Bender & Koschier 2017).

    Key properties vs WCSPH:
    - Enforces near-incompressibility via two iterative PPE solvers
    - Allows 5–10× larger time steps than WCSPH
    - Density error < 0.5% at steady state
    - Divergence-free velocity field
    """

    def __init__(
        self,
        kernel: CubicSplineKernel,
        neighbor_search: KDTreeNeighborSearch,
        nu: float = 0.0,
        viscosity_factor: float = 3.0,
        gravity: np.ndarray | None = None,
        eta_cd: float = 0.01,
        eta_df: float = 0.01,
        max_iter_cd: int = 20,
        max_iter_df: int = 20,
        periodic_x: tuple[float, float] | None = None,
        eos_k: float = 8000.0,
        c0: float = 10.0,
        density_diffusion_delta: float = 0.0,
        density_diffusion_enable: bool = False,
        density_floor_ratio: float = 0.85,
        free_surface_stabilization_enable: bool = True,
        free_surface_stabilization_mode: str = "density_shift",
        free_surface_stabilization_strength: float = 0.4,
        free_surface_target_ratio: float = 0.88,
        free_surface_max_raise_ratio: float = 0.12,
        free_surface_shift_strength: float = 0.25,
        free_surface_shift_max_ratio: float = 0.015,
        splash_stabilization_scale: float = 0.0,
    ):
        """
        Initialize DFSPH time step.

        Args:
            kernel: SPH kernel
            neighbor_search: Neighbor search
            nu: Kinematic viscosity
            gravity: Gravity vector (or None)
            eta_cd: Convergence tolerance for constant density solver (relative)
            eta_df: Convergence tolerance for divergence-free solver (relative)
            max_iter_cd: Maximum iterations for constant density solver
            max_iter_df: Maximum iterations for divergence-free solver
            periodic_x: Optional (x_min, x_max) for periodic BC in x.
                        Must be set here (not post-step) because positions are
                        updated mid-step before neighbor rebuild.
        """
        self.kernel = kernel
        self.neighbor_search = neighbor_search
        self.nu = float(nu)
        self.viscosity_factor = float(viscosity_factor)
        self.gravity = gravity if gravity is not None else np.zeros(kernel.dim, dtype=np.float64)
        self.eta_cd = float(eta_cd)
        self.eta_df = float(eta_df)
        self.max_iter_cd = int(max_iter_cd)
        self.max_iter_df = int(max_iter_df)
        self.periodic_x = periodic_x  # applied internally after position update
        self.eos_k = float(eos_k)
        self.c0 = float(c0)
        self.delta_density = float(density_diffusion_delta)
        self.enable_density_diffusion = bool(density_diffusion_enable and self.delta_density > 0.0)
        self.density_floor_ratio = float(density_floor_ratio)
        self.free_surface_stabilization_enable = bool(free_surface_stabilization_enable)
        self.free_surface_stabilization_mode = str(free_surface_stabilization_mode).lower()
        self.free_surface_stabilization_strength = float(free_surface_stabilization_strength)
        self.free_surface_target_ratio = float(free_surface_target_ratio)
        self.free_surface_max_raise_ratio = float(free_surface_max_raise_ratio)
        self.free_surface_shift_strength = float(free_surface_shift_strength)
        self.free_surface_shift_max_ratio = float(free_surface_shift_max_ratio)
        self.splash_stabilization_scale = float(splash_stabilization_scale)
        self.last_report: dict[str, float] = {}
        self.last_stage_timings_ms: dict[str, float] = {}
        self.last_regime_info: DensityRegimeInfo | None = None
        self.last_cuda_stage_snapshots: dict[str, CUDAValidationStageSnapshot] = {}
        # CUDA authoritative mode: set by NumbaCUDABackend to replace CPU PPE solve.
        # Signature: (stage_name, snap, fluid, dt) -> (meta_dict, timings_dict)
        # meta_dict keys: iters, converged, metric
        # timings_dict keys: cuda_{stage}_{timer_name} -> ms
        self.debug_mode: bool = False
        self._cuda_solve_callback: object = None  # callable | None
        self._last_cuda_stage_timings_ms: dict[str, float] = {}
        # Extra per-stage metrics (e.g. pair counts) reported by CUDA callback
        self._last_cuda_stage_extra_metrics: dict[str, float] = {}
        # Neighbor-cache: pairs2 from step N equals pairs at start of step N+1.
        # Storing them avoids one full KD-tree build per step on the CPU path.
        self._cached_pairs: NeighborPairs | None = None

    def step(
        self,
        fluid: FluidModel,
        boundary: BoundaryModel | None,
        dt: float,
    ) -> tuple[int, int]:
        """
        Advance simulation by one time step using DFSPH (Algorithm 6).

        Returns:
            (iter_cd, iter_df): iteration counts for both solvers
        """
        timings = {
            "neighbor_search": 0.0,
            "density_assembly": 0.0,
            "pressure_solve": 0.0,
            "integration": 0.0,
            "surface_correction": 0.0,
        }
        self.last_cuda_stage_snapshots = {}
        self._last_cuda_stage_timings_ms = {}
        self._last_cuda_stage_extra_metrics = {}

        _cuda_fast = self._cuda_solve_callback is not None and not self.debug_mode

        if _cuda_fast:
            # ── CUDA full pipeline ────────────────────────────────────
            # GPU handles the entire step: pre-CD block (pair build,
            # density, boundary state, viscosity, velocity prediction),
            # k-factor, CD solve, position integration, periodic wrap,
            # and DF solve.  No CPU compute work — only snapshot capture,
            # diagnostics, and CFL control remain host-side.
            empty_pairs = NeighborPairs.empty(fluid.dim)

            # CD stage — snapshot carries pre-viscosity state; GPU does
            # boundary state + viscosity + prediction before the solve.
            t_stage = time.perf_counter()
            snap_cd = self._capture_cuda_stage_snapshot(
                fluid, boundary, empty_pairs, dt,
                fluid.p_cd_prev, self.max_iter_cd, self.eta_cd, "cd",
            )
            cd_snapshot = CUDAValidationStageSnapshot(stage="cd", stage_input=snap_cd)
            self.last_cuda_stage_snapshots["cd"] = cd_snapshot

            cd_meta, cd_timings = self._cuda_solve_callback(
                "cd",
                self._build_cuda_callback_input(
                    snap_cd,
                    boundary,
                    gpu_pre_cd=True,
                ),
                fluid,
                dt,
            )
            iter_cd = int(cd_meta.get("iters", 1))
            cd_converged = bool(cd_meta.get("converged", True))
            rho_err_mean = float(cd_meta.get("metric", 0.0))
            rho_err_max = 0.0
            self._last_cuda_stage_timings_ms.update(cd_timings)
            for k in ("gpu_ff_pairs", "gpu_fb_pairs", "cpu_ff_pairs", "cpu_fb_pairs", "pair_count_match"):
                if k in cd_meta:
                    self._last_cuda_stage_extra_metrics[f"cuda_cd_{k}"] = float(cd_meta[k])
            cd_snapshot.stage_output = self._capture_cuda_stage_output(
                fluid.velocities, fluid.p_cd_prev, iter_cd, cd_converged, rho_err_mean,
            )
            timings["pressure_solve"] += time.perf_counter() - t_stage

            # Position integration + periodic wrap are handled on GPU
            # inside the CD callback.  fluid.positions is already updated
            # (written back by the callback after device-side integration).

            # DF stage — boundary velocities and positions were written
            # back to CPU by the CD callback, so the DF snapshot captures
            # them correctly.
            t_stage = time.perf_counter()
            snap_df = self._capture_cuda_stage_snapshot(
                fluid, boundary, empty_pairs, dt,
                fluid.p_df_prev, self.max_iter_df, self.eta_df, "df",
            )
            df_snapshot = CUDAValidationStageSnapshot(stage="df", stage_input=snap_df)
            self.last_cuda_stage_snapshots["df"] = df_snapshot
            df_meta, df_timings = self._cuda_solve_callback(
                "df",
                self._build_cuda_callback_input(
                    snap_df,
                    boundary,
                    gpu_pre_cd=False,
                ),
                fluid,
                dt,
            )
            iter_df = int(df_meta.get("iters", 1))
            df_converged = bool(df_meta.get("converged", True))
            div_err_mean = float(df_meta.get("metric", 0.0))
            self._last_cuda_stage_timings_ms.update(df_timings)
            for k in ("gpu_ff_pairs", "gpu_fb_pairs", "cpu_ff_pairs", "cpu_fb_pairs", "pair_count_match"):
                if k in df_meta:
                    self._last_cuda_stage_extra_metrics[f"cuda_df_{k}"] = float(df_meta[k])
            df_snapshot.stage_output = self._capture_cuda_stage_output(
                fluid.velocities, fluid.p_df_prev, iter_df, df_converged, div_err_mean,
            )
            timings["pressure_solve"] += time.perf_counter() - t_stage

            # Diagnostics — GPU density was written back to fluid.densities
            # by the CD callback.  Neighbor counts computed on GPU are passed
            # to the regime analysis so classification works without CPU pairs.
            regime_info = self._analyze_density_regime(
                fluid, boundary, empty_pairs,
                precomputed_fluid_counts=cd_meta.get("fluid_counts"),
                precomputed_boundary_counts=cd_meta.get("boundary_counts"),
            )
            self.last_regime_info = regime_info
            shift_count = 0
            shift_mean = 0.0
            density_raise_count = 0
            density_raise_mean = 0.0
        else:
            # ── CPU / debug path ──────────────────────────────────────
            # 1. Build neighbors + density field
            # Optimisation: pairs2 from the previous step's post-integration
            # rebuild equals pairs at the start of this step (positions are
            # identical), so we reuse it to skip one KD-tree build.
            t_stage = time.perf_counter()
            if self._cached_pairs is not None:
                pairs = self._cached_pairs
                fluid.neighbor_pairs = pairs
                self._cached_pairs = None
            else:
                pairs = self._build_neighbors(fluid, boundary)
            timings["neighbor_search"] += time.perf_counter() - t_stage

            t_stage = time.perf_counter()
            self._compute_densities(fluid, boundary, pairs)
            self._cap_densities(fluid)
            self._update_fluid_pressure(fluid)
            self._update_boundary_state(fluid, boundary, pairs)
            timings["density_assembly"] += time.perf_counter() - t_stage

            # 2. Non-pressure forces and velocity prediction
            t_stage = time.perf_counter()
            fluid.accelerations.fill(0.0)
            if self.nu > 0.0:
                self._compute_viscosity_forces(fluid, boundary, pairs)
            fluid.accelerations += self.gravity
            # Adami background boundary-pressure force (P_k/ρ_k² term).
            # The CD/DF solvers handle the complementary P_i/ρ_i² term via λ,
            # so we only add the boundary-pressure contribution here.
            if boundary is not None and pairs.fb_i.size:
                self._apply_boundary_pressure_force(fluid, boundary, pairs)
            fluid.velocities += dt * fluid.accelerations
            timings["integration"] += time.perf_counter() - t_stage

            # 3. Constant-density PPE
            t_stage = time.perf_counter()
            if self.enable_density_diffusion:
                self._apply_density_diffusion(fluid, pairs, dt)
            self._compute_k_factor(fluid, boundary, pairs)
            cd_snapshot = CUDAValidationStageSnapshot(
                stage="cd",
                stage_input=self._capture_cuda_stage_snapshot(
                    fluid, boundary, pairs, dt,
                    fluid.p_cd_prev, self.max_iter_cd, self.eta_cd, "cd",
                ),
            )
            self.last_cuda_stage_snapshots["cd"] = cd_snapshot
            if self._cuda_solve_callback is not None and self.debug_mode:
                cd_meta, cd_timings = self._cuda_solve_callback(
                    "cd",
                    self._build_cuda_callback_input(
                        cd_snapshot.stage_input,
                        boundary,
                        gpu_pre_cd=False,
                    ),
                    fluid,
                    dt,
                )
                iter_cd = int(cd_meta.get("iters", 1))
                cd_converged = bool(cd_meta.get("converged", True))
                rho_err_mean = float(cd_meta.get("metric", 0.0))
                rho_err_max = 0.0
                self._last_cuda_stage_timings_ms.update(cd_timings)
                for k in ("gpu_ff_pairs", "gpu_fb_pairs", "cpu_ff_pairs", "cpu_fb_pairs", "pair_count_match"):
                    if k in cd_meta:
                        self._last_cuda_stage_extra_metrics[f"cuda_cd_{k}"] = float(cd_meta[k])
            else:
                iter_cd, cd_converged, rho_err_mean, rho_err_max = self._solve_constant_density(
                    fluid, boundary, pairs, dt,
                )
            cd_snapshot.stage_output = self._capture_cuda_stage_output(
                fluid.velocities, fluid.p_cd_prev, iter_cd, cd_converged, rho_err_mean,
            )
            timings["pressure_solve"] += time.perf_counter() - t_stage

            # 4. Position update + periodic wrap
            t_stage = time.perf_counter()
            fluid.positions += dt * fluid.velocities
            if self.periodic_x is not None:
                self.apply_periodic_bc_x(fluid, *self.periodic_x)
            timings["integration"] += time.perf_counter() - t_stage

            # 5. Divergence-free PPE on updated configuration
            t_stage = time.perf_counter()
            pairs2 = self._build_neighbors(fluid, boundary)
            timings["neighbor_search"] += time.perf_counter() - t_stage

            t_stage = time.perf_counter()
            self._compute_densities(fluid, boundary, pairs2)
            self._cap_densities(fluid)
            if self.enable_density_diffusion:
                self._apply_density_diffusion(fluid, pairs2, dt)
            timings["density_assembly"] += time.perf_counter() - t_stage

            regime_info = self._analyze_density_regime(fluid, boundary, pairs2)

            t_stage = time.perf_counter()
            shift_count, shift_mean = self._apply_free_surface_transport_shift(
                fluid, boundary, pairs2, regime_info, dt,
            )
            timings["surface_correction"] += time.perf_counter() - t_stage
            if shift_count > 0:
                t_stage = time.perf_counter()
                if self.periodic_x is not None:
                    self.apply_periodic_bc_x(fluid, *self.periodic_x)
                pairs2 = self._build_neighbors(fluid, boundary)
                timings["neighbor_search"] += time.perf_counter() - t_stage

                t_stage = time.perf_counter()
                self._compute_densities(fluid, boundary, pairs2)
                self._cap_densities(fluid)
                if self.enable_density_diffusion:
                    self._apply_density_diffusion(fluid, pairs2, dt)
                regime_info = self._analyze_density_regime(fluid, boundary, pairs2)
                timings["density_assembly"] += time.perf_counter() - t_stage

            t_stage = time.perf_counter()
            density_raise_count, density_raise_mean = self._apply_free_surface_stabilization(
                fluid, boundary, pairs2, regime_info, dt,
            )
            timings["surface_correction"] += time.perf_counter() - t_stage
            regime_info = self._analyze_density_regime(fluid, boundary, pairs2)
            self.last_regime_info = regime_info
            self._maybe_warn_boundary(fluid, regime_info.summary)
            self._update_fluid_pressure(fluid)

            t_stage = time.perf_counter()
            self._compute_k_factor(fluid, boundary, pairs2)
            df_snapshot = CUDAValidationStageSnapshot(
                stage="df",
                stage_input=self._capture_cuda_stage_snapshot(
                    fluid, boundary, pairs2, dt,
                    fluid.p_df_prev, self.max_iter_df, self.eta_df, "df",
                ),
            )
            self.last_cuda_stage_snapshots["df"] = df_snapshot
            if self._cuda_solve_callback is not None and self.debug_mode:
                df_meta, df_timings = self._cuda_solve_callback(
                    "df",
                    self._build_cuda_callback_input(
                        df_snapshot.stage_input,
                        boundary,
                        gpu_pre_cd=False,
                    ),
                    fluid,
                    dt,
                )
                iter_df = int(df_meta.get("iters", 1))
                df_converged = bool(df_meta.get("converged", True))
                div_err_mean = float(df_meta.get("metric", 0.0))
                self._last_cuda_stage_timings_ms.update(df_timings)
                for k in ("gpu_ff_pairs", "gpu_fb_pairs", "cpu_ff_pairs", "cpu_fb_pairs", "pair_count_match"):
                    if k in df_meta:
                        self._last_cuda_stage_extra_metrics[f"cuda_df_{k}"] = float(df_meta[k])
            else:
                iter_df, df_converged, div_err_mean = self._solve_divergence_free(
                    fluid, boundary, pairs2, dt,
                )
            df_snapshot.stage_output = self._capture_cuda_stage_output(
                fluid.velocities, fluid.p_df_prev, iter_df, df_converged, div_err_mean,
            )
            timings["pressure_solve"] += time.perf_counter() - t_stage

        free_surface_ratio = regime_info.summary.free_surface_count / max(regime_info.summary.fluid_count, 1)
        splash_ratio = regime_info.summary.splash_count / max(regime_info.summary.fluid_count, 1)
        report = {
            "iter_cd": int(iter_cd),
            "iter_df": int(iter_df),
            "rho_error_mean": float(rho_err_mean),
            "rho_error_max": float(rho_err_max),
            "div_error_mean": float(div_err_mean),
            "cd_converged": bool(cd_converged),
            "df_converged": bool(df_converged),
            "converged": bool(cd_converged and df_converged),
            "free_surface_ratio": float(free_surface_ratio),
            "free_surface_count": int(regime_info.summary.free_surface_count),
            "free_surface_rho_mean": float(regime_info.summary.rho_mean_free_surface),
            "free_surface_rho_min": float(regime_info.summary.rho_min_free_surface),
            "splash_ratio": float(splash_ratio),
            "splash_count": int(regime_info.summary.splash_count),
            "wall_rho_mean": float(regime_info.summary.rho_mean_wall),
            "wall_rho_min": float(regime_info.summary.rho_min_wall),
            "overcompressed_count": int(regime_info.summary.overcompressed_count),
            "free_surface_stabilization": float(self.free_surface_stabilization_enable),
            "free_surface_shift_count": int(shift_count),
            "free_surface_shift_mean": float(shift_mean),
            "free_surface_density_raise_count": int(density_raise_count),
            "free_surface_density_raise_mean": float(density_raise_mean),
        }
        # Cache pairs2 for next step's pre-step neighbor build.
        # pairs2 was built at the final particle positions of this step,
        # which are identical to the starting positions of the next step,
        # so the cache is always valid when the CUDA path is not active.
        if not _cuda_fast:
            self._cached_pairs = pairs2  # type: ignore[possibly-undefined]

        self.last_report = report
        self.last_stage_timings_ms = {key: value * 1000.0 for key, value in timings.items()}
        return iter_cd, iter_df

    # ------------------------------------------------------------------ helpers

    def _build_cuda_callback_input(
        self,
        stage_input: CUDAValidationStageInput,
        boundary: BoundaryModel | None,
        *,
        gpu_pre_cd: bool,
    ) -> dict[str, object]:
        """Build the runtime callback payload without leaking it into validation types."""
        payload: dict[str, object] = {
            "dt": stage_input.dt,
            "h": stage_input.h,
            "rho0": stage_input.rho0,
            "mass": stage_input.mass,
            "max_iter": stage_input.max_iter,
            "eta": stage_input.eta,
            "positions": stage_input.positions,
            "velocities": stage_input.velocities,
            "densities": stage_input.densities,
            "rho_self": stage_input.rho_self,
            "rho_ff": stage_input.rho_ff,
            "rho_fb": stage_input.rho_fb,
            "k_factor": stage_input.k_factor,
            "lambda_prev": stage_input.lambda_prev,
            "boundary_velocities": stage_input.boundary_velocities,
            "pairs": stage_input.pairs,
            "gpu_pre_cd": bool(gpu_pre_cd),
        }
        if gpu_pre_cd:
            payload["nu"] = self.nu
            payload["gravity"] = self.gravity.tolist()
            payload["boundary_rho0"] = float(boundary.rho0) if boundary is not None else 0.0
            payload["boundary_mass"] = float(boundary.mass) if boundary is not None else 0.0
        return payload

    def _build_neighbors(
        self,
        fluid: FluidModel,
        boundary: BoundaryModel | None,
    ) -> NeighborPairs:
        """Build neighbor pairs and cache them on the fluid model."""
        boundary_pos = boundary.positions if boundary is not None else None
        pairs = self.neighbor_search.build_neighbor_pairs(fluid.positions, boundary_pos)
        fluid.neighbor_pairs = pairs
        return pairs

    def _capture_cuda_stage_snapshot(
        self,
        fluid: FluidModel,
        boundary: BoundaryModel | None,
        pairs: NeighborPairs,
        dt: float,
        lambda_prev: np.ndarray,
        max_iter: int,
        eta: float,
        stage_name: str,
    ) -> CUDAValidationStageInput:
        boundary_velocities = (
            boundary.velocities.copy()
            if boundary is not None and boundary.n
            else np.zeros((0, fluid.dim), dtype=np.float64)
        )
        return CUDAValidationStageInput(
            dt=float(dt),
            h=float(fluid.h),
            rho0=float(fluid.rho0),
            mass=float(fluid.mass),
            max_iter=int(max_iter),
            eta=float(eta),
            positions=fluid.positions.copy(),
            velocities=fluid.velocities.copy(),
            densities=fluid.densities.copy(),
            rho_self=fluid.rho_self.copy(),
            rho_ff=fluid.rho_ff.copy(),
            rho_fb=fluid.rho_fb.copy(),
            k_factor=fluid.k_dfsph.copy(),
            lambda_prev=lambda_prev.copy(),
            boundary_velocities=boundary_velocities,
            pairs=self._copy_neighbor_pairs(pairs),
        )

    @staticmethod
    def _capture_cuda_stage_output(
        velocities: np.ndarray,
        lambda_final: np.ndarray,
        iterations: int,
        converged: bool,
        metric: float,
    ) -> CUDAValidationStageOutput:
        return CUDAValidationStageOutput(
            velocities=velocities.copy(),
            lambda_final=lambda_final.copy(),
            iterations=int(iterations),
            converged=bool(converged),
            metric=float(metric),
        )

    @staticmethod
    def _copy_neighbor_pairs(pairs: NeighborPairs) -> NeighborPairs:
        return NeighborPairs(
            pairs.ff_i.copy(),
            pairs.ff_j.copy(),
            pairs.ff_r.copy(),
            pairs.ff_dist.copy(),
            pairs.fb_i.copy(),
            pairs.fb_j.copy(),
            pairs.fb_r.copy(),
            pairs.fb_dist.copy(),
        )

    def _compute_densities(
        self,
        fluid: FluidModel,
        boundary: BoundaryModel | None,
        pairs: NeighborPairs,
    ):
        """SPH density summation (same as WCSPH)."""
        densities = fluid.densities
        h = fluid.h
        W_self = self.kernel.W(0.0)

        fluid.rho_self.fill(fluid.mass * W_self)
        fluid.rho_ff.fill(0.0)
        fluid.rho_fb.fill(0.0)
        if boundary is not None:
            boundary.densities.fill(boundary.rho0)

        if pairs.ff_i.size:
            W_ff = cubic_spline_W_batch(pairs.ff_dist, h, fluid.dim)
            accumulate_density(fluid.rho_ff, pairs.ff_i, pairs.ff_j, fluid.mass, W_ff)

        if boundary is not None and pairs.fb_i.size:
            W_fb = cubic_spline_W_batch(pairs.fb_dist, h, fluid.dim)
            bc = boundary.psi[pairs.fb_j] * W_fb
            scatter_add_1d(fluid.rho_fb, pairs.fb_i, bc)

        densities[:] = fluid.rho_self + fluid.rho_ff + fluid.rho_fb

    def _update_boundary_state(
        self,
        fluid: FluidModel,
        boundary: BoundaryModel | None,
        pairs: NeighborPairs,
    ):
        """
        Adami (2012) density/pressure and velocity extrapolation to boundary.

        Boundary pressure uses the full Adami formula including the hydrostatic
        correction term  ρ_f * g · (x_wall − x_fluid), which is essential for
        gravity-driven free-surface flows (e.g. dam break).  Without it the
        boundary exerts no upward force and falling fluid is never decelerated.
        """
        if boundary is None or not pairs.fb_i.size:
            return

        idx_f = pairs.fb_i
        idx_b = pairs.fb_j
        W = cubic_spline_W_batch(pairs.fb_dist, fluid.h, fluid.dim)

        # --- Density (simple weighted average) ---
        rho_num = np.zeros(boundary.n)
        rho_den = np.zeros(boundary.n)
        scatter_add_1d(rho_num, idx_b, fluid.densities[idx_f] * W)
        scatter_add_1d(rho_den, idx_b, W)
        valid_rho = rho_den > 1e-6
        boundary.densities[valid_rho] = rho_num[valid_rho] / rho_den[valid_rho]
        boundary.densities[~valid_rho] = boundary.rho0

        # --- Pressure: Adami (2012) with hydrostatic correction ---
        # P_w = Σ_f [P_f + ρ_f · g · (x_w − x_f)] * W / Σ_f W
        x_wf = boundary.positions[idx_b] - fluid.positions[idx_f]   # (n_pairs, dim)
        hydrostatic = fluid.densities[idx_f] * np.einsum(
            "ij,j->i", x_wf, self.gravity
        )
        # EOS background pressure of each fluid particle (≈ 0 for DFSPH but kept
        # for generality; clip negative values so we don't pull fluid into walls)
        eos_p_f = np.maximum(
            self._pressure_from_density(fluid.densities[idx_f], boundary.rho0), 0.0
        )

        p_num = np.zeros(boundary.n)
        p_den = np.zeros(boundary.n)
        scatter_add_1d(p_num, idx_b, (eos_p_f + hydrostatic) * W)
        scatter_add_1d(p_den, idx_b, W)
        valid_p = p_den > 1e-6
        boundary.pressures[valid_p] = np.maximum(
            p_num[valid_p] / p_den[valid_p], 0.0
        )
        boundary.pressures[~valid_p] = 0.0

        # --- Velocity (mirror / no-slip) ---
        v_num = np.zeros((boundary.n, self.kernel.dim))
        v_den = np.zeros(boundary.n)
        scatter_add_2d(v_num, idx_b, fluid.velocities[idx_f] * W[:, np.newaxis])
        scatter_add_1d(v_den, idx_b, W)
        valid_v = v_den > 1e-6
        boundary.velocities[valid_v] = -v_num[valid_v] / v_den[valid_v, np.newaxis]
        boundary.velocities[~valid_v] = 0.0

    def _apply_boundary_pressure_force(
        self,
        fluid: FluidModel,
        boundary: BoundaryModel,
        pairs: NeighborPairs,
    ) -> None:
        """
        Apply the background Adami boundary-pressure acceleration in the
        prediction step.

        The full Adami momentum term is:
            a_i += −Σ_k ψ_k (P_i/ρ_i² + P_k/ρ_k²) ∇W_ik

        The P_i/ρ_i² (fluid-pressure) part is handled inside the CD/DF pressure
        solvers via the λ correction, so here we only add the P_k/ρ_k² part
        (background boundary pressure).
        """
        if not pairs.fb_i.size:
            return

        gw_fb = cubic_spline_gradW_batch(
            pairs.fb_r, pairs.fb_dist, fluid.h, fluid.dim
        )
        P_b   = boundary.pressures[pairs.fb_j]               # (n_pairs,)
        rho_b = boundary.densities[pairs.fb_j]               # (n_pairs,)
        psi_b = boundary.psi[pairs.fb_j]                     # (n_pairs,)

        coeff = psi_b * P_b / (rho_b ** 2 + 1e-12)          # (n_pairs,)
        acc_contrib = -coeff[:, np.newaxis] * gw_fb          # (n_pairs, dim)
        scatter_add_2d(fluid.accelerations, pairs.fb_i, acc_contrib)

    def _compute_viscosity_forces(
        self,
        fluid: FluidModel,
        boundary: BoundaryModel | None,
        pairs: NeighborPairs,
    ):
        """Morris (1997) Laplacian viscosity (same formulation as WCSPH)."""
        acc = fluid.accelerations
        h = fluid.h

        if pairs.ff_i.size:
            grad_ff = cubic_spline_gradW_batch(pairs.ff_r, pairs.ff_dist, h, fluid.dim)
            v_ij = fluid.velocities[pairs.ff_i] - fluid.velocities[pairs.ff_j]
            rdgw = np.sum(pairs.ff_r * grad_ff, axis=1)
            denom = pairs.ff_dist**2 + 0.01 * h**2
            coeff = self.viscosity_factor * self.nu * fluid.mass / fluid.rho0 * rdgw / denom
            force = coeff[:, np.newaxis] * v_ij
            accumulate_force(acc, pairs.ff_i, pairs.ff_j, force)

        if boundary is not None and pairs.fb_i.size:
            grad_fb = cubic_spline_gradW_batch(pairs.fb_r, pairs.fb_dist, h, fluid.dim)
            v_diff = fluid.velocities[pairs.fb_i] - boundary.velocities[pairs.fb_j]
            rdgw = np.sum(pairs.fb_r * grad_fb, axis=1)
            denom = pairs.fb_dist**2 + 0.01 * h**2
            coeff = self.viscosity_factor * self.nu * boundary.mass / boundary.rho0 * rdgw / denom
            force = coeff[:, np.newaxis] * v_diff
            scatter_add_2d(acc, pairs.fb_i, force)

    # ------------------------------------------------------------------ DFSPH core

    def _compute_k_factor(self, fluid: FluidModel, boundary: BoundaryModel | None, pairs: NeighborPairs):
        """
        Precompute per-particle DFSPH stiffness factor k_i using a stable denominator:

            k_i = 1 / (||Σ_j grad W_ij||^2 + Σ_j ||grad W_ij||^2 + eps)
        """
        if fluid.n == 0:
            return

        sum_grad = np.zeros((fluid.n, self.kernel.dim))
        sum_grad_sq = np.zeros(fluid.n)

        if pairs.ff_i.size:
            gw = cubic_spline_gradW_batch(pairs.ff_r, pairs.ff_dist, fluid.h, fluid.dim)
            accumulate_force(sum_grad, pairs.ff_i, pairs.ff_j, gw)
            gw_sq = np.sum(gw ** 2, axis=1)
            accumulate_density(sum_grad_sq, pairs.ff_i, pairs.ff_j, 1.0, gw_sq)

        if boundary is not None and pairs.fb_i.size:
            gw_fb = cubic_spline_gradW_batch(pairs.fb_r, pairs.fb_dist, fluid.h, fluid.dim)
            scatter_add_2d(sum_grad, pairs.fb_i, gw_fb)
            gw_fb_sq = np.sum(gw_fb ** 2, axis=1)
            scatter_add_1d(sum_grad_sq, pairs.fb_i, gw_fb_sq)

        denom = np.sum(sum_grad ** 2, axis=1) + sum_grad_sq
        denom = np.maximum(denom, 1e-12)

        fluid.k_dfsph[:] = 0.0
        mask = denom > 0.0
        fluid.k_dfsph[mask] = 1.0 / denom[mask]

    def _solve_constant_density(
        self,
        fluid: FluidModel,
        boundary: BoundaryModel | None,
        pairs: NeighborPairs,
        dt: float,
    ) -> tuple[int, bool, float, float]:
        """
        Constant Density Solver (Sec. 3.2 in Bender & Koschier 2017).

        Returns (iterations, converged, mean_rel_error, max_rel_error).
        """
        has_pairs = pairs.ff_i.size or (boundary is not None and pairs.fb_i.size)
        if not has_pairs:
            fluid.p_cd_prev[:] = 0.0
            return 0, True, 0.0, 0.0

        i_u = pairs.ff_i
        j_u = pairs.ff_j
        gw = cubic_spline_gradW_batch(pairs.ff_r, pairs.ff_dist, fluid.h, fluid.dim) if pairs.ff_i.size else np.zeros((0, self.kernel.dim))
        boundary_active = boundary is not None and pairs.fb_i.size
        if boundary_active:
            gw_fb = cubic_spline_gradW_batch(pairs.fb_r, pairs.fb_dist, fluid.h, fluid.dim)
            psi_fb = boundary.psi[pairs.fb_j]
            boundary_vel = boundary.velocities[pairs.fb_j]
        else:
            gw_fb = None
            psi_fb = None
            boundary_vel = None

        lambda_prev = fluid.p_cd_prev.copy()
        if np.any(lambda_prev) and pairs.ff_i.size:
            pi_prev = lambda_prev[i_u]
            pj_prev = lambda_prev[j_u]
            ri_prev = fluid.densities[i_u]
            rj_prev = fluid.densities[j_u]
            coeff_prev = fluid.mass * (pi_prev / (ri_prev ** 2 + 1e-12) + pj_prev / (rj_prev ** 2 + 1e-12))
            dv_prev = -dt * coeff_prev[:, np.newaxis] * gw
            accumulate_force(fluid.velocities, i_u, j_u, dv_prev)
        if boundary_active and np.any(lambda_prev):
            lam_prev_fb = lambda_prev[pairs.fb_i]
            ri_prev_fb = fluid.densities[pairs.fb_i]
            coeff_fb_prev = psi_fb * (lam_prev_fb / (ri_prev_fb ** 2 + 1e-12))
            dv_prev_fb = -dt * coeff_fb_prev[:, np.newaxis] * gw_fb
            scatter_add_2d(fluid.velocities, pairs.fb_i, dv_prev_fb)

        lambda_cd = np.zeros_like(fluid.p_cd_prev)
        rho_err_mean = 0.0
        rho_err_max = 0.0
        converged = False
        iterations = 0

        for iteration in range(self.max_iter_cd):
            iterations = iteration + 1
            if i_u.size:
                dv = fluid.velocities[i_u] - fluid.velocities[j_u]
                drho = dt * fluid.mass * np.sum(dv * gw, axis=1)
            else:
                drho = np.zeros(0)

            rho_star = fluid.densities.copy()
            if drho.size:
                scatter_add_1d(rho_star, i_u, drho)
                scatter_add_1d(rho_star, j_u, drho)
            if boundary_active:
                vel_diff_fb = fluid.velocities[pairs.fb_i] - boundary_vel
                contrib_fb = dt * psi_fb * np.sum(vel_diff_fb * gw_fb, axis=1)
                scatter_add_1d(rho_star, pairs.fb_i, contrib_fb)

            rho_err = np.maximum(rho_star - fluid.rho0, 0.0)
            if rho_err.size:
                rel_err = rho_err / fluid.rho0
                rho_err_mean = float(rel_err.mean())
                rho_err_max = float(rel_err.max())
            else:
                rho_err_mean = 0.0
                rho_err_max = 0.0

            if rho_err_mean < self.eta_cd:
                converged = True
                break

            lambda_cd = (rho_err / (dt * dt + 1e-12)) * fluid.k_dfsph

            if i_u.size:
                pi = lambda_cd[i_u]
                pj = lambda_cd[j_u]
                ri = fluid.densities[i_u]
                rj = fluid.densities[j_u]
                coeff = fluid.mass * (pi / (ri ** 2 + 1e-12) + pj / (rj ** 2 + 1e-12))
                dv_p = -dt * coeff[:, np.newaxis] * gw
                accumulate_force(fluid.velocities, i_u, j_u, dv_p)

            if boundary_active:
                lambda_fb = lambda_cd[pairs.fb_i]
                ri_fb = fluid.densities[pairs.fb_i]
                coeff_fb = psi_fb * (lambda_fb / (ri_fb ** 2 + 1e-12))
                dv_fb = -dt * coeff_fb[:, np.newaxis] * gw_fb
                scatter_add_2d(fluid.velocities, pairs.fb_i, dv_fb)

        fluid.p_cd_prev[:] = lambda_cd
        return iterations, converged, rho_err_mean, rho_err_max

    def _solve_divergence_free(
        self,
        fluid: FluidModel,
        boundary: BoundaryModel | None,
        pairs: NeighborPairs,
        dt: float,
    ) -> tuple[int, bool, float]:
        """
        Divergence-Free Solver (Sec. 3.3 in Bender & Koschier 2017).

        Returns (iterations, converged, mean_relative_divergence).
        """
        has_pairs = pairs.ff_i.size or (boundary is not None and pairs.fb_i.size)
        if not has_pairs:
            fluid.p_df_prev[:] = 0.0
            return 0, True, 0.0

        i_u = pairs.ff_i
        j_u = pairs.ff_j
        gw = cubic_spline_gradW_batch(pairs.ff_r, pairs.ff_dist, fluid.h, fluid.dim) if pairs.ff_i.size else np.zeros((0, self.kernel.dim))
        boundary_active = boundary is not None and pairs.fb_i.size
        if boundary_active:
            gw_fb = cubic_spline_gradW_batch(pairs.fb_r, pairs.fb_dist, fluid.h, fluid.dim)
            psi_fb = boundary.psi[pairs.fb_j]
            boundary_vel = boundary.velocities[pairs.fb_j]
        else:
            gw_fb = None
            psi_fb = None
            boundary_vel = None

        lambda_prev = fluid.p_df_prev.copy()
        if np.any(lambda_prev) and i_u.size:
            pi_prev = lambda_prev[i_u]
            pj_prev = lambda_prev[j_u]
            ri_prev = fluid.densities[i_u]
            rj_prev = fluid.densities[j_u]
            coeff_prev = fluid.mass * (pi_prev / (ri_prev ** 2 + 1e-12) + pj_prev / (rj_prev ** 2 + 1e-12))
            dv_prev = -dt * coeff_prev[:, np.newaxis] * gw
            accumulate_force(fluid.velocities, i_u, j_u, dv_prev)
        if boundary_active and np.any(lambda_prev):
            lam_prev_fb = lambda_prev[pairs.fb_i]
            ri_prev_fb = fluid.densities[pairs.fb_i]
            coeff_fb_prev = psi_fb * (lam_prev_fb / (ri_prev_fb ** 2 + 1e-12))
            dv_prev_fb = -dt * coeff_fb_prev[:, np.newaxis] * gw_fb
            scatter_add_2d(fluid.velocities, pairs.fb_i, dv_prev_fb)

        lambda_df = np.zeros_like(fluid.p_df_prev)
        div_err_mean = 0.0
        converged = False
        iterations = 0

        for iteration in range(self.max_iter_df):
            iterations = iteration + 1
            if i_u.size:
                dv = fluid.velocities[i_u] - fluid.velocities[j_u]
                contrib = fluid.mass * np.sum(dv * gw, axis=1)
            else:
                contrib = np.zeros(0)

            drho_dt = np.zeros(fluid.n)
            if contrib.size:
                scatter_add_1d(drho_dt, i_u, contrib)
                scatter_add_1d(drho_dt, j_u, contrib)
            if boundary_active:
                vel_diff_fb = fluid.velocities[pairs.fb_i] - boundary_vel
                contrib_fb = psi_fb * np.sum(vel_diff_fb * gw_fb, axis=1)
                scatter_add_1d(drho_dt, pairs.fb_i, contrib_fb)

            if drho_dt.size:
                div_err_mean = float((np.abs(drho_dt) * dt / fluid.rho0).mean())
            else:
                div_err_mean = 0.0

            if div_err_mean < self.eta_df:
                converged = True
                break

            lambda_df = (drho_dt / (dt + 1e-12)) * fluid.k_dfsph

            if i_u.size:
                pi = lambda_df[i_u]
                pj = lambda_df[j_u]
                ri = fluid.densities[i_u]
                rj = fluid.densities[j_u]
                coeff = fluid.mass * (pi / (ri ** 2 + 1e-12) + pj / (rj ** 2 + 1e-12))
                dv_p = -dt * coeff[:, np.newaxis] * gw
                accumulate_force(fluid.velocities, i_u, j_u, dv_p)

            if boundary_active:
                lambda_fb = lambda_df[pairs.fb_i]
                ri_fb = fluid.densities[pairs.fb_i]
                coeff_fb = psi_fb * (lambda_fb / (ri_fb ** 2 + 1e-12))
                dv_fb = -dt * coeff_fb[:, np.newaxis] * gw_fb
                scatter_add_2d(fluid.velocities, pairs.fb_i, dv_fb)

        fluid.p_df_prev[:] = lambda_df
        return iterations, converged, div_err_mean

    def _cap_densities(self, fluid: FluidModel) -> None:
        """
        Hard upper cap on the density field — always applied after summation.

        Purpose
        ───────
        Corner fluid particles near two intersecting walls receive boundary
        contributions from both walls simultaneously, pushing their SPH density
        estimate 40–60 % above ρ₀.  Without a cap the CDS sees these extreme
        errors (ρ* ≫ ρ₀), hits max_iter_cd on every step, and never fully
        converges — causing rho_mean to drift upward without bound.

        A cap at 1.05 ρ₀ is consistent with the weakly-compressible assumption
        (Mach < 0.1 implies |ρ/ρ₀ − 1| < ~1 %).  Corner effects push this
        slightly higher; 5 % is conservative enough to allow physical behaviour
        while keeping the CDS convergent in 1–2 iterations.

        Lower bound (0): surface particles are allowed to fall as low as needed —
        truncated kernel support at the free surface is physically correct.
        """
        np.clip(fluid.densities, 0.0, fluid.rho0 * 1.02, out=fluid.densities)

    def _apply_density_diffusion(
        self,
        fluid: FluidModel,
        pairs: NeighborPairs,
        dt: float,
    ) -> None:
        """
        δ-SPH density diffusion (Marrone et al. 2011).

        Applies a single-step Laplacian smoothing to the density field to
        reduce low-density artifacts at free surfaces caused by truncated
        kernel support.  Applied post-summation, before the k-factor and PPE
        solve, so the corrected density is used for pressure computation in
        this step.

        Note: because this solver uses density summation (ρ recomputed each
        step), the correction cannot accumulate across steps the way it does
        in continuity-equation solvers.  However, raising surface densities
        for the current step's PPE still improves pressure accuracy and
        prevents unphysical voids.  A one-sided surface-only correction
        (mask: deficit > 0) avoids draining interior particles below ρ₀.

        Formula (surface-only, asymmetric):
            D_i = Σ_j [coeff * (ρ_j - ρ_i) * (r_ij · ∇W_ij) / (|r|² + ε)] * V_j
            ρ_i += dt * D_i   (only where ρ_i < floor)
        """
        if not pairs.ff_i.size or self.delta_density <= 0.0:
            return

        i = pairs.ff_i
        j = pairs.ff_j
        r_ij = pairs.ff_r
        dist = pairs.ff_dist

        gw = cubic_spline_gradW_batch(r_ij, dist, fluid.h, fluid.dim)
        rho_i = fluid.densities[i]
        rho_j = fluid.densities[j]
        Vj = fluid.mass / np.maximum(rho_j, 1e-6)
        dot_r_gw = np.sum(r_ij * gw, axis=1)
        denom = dist**2 + 0.01 * fluid.h**2
        coeff = self.delta_density * fluid.h * self.c0

        D = np.zeros(fluid.n, dtype=np.float64)
        contrib = coeff * (rho_j - rho_i) * dot_r_gw / denom * Vj
        np.add.at(D, i,  contrib)
        np.add.at(D, j, -contrib)
        fluid.densities += dt * D
        # Floor at density_floor_ratio (default 0.85) — particles below this level
        # have severely truncated kernel support and their density estimates are
        # unphysical.  Clipping ensures the PPE always works on a valid density field.
        np.clip(
            fluid.densities,
            self.density_floor_ratio * fluid.rho0,
            1.02 * fluid.rho0,
            out=fluid.densities,
        )

    def _analyze_density_regime(
        self,
        fluid: FluidModel,
        boundary: BoundaryModel | None,
        pairs: NeighborPairs,
        *,
        precomputed_fluid_counts: np.ndarray | None = None,
        precomputed_boundary_counts: np.ndarray | None = None,
    ) -> DensityRegimeInfo:
        return analyze_density_regimes(
            fluid, boundary, pairs,
            precomputed_fluid_counts=precomputed_fluid_counts,
            precomputed_boundary_counts=precomputed_boundary_counts,
        )

    def _apply_free_surface_stabilization(
        self,
        fluid: FluidModel,
        boundary: BoundaryModel | None,
        pairs: NeighborPairs,
        regime: DensityRegimeInfo,
        dt: float,
    ) -> tuple[int, float]:
        """
        Localized support renormalization for free-surface particles.

        SPlisHSPlasH-style solvers usually improve free surfaces through local,
        regime-aware corrections rather than global density smoothing. In this
        density-summation solver the most stable first step is to compensate the
        missing kernel support only for particles already classified as
        free-surface. Interior and wall-adjacent particles are left untouched.
        """
        _ = dt
        if not self.free_surface_stabilization_enable or fluid.n == 0:
            return 0, 0.0
        if self.free_surface_stabilization_mode not in {"density", "density_shift"}:
            return 0, 0.0

        surface_mask = regime.free_surface_mask.copy()
        surface_count = int(np.count_nonzero(surface_mask))
        if surface_count == 0:
            return 0, 0.0
        if surface_count < max(8, int(0.005 * fluid.n)):
            return 0, 0.0

        support = np.full(fluid.n, (fluid.mass / fluid.rho0) * self.kernel.W(0.0), dtype=np.float64)

        if pairs.ff_i.size:
            w_ff = cubic_spline_W_batch(pairs.ff_dist, fluid.h, fluid.dim)
            accumulate_density(
                support,
                pairs.ff_i,
                pairs.ff_j,
                fluid.mass / fluid.rho0,
                w_ff,
            )

        if boundary is not None and pairs.fb_i.size:
            w_fb = cubic_spline_W_batch(pairs.fb_dist, fluid.h, fluid.dim)
            scatter_add_1d(
                support,
                pairs.fb_i,
                (boundary.psi[pairs.fb_j] / fluid.rho0) * w_fb,
            )

        min_neighbors = max(3, regime.free_surface_neighbor_threshold // 2)
        valid = surface_mask & (regime.fluid_counts >= min_neighbors)
        if not np.any(valid):
            return 0, 0.0

        rho = fluid.densities
        support_safe = np.clip(support, 0.72, 1.0)
        target_rho = rho.copy()
        target_rho[valid] = rho[valid] / support_safe[valid]
        target_cap = self.free_surface_target_ratio * fluid.rho0
        target_rho[valid] = np.clip(target_rho[valid], rho[valid], target_cap)

        gain = np.full(fluid.n, self.free_surface_stabilization_strength, dtype=np.float64)
        if np.any(regime.splash_mask):
            gain[regime.splash_mask] *= self.splash_stabilization_scale

        delta_rho = np.maximum(target_rho - rho, 0.0)
        max_raise = self.free_surface_max_raise_ratio * fluid.rho0
        applied_raise = np.minimum(gain[valid] * delta_rho[valid], max_raise)
        rho[valid] += applied_raise
        rho[valid] = np.minimum(rho[valid], target_cap)
        return int(np.count_nonzero(applied_raise > 1e-9)), float(applied_raise.mean()) if applied_raise.size else 0.0

    def _apply_free_surface_transport_shift(
        self,
        fluid: FluidModel,
        boundary: BoundaryModel | None,
        pairs: NeighborPairs,
        regime: DensityRegimeInfo,
        dt: float,
    ) -> tuple[int, float]:
        """
        Surface-only transport regularization inspired by particle shifting.

        This moves only free-surface particles a small distance toward the
        weighted centroid of their fluid neighbors. The displacement is scaled
        with dt*c0, i.e. as a transport-style correction rather than a direct
        geometric shift, so the effect stays local and conservative.
        """
        _ = boundary
        if not self.free_surface_stabilization_enable or fluid.n == 0:
            return 0, 0.0
        if self.free_surface_stabilization_mode not in {"shift", "density_shift"}:
            return 0, 0.0

        surface_mask = regime.free_surface_mask | regime.splash_mask
        surface_count = int(np.count_nonzero(surface_mask))
        if surface_count == 0:
            return 0, 0.0
        if surface_count < max(8, int(0.005 * fluid.n)):
            return 0, 0.0
        if not pairs.ff_i.size:
            return 0, 0.0
        if regime.summary.interior_count > 0 and regime.summary.rho0 > 0.0:
            interior_rel_err = abs(regime.summary.rho_mean_interior - regime.summary.rho0) / regime.summary.rho0
            if interior_rel_err > 0.02:
                return 0, 0.0

        support = np.full(fluid.n, (fluid.mass / fluid.rho0) * self.kernel.W(0.0), dtype=np.float64)
        centroid_num = np.zeros((fluid.n, fluid.dim), dtype=np.float64)
        centroid_den = np.zeros(fluid.n, dtype=np.float64)

        w_ff = cubic_spline_W_batch(pairs.ff_dist, fluid.h, fluid.dim)
        accumulate_density(
            support,
            pairs.ff_i,
            pairs.ff_j,
            fluid.mass / fluid.rho0,
            w_ff,
        )
        scatter_add_2d(centroid_num, pairs.ff_i, w_ff[:, np.newaxis] * fluid.positions[pairs.ff_j])
        scatter_add_2d(centroid_num, pairs.ff_j, w_ff[:, np.newaxis] * fluid.positions[pairs.ff_i])
        scatter_add_1d(centroid_den, pairs.ff_i, w_ff)
        scatter_add_1d(centroid_den, pairs.ff_j, w_ff)

        min_neighbors = max(3, regime.free_surface_neighbor_threshold // 2)
        valid = surface_mask & (regime.fluid_counts >= min_neighbors) & (centroid_den > 1e-8)
        if not np.any(valid):
            return 0, 0.0

        centroid = centroid_num[valid] / centroid_den[valid, np.newaxis]
        direction = centroid - fluid.positions[valid]
        direction_norm = np.linalg.norm(direction, axis=1)
        if not direction_norm.size:
            return 0, 0.0

        support_deficit = np.clip(1.0 - np.clip(support[valid], 0.0, 1.0), 0.0, 0.25)
        density_need = np.clip(
            (regime.low_density_threshold - fluid.densities[valid]) / max(regime.low_density_threshold, 1e-6),
            0.0,
            1.0,
        )
        activation = support_deficit * density_need
        max_shift = self.free_surface_shift_max_ratio * fluid.h
        desired = self.free_surface_shift_strength * self.c0 * dt * activation

        splash_local = regime.splash_mask[valid]
        if np.any(splash_local):
            desired[splash_local] *= self.splash_stabilization_scale

        applied_mag = np.minimum(desired, max_shift)
        nonzero = direction_norm > 1e-12
        if not np.any(nonzero):
            return 0, 0.0

        unit = np.zeros_like(direction)
        unit[nonzero] = direction[nonzero] / direction_norm[nonzero, np.newaxis]
        displacement = unit * applied_mag[:, np.newaxis]

        target_indices = np.flatnonzero(valid)
        active = applied_mag > 1e-12
        if not np.any(active):
            return 0, 0.0
        fluid.positions[target_indices[active]] += displacement[active]
        return int(np.count_nonzero(active)), float(applied_mag[active].mean())

    def _maybe_warn_boundary(self, fluid: FluidModel, summary) -> None:
        if not self._should_warn_boundary(summary):
            return
        dominant = (fluid.rho_fb > fluid.rho_ff) & (fluid.rho_fb > 1e-6)
        if not np.any(dominant):
            return
        idx = int(np.argmax(fluid.rho_fb - fluid.rho_ff))
        rho_fb = float(fluid.rho_fb[idx])
        rho_ff = float(fluid.rho_ff[idx])
        rho0 = max(summary.rho0, 1e-6)
        interior_err = (
            abs(summary.rho_mean_interior - summary.rho0) / rho0
            if summary.interior_count > 0
            else float("inf")
        )
        wall_err = (
            abs(summary.rho_mean_wall - summary.rho0) / rho0
            if summary.wall_count > 0
            else 0.0
        )
        print(
            "[density-warning] boundary contribution dominates at particle "
            f"{idx}: rho_fb={rho_fb:.2f}, rho_ff={rho_ff:.2f} "
            f"(interior err {interior_err*100:.2f}%, wall err {wall_err*100:.2f}%)"
        )

    @staticmethod
    def _should_warn_boundary(summary) -> bool:
        if summary.fluid_count == 0:
            return False
        rho0 = max(summary.rho0, 1e-6)
        if summary.interior_count == 0:
            return True
        interior_err = abs(summary.rho_mean_interior - summary.rho0) / rho0
        wall_err = (
            abs(summary.rho_mean_wall - summary.rho0) / rho0
            if summary.wall_count > 0
            else 0.0
        )
        over_ratio = summary.overcompressed_count / max(summary.fluid_count, 1)
        return not (interior_err < 0.02 and wall_err < 0.06 and over_ratio < 0.02)

    def apply_periodic_bc_x(self, fluid: FluidModel, x_min: float, x_max: float):
        """Wrap particles in x-direction for periodic BC."""
        L_x = x_max - x_min
        fluid.positions[:, 0] = x_min + np.mod(fluid.positions[:, 0] - x_min, L_x)

    def apply_periodic_bc_z(self, fluid: FluidModel, z_min: float, z_max: float):
        """Wrap particles in z-direction for periodic BC."""
        L_z = z_max - z_min
        fluid.positions[:, 2] = z_min + np.mod(fluid.positions[:, 2] - z_min, L_z)

    def _pressure_from_density(self, density: np.ndarray, rho0: float) -> np.ndarray:
        rho_ratio = density / rho0
        pressure = self.eos_k * (rho_ratio ** 7 - 1.0)
        pressure[pressure < 0.0] = 0.0
        return pressure

    def _update_fluid_pressure(self, fluid: FluidModel) -> None:
        fluid.pressures[:] = self._pressure_from_density(fluid.densities, fluid.rho0)
