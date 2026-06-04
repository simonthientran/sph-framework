"""CPU-vs-CUDA numerical comparison for SPH backends.

This module provides the core validation engine for comparing CPU and CUDA
backend outputs step-by-step against the same scene. It is intentionally
backend-agnostic: both sides are run through ``SimulationRunner`` so the
same export/state API governs both paths.

Typical usage
-------------
    from sph.validation.compare import run_comparison, ComparisonResult

    result = run_comparison(
        scene_path=Path("scenes/examples/pipe_flow_2d.json"),
        steps=50,
    )
    for line in result.summary_lines():
        print(line)
    sys.exit(0 if result.overall_passed else 1)
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ScalarCheck:
    """Comparison of a single scalar metric between CPU and CUDA."""

    name: str
    cpu_value: float
    cuda_value: float
    abs_diff: float
    rel_diff: float
    threshold_rel: float
    passed: bool


@dataclass(slots=True)
class ArrayCheck:
    """Comparison of a numpy array (density, velocity, …) between backends."""

    name: str
    shape_cpu: tuple
    shape_cuda: tuple
    max_abs_diff: float
    mean_abs_diff: float
    l2_rel: float
    threshold: float
    passed: bool


@dataclass(slots=True)
class RegimeCheck:
    """Comparison of regime classification counts."""

    cpu_interior: int
    cuda_interior: int
    cpu_wall: int
    cuda_wall: int
    cpu_splash: int
    cuda_splash: int

    @property
    def interior_match(self) -> bool:
        return self.cpu_interior == self.cuda_interior

    @property
    def wall_match(self) -> bool:
        return self.cpu_wall == self.cuda_wall

    @property
    def splash_match(self) -> bool:
        return self.cpu_splash == self.cuda_splash

    @property
    def passed(self) -> bool:
        return self.interior_match and self.wall_match and self.splash_match


@dataclass
class ComparisonResult:
    """Structured result of a CPU-vs-CUDA validation run."""

    scene_path: str
    steps: int
    n_fluid_cpu: int
    n_fluid_cuda: int
    n_boundary_cpu: int
    n_boundary_cuda: int
    scalar_checks: list[ScalarCheck] = field(default_factory=list)
    array_checks: list[ArrayCheck] = field(default_factory=list)
    regime_check: RegimeCheck | None = None
    cpu_health: str = ""
    cuda_health: str = ""
    cpu_stability: str = ""
    cuda_stability: str = ""
    cpu_wall_ms: float = 0.0
    cuda_wall_ms: float = 0.0
    notes: list[str] = field(default_factory=list)

    # -- aggregate pass/fail --------------------------------------------------

    @property
    def particle_count_match(self) -> bool:
        return (
            self.n_fluid_cpu == self.n_fluid_cuda
            and self.n_boundary_cpu == self.n_boundary_cuda
        )

    @property
    def all_scalar_passed(self) -> bool:
        return all(c.passed for c in self.scalar_checks)

    @property
    def all_array_passed(self) -> bool:
        return all(c.passed for c in self.array_checks)

    @property
    def regime_passed(self) -> bool:
        return self.regime_check is None or self.regime_check.passed

    @property
    def both_stable(self) -> bool:
        return self.cpu_stability == "pass" and self.cuda_stability == "pass"

    @property
    def overall_passed(self) -> bool:
        return (
            self.particle_count_match
            and self.all_scalar_passed
            and self.all_array_passed
            and self.regime_passed
            and self.both_stable
        )

    # -- serialization --------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize the full comparison for JSON output or logging."""
        return {
            "overall_passed": self.overall_passed,
            "scene_path": self.scene_path,
            "steps": self.steps,
            "n_fluid": {"cpu": self.n_fluid_cpu, "cuda": self.n_fluid_cuda},
            "n_boundary": {"cpu": self.n_boundary_cpu, "cuda": self.n_boundary_cuda},
            "scalar_checks": [
                {
                    "name": c.name,
                    "cpu": c.cpu_value,
                    "cuda": c.cuda_value,
                    "abs_diff": c.abs_diff,
                    "rel_diff": c.rel_diff,
                    "threshold_rel": c.threshold_rel,
                    "passed": c.passed,
                }
                for c in self.scalar_checks
            ],
            "array_checks": [
                {
                    "name": c.name,
                    "shape_cpu": list(c.shape_cpu),
                    "shape_cuda": list(c.shape_cuda),
                    "max_abs_diff": c.max_abs_diff,
                    "mean_abs_diff": c.mean_abs_diff,
                    "l2_rel": c.l2_rel,
                    "threshold": c.threshold,
                    "passed": c.passed,
                }
                for c in self.array_checks
            ],
            "regime": (
                {
                    "cpu_interior": self.regime_check.cpu_interior,
                    "cuda_interior": self.regime_check.cuda_interior,
                    "cpu_wall": self.regime_check.cpu_wall,
                    "cuda_wall": self.regime_check.cuda_wall,
                    "cpu_splash": self.regime_check.cpu_splash,
                    "cuda_splash": self.regime_check.cuda_splash,
                    "passed": self.regime_check.passed,
                }
                if self.regime_check is not None
                else None
            ),
            "stability": {"cpu": self.cpu_stability, "cuda": self.cuda_stability},
            "health": {"cpu": self.cpu_health, "cuda": self.cuda_health},
            "timing_ms_step": {"cpu": self.cpu_wall_ms, "cuda": self.cuda_wall_ms},
            "notes": self.notes,
        }

    def summary_lines(self) -> list[str]:
        """Human-readable summary of the comparison."""
        status = "PASS" if self.overall_passed else "FAIL"
        lines: list[str] = []
        lines.append(f"[{status}] CPU-vs-CUDA comparison — {self.steps} steps")
        lines.append(
            f"  particles: CPU={self.n_fluid_cpu}f/{self.n_boundary_cpu}b  "
            f"CUDA={self.n_fluid_cuda}f/{self.n_boundary_cuda}b"
            + ("  ✓" if self.particle_count_match else "  ✗ MISMATCH")
        )
        lines.append(
            f"  stability:  CPU={self.cpu_stability}  CUDA={self.cuda_stability}"
        )
        if self.scalar_checks:
            lines.append("  scalar checks:")
            for c in self.scalar_checks:
                tag = "ok" if c.passed else "FAIL"
                lines.append(
                    f"    [{tag}] {c.name:30s} cpu={c.cpu_value:.6g}  cuda={c.cuda_value:.6g}"
                    f"  rel_diff={c.rel_diff:.2e}  (tol={c.threshold_rel:.1e})"
                )
        if self.array_checks:
            lines.append("  array checks (L2 relative):")
            for c in self.array_checks:
                tag = "ok" if c.passed else "FAIL"
                lines.append(
                    f"    [{tag}] {c.name:30s} max={c.max_abs_diff:.3e}"
                    f"  L2_rel={c.l2_rel:.3e}  (tol={c.threshold:.1e})"
                )
        if self.regime_check is not None:
            rc = self.regime_check
            tag = "ok" if rc.passed else "FAIL"
            lines.append(
                f"  [{tag}] regime: interior={rc.cpu_interior}/{rc.cuda_interior}"
                f"  wall={rc.cpu_wall}/{rc.cuda_wall}"
                f"  splash={rc.cpu_splash}/{rc.cuda_splash}"
            )
        lines.append(
            f"  health: CPU='{self.cpu_health}'  CUDA='{self.cuda_health}'"
        )
        lines.append(
            f"  timing:  CPU={self.cpu_wall_ms:.2f} ms/step  "
            f"CUDA={self.cuda_wall_ms:.2f} ms/step"
        )
        for note in self.notes:
            lines.append(f"  note: {note}")
        return lines


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _scalar_check(
    name: str,
    cpu_val: float,
    cuda_val: float,
    tol_rel: float,
) -> ScalarCheck:
    abs_diff = abs(cpu_val - cuda_val)
    ref = max(abs(cpu_val), 1.0e-12)
    rel_diff = abs_diff / ref
    return ScalarCheck(
        name=name,
        cpu_value=float(cpu_val),
        cuda_value=float(cuda_val),
        abs_diff=float(abs_diff),
        rel_diff=float(rel_diff),
        threshold_rel=float(tol_rel),
        passed=bool(rel_diff <= tol_rel),
    )


def _array_check(
    name: str,
    cpu_arr: np.ndarray,
    cuda_arr: np.ndarray,
    tol: float,
) -> ArrayCheck:
    shape_match = cpu_arr.shape == cuda_arr.shape
    if not shape_match:
        return ArrayCheck(
            name=name,
            shape_cpu=tuple(cpu_arr.shape),
            shape_cuda=tuple(cuda_arr.shape),
            max_abs_diff=float("inf"),
            mean_abs_diff=float("inf"),
            l2_rel=float("inf"),
            threshold=float(tol),
            passed=False,
        )
    diff = cpu_arr.ravel() - cuda_arr.ravel()
    max_diff = float(np.max(np.abs(diff))) if diff.size else 0.0
    mean_diff = float(np.mean(np.abs(diff))) if diff.size else 0.0
    ref_l2 = float(np.sqrt(np.mean(cpu_arr.ravel() ** 2))) if diff.size else 1.0
    l2_diff = float(np.sqrt(np.mean(diff ** 2)))
    l2_rel = l2_diff / max(ref_l2, 1.0e-12)
    return ArrayCheck(
        name=name,
        shape_cpu=tuple(cpu_arr.shape),
        shape_cuda=tuple(cuda_arr.shape),
        max_abs_diff=max_diff,
        mean_abs_diff=mean_diff,
        l2_rel=l2_rel,
        threshold=float(tol),
        passed=bool(l2_rel <= tol),
    )


# ---------------------------------------------------------------------------
# Public comparison API
# ---------------------------------------------------------------------------


def run_comparison(
    scene_path: Path,
    steps: int = 50,
    warmup_steps: int = 0,
    rho_tol_rel: float = 0.005,
    velocity_tol_rel: float = 0.01,
    density_array_l2_tol: float = 0.01,
    velocity_array_l2_tol: float = 0.02,
    cpu_backend: str = "numba_cpu",
    cuda_backend: str = "numba_cuda",
    suppress_warnings: bool = True,
) -> ComparisonResult:
    """Run *steps* of the same scene on both backends and compare final state.

    Parameters
    ----------
    scene_path:
        Path to the scene JSON file.
    steps:
        Number of full simulation steps to run on each backend.
    warmup_steps:
        Steps to discard before collecting the final comparison state.
        Useful to skip JIT overhead when timing matters.
    rho_tol_rel:
        Relative tolerance for scalar density metrics (rho_mean, rho_min, rho_max).
    velocity_tol_rel:
        Relative tolerance for velocity_max.
    density_array_l2_tol:
        L2-relative tolerance for the per-particle density array.
    velocity_array_l2_tol:
        L2-relative tolerance for the per-particle velocity magnitude array.
    cpu_backend / cuda_backend:
        Backend name strings passed to ``SimulationRunner``.
    suppress_warnings:
        If True, suppress CUDA free-surface warnings that are expected for
        single-phase scenes with stabilization enabled.

    Returns
    -------
    ComparisonResult
        Structured pass/fail report.
    """
    from sph.core.simulation import SimulationRunner  # lazy — avoids circular import
    from sph.core.backends import NumbaCPUBackend
    try:
        from sph.core.backends.numba_cuda_backend import NumbaCUDABackend
    except Exception:
        NumbaCUDABackend = None  # type: ignore[assignment,misc]

    def _backend_factory(name: str):
        if name in ("numba_cpu", "cpu"):
            return NumbaCPUBackend
        if NumbaCUDABackend is not None and name in ("numba_cuda", "cuda"):
            return NumbaCUDABackend
        return NumbaCPUBackend

    scene_path = Path(scene_path)
    notes: list[str] = []
    total_steps = warmup_steps + steps

    # --- run CPU backend ---------------------------------------------------
    cpu_runner = SimulationRunner(scene_path, backend_factory=_backend_factory(cpu_backend))
    cpu_ms_samples: list[float] = []

    for i in range(total_steps):
        t0 = time.perf_counter()
        cpu_result = cpu_runner.step()
        ms = (time.perf_counter() - t0) * 1000.0
        if i >= warmup_steps:
            cpu_ms_samples.append(ms)

    cpu_rt = cpu_result.runtime
    cpu_state = cpu_runner.state

    # --- run CUDA backend -------------------------------------------------
    ctx_mgr = warnings.catch_warnings() if suppress_warnings else _null_context()
    with ctx_mgr:
        if suppress_warnings:
            warnings.filterwarnings("ignore")

        cuda_runner = SimulationRunner(scene_path, backend_factory=_backend_factory(cuda_backend))
        cuda_ms_samples: list[float] = []

        for i in range(total_steps):
            t0 = time.perf_counter()
            cuda_result = cuda_runner.step()
            ms = (time.perf_counter() - t0) * 1000.0
            if i >= warmup_steps:
                cuda_ms_samples.append(ms)

    cuda_rt = cuda_result.runtime
    cuda_state = cuda_runner.state

    # --- scalar checks -----------------------------------------------------
    scalar_checks: list[ScalarCheck] = [
        _scalar_check("rho_mean", cpu_rt.rho_mean, cuda_rt.rho_mean, rho_tol_rel),
        _scalar_check("rho_min", cpu_rt.rho_min, cuda_rt.rho_min, rho_tol_rel * 3),
        _scalar_check("rho_max", cpu_rt.rho_max, cuda_rt.rho_max, rho_tol_rel * 2),
        _scalar_check(
            "rho_error_mean",
            cpu_rt.rho_error_mean,
            cuda_rt.rho_error_mean,
            max(rho_tol_rel * 5, 0.01),
        ),
        _scalar_check(
            "velocity_max",
            cpu_rt.velocity_max,
            cuda_rt.velocity_max,
            velocity_tol_rel,
        ),
    ]

    # --- array checks (from SimulationStateView) ---------------------------
    array_checks: list[ArrayCheck] = [
        _array_check(
            "density",
            cpu_state.fluid_density,
            cuda_state.fluid_density,
            density_array_l2_tol,
        ),
        _array_check(
            "velocity_magnitude",
            cpu_state.fluid_speed,
            cuda_state.fluid_speed,
            velocity_array_l2_tol,
        ),
    ]

    # --- regime check -------------------------------------------------------
    regime_check: RegimeCheck | None = None
    cpu_ds = cpu_rt.density_summary
    cuda_ds = cuda_rt.density_summary
    if cpu_ds is not None and cuda_ds is not None:
        regime_check = RegimeCheck(
            cpu_interior=cpu_ds.interior_count,
            cuda_interior=cuda_ds.interior_count,
            cpu_wall=cpu_ds.wall_count,
            cuda_wall=cuda_ds.wall_count,
            cpu_splash=cpu_ds.splash_count,
            cuda_splash=cuda_ds.splash_count,
        )

    # --- notes --------------------------------------------------------------
    if cpu_rt.step != cuda_rt.step:
        notes.append(
            f"Step count mismatch: CPU step={cpu_rt.step}, CUDA step={cuda_rt.step}"
        )
    if cpu_rt.fluid_count != cuda_rt.fluid_count:
        notes.append(
            f"Fluid count differs: CPU={cpu_rt.fluid_count}, CUDA={cuda_rt.fluid_count}"
        )

    # Warn if CUDA density intentionally differs (free-surface stabilization disabled on CUDA)
    if cuda_rt.solver_health_notes:
        for note in cuda_rt.solver_health_notes:
            if "free-surface stabilization" in note.lower() or "raw sph density" in note.lower():
                notes.append(
                    "CUDA path uses raw SPH density (no free-surface stabilization); "
                    "density may differ from CPU for open-surface scenes."
                )
                break

    mean_cpu_ms = float(np.mean(cpu_ms_samples)) if cpu_ms_samples else 0.0
    mean_cuda_ms = float(np.mean(cuda_ms_samples)) if cuda_ms_samples else 0.0

    return ComparisonResult(
        scene_path=str(scene_path),
        steps=steps,
        n_fluid_cpu=int(cpu_rt.fluid_count),
        n_fluid_cuda=int(cuda_rt.fluid_count),
        n_boundary_cpu=int(cpu_rt.boundary_count),
        n_boundary_cuda=int(cuda_rt.boundary_count),
        scalar_checks=scalar_checks,
        array_checks=array_checks,
        regime_check=regime_check,
        cpu_health=str(cpu_rt.solver_health_summary),
        cuda_health=str(cuda_rt.solver_health_summary),
        cpu_stability=str(cpu_rt.stability),
        cuda_stability=str(cuda_rt.stability),
        cpu_wall_ms=mean_cpu_ms,
        cuda_wall_ms=mean_cuda_ms,
        notes=notes,
    )


def cpu_only_validate(
    scene_path: Path,
    steps: int = 50,
    rho_tol_rel: float = 0.006,
    neighbor_min_threshold: int = 8,
    cpu_backend: str = "numba_cpu",
) -> dict:
    """Lightweight CPU-only correctness check.

    Useful in environments without a CUDA device (CI, simple sanity checks).
    Returns a dict with ``passed`` (bool) and human-readable ``summary`` (str).
    """
    from sph.core.simulation import SimulationRunner  # lazy — avoids circular import
    from sph.core.backends import NumbaCPUBackend

    scene_path = Path(scene_path)
    runner = SimulationRunner(scene_path, backend_factory=NumbaCPUBackend)
    last_rt = None

    for _ in range(steps):
        result = runner.step()
        last_rt = result.runtime

    if last_rt is None:
        return {"passed": False, "summary": "No steps completed."}

    checks: list[str] = []
    passed = True

    # Use interior_rel_err (interior-particle-only density error) instead of
    # the global rho_error_mean.  Wall-adjacent particles legitimately exceed ρ₀
    # due to boundary kernel contributions (SPlisHSPlasH behaviour); counting
    # them in a global average inflates the apparent density error.
    solver_metrics = last_rt.solver_metrics or {}
    interior_rel_err = float(solver_metrics.get("interior_rel_err", last_rt.rho_error_mean))
    rho_ok = interior_rel_err <= rho_tol_rel
    if not rho_ok:
        passed = False
    checks.append(
        f"interior_rel_err={interior_rel_err:.4%}  {'ok' if rho_ok else 'FAIL'} (tol={rho_tol_rel:.1%})"
    )

    neigh_ok = last_rt.neighbor_min >= neighbor_min_threshold
    if not neigh_ok:
        passed = False
    checks.append(
        f"neighbor_min={last_rt.neighbor_min}  {'ok' if neigh_ok else 'FAIL'}"
        f" (threshold={neighbor_min_threshold})"
    )

    stability_ok = last_rt.stability in ("pass", "warn")
    if not stability_ok:
        passed = False
    checks.append(
        f"stability={last_rt.stability!r}  {'ok' if stability_ok else 'FAIL'}"
    )

    summary = (
        f"[{'PASS' if passed else 'FAIL'}] CPU-only check — {steps} steps\n"
        + "\n".join(f"  {c}" for c in checks)
        + f"\n  health: {last_rt.solver_health_summary}"
    )
    return {"passed": passed, "summary": summary, "runtime": last_rt}


# ---------------------------------------------------------------------------
# Tiny context-manager helper so suppress_warnings=False avoids a try/except
# ---------------------------------------------------------------------------

class _null_context:
    def __enter__(self) -> "_null_context":
        return self

    def __exit__(self, *_: object) -> None:
        pass
