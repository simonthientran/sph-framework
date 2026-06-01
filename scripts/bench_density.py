"""
Headless benchmark runner with single-phase-focused diagnostics and telemetry.

Usage:
    python scripts/bench_density.py scenes/examples/pipe_flow_2d.json 500
    python scripts/bench_density.py scenes/examples/00_showcase_basin.json 500
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sph.core.simulation import SimulationRunner


def _is_single_phase_mode(intended_use: str) -> bool:
    use = (intended_use or "").lower()
    return use in {"single_phase_benchmark", "single_phase", "closed_flow", "internal_flow"}


def _fmt_stage(stage_timings_ms: dict[str, float], key: str) -> str:
    return f"{stage_timings_ms.get(key, 0.0):6.2f}"


def run(scene_path: str, n_steps: int, log_every: int = 50, backend_name: str | None = None) -> None:
    runner = SimulationRunner(Path(scene_path), backend_name=backend_name)
    metadata = runner.scene_metadata
    single_phase = _is_single_phase_mode(metadata.intended_use)

    stage_totals: dict[str, float] = {}

    print(f"\n{'='*96}")
    print(f"Scene      : {scene_path}")
    print(f"Name       : {metadata.name}")
    print(f"Use        : {metadata.intended_use}")
    print(f"Backend    : {runner.backend.backend_name}")
    print(f"Solver     : {runner.solver_name}")
    if single_phase:
        print("Validation : SINGLE-PHASE BENCHMARK")
    else:
        print("Validation : GENERAL REGIME BENCHMARK")
    if metadata.notes:
        print(f"Notes      : {metadata.notes}")
    state = runner.state
    print(
        f"Particles  : fluid={state.fluid_positions.shape[0]} boundary={state.boundary_positions.shape[0]} "
        f"rho0={state.fluid_rest_density:.1f}"
    )
    print(f"Steps      : {n_steps}   log_every={log_every}")
    print(f"{'='*96}")

    if single_phase:
        print(
            " step        dt     vmax   int_mean    int_min    int_max   wall_mean   wall_min   wall_max "
            " it_cd  it_df   ms/step      sps   roll_ms   roll_sps"
        )
    else:
        print(
            " step        dt     vmax   rho_min   rho_mean    rho_max  free_mean   free_min  wall_mean "
            " it_cd  it_df   ms/step      sps   roll_ms   roll_sps"
        )
    print("        timings [ms]  neighbor density solve integrate xsph dt")
    print(f"{'-'*96}")

    t0 = time.perf_counter()
    last_result = None
    for _ in range(n_steps):
        result = runner.step()
        last_result = result
        runtime = result.runtime
        diagnostics = result.diagnostics
        summary = diagnostics.density_summary
        assert summary is not None

        for key, value in runtime.stage_timings_ms.items():
            stage_totals[key] = stage_totals.get(key, 0.0) + value

        if diagnostics.step == 1 or diagnostics.step % log_every == 0:
            iter_cd = int(runtime.solver_metrics.get("iter_cd", 0))
            iter_df = int(runtime.solver_metrics.get("iter_df", 0))
            if single_phase:
                print(
                    f"{diagnostics.step:5d} {diagnostics.dt:9.3e} {diagnostics.velocity_max:8.4f} "
                    f"{summary.rho_mean_interior:10.2f} {summary.rho_min_interior:10.2f} {summary.rho_max_interior:10.2f} "
                    f"{summary.rho_mean_wall:10.2f} {summary.rho_min_wall:10.2f} {summary.rho_max_wall:10.2f} "
                    f"{iter_cd:6d} {iter_df:6d} {diagnostics.performance.wall_time_ms:9.2f} "
                    f"{diagnostics.performance.steps_per_second:8.2f} {diagnostics.performance.rolling_wall_time_ms:9.2f} "
                    f"{diagnostics.performance.rolling_steps_per_second:9.2f}"
                )
            else:
                print(
                    f"{diagnostics.step:5d} {diagnostics.dt:9.3e} {diagnostics.velocity_max:8.4f} "
                    f"{diagnostics.rho_min:9.2f} {diagnostics.rho_mean:10.2f} {diagnostics.rho_max:10.2f} "
                    f"{summary.rho_mean_free_surface:10.2f} {summary.rho_min_free_surface:10.2f} {summary.rho_mean_wall:10.2f} "
                    f"{iter_cd:6d} {iter_df:6d} {diagnostics.performance.wall_time_ms:9.2f} "
                    f"{diagnostics.performance.steps_per_second:8.2f} {diagnostics.performance.rolling_wall_time_ms:9.2f} "
                    f"{diagnostics.performance.rolling_steps_per_second:9.2f}"
                )
            print(
                "        timings [ms]  "
                f"{_fmt_stage(runtime.stage_timings_ms, 'neighbor_search')} "
                f"{_fmt_stage(runtime.stage_timings_ms, 'density_assembly')} "
                f"{_fmt_stage(runtime.stage_timings_ms, 'pressure_solve')} "
                f"{_fmt_stage(runtime.stage_timings_ms, 'integration')} "
                f"{_fmt_stage(runtime.stage_timings_ms, 'xsph')} "
                f"{_fmt_stage(runtime.stage_timings_ms, 'dt_control')}"
            )
            if single_phase and (summary.free_surface_count > 0 or summary.splash_count > 0):
                print(
                    "        note          "
                    f"transient free-surface classification: free={summary.free_surface_count} splash={summary.splash_count}"
                )
            elif not single_phase:
                free_ratio = summary.free_surface_count / max(summary.fluid_count, 1)
                print(
                    "        regime        "
                    f"interior err={runtime.solver_metrics.get('interior_rel_err', 0.0) * 100:.2f}% "
                    f"free={free_ratio:.1%} splash={summary.splash_count} wall_err={runtime.solver_metrics.get('wall_rel_err', 0.0) * 100:.2f}%"
                )

    elapsed = time.perf_counter() - t0
    assert last_result is not None
    runtime = last_result.runtime
    diagnostics = last_result.diagnostics
    summary = diagnostics.density_summary
    assert summary is not None
    steps_done = max(diagnostics.step, 1)

    avg_stage = {key: value / steps_done for key, value in stage_totals.items()}
    print(f"\n{'='*96}")
    print(f"FINAL  step={diagnostics.step}  elapsed={elapsed:.2f}s  backend={runtime.backend_name}  solver={runtime.solver}")
    print(
        f"PERF   ms/step={diagnostics.performance.wall_time_ms:.2f}  sps={diagnostics.performance.steps_per_second:.2f}  "
        f"roll_ms={diagnostics.performance.rolling_wall_time_ms:.2f}  roll_sps={diagnostics.performance.rolling_steps_per_second:.2f}"
    )
    print(
        f"STAGES avg_ms neighbor={avg_stage.get('neighbor_search', 0.0):.2f} "
        f"density={avg_stage.get('density_assembly', 0.0):.2f} "
        f"solve={avg_stage.get('pressure_solve', 0.0):.2f} "
        f"integrate={avg_stage.get('integration', 0.0):.2f} "
        f"xsph={avg_stage.get('xsph', 0.0):.2f} dt={avg_stage.get('dt_control', 0.0):.2f}"
    )
    extra_stage_keys = sorted(
        key for key in avg_stage
        if key.startswith("cuda_") or key == "gpu_sync"
    )
    if extra_stage_keys:
        extra = " ".join(f"{key}={avg_stage[key]:.2f}" for key in extra_stage_keys)
        print(f"CUDA   avg_ms {extra}")
    if "host_step_ms" in runtime.solver_metrics:
        print(f"HOST   reference_step_ms={runtime.solver_metrics['host_step_ms']:.2f}")
    if single_phase:
        print(
            f"QUALITY interior(mean/min/max)=({summary.rho_mean_interior:.2f}, {summary.rho_min_interior:.2f}, {summary.rho_max_interior:.2f}) "
            f"wall(mean/min/max)=({summary.rho_mean_wall:.2f}, {summary.rho_min_wall:.2f}, {summary.rho_max_wall:.2f})"
        )
    else:
        print(
            f"QUALITY rho(min/mean/max)=({diagnostics.rho_min:.2f}, {diagnostics.rho_mean:.2f}, {diagnostics.rho_max:.2f}) "
            f"free(mean/min)=({summary.rho_mean_free_surface:.2f}, {summary.rho_min_free_surface:.2f}) wall_mean={summary.rho_mean_wall:.2f}"
        )
    print(
        f"HEALTH {runtime.solver_health_summary}  ·  dt={diagnostics.dt:.3e}  vmax={diagnostics.velocity_max:.4f}  "
        f"iter_cd={int(runtime.solver_metrics.get('iter_cd', 0))}  iter_df={int(runtime.solver_metrics.get('iter_df', 0))}"
    )
    for note in runtime.solver_health_notes:
        print(f"NOTE   {note}")
    print(f"{'='*96}\n")


if __name__ == "__main__":
    scene = sys.argv[1] if len(sys.argv) > 1 else "scenes/examples/pipe_flow_2d.json"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    log_ev = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    backend = sys.argv[4] if len(sys.argv) > 4 else None
    run(scene, steps, log_ev, backend)
