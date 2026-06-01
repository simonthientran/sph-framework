#!/usr/bin/env python3
"""
Aggregate CUDA stage timings from SimulationRunner (real device only).

Usage:
    NUMBA_ENABLE_CUDASIM=0 PYTHONPATH=src python scripts/profile_cuda_stages.py
    NUMBA_ENABLE_CUDASIM=0 PYTHONPATH=src python scripts/profile_cuda_stages.py \\
        --scene scenes/examples/pipe_flow_2d_dense.json --warmup 5 --steps 500

Refuses to run under CUDASIM so results reflect GPU behavior.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sph.validation.baseline_registry import BASELINE_RUN_PRESETS, baseline_scene_path


def _refuse_cudasim() -> None:
    if os.environ.get("NUMBA_ENABLE_CUDASIM", "").strip() == "1":
        print(
            "Refusing to run: NUMBA_ENABLE_CUDASIM=1 (use NUMBA_ENABLE_CUDASIM=0 for real GPU).",
            file=sys.stderr,
        )
        sys.exit(2)


def _sum_cd_df_by_suffix(cuda_keys: dict[str, float], suffix: str) -> float:
    return (
        float(cuda_keys.get(f"cuda_cd_{suffix}", 0.0))
        + float(cuda_keys.get(f"cuda_df_{suffix}", 0.0))
    )


def run_profile(scene: Path, warmup: int, steps: int, sub_phase_timing: bool = False) -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    from sph.core.simulation import SimulationRunner
    from sph.core.backends.numba_cuda_backend import NumbaCUDABackend

    runner = SimulationRunner(scene, backend_name="numba_cuda")

    if sub_phase_timing and isinstance(runner.backend, NumbaCUDABackend):
        runner.backend.enable_pair_build_diagnostics(True)

    wall_warmup: list[float] = []
    for _ in range(warmup):
        t0 = time.perf_counter()
        runner.step()
        wall_warmup.append((time.perf_counter() - t0) * 1000.0)

    sums: dict[str, float] = defaultdict(float)
    sum_pressure_solve = 0.0
    sum_integration = 0.0
    wall_steady: list[float] = []
    rolling_last = 0.0
    rolling_sps_last = 0.0

    for _ in range(steps):
        result = runner.step()
        st = result.runtime.stage_timings_ms
        w = float(result.runtime.wall_time_ms)
        wall_steady.append(w)
        rolling_last = result.diagnostics.performance.rolling_wall_time_ms
        rolling_sps_last = result.diagnostics.performance.rolling_steps_per_second
        sum_pressure_solve += float(st.get("pressure_solve", 0.0))
        sum_integration += float(st.get("integration", 0.0))
        for k, v in st.items():
            if k.startswith("cuda_"):
                sums[k] += float(v)

    n = float(steps)
    avgs = {k: sums[k] / n for k in sorted(sums.keys())}

    # Combined CD+DF per logical stage
    suffixes = [
        "upload",
        "pair_build",
        "neighbor_count",
        "pair_geometry",
        "density",
        "boundary_state",
        "viscosity",
        "velocity_predict",
        "k_factor",
        "solve",
        "metric_sync",
        "position_integrate",
        "download",
    ]
    pair_build_subphase_suffixes = [
        "pair_build_hash_assign",
        "pair_build_count_scan_scatter",
        "pair_build_boundary_grid",
        "pair_build_ff_emit",
        "pair_build_fb_emit",
        "pair_build_count_read",
        "pair_build_materialize",
    ]
    combined: dict[str, float] = {}
    for s in suffixes:
        combined[s] = _sum_cd_df_by_suffix(avgs, s)
    pair_build_subphases = {
        s.removeprefix("pair_build_"): _sum_cd_df_by_suffix(avgs, s)
        for s in pair_build_subphase_suffixes
    }

    sum_cuda = sum(combined.values())
    mean_wall = statistics.mean(wall_steady) if wall_steady else 0.0
    host_orchestration = max(mean_wall - sum_cuda, 0.0)

    fluid_n = int(result.runtime.fluid_count)
    bnd_n = int(result.runtime.boundary_count)

    return {
        "scene": scene,
        "warmup": warmup,
        "steps": steps,
        "fluid_n": fluid_n,
        "boundary_n": bnd_n,
        "wall_warmup_ms": wall_warmup,
        "wall_steady_ms": wall_steady,
        "mean_wall_steady_ms": mean_wall,
        "stdev_wall_steady_ms": statistics.pstdev(wall_steady) if len(wall_steady) > 1 else 0.0,
        "rolling_wall_ms_final": rolling_last,
        "rolling_sps_final": rolling_sps_last,
        "cuda_per_key_avg": avgs,
        "combined_cd_df_avg_ms": combined,
        "pair_build_subphases_ms": pair_build_subphases,
        "sum_cuda_stages_ms": sum_cuda,
        "host_orchestration_ms": host_orchestration,
        "pressure_solve_avg_ms": sum_pressure_solve / n,
        "integration_avg_ms": sum_integration / n,
    }


def _print_report(data: dict) -> None:
    print("=" * 80)
    print(f"Scene: {data['scene']}")
    print(
        f"Particles: fluid={data['fluid_n']}, boundary={data['boundary_n']}  "
        f"(warmup={data['warmup']}, steady steps={data['steps']})"
    )
    print("=" * 80)
    wwu = data["wall_warmup_ms"]
    if wwu:
        print(
            f"Warmup wall time: first={wwu[0]:.2f} ms, "
            f"last={wwu[-1]:.2f} ms, mean={statistics.mean(wwu):.2f} ms"
        )
    ws = data["wall_steady_ms"]
    print(
        f"Steady-state wall: mean={data['mean_wall_steady_ms']:.3f} ms, "
        f"stdev={data['stdev_wall_steady_ms']:.4f} ms, "
        f"steps/s={1000.0 / data['mean_wall_steady_ms']:.1f} (1/mean)"
    )
    print(
        f"Rolling (window 60) at last step: {data['rolling_wall_ms_final']:.3f} ms/step, "
        f"{data['rolling_sps_final']:.1f} steps/s"
    )
    print(
        f"DFSPH coarse timers (avg steady): pressure_solve={data['pressure_solve_avg_ms']:.3f} ms, "
        f"integration={data['integration_avg_ms']:.3f} ms"
    )
    print()
    comb = data["combined_cd_df_avg_ms"]
    total = data["sum_cuda_stages_ms"]
    print(f"Sum of CUDA stage timers (CD+DF combined, avg per step): {total:.3f} ms")
    print(f"Host/orchestration not covered by CUDA timers (avg): {data['host_orchestration_ms']:.3f} ms")
    print()
    print(f"{'Stage (CD+DF sum)':<22} {'avg ms':>10} {'% of Σcuda':>12}")
    print("-" * 46)
    for name in sorted(comb.keys(), key=lambda k: -comb[k]):
        pct = 100.0 * comb[name] / total if total > 0 else 0.0
        print(f"{name:<22} {comb[name]:>10.3f} {pct:>11.1f}%")
    print()
    pair_sub = data["pair_build_subphases_ms"]
    pair_total = comb.get("pair_build", 0.0)
    if pair_total > 0.0:
        print(f"{'Pair-build sub-phase':<28} {'avg ms':>10} {'% of pair_build':>16}")
        print("-" * 56)
        for name in sorted(pair_sub.keys(), key=lambda k: -pair_sub[k]):
            pct = 100.0 * pair_sub[name] / pair_total if pair_total > 0 else 0.0
            print(f"{name:<28} {pair_sub[name]:>10.3f} {pct:>15.1f}%")
        print()
    print("Per-key averages (cuda_cd_* / cuda_df_*) — sample:")
    for k in sorted(data["cuda_per_key_avg"].keys())[:8]:
        print(f"  {k}: {data['cuda_per_key_avg'][k]:.3f} ms")
    if len(data["cuda_per_key_avg"]) > 8:
        print("  ...")
    print("=" * 80)


def main() -> None:
    _refuse_cudasim()
    ap = argparse.ArgumentParser(description="Profile CUDA stage timings on real hardware.")
    ap.add_argument(
        "--scene",
        type=Path,
        default=baseline_scene_path(ROOT, "base"),
        help="Path to scene JSON",
    )
    ap.add_argument(
        "--warmup",
        type=int,
        default=int(BASELINE_RUN_PRESETS["cuda-profile-base"].warmup or 5),
        help="JIT / cache warmup steps (excluded from steady avg)",
    )
    ap.add_argument(
        "--steps",
        type=int,
        default=int(BASELINE_RUN_PRESETS["cuda-profile-base"].steps),
        help="Steady-state steps to average",
    )
    ap.add_argument(
        "--sub-phase-timing",
        action="store_true",
        default=False,
        help=(
            "Enable sub-phase GPU event timing inside pair_build.  "
            "Adds ~4 GPU sync points per build call and inflates pair_build "
            "timings; use only for sub-phase analysis, not wall-time benchmarking."
        ),
    )
    args = ap.parse_args()

    data = run_profile(args.scene.resolve(), args.warmup, args.steps,
                       sub_phase_timing=args.sub_phase_timing)
    _print_report(data)


if __name__ == "__main__":
    main()
