"""
500-step CUDA-authoritative regression run.

CUDA owns the PPE solve; CPU handles neighbors, density, forces, and integration.
CPU solve runs only if debug_mode=True.

Usage (CUDASIM):
    NUMBA_ENABLE_CUDASIM=1 python scripts/run_cuda_authoritative.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import os
os.environ.setdefault("NUMBA_ENABLE_CUDASIM", "1")

from sph.core.simulation import SimulationRunner

SCENE = ROOT / "scenes/examples/pipe_flow_2d.json"
TARGET_STEPS = 500
LOG_EVERY = 50

# CUDASIM executes CUDA threads sequentially on the CPU; each step takes ~200 s.
# Cap at 10 steps for CUDASIM runs so the script finishes in ~35 min instead of
# 28 hours.  On a real GPU, remove the cap and run the full 500.
CUDASIM = os.environ.get("NUMBA_ENABLE_CUDASIM", "").strip() == "1"
N_STEPS = 10 if CUDASIM else TARGET_STEPS


def main() -> None:
    runner = SimulationRunner(SCENE, backend_name="numba_cuda")
    backend = runner.backend  # NumbaCUDABackend

    # Authoritative mode is the default (debug_mode=False)
    assert not backend.debug_mode, "Expected debug_mode=False for authoritative run"

    print("=" * 80)
    print(f"CUDA Authoritative regression  ({'CUDASIM – capped at ' + str(N_STEPS) + ' steps' if CUDASIM else str(TARGET_STEPS) + ' steps'})")
    print(f"Scene : {SCENE.name}")
    print(f"Mode  : {'DEBUG (CPU+CUDA compare)' if backend.debug_mode else 'AUTHORITATIVE (CUDA sole solver)'}")
    print(f"Steps : {N_STEPS}  log_every={min(LOG_EVERY, N_STEPS)}")
    if CUDASIM:
        print(f"NOTE  : CUDASIM ~200 s/step; full 500-step run requires a real GPU.")
    print("=" * 80)
    log_every = min(LOG_EVERY, N_STEPS)
    header = (
        f"{'Step':>6} | {'dt':>10} | {'iter_cd':>7} | {'iter_df':>7} | "
        f"{'rho_mean':>10} | {'rho_err':>9} | {'vmax':>10}"
    )
    print(header)
    print("-" * len(header))

    wall_start = time.perf_counter()

    for i in range(1, N_STEPS + 1):
        result = runner.step()
        stats = result.runtime
        metrics = stats.solver_metrics

        if i % log_every == 0 or i == 1:
            iter_cd = int(metrics.get("iter_cd", -1))
            iter_df = int(metrics.get("iter_df", -1))
            rho_err_pct = stats.rho_error_mean * 100.0
            print(
                f"Step {i:>4} | dt={stats.dt:.3e} | iter_cd={iter_cd} | iter_df={iter_df} | "
                f"rho_mean={stats.rho_mean:>8.2f} | rho_err={rho_err_pct:>7.4f}% | vmax={stats.velocity_max:.4e}"
            )

    total_wall_s = time.perf_counter() - wall_start
    ms_per_step = total_wall_s / N_STEPS * 1000.0
    print("=" * 80)
    print(f"Done.  {N_STEPS} steps in {total_wall_s:.1f}s  ({ms_per_step:.0f} ms/step avg)")
    if CUDASIM:
        print(f"Extrapolated 500-step wall time: {500 * ms_per_step / 1000:.0f} s  (real GPU: <1 s expected)")
    print(f"CPU solve: {'ENABLED (debug)' if backend.debug_mode else 'DISABLED (CUDA authoritative)'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
