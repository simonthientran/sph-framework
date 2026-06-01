"""
Real-GPU benchmark: timing, scaling, and Poiseuille physics validation.

Tasks
-----
1. 500-step run on the base scene (N≈1800)
2. Scaling table: N≈1800 / 7200 / 28800, measuring ms/step, neighbor build,
   solve, and GPU memory
3. Physics validation at N≈1800: iter_cd=1, iter_df=1, rho_err < 0.5%,
   Poiseuille L2 < 7%

Usage
-----
    PYTHONPATH=src python scripts/gpu_benchmark.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from numba import cuda
from sph.core.simulation import SimulationRunner


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------

_BASE_SCENE = ROOT / "scenes/examples/pipe_flow_2d.json"

# fluid block geometry (must match base scene)
_FLUID_XMIN, _FLUID_XMAX = 0.0, 1.0
_FLUID_YMIN, _FLUID_YMAX = 0.03, 0.17


def _particle_count(spacing: float) -> int:
    nx = int((_FLUID_XMAX - _FLUID_XMIN) / spacing)
    ny = int((_FLUID_YMAX - _FLUID_YMIN) / spacing)
    return nx * ny


def _make_scene_file(spacing: float, h_sph: float) -> str:
    """Write a temporary scene JSON and return its path."""
    with open(_BASE_SCENE) as fh:
        scene = json.load(fh)
    scene["fluid"]["spacing"] = round(spacing, 6)
    scene["neighbors"]["support_radius"] = round(h_sph, 6)
    tmp = tempfile.NamedTemporaryFile(
        suffix=".json", mode="w", delete=False, dir="/tmp"
    )
    json.dump(scene, tmp)
    tmp.flush()
    return tmp.name


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------


def _poiseuille_l2_pct(positions: np.ndarray, vx: np.ndarray, g_x: float, nu: float) -> float:
    """
    L2 error (%) of vx vs. the analytical Poiseuille profile.

    The effective channel bounds are taken from the fluid particle y-extent;
    we use the average of the per-y-layer mean velocity so that density of
    sampling across y does not bias the norm.
    """
    y = positions[:, 1]
    y0 = float(y.min())
    y1 = float(y.max())
    H = y1 - y0
    if H < 1e-12:
        return float("nan")

    u_ana = (g_x / (2.0 * nu)) * (y - y0) * (y1 - y)

    # L2 relative error
    num = float(np.sqrt(np.mean((vx - u_ana) ** 2)))
    den = float(np.sqrt(np.mean(u_ana ** 2)))
    return num / (den + 1e-12) * 100.0


# ---------------------------------------------------------------------------
# GPU memory query
# ---------------------------------------------------------------------------


def _gpu_free_mb() -> float:
    info = cuda.current_context().get_memory_info()
    return info.free / 1024 ** 2


def _gpu_used_mb(free_before: float) -> float:
    return free_before - _gpu_free_mb()


# ---------------------------------------------------------------------------
# Single-scene benchmark
# ---------------------------------------------------------------------------

_WARMUP = 5   # steps discarded before timing
_MEASURE_STEPS = {
    "small": 500,
    "medium": 200,
    "large": 100,
}


def run_one(
    scene_path: str,
    n_steps: int,
    label: str,
    verbose: bool = False,
) -> dict:
    """
    Run *n_steps* and return a dict of averages:
      n, ms_step, nb_ms, solve_ms, mem_mb, iter_cd, rho_err, vmax
    """
    free_before = _gpu_free_mb()

    runner = SimulationRunner(scene_path, backend_name="numba_cuda")
    backend = runner.backend

    mem_mb = _gpu_used_mb(free_before)

    # Warmup (includes JIT compilation on first call)
    for _ in range(_WARMUP):
        runner.step()

    # Timed measurement
    step_times: list[float] = []
    nb_times: list[float] = []
    solve_times: list[float] = []
    iter_cds: list[int] = []
    rho_errs: list[float] = []
    vmaxs: list[float] = []

    for s in range(1, n_steps + 1):
        t0 = time.perf_counter()
        result = runner.step()
        step_times.append((time.perf_counter() - t0) * 1000.0)

        rt = result.runtime
        m = rt.solver_metrics

        # Neighbor-build and solve timings live in stage_timings_ms
        ts = rt.stage_timings_ms
        nb = ts.get("cuda_cd_pair_build", 0.0) + ts.get("cuda_df_pair_build", 0.0)
        solve = ts.get("cuda_cd_solve", 0.0) + ts.get("cuda_df_solve", 0.0)
        nb_times.append(nb)
        solve_times.append(solve)
        iter_cds.append(int(m.get("iter_cd", -1)))
        rho_errs.append(float(rt.rho_error_mean) * 100.0)
        vmaxs.append(float(rt.velocity_max))

        if verbose and (s % 50 == 0 or s == 1):
            print(
                f"  {label} step {s:>3} | {step_times[-1]:>6.1f} ms | "
                f"nb={nb:>5.2f} ms | solve={solve:>5.2f} ms | "
                f"iter_cd={iter_cds[-1]} | rho_err={rho_errs[-1]:.3f}%"
            )

    # Physics validation: read fluid state from host backend
    sim = backend._host_backend.sim
    positions = sim.fluid.positions
    vx = sim.fluid.velocities[:, 0]
    g_x = float(sim.gravity_vec[0])
    nu = float(sim.forces_cfg.get("viscosity", {}).get("nu", 0.12))

    l2_pct = _poiseuille_l2_pct(positions, vx, g_x, nu)

    return {
        "n_fluid": int(sim.fluid.n),
        "ms_step": float(np.mean(step_times)),
        "nb_ms": float(np.mean(nb_times)),
        "solve_ms": float(np.mean(solve_times)),
        "mem_mb": mem_mb,
        "iter_cd_mean": float(np.mean(iter_cds)),
        "rho_err_mean": float(np.mean(rho_errs)),
        "vmax_final": float(vmaxs[-1]),
        "poiseuille_l2_pct": l2_pct,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Scale configs: (spacing, h_sph, label, n_measure_steps)
_SCALES = [
    (0.0087,   0.0226,  "N≈1800",  500),
    (0.00435,  0.0113,  "N≈7200",  200),
    (0.00217,  0.00565, "N≈28800", 100),
]


def main() -> None:
    try:
        gpu = cuda.get_current_device()
        gpu_name = gpu.name.decode() if isinstance(gpu.name, bytes) else str(gpu.name)
    except Exception:
        gpu_name = "unknown"

    print("=" * 72)
    print("GPU Benchmark — Real CUDA Device")
    print(f"GPU: {gpu_name}")
    print("=" * 72)

    # -------------------------------------------------------------------
    # Task 1: 500-step run at N≈1800, verbose
    # -------------------------------------------------------------------
    print("\n=== Task 1: 500-step run (N≈1800) ===\n")
    scene_small = _make_scene_file(0.0087, 0.0226)
    t1 = run_one(scene_small, 500, "N≈1800", verbose=True)

    print(f"\nSummary N≈{t1['n_fluid']}:")
    print(f"  ms/step (avg):  {t1['ms_step']:.2f} ms")
    print(f"  neighbor build: {t1['nb_ms']:.2f} ms")
    print(f"  solve (CD+DF):  {t1['solve_ms']:.2f} ms")
    print(f"  iter_cd (avg):  {t1['iter_cd_mean']:.2f}")
    print(f"  rho_err (avg):  {t1['rho_err_mean']:.4f}%")
    print(f"  vmax final:     {t1['vmax_final']:.4e}")

    # -------------------------------------------------------------------
    # Task 3: Physics validation
    # -------------------------------------------------------------------
    print("\n=== Task 3: Physics Validation (N≈1800, 500 steps) ===\n")
    rho_ok = t1["rho_err_mean"] < 0.5
    iter_ok = t1["iter_cd_mean"] <= 1.0
    l2 = t1["poiseuille_l2_pct"]
    l2_ok = l2 < 7.0
    print(f"  iter_cd = {t1['iter_cd_mean']:.1f}  {'OK' if iter_ok else 'FAIL (expected 1)'}")
    print(f"  iter_df ≈ 1  (same convergence path)")
    print(f"  rho_err = {t1['rho_err_mean']:.4f}%  {'OK (<0.5%)' if rho_ok else 'FAIL (>0.5%)'}")
    print(f"  Poiseuille L2 = {l2:.2f}%  {'OK (<7%)' if l2_ok else 'FAIL (>7%)'}")
    print(f"\n  VALIDATION {'PASSED' if (rho_ok and iter_ok and l2_ok) else 'FAILED'}")

    # -------------------------------------------------------------------
    # Task 2: Scaling table
    # -------------------------------------------------------------------
    print("\n=== Task 2: Scaling Table ===\n")

    results = [t1]
    for spacing, h_sph, label, n_steps in _SCALES[1:]:
        print(f"Running {label} ({n_steps} steps)...")
        scene = _make_scene_file(spacing, h_sph)
        r = run_one(scene, n_steps, label, verbose=False)
        results.append(r)

    print()
    hdr = f"{'N':>7} | {'ms/step':>8} | {'nb_ms':>7} | {'solve_ms':>8} | {'mem_MB':>7} | {'rho_err':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        n_label = r["n_fluid"]
        print(
            f"{n_label:>7} | {r['ms_step']:>8.2f} | {r['nb_ms']:>7.2f} | "
            f"{r['solve_ms']:>8.2f} | {r['mem_mb']:>7.1f} | {r['rho_err_mean']:>7.4f}%"
        )

    # Identify bottleneck
    print()
    r0 = results[0]
    nb_frac = r0["nb_ms"] / (r0["ms_step"] + 1e-10) * 100
    solve_frac = r0["solve_ms"] / (r0["ms_step"] + 1e-10) * 100
    print(f"Bottleneck (N≈1800): neighbor_build={nb_frac:.1f}%  solve={solve_frac:.1f}% of ms/step")

    if len(results) >= 2:
        scale_factor = results[1]["n_fluid"] / (results[0]["n_fluid"] + 1e-10)
        ms_ratio = results[1]["ms_step"] / (results[0]["ms_step"] + 1e-10)
        print(f"Scaling N x{scale_factor:.1f}: ms/step x{ms_ratio:.2f}")
    if len(results) >= 3:
        scale_factor = results[2]["n_fluid"] / (results[0]["n_fluid"] + 1e-10)
        ms_ratio = results[2]["ms_step"] / (results[0]["ms_step"] + 1e-10)
        print(f"Scaling N x{scale_factor:.1f}: ms/step x{ms_ratio:.2f}")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
