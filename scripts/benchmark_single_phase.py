"""
Automatic benchmark for single-phase 3D pipe flow (Poiseuille).

Compares against the analytical Hagen-Poiseuille solution for a
square-channel driven by body force.  Reports L2 velocity profile
error, vmax error, and density error.  Pass/fail against
SPlisHSPlasH-level thresholds.

Usage (from repo root):
    PYTHONPATH=src python scripts/benchmark_single_phase.py

Thresholds (SPlisHSPlasH standard):
    L2 error     < 7%
    vmax error   < 15%
    density err  < 2%
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure src/ is on the path when run from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def run_poiseuille_benchmark(n_steps: int = 3000, verbose: bool = True) -> bool:
    from sph.core.simulation import SimulationRunner

    scene_path = Path(__file__).parent.parent / "scenes" / "benchmark_pipe_flow_3d.json"
    if not scene_path.exists():
        print(f"ERROR: scene not found: {scene_path}")
        return False

    runner = SimulationRunner(scene_path, backend_name="numba_cpu")
    fl = runner.backend.sim.fluid

    # ── Scene parameters ──────────────────────────────────────────────────────
    # Fluid block: y in [0.04, 0.16], z in [0.04, 0.16] → half-width L = 0.06
    f_body = 0.003      # body force in x (from scene JSON gravity[0])
    nu = 0.12           # kinematic viscosity (from scene JSON forces.viscosity.nu)
    # Effective half-width: measured from actual particle positions
    y_min_fl = float(fl.positions[:, 1].min())
    y_max_fl = float(fl.positions[:, 1].max())
    y_center = 0.5 * (y_min_fl + y_max_fl)
    L = 0.5 * (y_max_fl - y_min_fl)

    vmax_analytical = f_body * L ** 2 / (2.0 * nu)
    if verbose:
        print(f"=== Poiseuille Benchmark (3D pipe, {n_steps} steps) ===")
        print(f"  f_body={f_body}, nu={nu}, L={L:.4f}")
        print(f"  Analytical vmax: {vmax_analytical:.6f} m/s")
        print(f"  n_fluid: {fl.n}")

    # ── Run simulation ────────────────────────────────────────────────────────
    for step in range(n_steps):
        runner.step()
        if verbose and (step + 1) % 500 == 0:
            vx_max = float(fl.velocities[:, 0].max())
            rho_err = float(np.abs(fl.densities - fl.rho0).mean() / fl.rho0)
            print(f"  step {step+1:4d}: vx_max={vx_max:.4e}  rho_err={rho_err*100:.2f}%")

    # ── Velocity profile analysis ─────────────────────────────────────────────
    y = fl.positions[:, 1]
    vx = fl.velocities[:, 0]

    n_bins = 20
    bins = np.linspace(y_min_fl - 1e-6, y_max_fl + 1e-6, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    vx_mean = np.zeros(n_bins)
    vx_count = np.zeros(n_bins, dtype=int)

    for k in range(n_bins):
        mask = (y >= bins[k]) & (y < bins[k + 1])
        cnt = int(mask.sum())
        if cnt > 0:
            vx_mean[k] = float(vx[mask].mean())
            vx_count[k] = cnt

    # Analytical Poiseuille profile: parabola centred at y_center
    vx_ana = np.maximum(
        f_body / (2.0 * nu) * (L ** 2 - (centers - y_center) ** 2),
        0.0,
    )

    # L2 error — interior bins only (avoid near-wall truncation artefacts)
    interior = (np.abs(centers - y_center) < 0.85 * L) & (vx_count > 0)
    if interior.sum() > 0 and vx_ana[interior].max() > 1e-15:
        residuals = vx_mean[interior] - vx_ana[interior]
        L2 = float(np.sqrt(np.mean(residuals ** 2)))
        L2_ref = float(np.sqrt(np.mean(vx_ana[interior] ** 2)))
        L2_rel = L2 / L2_ref if L2_ref > 1e-15 else 1.0
    else:
        L2_rel = 1.0

    vmax_sim = float(vx_mean.max())
    vmax_err = abs(vmax_sim - vmax_analytical) / max(vmax_analytical, 1e-15)

    rho_err_mean = float(np.abs(fl.densities - fl.rho0).mean() / fl.rho0)

    # ── Report ────────────────────────────────────────────────────────────────
    pass_L2 = L2_rel < 0.07
    pass_vmax = vmax_err < 0.15
    pass_rho = rho_err_mean < 0.02

    print(f"\n=== POISEUILLE BENCHMARK RESULTS ===")
    print(f"  L2 error:    {L2_rel*100:6.2f}%  {'PASS' if pass_L2   else 'FAIL'}  (threshold  7%)")
    print(f"  vmax error:  {vmax_err*100:6.2f}%  {'PASS' if pass_vmax else 'FAIL'}  (threshold 15%)")
    print(f"  density err: {rho_err_mean*100:6.2f}%  {'PASS' if pass_rho  else 'FAIL'}  (threshold  2%)")
    print(f"  vmax sim:    {vmax_sim:.6e} m/s")
    print(f"  vmax ana:    {vmax_analytical:.6e} m/s")

    all_pass = pass_L2 and pass_vmax and pass_rho
    print(f"\n  OVERALL: {'PASS ✓' if all_pass else 'FAIL ✗'}")
    return all_pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Poiseuille benchmark for single-phase SPH")
    parser.add_argument("--steps", type=int, default=3000, help="Simulation steps (default 3000)")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-step output")
    args = parser.parse_args()

    ok = run_poiseuille_benchmark(n_steps=args.steps, verbose=not args.quiet)
    sys.exit(0 if ok else 1)
