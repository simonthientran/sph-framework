"""
Hagen-Poiseuille benchmark: 3D cylindrical pipe (round cross-section).

Measures the steady-state axial velocity profile of body-force-driven flow
inside a circular-cross-section pipe.  Compares to the analytical solution:

    u(r) = (f / (4*nu)) * (R² - r²)   →  u_max = f*R² / (4*nu)

Geometry
--------
  R = 0.10 m   (10 particle layers at spacing = 0.01 m)
  L = 0.20 m   (pipe length, periodic in x)
  Cylinder center: (y=0.14, z=0.14) — all coords positive for cKDTree

Pass / Fail criteria
--------------------
  L2 velocity error  < 7 %
  Mean density error < 2 %

Usage
-----
  python scripts/benchmark_cylinder_pipe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sph.core.simulation import SimulationRunner  # noqa: E402


# ── Scene parameters ─────────────────────────────────────────────────────────
SCENE = ROOT / "scenes" / "pipe_flow_cylinder_3d.json"

F    = 0.002   # body force (gravity x-component) [m/s²]
NU   = 0.001   # kinematic viscosity [m²/s]
R    = 0.10    # pipe radius [m]
CY   = 0.14    # cylinder center y [m]
CZ   = 0.14    # cylinder center z [m]
N_WARMUP = 6000   # warm-up steps before measuring profile (~3.5 viscous diffusion times)
N_MEASURE = 500   # additional steps over which to average

VMAX_ANALYTICAL = F * R**2 / (4.0 * NU)   # 0.005 m/s

L2_THRESHOLD   = 0.07   # 7 % relative L2 error
DRHO_THRESHOLD = 0.02   # 2 % mean density error


def main() -> None:
    print("=" * 60)
    print("Hagen-Poiseuille cylinder benchmark")
    print(f"  Analytical vmax = f*R²/(4*nu) = {VMAX_ANALYTICAL:.5f} m/s")
    print(f"  (f={F}, R={R}, nu={NU})")
    print("=" * 60)

    runner = SimulationRunner(SCENE, backend_name="numba_cpu")
    sim = runner.backend.sim

    # Disable emitter and outlet for pure periodic benchmark
    sim._emitters = ()
    sim._outlets = ()

    rho0 = float(sim.fluid.rho0)

    print(f"Particles: fluid={sim.fluid.n}  boundary={sim.boundary.n}")
    print(f"Warm-up: {N_WARMUP} steps …", flush=True)

    for step in range(N_WARMUP):
        runner.backend.step()
        if (step + 1) % 500 == 0:
            vx_mean = sim.fluid.velocities[:, 0].mean()
            rho_err = abs(sim.fluid.densities.mean() - rho0) / rho0 * 100
            print(f"  step {step+1:4d}  vx_mean={vx_mean:.5f}  rho_err={rho_err:.2f}%")

    print(f"Measuring over {N_MEASURE} steps …", flush=True)

    # Accumulate radial profile: bin particles by r
    N_BINS = 10
    r_bins = np.linspace(0.0, R, N_BINS + 1)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    vx_accum = np.zeros(N_BINS)
    cnt_accum = np.zeros(N_BINS, dtype=int)

    for _ in range(N_MEASURE):
        runner.backend.step()
        pos = sim.fluid.positions
        vel = sim.fluid.velocities[:, 0]
        r = np.sqrt((pos[:, 1] - CY) ** 2 + (pos[:, 2] - CZ) ** 2)
        for b in range(N_BINS):
            mask = (r >= r_bins[b]) & (r < r_bins[b + 1])
            if np.any(mask):
                vx_accum[b] += vel[mask].mean()
                cnt_accum[b] += 1

    with np.errstate(invalid="ignore"):
        vx_profile = np.where(cnt_accum > 0, vx_accum / cnt_accum, np.nan)

    # Analytical profile at bin centers
    vx_analytical = (F / (4.0 * NU)) * (R**2 - r_centers**2)

    # L2 relative error (only bins with data)
    valid = ~np.isnan(vx_profile)
    if np.any(valid):
        num = np.sqrt(np.mean((vx_profile[valid] - vx_analytical[valid]) ** 2))
        den = np.sqrt(np.mean(vx_analytical[valid] ** 2))
        l2_err = num / den
    else:
        l2_err = float("inf")

    # Density error — interior particles only (r < 0.8*R); wall-adjacent particles
    # are under-supported by the Adami boundary and have systematically lower density.
    r_all = np.sqrt(
        (sim.fluid.positions[:, 1] - CY) ** 2
        + (sim.fluid.positions[:, 2] - CZ) ** 2
    )
    interior_mask = r_all < 0.8 * R
    if interior_mask.any():
        rho_mean = sim.fluid.densities[interior_mask].mean()
    else:
        rho_mean = sim.fluid.densities.mean()
    drho = abs(rho_mean - rho0) / rho0

    # No-penetration check
    n_outside = int(np.sum(r_all > R + 0.005))

    print()
    print("── Results ─────────────────────────────────────────────")
    print(f"  Simulated vmax  : {vx_profile[0]:.5f} m/s  (bin 0, r≈0)")
    print(f"  Analytical vmax : {VMAX_ANALYTICAL:.5f} m/s")
    print(f"  L2 velocity err : {l2_err * 100:.2f}%  (threshold {L2_THRESHOLD*100:.0f}%)")
    print(f"  Mean density    : {rho_mean:.1f} kg/m³  (rho0={rho0:.0f})")
    print(f"  Density error   : {drho * 100:.2f}%  (threshold {DRHO_THRESHOLD*100:.0f}%)")
    print(f"  Particles outside R+tol: {n_outside}")
    print()

    print("── Radial profile ───────────────────────────────────────")
    print(f"{'r [m]':>8}  {'vx_sim [m/s]':>14}  {'vx_analytic':>12}  {'err%':>7}")
    for i in range(N_BINS):
        vs = vx_profile[i]
        va = vx_analytical[i]
        if not np.isnan(vs):
            err = (vs - va) / max(abs(va), 1e-12) * 100
            print(f"{r_centers[i]:8.4f}  {vs:14.6f}  {va:12.6f}  {err:7.1f}%")

    print()

    # Verdict
    ok_l2  = l2_err  <= L2_THRESHOLD
    ok_rho = drho    <= DRHO_THRESHOLD
    ok_pen = n_outside == 0

    verdict = "PASS" if (ok_l2 and ok_rho and ok_pen) else "FAIL"
    print(f"Overall: {verdict}")
    if not ok_l2:
        print(f"  FAIL  L2 error {l2_err*100:.2f}% > {L2_THRESHOLD*100:.0f}%")
    if not ok_rho:
        print(f"  FAIL  density error {drho*100:.2f}% > {DRHO_THRESHOLD*100:.0f}%")
    if not ok_pen:
        print(f"  FAIL  {n_outside} particle(s) outside cylinder")

    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
