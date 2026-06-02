"""
physics_report.py — SPH Physics Validation Report

Runs a scene for n_steps, prints per-500-step diagnostics (Re, regime,
rho_err), then computes the radial velocity profile L2 error vs the
Hagen-Poiseuille analytical solution and prints OVERALL: PASS or FAIL.

Usage:
    PYTHONPATH=src python scripts/physics_report.py [scene_path]

Default scene: scenes/benchmark_cylinder_pipe_3d.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sph.core.simulation import SimulationRunner  # noqa: E402
from sph.simulator_new import compute_reynolds_number  # noqa: E402


# ── Configuration ─────────────────────────────────────────────────────────────

SCENE_DEFAULT = ROOT / "scenes" / "benchmark_cylinder_pipe_3d.json"
N_STEPS = 3000
PRINT_EVERY = 500
N_BINS = 10


def _read_scene(scene_path: Path) -> dict:
    with scene_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _get_nu(scene: dict) -> float:
    """Read kinematic viscosity from scene, checking multiple locations."""
    nu = scene.get("forces", {}).get("viscosity", {}).get("nu", None)
    if nu is not None:
        return float(nu)
    fluids = scene.get("fluids", None)
    if isinstance(fluids, list) and fluids:
        nu = fluids[0].get("nu", None)
        if nu is not None:
            return float(nu)
    fluid = scene.get("fluid", None)
    if isinstance(fluid, dict):
        nu = fluid.get("nu", None)
        if nu is not None:
            return float(nu)
    return 0.001


def _get_geometry(scene: dict) -> tuple[float, float, float]:
    """Return (cy, cz, R) of the pipe cross-section."""
    cy, cz, R = 0.14, 0.14, 0.10
    wall = scene.get("domain", {}).get("cylinder_wall")
    if wall is None:
        fluids = scene.get("fluids")
        if isinstance(fluids, list) and fluids:
            wall = fluids[0]
        else:
            wall = scene.get("fluid")
    if wall is not None:
        center = wall.get("center")
        if center is not None and len(center) >= 2:
            cy, cz = float(center[0]), float(center[1])
        if wall.get("radius") is not None:
            R = float(wall["radius"])
    return cy, cz, R


def main() -> None:
    scene_path = Path(sys.argv[1]) if len(sys.argv) > 1 else SCENE_DEFAULT
    if not scene_path.is_absolute():
        scene_path = ROOT / scene_path

    print("=" * 65)
    print(f"Physics Report: {scene_path.name}")
    print("=" * 65)

    scene = _read_scene(scene_path)
    nu = _get_nu(scene)
    L_ref = float(scene.get("hydraulic_diameter", 0.14))
    physics_target = scene.get("physics_target", {})
    L2_threshold = float(physics_target.get("L2_threshold", 0.10))
    gravity = scene.get("forces", {}).get("gravity", [0.0])
    f = float(gravity[0]) if gravity else 0.003
    cy, cz, R = _get_geometry(scene)

    v_max_analytical = f * R ** 2 / (4.0 * nu) if nu > 0.0 else 0.0

    print(f"  nu           = {nu}")
    print(f"  L_ref (D_h)  = {L_ref}")
    print(f"  gravity_x    = {f}")
    print(f"  R            = {R}, center = ({cy:.3f}, {cz:.3f})")
    print(f"  vmax_analyt  = {v_max_analytical:.6f} m/s")
    print(f"  L2_threshold = {L2_threshold * 100:.1f}%")
    print(f"  Running {N_STEPS} steps…")
    print()

    runner = SimulationRunner(scene_path)
    sim = runner.backend.sim
    sim._inlet_yz = None  # disable inlet/outlet for clean periodic benchmark

    rho0 = float(sim.fluid.rho0)
    print(f"  Particles: fluid={sim.fluid.n}  boundary={getattr(sim.boundary, 'n', 0)}")
    print()

    n_warmup = max(0, N_STEPS - PRINT_EVERY)
    n_measure = N_STEPS - n_warmup

    # ── Warm-up phase ─────────────────────────────────────────────────────────
    for step in range(n_warmup):
        rt = runner.step()
        if (step + 1) % PRINT_EVERY == 0:
            fl = sim.fluid
            Re, regime = compute_reynolds_number(fl, nu, L_ref)
            rho_err = float(rt.runtime.rho_error_mean) * 100.0
            vx_mean = float(fl.velocities[:, 0].mean()) if fl.n else 0.0
            print(
                f"  step {step+1:5d} | Re={Re:8.3f} ({regime:12s}) | "
                f"rho_err={rho_err:.3f}% | vx_mean={vx_mean:.6f}"
            )

    # ── Measurement phase ─────────────────────────────────────────────────────
    r_bins = np.linspace(0.0, R, N_BINS + 1)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    vx_accum = np.zeros(N_BINS)
    cnt_accum = np.zeros(N_BINS, dtype=int)

    for step in range(n_measure):
        rt = runner.step()
        global_step = n_warmup + step + 1
        fl = sim.fluid
        if (global_step) % PRINT_EVERY == 0:
            Re, regime = compute_reynolds_number(fl, nu, L_ref)
            rho_err = float(rt.runtime.rho_error_mean) * 100.0
            vx_mean = float(fl.velocities[:, 0].mean()) if fl.n else 0.0
            print(
                f"  step {global_step:5d} | Re={Re:8.3f} ({regime:12s}) | "
                f"rho_err={rho_err:.3f}% | vx_mean={vx_mean:.6f}"
            )

        pos = fl.positions
        vel = fl.velocities[:, 0]
        r = np.sqrt((pos[:, 1] - cy) ** 2 + (pos[:, 2] - cz) ** 2)
        for b in range(N_BINS):
            mask = (r >= r_bins[b]) & (r < r_bins[b + 1])
            if np.any(mask):
                vx_accum[b] += float(vel[mask].mean())
                cnt_accum[b] += 1

    with np.errstate(invalid="ignore"):
        vx_profile = np.where(cnt_accum > 0, vx_accum / cnt_accum, np.nan)

    # ── Analytics ─────────────────────────────────────────────────────────────
    vx_analytical = (f / (4.0 * nu)) * (R ** 2 - r_centers ** 2) if nu > 0.0 else np.zeros(N_BINS)

    valid = (~np.isnan(vx_profile)) & (r_centers < 0.85 * R)
    if np.any(valid) and np.any(np.abs(vx_analytical[valid]) > 1e-12):
        num = np.sqrt(np.mean((vx_profile[valid] - vx_analytical[valid]) ** 2))
        den = np.sqrt(np.mean(vx_analytical[valid] ** 2))
        l2_err = float(num / den) if den > 1e-12 else float("inf")
    else:
        l2_err = float("inf")
        print("  WARNING: no valid bins or zero analytical velocity — L2 undefined")

    fl_final = sim.fluid
    Re_final, regime_final = compute_reynolds_number(fl_final, nu, L_ref)
    rho_err_final = float(np.abs(fl_final.densities - rho0).mean() / rho0) * 100.0

    print()
    print("── Final state ─────────────────────────────────────────────────")
    print(f"  Re           = {Re_final:.3f}  ({regime_final})")
    print(f"  rho_err      = {rho_err_final:.3f}%")
    print(f"  L2 vel error = {l2_err * 100:.2f}%  (threshold {L2_threshold * 100:.1f}%)")
    print()

    print("── Radial profile ──────────────────────────────────────────────")
    print(f"{'r [m]':>8}  {'vx_sim':>12}  {'vx_analyt':>12}  {'err%':>7}")
    for i in range(N_BINS):
        vs = vx_profile[i]
        va = vx_analytical[i]
        if not np.isnan(vs) and abs(va) > 1e-12:
            err = (vs - va) / abs(va) * 100.0
            print(f"{r_centers[i]:8.4f}  {vs:12.6f}  {va:12.6f}  {err:7.1f}%")
        elif not np.isnan(vs):
            print(f"{r_centers[i]:8.4f}  {vs:12.6f}  {'N/A':>12}")
    print()

    verdict = "PASS" if l2_err <= L2_threshold else "FAIL"
    print(f"OVERALL: {verdict}")
    if verdict == "FAIL":
        print(f"  L2 error {l2_err * 100:.2f}% > threshold {L2_threshold * 100:.1f}%")


if __name__ == "__main__":
    main()
