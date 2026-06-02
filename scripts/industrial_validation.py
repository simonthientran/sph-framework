"""
Industrial Hagen-Poiseuille Validation Suite.

Validates steady-state Poiseuille flow against the analytical solution for
body-force-driven flow in a cylindrical pipe:

    u(r) = f / (4*nu) * (R² - r²)    →    u_max = f*R² / (4*nu)

Benchmarks
----------
  B1  vmax relative error    < 15 %
  B2  Reynolds number        (informational)
  B3  L2 velocity profile    < 7 %
  B4  Mean density error     < 2 %
  B5  No NaN in positions/velocities
  B6  No-penetration (zero particles outside R)

Usage
-----
  PYTHONPATH=src python scripts/industrial_validation.py <scene.json> [n_steps]

Arguments
---------
  scene.json   Path to the scene JSON file
  n_steps      Number of simulation steps (default: 8000)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sph.core.simulation import SimulationRunner  # noqa: E402


# ── Thresholds ────────────────────────────────────────────────────────────────
VMAX_ERR_THRESHOLD = 0.15   # B1: 15 %
L2_THRESHOLD       = 0.07   # B3:  7 %
DRHO_THRESHOLD     = 0.02   # B4:  2 %
NEAR_WALL_FRAC     = 0.85   # interior fraction for L2 error (mask near-wall particles)
PENETRATION_TOL    = 0.005  # B6: particles outside R + tol are a violation


def _get_nu(sim) -> float:
    """Extract kinematic viscosity from simulator, trying multiple locations."""
    nu = getattr(sim, "_nu", None)
    if nu is not None and nu > 0:
        return float(nu)
    fluids = sim.scene.get("fluids", [{}])
    nu = fluids[0].get("nu", None) if fluids else None
    if nu is not None and nu > 0:
        return float(nu)
    nu = sim.scene.get("forces", {}).get("viscosity", {}).get("nu", None)
    if nu is not None and nu > 0:
        return float(nu)
    return 0.001


def _detect_pipe_geometry(sim) -> tuple[int, float, float, float]:
    """
    Detect pipe axis, radius and center from scene configuration.

    Returns
    -------
    axis   : int      — pipe axis index (0=x, 1=y, 2=z)
    radius : float    — pipe radius R
    center_a : float  — center coordinate in the first transverse direction
    center_b : float  — center coordinate in the second transverse direction
    """
    fluid_cfg = sim.scene.get("fluid", sim.scene.get("fluids", [{}])[0])
    fluid_type = fluid_cfg.get("type", "box")

    if fluid_type == "cylinder_x":
        axis = 0
        R = float(fluid_cfg.get("radius", 0.10))
        c = fluid_cfg.get("center", [0.0, 0.0])
        return axis, R, float(c[0]), float(c[1])

    # Try to infer from cylinder_wall domain config
    domain_cfg = sim.scene.get("domain", {})
    if "cylinder_wall" in domain_cfg:
        wall = domain_cfg["cylinder_wall"]
        R = float(wall.get("radius", 0.10))
        c = wall.get("center", [0.0, 0.0])
        axis = 0  # default x-axis for periodic_x flows
        return axis, R, float(c[0]), float(c[1])

    # Fallback: detect axis as the direction with the largest span
    pos = sim.fluid.positions
    span = pos.max(axis=0) - pos.min(axis=0)
    axis = int(np.argmax(span))

    # Estimate radius from transverse directions
    transverse = [i for i in range(sim.dim) if i != axis]
    center_coords = [(pos[:, t].max() + pos[:, t].min()) * 0.5 for t in transverse]
    radii = [(pos[:, t].max() - pos[:, t].min()) * 0.5 for t in transverse]
    R = float(np.mean(radii))
    ca = float(center_coords[0]) if len(center_coords) > 0 else 0.0
    cb = float(center_coords[1]) if len(center_coords) > 1 else 0.0
    return axis, R, ca, cb


def main(scene_path: Path, n_steps: int) -> dict:
    print("=" * 65)
    print(f"Industrial Hagen-Poiseuille Validation")
    print(f"  Scene : {scene_path.name}")
    print(f"  Steps : {n_steps}")
    print("=" * 65)

    runner = SimulationRunner(scene_path)
    sim = runner.backend.sim
    fl = sim.fluid
    bd = sim.boundary

    # Disable inlet/outlet if present
    sim._inlet_yz = None

    nu = _get_nu(sim)
    axis, R, ca, cb = _detect_pipe_geometry(sim)
    f_body = float(sim.gravity_vec[axis])

    vmax_analytical = f_body * R**2 / (4.0 * nu) if nu > 0 else 0.0
    vmean_analytical = vmax_analytical / 2.0
    hydraulic_diameter = float(sim.scene.get("hydraulic_diameter", 2.0 * R))
    Re = vmean_analytical * hydraulic_diameter / nu if nu > 0 else 0.0

    n_bd = len(bd.positions) if bd is not None else 0
    print(f"  Particles : fluid={fl.n}  boundary={n_bd}")
    print(f"  nu={nu:.4f}  f={f_body:.4f}  R={R:.4f}")
    print(f"  Analytical vmax = f*R²/(4*nu) = {vmax_analytical:.5e} m/s")
    print(f"  Reynolds number (steady state) Re ≈ {Re:.4f}")
    print()

    print(f"Running {n_steps} steps …", flush=True)
    log_interval = max(1, n_steps // 10)
    for step in range(n_steps):
        runner.backend.step()
        if (step + 1) % log_interval == 0:
            ax_vel = fl.velocities[:, axis]
            vx_mean = float(ax_vel.mean())
            rho_err = float(abs(fl.densities - fl.rho0).mean() / fl.rho0 * 100)
            print(f"  step {step+1:5d}  v_mean={vx_mean:.4e}  rho_err={rho_err:.2f}%", flush=True)

    print()

    # ── B5: NaN check ────────────────────────────────────────────────────────
    nan_pos = bool(np.isnan(fl.positions).any())
    nan_vel = bool(np.isnan(fl.velocities).any())
    b5_pass = not nan_pos and not nan_vel
    print(f"B5 NaN check : pos={nan_pos}  vel={nan_vel}  →  {'PASS' if b5_pass else 'FAIL'}")

    # ── Radial profile ────────────────────────────────────────────────────────
    t1, t2 = [i for i in range(3) if i != axis]
    r = np.sqrt((fl.positions[:, t1] - ca)**2 + (fl.positions[:, t2] - cb)**2)
    ax_vel = fl.velocities[:, axis]

    N_BINS = 10
    r_bins = np.linspace(0.0, R, N_BINS + 1)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    vx_profile = np.full(N_BINS, np.nan)
    for b in range(N_BINS):
        mask = (r >= r_bins[b]) & (r < r_bins[b + 1])
        if np.any(mask):
            vx_profile[b] = float(ax_vel[mask].mean())

    vx_analytical = (f_body / (4.0 * nu)) * (R**2 - r_centers**2) if nu > 0 else np.zeros(N_BINS)

    # ── B1: vmax error ────────────────────────────────────────────────────────
    vmax_sim = float(np.nanmax(vx_profile)) if not np.all(np.isnan(vx_profile)) else 0.0
    if abs(vmax_analytical) > 1e-14:
        b1_err = abs(vmax_sim - vmax_analytical) / abs(vmax_analytical)
    else:
        b1_err = 0.0
    b1_pass = b1_err <= VMAX_ERR_THRESHOLD
    print(f"B1 vmax     : sim={vmax_sim:.5e}  ana={vmax_analytical:.5e}  err={b1_err*100:.1f}%  →  {'PASS' if b1_pass else 'FAIL'}")

    # ── B2: Reynolds number ───────────────────────────────────────────────────
    v_mean_sim = float(ax_vel.mean())
    Re_sim = v_mean_sim * hydraulic_diameter / nu if nu > 0 else 0.0
    print(f"B2 Reynolds : Re_target={Re:.4f}  Re_sim={Re_sim:.4f}")

    # ── B3: L2 velocity profile error ────────────────────────────────────────
    valid = (~np.isnan(vx_profile)) & (r_centers < NEAR_WALL_FRAC * R)
    if np.any(valid) and np.any(vx_analytical[valid] != 0):
        num = np.sqrt(np.mean((vx_profile[valid] - vx_analytical[valid])**2))
        den = np.sqrt(np.mean(vx_analytical[valid]**2))
        l2_err = float(num / den) if den > 0 else 1.0
    else:
        l2_err = 1.0
    b3_pass = l2_err <= L2_THRESHOLD
    print(f"B3 L2 err   : {l2_err*100:.2f}%  (threshold {L2_THRESHOLD*100:.0f}%)  →  {'PASS' if b3_pass else 'FAIL'}")

    # ── B4: density error ─────────────────────────────────────────────────────
    interior = r < 0.8 * R
    if interior.any():
        rho_mean = float(fl.densities[interior].mean())
    else:
        rho_mean = float(fl.densities.mean())
    drho = abs(rho_mean - fl.rho0) / fl.rho0
    b4_pass = drho <= DRHO_THRESHOLD
    print(f"B4 density  : rho_mean={rho_mean:.2f}  rho0={fl.rho0:.0f}  err={drho*100:.2f}%  →  {'PASS' if b4_pass else 'FAIL'}")

    # ── B6: no-penetration ────────────────────────────────────────────────────
    n_outside = int(np.sum(r > R + PENETRATION_TOL))
    b6_pass = n_outside == 0
    print(f"B6 penetrat.: {n_outside} particles outside R+{PENETRATION_TOL}  →  {'PASS' if b6_pass else 'FAIL'}")

    # ── Radial profile table ──────────────────────────────────────────────────
    print()
    print("── Radial velocity profile ────────────────────────────────────")
    print(f"{'r [m]':>8}  {'vx_sim':>12}  {'vx_ana':>12}  {'err%':>7}")
    for i in range(N_BINS):
        vs = vx_profile[i]
        va = vx_analytical[i]
        if not np.isnan(vs):
            err = (vs - va) / max(abs(va), 1e-14) * 100 if abs(va) > 1e-14 else 0.0
            print(f"{r_centers[i]:8.4f}  {vs:12.6e}  {va:12.6e}  {err:7.1f}%")

    print()

    # ── Overall verdict ───────────────────────────────────────────────────────
    passes = {"B1": b1_pass, "B3": b3_pass, "B4": b4_pass, "B5": b5_pass, "B6": b6_pass}
    all_pass = all(passes.values())
    n_pass = sum(passes.values())
    verdict = "PASS" if all_pass else f"PARTIAL ({n_pass}/5)"

    print("── Summary ────────────────────────────────────────────────────")
    print(f"  B1 vmax err  : {b1_err*100:.1f}%  {'PASS' if b1_pass else 'FAIL'}")
    print(f"  B2 Re        : {Re_sim:.4f}  (info)")
    print(f"  B3 L2 err    : {l2_err*100:.2f}%  {'PASS' if b3_pass else 'FAIL'}")
    print(f"  B4 rho err   : {drho*100:.2f}%  {'PASS' if b4_pass else 'FAIL'}")
    print(f"  B5 no NaN    : {'PASS' if b5_pass else 'FAIL'}")
    print(f"  B6 no-pentr  : {'PASS' if b6_pass else 'FAIL'}")
    print(f"\nOverall : {verdict}")

    return {
        "b1_vmax_err": b1_err,
        "b2_re": Re_sim,
        "b3_l2_err": l2_err,
        "b4_drho": drho,
        "b5_no_nan": b5_pass,
        "b6_no_penetration": b6_pass,
        "verdict": verdict,
        "all_pass": all_pass,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Industrial Hagen-Poiseuille validation")
    parser.add_argument("scene", type=Path, help="Scene JSON file")
    parser.add_argument("n_steps", type=int, nargs="?", default=8000, help="Number of steps (default 8000)")
    args = parser.parse_args()

    result = main(args.scene, args.n_steps)
    sys.exit(0 if result["all_pass"] else 1)
