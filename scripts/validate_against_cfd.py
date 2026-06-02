"""
Validate SPH solver against Hagen-Poiseuille analytical solution.

For body-force-driven pipe flow:
    u(r) = (f / (4*nu)) * (R² - r²)
    u_max = f * R² / (4 * nu)

Returns True if the L2 relative velocity error (interior bins) < 10%.

Usage
-----
    PYTHONPATH=src python scripts/validate_against_cfd.py scenes/benchmark_cylinder_pipe_3d.json
    PYTHONPATH=src python scripts/validate_against_cfd.py scenes/benchmark_cylinder_pipe_3d.json 2000
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


def validate_hagen_poiseuille(
    scene_path: str | Path,
    n_steps: int = 3000,
) -> bool:
    """
    Run the SPH simulation for ``n_steps`` and compare the radial velocity
    profile against the Hagen-Poiseuille analytical solution.

    Parameters
    ----------
    scene_path : path to scene JSON file
    n_steps    : number of simulation steps to run

    Returns
    -------
    True if the interior-bin L2 relative error < 10 %.
    """
    from sph.core.simulation import SimulationRunner  # noqa: E402

    scene_path = Path(scene_path)
    print("=" * 62)
    print(f"CFD Validation — Hagen-Poiseuille")
    print(f"  Scene : {scene_path.name}")
    print(f"  Steps : {n_steps}")
    print("=" * 62)

    runner = SimulationRunner(scene_path)
    sim = runner.backend.sim

    # Disable inlet/outlet for a pure periodic benchmark
    sim._inlet_yz = None

    # ── Extract scene parameters ──────────────────────────────────────────────
    # Body force (gravity x-component)
    forces_cfg = getattr(sim, 'forces_cfg', {})
    gravity_raw = forces_cfg.get("gravity", sim.scene.get("forces", {}).get("gravity", [0.0]))
    F = float(gravity_raw[0]) if len(gravity_raw) > 0 else 0.0

    # Kinematic viscosity
    nu = getattr(sim, '_nu', None)
    if nu is None or nu <= 0.0:
        nu = (
            sim.scene.get("forces", {})
                     .get("viscosity", {})
                     .get("nu", 0.001)
        )
    nu = float(nu)

    # Pipe geometry from domain.cylinder_wall
    domain_cfg = sim.scene.get("domain", {})
    wall_cfg = domain_cfg.get("cylinder_wall", {})
    R = float(wall_cfg.get("radius", 0.07))
    center = wall_cfg.get("center", [0.0, 0.0])
    CY = float(center[0])
    CZ = float(center[1])

    rho0 = float(sim.fluid.rho0)
    vmax_analytical = F * R ** 2 / (4.0 * nu) if nu > 0.0 else 0.0

    print(f"  F    = {F}")
    print(f"  nu   = {nu}")
    print(f"  R    = {R}   center = ({CY}, {CZ})")
    print(f"  Analytical vmax = F*R²/(4*nu) = {vmax_analytical:.6g} m/s")
    print(f"  Particles: fluid={sim.fluid.n}", end="")
    if sim.boundary is not None:
        print(f"  boundary={sim.boundary.n}", end="")
    print()
    print()

    # ── Run simulation ────────────────────────────────────────────────────────
    print(f"Running {n_steps} steps …", flush=True)
    for step in range(n_steps):
        runner.backend.step()
        if (step + 1) % max(1, n_steps // 6) == 0:
            vx_mean = float(sim.fluid.velocities[:, 0].mean())
            rho_err = abs(sim.fluid.densities.mean() - rho0) / rho0 * 100
            print(f"  step {step+1:5d}  vx_mean={vx_mean:.5g}  rho_err={rho_err:.2f}%")

    # ── Radial profile ────────────────────────────────────────────────────────
    N_BINS = 12
    r_bins = np.linspace(0.0, R, N_BINS + 1)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

    pos = sim.fluid.positions
    vx = sim.fluid.velocities[:, 0]
    r = np.sqrt((pos[:, 1] - CY) ** 2 + (pos[:, 2] - CZ) ** 2)

    vx_profile = np.full(N_BINS, np.nan)
    for b in range(N_BINS):
        mask = (r >= r_bins[b]) & (r < r_bins[b + 1])
        if mask.sum() > 0:
            vx_profile[b] = float(vx[mask].mean())

    vx_analytical = (F / (4.0 * nu)) * np.maximum(R ** 2 - r_centers ** 2, 0.0)

    # L2 error — interior bins only (r < 0.85 R)
    valid = (~np.isnan(vx_profile)) & (r_centers < 0.85 * R)
    if np.any(valid):
        num = float(np.sqrt(np.mean((vx_profile[valid] - vx_analytical[valid]) ** 2)))
        den = float(np.sqrt(np.mean(vx_analytical[valid] ** 2)))
        l2_err = num / den if den > 1e-12 else float("inf")
    else:
        l2_err = float("inf")

    # Density error (interior)
    r_all = np.sqrt((pos[:, 1] - CY) ** 2 + (pos[:, 2] - CZ) ** 2)
    interior = r_all < 0.8 * R
    rho_mean = float(sim.fluid.densities[interior].mean()) if interior.any() else float(sim.fluid.densities.mean())
    drho = abs(rho_mean - rho0) / rho0

    # ── Table ─────────────────────────────────────────────────────────────────
    BAR_WIDTH = 20
    vmax_display = max(float(np.nanmax(vx_analytical)), 1e-12)

    print()
    print("── Radial velocity profile ──────────────────────────────────────")
    print(f"  {'r/R':>5}  {'vx_sim':>10}  {'vx_ana':>10}  {'err%':>7}  bar")
    print("  " + "-" * 52)
    for i in range(N_BINS):
        vs = vx_profile[i]
        va = vx_analytical[i]
        r_ratio = r_centers[i] / R
        if not np.isnan(vs):
            err_pct = (vs - va) / max(abs(va), 1e-12) * 100
            bar_len = int(BAR_WIDTH * min(vs / vmax_display, 1.0))
            bar_len = max(0, bar_len)
            bar = "█" * bar_len + "░" * (BAR_WIDTH - bar_len)
            print(f"  {r_ratio:5.3f}  {vs:10.5g}  {va:10.5g}  {err_pct:+7.1f}%  {bar}")
        else:
            print(f"  {r_ratio:5.3f}  {'(empty)':>10}  {va:10.5g}  {'—':>7}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("── Summary ──────────────────────────────────────────────────────")
    print(f"  L2 velocity error (interior) : {l2_err * 100:.2f}%  (threshold 10 %)")
    print(f"  Mean density error           : {drho * 100:.2f}%  (threshold  2 %)")
    n_outside = int(np.sum(r_all > R + 0.005))
    print(f"  Particles outside R+tol      : {n_outside}")
    print()

    passed = (l2_err < 0.10)
    verdict = "PASS" if passed else "FAIL"
    print(f"Validation: {verdict}  (L2={l2_err*100:.2f}%)")
    return passed


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_against_cfd.py <scene.json> [n_steps]")
        sys.exit(1)

    scene = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 3000

    ok = validate_hagen_poiseuille(scene, n_steps=steps)
    sys.exit(0 if ok else 1)
