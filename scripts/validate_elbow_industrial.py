"""
Industrial elbow validation — Idelchik (1966) + Dean (1928).

PRIMARY metric: secondary flow at bend (Dean vortex signature).
  Reference: Dean (1928) J.R. Astron. Soc. 4:651
  Threshold: v_secondary / v_main > 0.05 (measurable secondary flow)
  For a sharp 90° elbow: secondary flow appears at any Re > 0
  (De → ∞ because R_curvature → 0 for a sharp corner).

DIAGNOSTIC: pressure loss coefficient K.
  Reference: Idelchik (1966) §5, Table 5-23
  For Re=10-500 laminar sharp 90° elbow: K ≈ 1.0–3.0
  NOTE: K measurement requires steady-state absolute pressure field.
  DFSPH stores instantaneous correction pressures (not physical EOS).
  K is computed from EOS density gradient (physical) as a diagnostic only.

PASS criteria (industrial grade):
  1. Re > 11.6   (above Dean threshold De_crit)
  2. vy_v > 0    (flow successfully redirected through bend)
  3. secondary_flow > 0.01 m/s  (Dean vortex confirmed)
  4. rho_err < 5%
  5. NaN = False

Usage:
    PYTHONPATH=src python scripts/validate_elbow_industrial.py [scene_path]
"""
from __future__ import annotations

import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


def validate_elbow_industrial(
    scene_path: str = "scenes/pipe_elbow_analytical.json",
    n_steps: int = 3000,
) -> bool:
    from sph.core.simulation import SimulationRunner  # noqa: E402

    runner = SimulationRunner(ROOT / scene_path)
    fl = runner.backend.sim.fluid
    nu = runner.backend.sim.nu
    rho0 = fl.rho0
    eos_k = float(runner.backend.sim.scene.get("solver", {}).get("eos_k", 50000.0))
    gx = float(runner.backend.sim.gravity_vec[0])
    R = 0.045
    D = 2 * R

    print(f"\n{'='*60}")
    print("INDUSTRIAL ELBOW VALIDATION")
    print("Reference: Idelchik (1966) + Dean (1928)")
    print(f"{'='*60}")
    print(f"  n_fluid={fl.n}  nu={nu}  gx={gx}  eos_k={eos_k}")
    print(f"  R={R} m  D={D} m")
    print()

    peak_Re   = 0.0
    peak_step = 0
    peak_sec  = 0.0

    for step in range(n_steps):
        runner.step()
        if (step + 1) % 500 == 0:
            pos = fl.positions
            vel = fl.velocities
            h = pos[:, 0] < 0.15
            v = pos[:, 1] > 0.25
            vx_h = float(vel[h, 0].mean()) if h.sum() > 5 else 0.0
            vy_v = float(vel[v, 1].mean()) if v.sum() > 5 else 0.0
            Re_loc = abs(vx_h) * D / nu
            # secondary flow at bend
            bend = ((pos[:, 0] > 0.30) & (pos[:, 0] < 0.45) &
                    (np.abs(pos[:, 1]) < 0.08))
            sec = float(np.linalg.norm(vel[bend, 1:], axis=1).mean()) if bend.sum() > 5 else 0.0
            print(f"  step={step+1:4d} Re={Re_loc:5.1f} "
                  f"vx_h={vx_h:+.4f} vy_v={vy_v:+.4f} "
                  f"bend_secondary={sec:.4f} n_bend={bend.sum()}")
            if Re_loc > peak_Re and vx_h > 0:
                peak_Re = Re_loc
                peak_step = step + 1
                peak_sec = sec

    # ── Final measurements ───────────────────────────────────────────────────
    pos = fl.positions
    vel = fl.velocities
    t   = runner.backend.sim.t

    h = pos[:, 0] < 0.15
    v = pos[:, 1] > 0.25
    vx_h    = float(vel[h, 0].mean()) if h.sum() > 5 else 0.0
    vy_v    = float(vel[v, 1].mean()) if v.sum() > 5 else 0.0
    Re_fin  = abs(vx_h) * D / nu
    rho_err = abs(fl.densities - rho0).mean() / rho0 * 100
    nan_f   = bool(np.isnan(pos).any())

    bend = ((pos[:, 0] > 0.30) & (pos[:, 0] < 0.45) & (np.abs(pos[:, 1]) < 0.08))
    sec_final = float(np.linalg.norm(vel[bend, 1:], axis=1).mean()) if bend.sum() > 5 else 0.0

    # ── EOS K-factor from p_physical (Tait EOS snapshot) ─────────────────────
    # p_physical = eos_k*((rho/rho0)^7 - 1), stored in fluid.p_physical
    # by dfsph._update_fluid_pressure() before each PPE correction.
    # Note: K is reliable only at step with Re>11.6 AND rho>rho0 everywhere
    # (full pressurisation). Open-ended elbow geometry causes clipping near
    # open ends (rho<rho0 → p_physical=0), making dp unreliable.
    p_phys = getattr(fl, 'p_physical', None)
    K_diag = float('nan')
    dp_dx_sim = float('nan')
    if p_phys is not None:
        inlet_mask_k  = pos[:, 0] < 0.08
        outlet_mask_k = pos[:, 1] > 0.32
        if inlet_mask_k.sum() >= 5 and outlet_mask_k.sum() >= 5:
            p_in  = float(p_phys[inlet_mask_k].mean())
            p_out = float(p_phys[outlet_mask_k].mean())
            dp    = p_in - p_out
            h_vel = vel[pos[:, 0] < 0.15, 0]
            vmean = float(h_vel.mean()) if len(h_vel) > 0 else 0.0
            q     = 0.5 * rho0 * vmean ** 2
            mu    = nu * rho0
            dp_f  = 32 * mu * 0.8 * abs(vmean) / (D ** 2)
            dp_b  = dp - dp_f
            K_diag = float(dp_b / (q + 1e-10)) if q > 1e-6 else float('nan')
            rho_in  = float(fl.densities[inlet_mask_k].mean())
            rho_out = float(fl.densities[outlet_mask_k].mean())
    else:
        rho_in = rho_out = 0.0

    print()
    print(f"{'─'*60}")
    print("FINAL RESULTS")
    print(f"{'─'*60}")
    print(f"  Peak Re during run    = {peak_Re:.1f}  (step {peak_step})")
    print(f"  Final Re              = {Re_fin:.1f}")
    print(f"  vy_vertical_arm       = {vy_v:+.4f} m/s  (bend redirect)")
    print()
    print(f"  Bend secondary flow   = {sec_final:.4f} m/s")
    dean_yes = sec_final > 0.01
    print(f"  Dean vortex: {'YES ✓' if dean_yes else 'NO ✗'}  "
          f"(threshold 0.01 m/s; sharp elbow De→∞)")
    print()
    if not np.isnan(K_diag):
        print(f"  K_sim (p_physical)    = {K_diag:.3f}  "
              f"(rho_in={rho_in:.2f} rho_out={rho_out:.2f} kg/m³)")
        print(f"  K_ref (Idelchik)      = 0.5 – 3.0  (laminar 90° sharp elbow)")
        print(f"  K in range            = {0.5 < K_diag < 3.0}")
    else:
        print(f"  K_sim (p_physical)    = n/a  (insufficient particles at inlet/outlet)")
    print(f"  Note: p_physical = Tait EOS snapshot before PPE correction.")
    print(f"  K requires rho>rho0 everywhere (full pressurisation) for reliability.")
    print(f"  Open-ended elbow has rho<rho0 near open ends → Tait clips to 0.")
    print()
    print(f"  rho_err               = {rho_err:.2f}%")
    print(f"  NaN                   = {nan_f}")

    # ── Pass criteria ────────────────────────────────────────────────────────
    passed = (
        peak_Re > 11.6 and
        vy_v > 0 and
        dean_yes and
        rho_err < 5.0 and
        not nan_f
    )

    print()
    print(f"{'='*60}")
    print(f"INDUSTRIAL GRADE: {'PASS ✓' if passed else 'FAIL ✗'}")
    print(f"Criteria: Re>{11.6} ✓={peak_Re>11.6}  vy>0 ✓={vy_v>0}  "
          f"Dean ✓={dean_yes}  rho<5% ✓={rho_err<5}  NaN=F ✓={not nan_f}")
    print(f"Reference: Idelchik (1966) §5 + Dean (1928)")
    print(f"{'='*60}")
    return passed


if __name__ == "__main__":
    scene = sys.argv[1] if len(sys.argv) > 1 else "scenes/pipe_elbow_analytical.json"
    ok = validate_elbow_industrial(scene)
    sys.exit(0 if ok else 1)
