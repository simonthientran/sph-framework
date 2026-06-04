import sys; sys.path.insert(0, 'src')
from pathlib import Path
from sph.core.simulation import SimulationRunner
import numpy as np

def validate_elbow(scene='scenes/pipe_elbow_analytical.json', steps=800):
    print(f"\n{'='*50}")
    print(f"ELBOW VALIDATION — {steps} steps")
    print(f"{'='*50}")

    r = SimulationRunner(Path(scene))
    fl = r.backend.sim.fluid
    print(f"n_fluid: {fl.n}  n_boundary: {len(r.backend.sim.boundary.positions)}")

    max_iter_cd = 0
    for step in range(steps):
        rt = r.step()
        rm = rt.runtime if hasattr(rt, 'runtime') else rt
        sm = getattr(rm, 'solver_metrics', {})
        ic = int(sm.get('iter_cd', 0))
        max_iter_cd = max(max_iter_cd, ic)

        if step % 200 == 0:
            pos = fl.positions
            vel = fl.velocities
            h = pos[:, 0] < 0.20
            v = pos[:, 1] > 0.20
            vx_h = vel[h, 0].mean() if h.sum() > 5 else float('nan')
            vy_v = vel[v, 1].mean() if v.sum() > 5 else float('nan')
            rho_e = abs(fl.densities - fl.rho0).mean() / fl.rho0 * 100
            nan_f = np.isnan(fl.positions).any()
            print(f"  step={step:4d} | "
                  f"vx_h={vx_h:+.4f} | "
                  f"vy_v={vy_v:+.4f} | "
                  f"rho={rho_e:.2f}% | "
                  f"iter_cd={ic} | "
                  f"NaN={nan_f}")

    pos = fl.positions
    vel = fl.velocities
    h = pos[:, 0] < 0.20
    v = pos[:, 1] > 0.20
    vx_h = vel[h, 0].mean() if h.sum() > 5 else 0
    vy_v = vel[v, 1].mean() if v.sum() > 5 else 0
    rho_e = abs(fl.densities - fl.rho0).mean() / fl.rho0 * 100
    nan_f = np.isnan(fl.positions).any()

    results = {
        'vx_horizontal': vx_h,
        'vy_vertical':   vy_v,
        'rho_err':       rho_e,
        'NaN':           nan_f,
        'max_iter_cd':   max_iter_cd,
        'n_fluid':       fl.n,
    }

    print(f"\n{'─'*50}")
    print(f"FINAL RESULTS:")
    print(f"  vx horizontal arm: {vx_h:+.5f}  (need > +0.005)")
    print(f"  vy vertical arm:   {vy_v:+.5f}  (need > +0.001)")
    print(f"  rho_err:           {rho_e:.2f}%   (need < 5%)")
    print(f"  NaN:               {nan_f}        (need False)")
    print(f"  max iter_cd:       {max_iter_cd}        (need < 10)")

    passed = (vx_h > 0.005 and vy_v > 0.001 and
              rho_e < 5.0 and not nan_f and max_iter_cd < 10)
    print(f"\n  OVERALL: {'PASS ✓' if passed else 'FAIL ✗'}")
    return passed, results

if __name__ == '__main__':
    passed, _ = validate_elbow()
    sys.exit(0 if passed else 1)
