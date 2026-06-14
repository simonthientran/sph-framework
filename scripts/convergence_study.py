"""
Spatial-convergence study: L2 velocity error vs particle spacing dx.

Gold-standard CFD validation artifact for the 3D Hagen-Poiseuille benchmark
(scenes/benchmark_cylinder_pipe_3d.json, body-force driven, periodic-x):

    u(r) = (f / (4*nu)) * (R^2 - r^2)

ONLY the particle spacing dx is varied (and support_radius = 2*dx so the
smoothing-length ratio h/dx stays constant — standard SPH refinement).
Physics (f, nu, R, rho0, EOS) is identical across all resolutions, so the
measured L2 change isolates the spatial discretization error.

Each run is advanced to the SAME physical time t_warm = 0.15 s, which is
~10.4 viscous time constants tau = R^2/(lambda1^2 * nu) = 0.0144 s — the
startup transient has decayed to e^-10 ≈ 5e-5, i.e. fully developed Stokes
flow at every resolution. The profile is then averaged over N_MEASURE steps.

Usage
-----
  PYTHONPATH=src python scripts/convergence_study.py --dx 0.02 0.015
  PYTHONPATH=src python scripts/convergence_study.py --dx 0.01
  PYTHONPATH=src python scripts/convergence_study.py --plot --table

Results accumulate in out/convergence/results.csv; --plot writes
out/convergence/convergence.png (log-log, with slope-1/slope-2 references).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

BASE_SCENE = ROOT / "scenes" / "benchmark_cylinder_pipe_3d.json"
OUT_DIR = ROOT / "out" / "convergence"
CSV_PATH = OUT_DIR / "results.csv"
PNG_PATH = OUT_DIR / "convergence.png"

# ── Fixed physics (must match the scene; NEVER varied in this study) ─────────
F = 0.003       # body force [m/s^2]
NU = 0.12       # kinematic viscosity [m^2/s]
R = 0.10        # pipe radius [m]
CY, CZ = 0.14, 0.14
VMAX_ANA = F * R ** 2 / (4.0 * NU)        # 6.25e-5 m/s

T_WARM = 0.15   # physical warm-up time [s]  (≈10.4 viscous time constants)
N_MEASURE = 100  # steps to average the steady profile over
N_BINS = 10      # same radial binning as scripts/benchmark_cylinder_pipe.py
INTERIOR_FRAC = 0.85  # same interior mask (r < 0.85 R) as the benchmark

CSV_FIELDS = ["dx", "n_fluid", "n_boundary", "L2_pct", "L2p_pct", "vmax_ratio",
              "rho_err_pct", "iter_cd", "steps", "steps_per_sec", "t_phys"]


def _make_scene(dx: float) -> Path:
    """Write a resolution-variant of the base scene (only dx + support change)."""
    scene = json.loads(BASE_SCENE.read_text())
    scene["fluids"][0]["spacing"] = dx
    scene["neighbors"]["support_radius"] = 2.0 * dx   # keep h/dx constant
    scene["meta"]["description"] = (
        f"Convergence-study variant of benchmark_cylinder_pipe_3d: dx={dx}. "
        "Physics identical to base; only spatial resolution differs."
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"scene_dx{dx:g}.json"
    path.write_text(json.dumps(scene, indent=2))
    return path


def run_one(dx: float) -> dict:
    """Run a single resolution to developed flow and measure the L2 error."""
    from sph.core.simulation import SimulationRunner

    scene_path = _make_scene(dx)
    runner = SimulationRunner(scene_path)
    sim = runner.backend.sim
    sim._inlet_yz = None   # pure periodic benchmark (defensive, as in benchmark)
    rho0 = float(sim.fluid.rho0)

    print(f"[dx={dx:g}] n_fluid={sim.fluid.n} n_boundary={sim.boundary.n} "
          f"h={sim.h:g} — warm-up to t={T_WARM}s ...", flush=True)

    t0 = time.perf_counter()
    steps = 0
    while sim.t < T_WARM:
        runner.backend.step()
        steps += 1
        if steps % 500 == 0:
            print(f"  step {steps}  t={sim.t:.4f}s  "
                  f"vx_mean={sim.fluid.velocities[:, 0].mean():.3e}", flush=True)

    # ── Measure: radial profile averaged over N_MEASURE steps ────────────────
    r_bins = np.linspace(0.0, R, N_BINS + 1)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    vx_accum = np.zeros(N_BINS)
    cnt_accum = np.zeros(N_BINS, dtype=int)
    iter_cd_last = 0
    # Secondary, binning-free metric: per-particle L2 (analytic evaluated at
    # each particle's own r, interior r<0.85R), accumulated over the window.
    pp_num = 0.0
    pp_den = 0.0

    for _ in range(N_MEASURE):
        runner.backend.step()
        steps += 1
        pos = sim.fluid.positions
        vel = sim.fluid.velocities[:, 0]
        r = np.sqrt((pos[:, 1] - CY) ** 2 + (pos[:, 2] - CZ) ** 2)
        for b in range(N_BINS):
            mask = (r >= r_bins[b]) & (r < r_bins[b + 1])
            if np.any(mask):
                vx_accum[b] += vel[mask].mean()
                cnt_accum[b] += 1
        inner = r < INTERIOR_FRAC * R
        v_ana_p = (F / (4.0 * NU)) * (R ** 2 - r[inner] ** 2)
        pp_num += float(np.sum((vel[inner] - v_ana_p) ** 2))
        pp_den += float(np.sum(v_ana_p ** 2))
        iter_cd_last = int(sim.last_solver_stats.get("iter_cd", 0))

    wall = time.perf_counter() - t0
    with np.errstate(invalid="ignore"):
        vx_profile = np.where(cnt_accum > 0, vx_accum / cnt_accum, np.nan)
    vx_analytical = (F / (4.0 * NU)) * (R ** 2 - r_centers ** 2)

    # Identical L2 definition to scripts/benchmark_cylinder_pipe.py
    valid = (~np.isnan(vx_profile)) & (r_centers < INTERIOR_FRAC * R)
    num = np.sqrt(np.mean((vx_profile[valid] - vx_analytical[valid]) ** 2))
    den = np.sqrt(np.mean(vx_analytical[valid] ** 2))
    l2_pct = 100.0 * num / den
    l2p_pct = 100.0 * np.sqrt(pp_num / max(pp_den, 1e-30))

    vmax_ratio = float(vx_profile[0] / VMAX_ANA) if not np.isnan(vx_profile[0]) else float("nan")

    r_all = np.sqrt((sim.fluid.positions[:, 1] - CY) ** 2
                    + (sim.fluid.positions[:, 2] - CZ) ** 2)
    interior = r_all < 0.8 * R
    rho_err_pct = 100.0 * abs(sim.fluid.densities[interior].mean() - rho0) / rho0

    row = {
        "dx": dx,
        "n_fluid": int(sim.fluid.n),
        "n_boundary": int(sim.boundary.n),
        "L2_pct": round(l2_pct, 3),
        "L2p_pct": round(l2p_pct, 3),
        "vmax_ratio": round(vmax_ratio, 4),
        "rho_err_pct": round(rho_err_pct, 3),
        "iter_cd": iter_cd_last,
        "steps": steps,
        "steps_per_sec": round(steps / wall, 1),
        "t_phys": round(float(sim.t), 4),
    }
    print(f"[dx={dx:g}] DONE  L2={l2_pct:.2f}%  L2p={l2p_pct:.2f}%  "
          f"vmax_ratio={vmax_ratio:.3f}  rho_err={rho_err_pct:.2f}%  "
          f"steps={steps}  ({steps / wall:.1f} steps/s)", flush=True)
    return row


def _append_csv(row: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    new_file = not CSV_PATH.exists()
    # De-duplicate: replace any existing row with the same dx.
    rows: list[dict] = []
    if not new_file:
        with CSV_PATH.open() as f:
            rows = [r for r in csv.DictReader(f) if abs(float(r["dx"]) - row["dx"]) > 1e-12]
    rows.append({k: str(row[k]) for k in CSV_FIELDS})
    rows.sort(key=lambda r: -float(r["dx"]))
    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def _load_csv() -> list[dict]:
    if not CSV_PATH.exists():
        return []
    with CSV_PATH.open() as f:
        return sorted(csv.DictReader(f), key=lambda r: -float(r["dx"]))


def print_table() -> None:
    rows = _load_csv()
    if not rows:
        print("No results yet.")
        return
    hdr = (f"{'dx':>7} {'n_fluid':>8} {'L2%':>7} {'L2p%':>7} {'vmax_ratio':>11} "
           f"{'rho_err%':>9} {'iter_cd':>8} {'steps':>6} {'steps/s':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{float(r['dx']):>7.4f} {int(r['n_fluid']):>8d} {float(r['L2_pct']):>7.2f} "
              f"{float(r.get('L2p_pct', 'nan')):>7.2f} {float(r['vmax_ratio']):>11.3f} "
              f"{float(r['rho_err_pct']):>9.2f} "
              f"{int(r['iter_cd']):>8d} {int(r['steps']):>6d} {float(r['steps_per_sec']):>8.1f}")
    # Observed order of convergence from consecutive pairs (both metrics)
    if len(rows) >= 2:
        for key, label in (("L2_pct", "binned L2"), ("L2p_pct", "per-particle L2")):
            if any(key not in r for r in rows):
                continue
            print(f"\nObserved order p — {label} (p = ln(e_c/e_f)/ln(dx_c/dx_f)):")
            for a, b in zip(rows[:-1], rows[1:]):
                dxa, dxb = float(a["dx"]), float(b["dx"])
                ea, eb = float(a[key]), float(b[key])
                if ea > 0 and eb > 0:
                    p = np.log(ea / eb) / np.log(dxa / dxb)
                    print(f"  dx {dxa:g} → {dxb:g}:  p = {p:.2f}")
            e_list = [float(r[key]) for r in rows]
            mono = all(e_list[i] >= e_list[i + 1] for i in range(len(e_list) - 1))
            print(f"  monotonically decreasing: {'YES' if mono else 'NO'}")


def make_plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _load_csv()
    if len(rows) < 2:
        print("Need >=2 results to plot.")
        return
    dx = np.array([float(r["dx"]) for r in rows])
    l2 = np.array([float(r["L2_pct"]) for r in rows])

    fig, ax = plt.subplots(figsize=(6.0, 4.5), dpi=150)
    ax.loglog(dx, l2, "o-", color="#0057a8", lw=1.8, ms=7,
              label="binned L2 (benchmark metric)", zorder=3)
    if all("L2p_pct" in r for r in rows):
        l2p = np.array([float(r["L2p_pct"]) for r in rows])
        ax.loglog(dx, l2p, "s-", color="#c44e52", lw=1.5, ms=6,
                  label="per-particle L2 (binning-free)", zorder=3)

    # Reference slopes anchored at the coarsest point
    x_ref = np.array([dx.min(), dx.max()])
    for slope, style, col in ((1, "--", "#888888"), (2, ":", "#bbbbbb")):
        y_ref = l2[0] * (x_ref / dx[0]) ** slope
        ax.loglog(x_ref, y_ref, style, color=col, lw=1.2,
                  label=f"slope {slope} (order-{slope} ref.)")

    ax.set_xlabel("particle spacing dx [m]")
    ax.set_ylabel("L2 velocity error [%]")
    ax.set_title("Spatial convergence — Hagen-Poiseuille")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH)
    print(f"Saved {PNG_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dx", type=float, nargs="*", default=[],
                    help="resolutions to run (appends/updates results.csv)")
    ap.add_argument("--plot", action="store_true", help="write convergence.png")
    ap.add_argument("--table", action="store_true", help="print results table")
    args = ap.parse_args()

    for dx in args.dx:
        row = run_one(dx)
        _append_csv(row)

    if args.table or args.dx:
        print()
        print_table()
    if args.plot:
        make_plot()


if __name__ == "__main__":
    main()
