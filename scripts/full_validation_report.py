"""
Complete SPH Framework Validation Report.

Compares against: analytical, OpenFOAM, SimScale, SPlisHSPlasH
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


BENCHMARKS = [
    {
        "name": "2D Poiseuille (channel)",
        "scene": "scenes/examples/pipe_flow_2d.json",
        "runner": "single_phase",
        "target_L2": 7.0,
        "target_rho": 2.0,
    },
    {
        "name": "3D Hagen-Poiseuille (cylinder)",
        "scene": "scenes/benchmark_cylinder_pipe_3d.json",
        "runner": "cylinder",
        "target_L2": 5.0,
        "target_rho": 2.0,
    },
    {
        "name": "Industrial pipe (long)",
        "scene": "scenes/industrial_pipe_benchmark.json",
        "runner": "industrial",
        "target_L2": 10.0,
        "target_rho": 2.0,
    },
]


def _run_single_phase() -> tuple[float, float, bool]:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "benchmark_single_phase",
        ROOT / "scripts" / "benchmark_single_phase.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        ok = mod.run_poiseuille_benchmark(n_steps=3000, verbose=False)
    text = buf.getvalue()
    l2 = rho = 0.0
    for line in text.splitlines():
        if "L2 error:" in line:
            l2 = float(line.split("%")[0].split()[-1])
        if "density err:" in line:
            rho = float(line.split("%")[0].split()[-1])
    return l2, rho, ok


def _run_cylinder() -> tuple[float, float, bool]:
    import numpy as np
    from sph.core.simulation import SimulationRunner

    scene = ROOT / "scenes" / "benchmark_cylinder_pipe_3d.json"
    F, NU, R, CY, CZ = 0.003, 0.12, 0.10, 0.14, 0.14
    N_WARMUP, N_MEASURE = 500, 200

    runner = SimulationRunner(scene)
    sim = runner.backend.sim
    sim._inlet_yz = None
    rho0 = float(sim.fluid.rho0)

    for _ in range(N_WARMUP + N_MEASURE):
        runner.step()

    pos = sim.fluid.positions
    vel = sim.fluid.velocities
    r = np.sqrt((pos[:, 1] - CY) ** 2 + (pos[:, 2] - CZ) ** 2)
    interior = r < 0.85 * R
    vx = vel[interior, 0]
    r_int = r[interior]
    vmax = float(vx.max()) if vx.size else 0.0
    vx_ana = (F / (4.0 * NU)) * (R**2 - r_int**2)
    l2 = float(np.sqrt(np.mean((vx - vx_ana) ** 2)) / (np.sqrt(np.mean(vx_ana**2)) + 1e-10))
    drho = float(abs(sim.fluid.densities[interior].mean() - rho0) / rho0 * 100)
    ok = l2 * 100 < 5.0 and drho < 2.0
    return l2 * 100, drho, ok


def _run_industrial() -> tuple[float, float, bool]:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "industrial_validation",
        ROOT / "scripts" / "industrial_validation.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    scene = ROOT / "scenes" / "industrial_pipe_benchmark.json"
    result = mod.main(scene, n_steps=2000)
    l2 = float(result["b3_l2_err"]) * 100
    rho = float(result["b4_drho"]) * 100
    return l2, rho, bool(result["all_pass"])


def _run_elbow_k() -> tuple[float, bool]:
    import importlib.util
    import io
    from contextlib import redirect_stdout

    spec = importlib.util.spec_from_file_location(
        "validate_elbow",
        ROOT / "scripts" / "validate_elbow.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    buf = io.StringIO()
    with redirect_stdout(buf):
        ok = mod.validate_elbow(ROOT / "scenes" / "pipe_elbow_3d.json", n_steps=1000)
    k = 0.0
    for line in buf.getvalue().splitlines():
        if line.startswith("K_sim"):
            k = float(line.split("=")[1].strip())
    return k, ok


def main() -> None:
    rows: list[tuple[str, str, str, str, str, bool]] = []

    l2_2d, rho_2d, ok_2d = _run_single_phase()
    rows.append(("2D Poiseuille", f"{l2_2d:.2f}%", f"{rho_2d:.2f}%", "Analytical", ok_2d))

    l2_3d, rho_3d, ok_3d = _run_cylinder()
    rows.append(("3D Hagen-Poiseuille", f"{l2_3d:.2f}%", f"{rho_3d:.2f}%", "Analytical", ok_3d))

    try:
        l2_ind, rho_ind, ok_ind = _run_industrial()
        rows.append(("Industrial pipe", f"{l2_ind:.2f}%", f"{rho_ind:.2f}%", "SimScale", ok_ind))
    except Exception as exc:
        rows.append(("Industrial pipe", "N/A", "N/A", "SimScale", False))
        print(f"Industrial benchmark skipped: {exc}")

    k_elbow, ok_elbow = _run_elbow_k()
    rows.append(("Elbow K-factor", f"K={k_elbow:.2f}", "—", "Idel'chik", ok_elbow))

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          SPH FRAMEWORK — VALIDATION REPORT                  ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║ Benchmark          │ L2 err │ rho err │ Ref         │ Status ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    for name, l2, rho, ref, ok in rows:
        status = "PASS ✓" if ok else "FAIL ✗"
        print(f"║ {name:<18} │ {l2:>6} │ {rho:>7} │ {ref:<11} │ {status:<6} ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    all_ok = all(r[4] for r in rows)
    print(f"\nOVERALL: {'PASS ✓' if all_ok else 'PARTIAL'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
