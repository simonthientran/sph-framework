#!/usr/bin/env python3
"""
splash — SPH Framework CLI.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sph.core.simulation import SimulationRunner
from sph.scene.schema import Scene


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="splash",
        description="SPH Framework — SPlisHSPlasH-compatible CLI",
    )
    parser.add_argument("--scene", "-s", required=True, help="Path to scene JSON file")
    parser.add_argument("--no-gui", action="store_true", help="Run headless")
    parser.add_argument("--stopAt", type=float, default=-1.0, help="Stop simulation at this time")
    parser.add_argument("--pauseAt", type=float, default=-1.0, help="Pause time (ignored in headless mode)")
    parser.add_argument("--output-dir", default=None, help="Override VTK output directory")
    parser.add_argument(
        "--backend",
        default="numba_cpu",
        choices=["numba_cpu", "numba_cuda"],
        help="Simulation backend",
    )
    parser.add_argument("--steps", type=int, default=-1, help="Override: run exactly N steps")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    scene_path = Path(args.scene)
    if not scene_path.exists():
        print(f"Error: scene not found: {scene_path}")
        return 1

    scene = Scene.from_json(scene_path)
    if args.stopAt > 0.0:
        scene.configuration.stop_at = args.stopAt
    if args.pauseAt > 0.0:
        scene.configuration.pause_at = args.pauseAt
    if args.no_gui:
        _ = args.no_gui

    print("=" * 60)
    print("  SPH FRAMEWORK  |  SPlisHSPlasH-compatible")
    print("=" * 60)
    print(f"  Scene:    {scene_path.name}")
    print("  Solver:   DFSPH")
    print(f"  Backend:  {args.backend}")
    print(f"  Radius:   {scene.configuration.particle_radius}")
    print(
        f"  CFL:      method={scene.configuration.cfl_method}  "
        f"factor={scene.configuration.cfl_factor}"
    )
    if scene.configuration.stop_at > 0.0:
        print(f"  Stop at:  {scene.configuration.stop_at}s")
    print("=" * 60)

    runner = SimulationRunner(scene_path, backend_name=args.backend)
    if args.output_dir:
        out_dir = Path(args.output_dir)
        runner._export_manager.settings.vtk.directory = out_dir
        runner.backend.export_settings.vtk.directory = out_dir

    t0 = time.perf_counter()
    sim_time = 0.0
    step = 0
    log_every = 50

    try:
        while True:
            result = runner.step()
            step += 1
            sim_time += float(result.runtime.dt)

            if args.verbose or step % log_every == 0:
                elapsed = time.perf_counter() - t0
                print(
                    f"  t={sim_time:.4f}s  "
                    f"step={step:6d}  "
                    f"dt={result.runtime.dt:.3e}  "
                    f"vmax={result.runtime.velocity_max:.4f}  "
                    f"rho_err={result.runtime.rho_error_mean * 100:.3f}%  "
                    f"[{elapsed:.1f}s elapsed]"
                )

            if scene.configuration.stop_at > 0.0 and sim_time >= scene.configuration.stop_at:
                print(f"\n  Reached stopAt={scene.configuration.stop_at}s")
                break
            if args.steps > 0 and step >= args.steps:
                print(f"\n  Reached {args.steps} steps")
                break
    except KeyboardInterrupt:
        print("\n  Interrupted by user")

    elapsed = time.perf_counter() - t0
    realtime = sim_time / elapsed if elapsed > 0.0 else 0.0
    print("=" * 60)
    print(
        f"  Done: {step} steps, t={sim_time:.4f}s, elapsed={elapsed:.1f}s, "
        f"perf={realtime:.3f}x realtime"
    )
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
