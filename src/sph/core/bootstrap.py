from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from sph.core.simulator import SimConfig, step_wc_sph
from sph.core.state_builder import build_fluid_block


def main() -> int:
    # Marker print to verify we're executing the currently edited file.
    # This does not change any simulation logic, math, physics equations,
    # solver ordering, or numerical formulations.
    print("[BOOT] NEW BOOTSTRAP ACTIVE ✅")
    print("[BOOT] bootstrap started")

    if len(sys.argv) < 2:
        print("Usage: python -m sph.core.bootstrap <scene.json>")
        return 2

    scene_path = Path(sys.argv[1]).resolve()
    if not scene_path.exists():
        print("[ERROR] scene file not found")
        return 1

    with scene_path.open("r", encoding="utf-8") as f:
        scene = json.load(f)

    state = build_fluid_block(scene)

    dim = state.dim
    h = float(scene["neighbors"]["support_radius"])
    rho0 = float(scene["material"]["rho0"])

    # particle size h_tilde in Eq. (33) is the particle diameter / spacing scale.
    particle_size = float(scene["fluid"]["spacing"])

    # gravity from scene (fallback: -9.81 in y)
    g_list = scene.get("forces", {}).get("gravity", [0.0, -9.81, 0.0])
    g = np.array(g_list[:dim], dtype=np.float64)

    # time settings
    time_cfg = scene.get("time", {})
    use_cfl = (time_cfg.get("mode", "cfl") == "cfl")
    dt_fixed = float(time_cfg.get("dt_fixed", 0.0005))
    cfl_lambda = float(time_cfg.get("cfl", 0.4))  # tutorial suggests ~0.4 in practice (text around Eq. (33))
    dt_min = float(time_cfg.get("dt_min", 1e-5))
    dt_max = float(time_cfg.get("dt_max", 5e-3))
    steps = int(time_cfg.get("steps", 10))

    # EOS stiffness (Section 4.4 examples: p = k (rho - rho0))
    eos_k = float(scene.get("material", {}).get("eos", {}).get("k", 2000.0))

    # viscosity (optional)
    enable_visc = bool(scene.get("material", {}).get("viscosity", {}).get("enable", False))
    nu = float(scene.get("material", {}).get("viscosity", {}).get("nu", 0.0))

    cfg = SimConfig(
        support_radius=h,
        rho0=rho0,
        eos_k=eos_k,
        g=g,
        cfl_lambda=cfl_lambda,
        dt_min=dt_min,
        dt_max=dt_max,
        dt_fixed=dt_fixed,
        use_cfl=use_cfl,
        enable_viscosity=enable_visc,
        kinematic_viscosity=nu,
    )

    print(f"[BOOT] N={state.n}, dim={dim}, h={h}, spacing={particle_size}")
    print(f"[BOOT] steps={steps}, use_cfl={use_cfl}, dt_fixed={dt_fixed}")
    print(f"[BOOT] g={g}, eos_k={eos_k}, viscosity_enable={enable_visc}, nu={nu}")

    for s in range(steps):
        dt = step_wc_sph(state, cfg=cfg, particle_size=particle_size)

        if s == 0 or (s + 1) % max(1, steps // 5) == 0:
            vnorm = np.linalg.norm(state.vel, axis=1)
            print(f"[STEP {s+1:04d}] dt={dt:.3e} |v| max={vnorm.max():.3e} pos_y min={state.pos[:,1].min():.3e}")

    print("[BOOT] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


