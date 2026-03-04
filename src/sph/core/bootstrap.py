"""
Bootstrap / CLI entry point for the SPH framework.

What this file does:
- Loads a JSON scene configuration.
- Builds the particle state (fluid + static boundary particles).
- Runs a selectable SPH solver loop:
  - WCSPH baseline (Algorithm 1 style) OR
  - PCISPH (Predictive–Corrective Incompressible SPH) using the equations below.
- Logs per-step diagnostics (rho/p/v/neighbors).
- Optionally exports CSV and VTK snapshots for ParaView/analysis.

References:
- "SPH Techniques for the Physics Based Simulation of Fluids and Solids - SPH_Tutorial.pdf"
  - Algorithm 1: baseline loop structure (density -> forces -> integrate)
  - Eq. (33): CFL time step restriction (dt selected inside solver step)
  - Eq. (83): density including boundary contributions (boundary-aware density)
  - Eq. (84): pressure forces including boundary contributions (boundary-aware pressure)
  - PCISPH section:
    - Eq. (51), (53), (57), (58), (59), (60) (implemented in sph/solver/pcisph.py)

Important constraint:
- This file must not change any solver math/physics. It only wires together
  existing components and adds observability/export around them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from sph.core.diagnostics import compute_step_diagnostics
from sph.core.simulator import SimConfig, step_simulation
from sph.core.state_builder import build_scene_state
from sph.core.vx_profile import (
    compute_vx_profile,
    export_vx_profile_csv,
    format_vx_profile_log_line,
)
from sph.io.csv_export import export_particles_csv
from sph.io.vtk_export import export_particles_vtk_legacy
from sph.neighbors.spatial_hash import SpatialHash


def main() -> int:
    print("[BOOT] NEW BOOTSTRAP ACTIVE ✅")

    if len(sys.argv) < 2:
        print("Usage: python -m sph.core.bootstrap <scene.json>")
        return 2

    scene_path = Path(sys.argv[1]).resolve()
    if not scene_path.exists():
        print("[ERROR] scene file not found")
        return 1

    with scene_path.open("r", encoding="utf-8") as f:
        scene = json.load(f)

    solver_cfg = scene.get("solver", {"type": "wcsph"})
    solver_type = str(solver_cfg.get("type", "wcsph")).lower()
    # Print solver params for reproducibility/observability (no physics impact).
    print(f"[BOOT] solver={solver_type} cfg={json.dumps(solver_cfg, sort_keys=True)}")

    # -------------------------------------------------------------------------
    # Build state (fluid + boundary particles)
    # Boundary particles are static; fluid particles are integrated each step.
    # This matches the particle-based boundary handling idea around Eq. (83)/(84).
    # -------------------------------------------------------------------------
    state = build_scene_state(scene)

    dim = int(state.dim)
    spacing = float(scene["fluid"]["spacing"])
    h = float(scene["neighbors"]["support_radius"])
    rho0 = float(scene["material"]["rho0"])

    # Gravity from scene (fallback: -9.81 in y for 2D)
    g = np.array(scene.get("forces", {}).get("gravity", [0.0, -9.81])[:dim], dtype=np.float64)

    # Time settings (dt is selected inside the solver according to Eq. (33) if enabled)
    time_cfg = scene.get("time", {})
    use_cfl = (time_cfg.get("mode", "cfl") == "cfl")
    steps = int(time_cfg.get("steps", 50))
    log_every = int(time_cfg.get("log_every", 10))

    # Domain / Boundary constraints
    domain_cfg = scene.get("domain", {})
    boundary_cfg = scene.get("boundary", {})  # optional (preferred for collision params)
    domain_min = None
    domain_max = None
    if "min" in domain_cfg and "max" in domain_cfg:
        domain_min = np.array(domain_cfg["min"], dtype=np.float64)
        domain_max = np.array(domain_cfg["max"], dtype=np.float64)

    # Viscosity from scene (for solver and vx_profile analytic vmax)
    visc_cfg = scene.get("material", {}).get("viscosity", {})
    enable_viscosity = bool(visc_cfg.get("enable", False))
    kinematic_viscosity = float(visc_cfg.get("nu", 0.0))

    cfg = SimConfig(
        support_radius=h,
        rho0=rho0,
        eos_k=float(scene.get("material", {}).get("eos", {}).get("k", 500.0)),
        g=g,
        cfl_lambda=float(time_cfg.get("cfl", 0.4)),
        dt_min=float(time_cfg.get("dt_min", 1e-5)),
        dt_max=float(time_cfg.get("dt_max", 5e-4)),
        dt_fixed=float(time_cfg.get("dt_fixed", 5e-4)),
        use_cfl=bool(use_cfl),
        enable_viscosity=enable_viscosity,
        kinematic_viscosity=kinematic_viscosity,
        # domain collision
        domain_min=domain_min,
        domain_max=domain_max,
        boundary_restitution=float(boundary_cfg.get("restitution", domain_cfg.get("restitution", 0.0))),
        boundary_friction=float(boundary_cfg.get("friction", domain_cfg.get("friction", 0.05))),
        boundary_eps=(
            float(boundary_cfg.get("eps"))
            if boundary_cfg.get("eps", None) is not None
            else (float(domain_cfg.get("eps")) if domain_cfg.get("eps", None) is not None else None)
        ),
    )

    # -------------------------------------------------------------------------
    # Optional exports controlled by scene:
    #   export.csv.enable/every/dir
    #   export.vtk.enable/every/dir
    # -------------------------------------------------------------------------
    export_cfg = scene.get("export", {})

    csv_cfg = export_cfg.get("csv", {})
    csv_enabled = bool(csv_cfg.get("enable", False))
    csv_every = int(csv_cfg.get("every", 10))
    csv_dir = Path(csv_cfg.get("dir", "out/csv"))

    vtk_cfg = export_cfg.get("vtk", {})
    vtk_enabled = bool(vtk_cfg.get("enable", False))
    vtk_every = int(vtk_cfg.get("every", 10))
    vtk_dir = Path(vtk_cfg.get("dir", "out/vtk"))

    # Vx-profile debug (optional): scene.debug.vx_profile.enable, y_extent_mode, n_bins, gx, nu
    debug_cfg = scene.get("debug", {})
    vx_profile_cfg = debug_cfg.get("vx_profile", {})
    vx_profile_enabled = bool(vx_profile_cfg.get("enable", False))
    vx_profile_mode = str(vx_profile_cfg.get("y_extent_mode", "walls_inner"))
    vx_profile_n_bins = int(vx_profile_cfg.get("n_bins", 8))
    vx_profile_gx = vx_profile_cfg.get("gx", None)
    if vx_profile_gx is not None:
        vx_profile_gx = float(vx_profile_gx)
    else:
        vx_profile_gx = float(g[0]) if dim >= 1 else 0.0
    vx_profile_nu = vx_profile_cfg.get("nu", None)
    if vx_profile_nu is not None:
        vx_profile_nu = float(vx_profile_nu)
    else:
        vx_profile_nu = kinematic_viscosity if enable_viscosity else None
    vx_profile_csv_enabled = bool(vx_profile_cfg.get("csv", {}).get("enable", False))
    vx_profile_csv_every = int(vx_profile_cfg.get("csv", {}).get("every", 10))
    vx_profile_csv_dir = Path(vx_profile_cfg.get("csv", {}).get("dir", "out/vx_profile"))

    # Export step 0000 if enabled (pre-step snapshot)
    if csv_enabled:
        export_particles_csv(csv_dir / "particles_step_0000.csv", state)
    if vtk_enabled:
        export_particles_vtk_legacy(vtk_dir / "particles_step_0000.vtk", state)

    # -------------------------------------------------------------------------
    # Main simulation loop
    #
    # Ordering matches the selected solver:
    # - WCSPH uses Algorithm 1 structure and boundary handling via Eq. (83)/(84).
    # - PCISPH uses Eq. (51),(53),(57),(58),(59),(60) (see sph/solver/pcisph.py).
    #
    # This loop only:
    # - dispatches to the existing solver step
    # - builds a neighbor search for diagnostics
    # - prints/export snapshots for observability
    # -------------------------------------------------------------------------
    for s in range(steps):
        dt = step_simulation(
            state=state,
            cfg=cfg,
            particle_size=spacing,
            solver_cfg_dict=solver_cfg,
            step_idx=s + 1,
        )

        # Diagnostics neighbor search on current positions (read-only)
        ns = SpatialHash(support_radius=h, dim=dim)
        ns.build(state.pos)
        diag = compute_step_diagnostics(step=s + 1, dt=dt, state=state, rho0=rho0, neighbor_search=ns)

        if (s == 0) or ((s + 1) % max(1, log_every) == 0):
            print(
                f"[STEP {diag.step:04d}] dt={diag.dt:.3e} "
                f"|v|max={diag.v_max:.3e} "
                f"rho(min/avg/max)={diag.rho_min:.2f}/{diag.rho_mean:.2f}/{diag.rho_max:.2f} "
                f"err% (avg)={100.0 * diag.rho_rel_err_mean:.2f} "
                f"p(min/avg/max)={diag.p_min:.2f}/{diag.p_mean:.2f}/{diag.p_max:.2f} "
                f"neigh(min/avg/max)={diag.neigh_min}/{diag.neigh_mean:.1f}/{diag.neigh_max}"
            )

        if vx_profile_enabled and ((s == 0) or ((s + 1) % max(1, log_every) == 0)):
            vx_result = compute_vx_profile(
                step=diag.step,
                state=state,
                scene=scene,
                y_extent_mode=vx_profile_mode,
                n_bins=vx_profile_n_bins,
                gx=vx_profile_gx,
                nu=vx_profile_nu,
            )
            print(f"[STEP {diag.step:04d}] {format_vx_profile_log_line(vx_result)}")
            if vx_profile_csv_enabled and ((s + 1) % max(1, vx_profile_csv_every) == 0):
                export_vx_profile_csv(
                    vx_profile_csv_dir / f"vx_profile_step_{diag.step:04d}.csv",
                    vx_result,
                )

        if csv_enabled and ((s + 1) % max(1, csv_every) == 0):
            export_particles_csv(csv_dir / f"particles_step_{diag.step:04d}.csv", state)

        if vtk_enabled and ((s + 1) % max(1, vtk_every) == 0):
            export_particles_vtk_legacy(vtk_dir / f"particles_step_{diag.step:04d}.vtk", state)

    print("[BOOT] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


