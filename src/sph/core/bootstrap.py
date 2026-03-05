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

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from sph.boundary.mesh_sampling import (
    compute_fluid_boundary_distance_stats,
    estimate_point_spacing,
    write_boundary_cloud_csv,
)
from sph.core.diagnostics import compute_step_diagnostics
from sph.core.startup_sanity import evaluate_startup_sanity, format_startup_sanity_block
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
from sph.verification.harness import run_verification


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SPH scene bootstrap.")
    parser.add_argument("scene", help="Path to scene JSON")
    parser.add_argument("--debug-geom", action="store_true", help="Enable detailed geometry diagnostics and cloud dump.")
    parser.add_argument("--verify", action="store_true", help="Run verification harness and enforce pass/fail gates.")
    parser.add_argument("--verify-fast", action="store_true", help="Run verification in fast mode (reduced steps from scene verification.fast).")
    parser.add_argument(
        "--verify-report",
        default=None,
        help="Optional report JSON path for --verify mode. Defaults to out/verification/<scene>_<solver>.json",
    )
    return parser.parse_args()


def _print_verify_table(report: dict) -> None:
    gates = report.get("gates", {})
    if not gates:
        return
    print("[VERIFY] metric                         value           min             max             pass")
    print("[VERIFY] ------------------------------------------------------------------------------------")
    for name in sorted(gates.keys()):
        g = gates[name]
        value = g.get("value", "")
        gmin = g.get("min", None)
        gmax = g.get("max", None)
        p = bool(g.get("pass", False))
        def _fmt(v):
            if v is None:
                return "-"
            if isinstance(v, bool):
                return "true" if v else "false"
            try:
                vf = float(v)
                return f"{vf:.6g}"
            except Exception:
                return str(v)
        print(f"[VERIFY] {name:<30} {_fmt(value):<14} {_fmt(gmin):<14} {_fmt(gmax):<14} {'PASS' if p else 'FAIL'}")


def _print_geom_step0_report(state, h: float, spacing: float, *, debug_geom: bool) -> None:
    ns0 = SpatialHash(support_radius=float(h), dim=state.dim)
    ns0.build(state.pos)
    fluid_ids = state.fluid_indices
    bound_ids = state.boundary_indices
    if fluid_ids.size:
        fluid_idx = fluid_ids[: min(200, fluid_ids.size)]
        fluid_neigh = np.array([len(ns0.query(int(i), state.pos)) for i in fluid_idx], dtype=np.int64)
    else:
        fluid_neigh = np.zeros((0,), dtype=np.int64)
    if bound_ids.size:
        bound_idx = bound_ids[: min(200, bound_ids.size)]
        bound_neigh = np.array([len(ns0.query(int(i), state.pos)) for i in bound_idx], dtype=np.int64)
    else:
        bound_neigh = np.zeros((0,), dtype=np.int64)

    bpos = state.pos[bound_ids]
    fpos = state.pos[fluid_ids]
    overlap = compute_fluid_boundary_distance_stats(
        fluid_positions=fpos,
        boundary_positions=bpos,
        threshold=0.5 * float(spacing),
    )
    spacing_est = estimate_point_spacing(bpos) if bpos.size else float("nan")
    bmin = np.min(bpos, axis=0) if bpos.size else np.array([float("nan")] * state.dim)
    bmax = np.max(bpos, axis=0) if bpos.size else np.array([float("nan")] * state.dim)
    print(
        f"[GEOM][STEP0] boundary_count={bound_ids.size} "
        f"bbox_min={bmin.tolist()} bbox_max={bmax.tolist()} "
        f"boundary_spacing_est={spacing_est:.6e} "
        f"fluid_neigh(avg)={float(np.mean(fluid_neigh)) if fluid_neigh.size else 0.0:.2f} "
        f"boundary_neigh(avg)={float(np.mean(bound_neigh)) if bound_neigh.size else 0.0:.2f} "
        f"fluid_boundary_min_dist={overlap.min_distance:.6e}"
    )
    if debug_geom and bpos.size:
        write_boundary_cloud_csv("out/geom/boundary_cloud_step0.csv", np.pad(bpos, ((0, 0), (0, 3 - state.dim))), np.tile(np.array([[0.0, 0.0, 1.0]]), (bpos.shape[0], 1)))
        print("[GEOM][STEP0] wrote boundary cloud CSV: out/geom/boundary_cloud_step0.csv")


def main() -> int:
    print("[BOOT] NEW BOOTSTRAP ACTIVE")

    args = _parse_args()
    scene_path = Path(args.scene).resolve()
    if not scene_path.exists():
        print("[ERROR] scene file not found")
        return 1

    with scene_path.open("r", encoding="utf-8") as f:
        scene = json.load(f)
    # Provide scene directory so relative asset paths (e.g., STL files) resolve predictably.
    scene["__scene_dir__"] = str(scene_path.parent)
    scene["__debug_geom__"] = bool(args.debug_geom)

    if bool(args.verify):
        solver_name = str(scene.get("solver", {}).get("type", "wcsph")).lower()
        default_report = Path("out/verification") / f"{scene_path.stem}_{solver_name}.json"
        report_path = Path(args.verify_report) if args.verify_report is not None else default_report
        passed, report = run_verification(
            scene,
            scene_path=str(scene_path),
            report_path=report_path,
            fast=bool(args.verify_fast),
        )
        print(
            f"[VERIFY] scene={report.get('scene_name')} solver={report.get('solver')} "
            f"passed={passed} report={report_path}"
        )
        _print_verify_table(report)
        print(f"[VERIFY] overall={'PASS' if passed else 'FAIL'}")
        return 0 if passed else 1

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
        eos_c0=(
            float(scene.get("material", {}).get("eos", {}).get("c0"))
            if scene.get("material", {}).get("eos", {}).get("c0", None) is not None
            else None
        ),
        enable_viscosity=enable_viscosity,
        kinematic_viscosity=kinematic_viscosity,
        # domain collision
        domain_min=domain_min,
        domain_max=domain_max,
        boundary_restitution=float(boundary_cfg.get("restitution", domain_cfg.get("restitution", 0.0))),
        boundary_friction=float(boundary_cfg.get("friction", domain_cfg.get("friction", 0.05))),
        boundary_tangent_friction=float(
            boundary_cfg.get("tangent_friction", boundary_cfg.get("friction", domain_cfg.get("friction", 0.1)))
        ),
        boundary_normal_damping=float(boundary_cfg.get("normal_damping", 0.2)),
        max_penetration_push_frac_of_dx=float(boundary_cfg.get("max_penetration_push_frac_of_dx", 0.25)),
        boundary_log_speed_threshold=float(boundary_cfg.get("log_speed_threshold", 30.0)),
        boundary_eps=(
            float(boundary_cfg.get("eps"))
            if boundary_cfg.get("eps", None) is not None
            else (float(domain_cfg.get("eps")) if domain_cfg.get("eps", None) is not None else None)
        ),
        boundary_force_accel_clamp=(
            float(boundary_cfg.get("force_accel_clamp"))
            if boundary_cfg.get("force_accel_clamp", None) is not None
            else None
        ),
    )

    # Startup numerical safety checks before stepping.
    startup_cfg = scene.get("startup_safety", {})
    density_warn_threshold = float(startup_cfg.get("density_error_warn_rel", 0.05))
    density_abort_threshold = float(startup_cfg.get("density_error_abort_rel", density_warn_threshold))
    dt_ramp_enabled = bool(startup_cfg.get("dt_ramp_if_high_density_error", True))
    dt_ramp_steps = int(startup_cfg.get("dt_ramp_steps", 10))
    dt_ramp_factor = float(startup_cfg.get("dt_ramp_factor", 0.25))
    sanity = evaluate_startup_sanity(
        scene=scene,
        state=state,
        h=h,
        spacing=spacing,
        rho0=rho0,
        startup_cfg=startup_cfg,
    )
    print(format_startup_sanity_block(sanity))
    if sanity.auto_tuned_support_radius is not None:
        h = float(sanity.auto_tuned_support_radius)
        print(f"[STARTUP][FIX] support_radius auto-tuned to {h:.6e}")
        cfg = replace(cfg, support_radius=h)

    init_rel_err = float(sanity.rho_rel_err_mean)
    dt_ramp_active = bool(dt_ramp_enabled and init_rel_err > density_warn_threshold and cfg.use_cfl)
    if init_rel_err > density_warn_threshold:
        print(
            f"[STARTUP][WARN] initial density rel error={init_rel_err:.3%} (> {density_warn_threshold:.3%}). "
            "Likely causes: mesh scale mismatch, fluid-boundary overlap, insufficient boundary resolution, "
            "or very low STL triangle count."
        )
    if sanity.should_abort:
        print(
            f"[STARTUP][ERROR] startup sanity failed ({sanity.abort_reason}). "
            f"Initial density avg rel error={init_rel_err:.3%}, abort threshold={density_abort_threshold:.3%}. "
            "Fix scene scale/units, overlap, boundary resolution, or support radius."
        )
        return 1
    if dt_ramp_active:
        print(
            f"[STARTUP] enabling dt_max ramp for first {dt_ramp_steps} steps "
            f"(factor={dt_ramp_factor:.3f} -> 1.0)"
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
    vx_profile_use_x_slice = bool(vx_profile_cfg.get("use_x_slice", False))
    vx_profile_x_slice_width = vx_profile_cfg.get("x_slice_width", None)
    if vx_profile_x_slice_width is not None:
        vx_profile_x_slice_width = float(vx_profile_x_slice_width)
    vx_profile_gx = vx_profile_cfg.get("gx", None)
    if vx_profile_gx is not None:
        vx_profile_gx = float(vx_profile_gx)
    else:
        vx_profile_gx = float(g[0]) if dim >= 1 else 0.0
    vx_profile_nu = vx_profile_cfg.get("nu", None)
    if vx_profile_nu is not None:
        vx_profile_nu = float(vx_profile_nu)
    else:
        vx_profile_nu = kinematic_viscosity
    vx_profile_csv_cfg = vx_profile_cfg.get("csv", {})
    vx_profile_csv_enabled = bool(vx_profile_csv_cfg.get("enable", False))
    vx_profile_csv_every = int(vx_profile_csv_cfg.get("every", 10))
    vx_profile_csv_dir = Path(vx_profile_csv_cfg.get("dir", "out/pipe_flow_2d"))
    vx_profile_csv_file = str(vx_profile_csv_cfg.get("file", "vx_profile_bins.csv"))

    # Export step 0000 if enabled (pre-step snapshot)
    if csv_enabled:
        export_particles_csv(csv_dir / "particles_step_0000.csv", state)
    if vtk_enabled:
        export_particles_vtk_legacy(vtk_dir / "particles_step_0000.vtk", state)
    if args.debug_geom or scene.get("geometry", {}).get("meshes"):
        _print_geom_step0_report(state, h=h, spacing=spacing, debug_geom=bool(args.debug_geom))

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
    sim_time = 0.0
    for s in range(steps):
        step_cfg = cfg
        if dt_ramp_active and s < dt_ramp_steps:
            alpha = float((s + 1) / max(1, dt_ramp_steps))
            scale = float(dt_ramp_factor + (1.0 - dt_ramp_factor) * alpha)
            step_cfg = replace(cfg, dt_max=float(cfg.dt_max * scale))
        dt = step_simulation(
            state=state,
            cfg=step_cfg,
            particle_size=spacing,
            solver_cfg_dict=solver_cfg,
            step_idx=s + 1,
        )

        # Diagnostics neighbor search on current positions (read-only)
        ns = SpatialHash(support_radius=h, dim=dim)
        ns.build(state.pos)
        diag = compute_step_diagnostics(step=s + 1, dt=dt, state=state, rho0=rho0, neighbor_search=ns)
        sim_time += float(dt)

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
                use_x_slice=vx_profile_use_x_slice,
                x_slice_width=vx_profile_x_slice_width,
            )
            print(f"[STEP {diag.step:04d}] {format_vx_profile_log_line(vx_result)}")
            if vx_profile_csv_enabled and ((s + 1) % max(1, vx_profile_csv_every) == 0):
                export_vx_profile_csv(
                    vx_profile_csv_dir / vx_profile_csv_file,
                    vx_result,
                    scene_name=str(scene.get("meta", {}).get("name", "unknown")),
                    sim_time=sim_time,
                )

        if csv_enabled and ((s + 1) % max(1, csv_every) == 0):
            export_particles_csv(csv_dir / f"particles_step_{diag.step:04d}.csv", state)

        if vtk_enabled and ((s + 1) % max(1, vtk_every) == 0):
            export_particles_vtk_legacy(vtk_dir / f"particles_step_{diag.step:04d}.vtk", state)

    print("[BOOT] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


