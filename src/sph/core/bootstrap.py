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
import signal
import sys
from pathlib import Path

import numpy as np

from sph.core.diagnostics import compute_step_diagnostics
from sph.core.simulator import SimConfig, step_simulation
from sph.core.state_builder import build_scene_state
from sph.io.csv_export import export_particles_csv
from sph.io.results_export import ResultsLogger, StepMetrics, VxProfileMetrics
from sph.io.vtk_export import export_particles_vtk_legacy
from sph.neighbors.spatial_hash import SpatialHash
from sph.boundaries import BoundaryManager, WallBoundary, InflowBoundary, OutflowBoundary
from sph.diagnostics.flow_metrics import FlowMetrics




def main() -> int:
    # Make `python -m ... | head` not spam BrokenPipeError during interpreter shutdown.
    # This is logging/CLI robustness only; it does not affect simulation math.
    try:
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except Exception:
        pass
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
        # -------------------------------------------------------------------------
        state = build_scene_state(scene)

        dim = int(state.dim)
        spacing = float(scene["fluid"]["spacing"])
        h = float(scene["neighbors"]["support_radius"])
        rho0 = float(scene["material"]["rho0"])

        # External body forces (gravity + optional constant body_force).
        forces_cfg = scene.get("forces", {})
        gravity = np.array(forces_cfg.get("gravity", [0.0, -9.81])[:dim], dtype=np.float64)
        body_force = np.array(forces_cfg.get("body_force", [0.0] * dim)[:dim], dtype=np.float64)
        g = gravity + body_force

        time_cfg = scene.get("time", {})
        use_cfl = (time_cfg.get("mode", "cfl") == "cfl")
        steps = int(time_cfg.get("steps", 50))
        log_every = int(time_cfg.get("log_every", 10))

        domain_cfg = scene.get("domain", {})
        boundary_cfg = scene.get("boundary", {})  # optional (preferred for collision params)
        domain_min = None
        domain_max = None
        if "min" in domain_cfg and "max" in domain_cfg:
            domain_min = np.array(domain_cfg["min"], dtype=np.float64)
            domain_max = np.array(domain_cfg["max"], dtype=np.float64)
        # Periodic axes (geometry / neighbor distance handling).
        # Example: "periodic_axes": ["x"] for periodic in x.
        axes_map = {"x": 0, "y": 1, "z": 2}
        periodic_axes_names = domain_cfg.get("periodic_axes", [])
        periodic_axes: tuple[int, ...] = ()
        if periodic_axes_names:
            periodic_axes = tuple(axes_map[str(a).lower()] for a in periodic_axes_names)

        visc_cfg = scene.get("material", {}).get("viscosity", {})
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
            enable_viscosity=bool(visc_cfg.get("enable", False)),
            kinematic_viscosity=float(visc_cfg.get("nu", 0.0)),
            domain_min=domain_min,
            domain_max=domain_max,
            boundary_restitution=float(boundary_cfg.get("restitution", domain_cfg.get("restitution", 0.0))),
            boundary_friction=float(boundary_cfg.get("friction", domain_cfg.get("friction", 0.05))),
            boundary_eps=(
                float(boundary_cfg.get("eps"))
                if boundary_cfg.get("eps", None) is not None
                else (float(domain_cfg.get("eps")) if domain_cfg.get("eps", None) is not None else None)
            ),
            periodic_axes=periodic_axes,
        )

        export_cfg = scene.get("export", {})
        csv_cfg = export_cfg.get("csv", {})
        csv_enabled = bool(csv_cfg.get("enable", False))
        csv_every = int(csv_cfg.get("every", 10))
        csv_dir = Path(csv_cfg.get("dir", "out/csv"))

        vtk_cfg = export_cfg.get("vtk", {})
        vtk_enabled = bool(vtk_cfg.get("enable", False))
        vtk_every = int(vtk_cfg.get("every", 10))
        vtk_dir = Path(vtk_cfg.get("dir", "out/vtk"))

        if csv_enabled:
            export_particles_csv(csv_dir / "particles_step_0000.csv", state)
        if vtk_enabled:
            export_particles_vtk_legacy(vtk_dir / "particles_step_0000.vtk", state)

        # ---------------------------------------------------------------------
        # Structured results export (CSV + XLSX) without parsing stdout.
        #
        # Scene format (new):
        #   "export": {
        #       "results": { "enable": true, "out_dir": "out", "base_name": "run",
        #                    "formats": ["csv","xlsx"], "every": 0 }
        #   }
        #
        # Backwards-compatible convenience:
        # - If "export" contains "enable", treat it as the results config.
        # ---------------------------------------------------------------------
        results_cfg = export_cfg.get("results", export_cfg if "enable" in export_cfg else {})
        results_enabled = bool(results_cfg.get("enable", False))
        results_every = int(results_cfg.get("every", 0))
        results_out_dir = str(results_cfg.get("out_dir", "out"))
        results_base_name = str(results_cfg.get("base_name", "results"))
        results_formats = tuple(results_cfg.get("formats", ["csv", "xlsx"]))

        results_logger: ResultsLogger | None = None
        if results_enabled:
            results_logger = ResultsLogger(
                meta={
                    "scene_file": str(scene_path),
                    "solver_type": solver_type,
                    "solver_cfg": solver_cfg,
                    "n_particles": int(state.n),
                    "dim": int(dim),
                    "dt_mode": str(time_cfg.get("mode", "cfl")),
                    "dt_min": float(time_cfg.get("dt_min", 1e-5)),
                    "dt_max": float(time_cfg.get("dt_max", 5e-4)),
                    "dt_fixed": float(time_cfg.get("dt_fixed", 5e-4)),
                    "cfl": float(time_cfg.get("cfl", 0.4)),
                }
            )

        # ---------------------------------------------------------------------
        # Flow Metrics Setup
        # ---------------------------------------------------------------------
        metrics_cfg = scene.get("metrics", {})
        flow_metrics: FlowMetrics | None = None
        if metrics_cfg.get("enable", False):
            base_name = results_cfg.get("base_name", scene_path.stem)
            flow_metrics = FlowMetrics(
                output_dir=results_out_dir,
                scene_name=base_name,
                config=metrics_cfg
            )
            
        # ---------------------------------------------------------------------
        # Boundary Manager Setup
        # ---------------------------------------------------------------------
        bm = BoundaryManager()
        scene_boundaries = scene.get("boundaries", [])
        for b_cfg in scene_boundaries:
            b_type = b_cfg.get("type", "").lower()
            if b_type == "wall":
                bm.add_boundary(WallBoundary(
                    domain_min=b_cfg.get("min"),
                    domain_max=b_cfg.get("max"),
                    slip_mode=b_cfg.get("slip_mode", "no-slip"),
                    restitution=b_cfg.get("restitution", 0.0)
                ))
            elif b_type == "inflow":
                bm.add_boundary(InflowBoundary(
                    region_min=b_cfg["min"],
                    region_max=b_cfg["max"],
                    velocity=b_cfg["velocity"],
                    spacing=spacing
                ))
            elif b_type == "outflow":
                bm.add_boundary(OutflowBoundary(
                    region_min=b_cfg["min"],
                    region_max=b_cfg["max"]
                ))
                
        # Fallback to legacy implicit domain bounds if no specific boundaries found
        # but the scene defines `domain.min` and `domain.max` (and it's not a block-only scene).
        if not scene_boundaries and cfg.domain_min is not None and cfg.domain_max is not None:
            bm.add_boundary(WallBoundary(
                domain_min=list(cfg.domain_min),
                domain_max=list(cfg.domain_max),
                slip_mode="no-slip",
                restitution=cfg.boundary_restitution
            ))

        t = 0.0
        prev_dt = cfg.dt_fixed
        for s in range(steps):
            bm.pre_step(state, prev_dt)

            dt = step_simulation(
                state=state,
                cfg=cfg,
                particle_size=spacing,
                solver_cfg_dict=solver_cfg,
                step_idx=s + 1,
            )
            bm.apply_walls(state, cfg)
            bm.post_step(state)
            
            t += float(dt)
            prev_dt = dt

            ns = SpatialHash(
                support_radius=h,
                dim=dim,
                periodic_min=cfg.domain_min,
                periodic_max=cfg.domain_max,
                periodic_axes=cfg.periodic_axes,
            )
            ns.build(state.pos)
            diag = compute_step_diagnostics(step=s + 1, dt=dt, state=state, rho0=rho0, neighbor_search=ns)

            if results_logger is not None:
                results_logger.log_step(
                    StepMetrics(
                        step=int(diag.step),
                        t=float(t),
                        dt=float(diag.dt),
                        vmax=float(diag.v_max),
                        rho_min=float(diag.rho_min),
                        rho_avg=float(diag.rho_mean),
                        rho_max=float(diag.rho_max),
                        err_avg_pct=float(100.0 * diag.rho_rel_err_mean),
                        p_min=float(diag.p_min),
                        p_avg=float(diag.p_mean),
                        p_max=float(diag.p_max),
                        neigh_min=int(diag.neigh_min),
                        neigh_avg=float(diag.neigh_mean),
                        neigh_max=int(diag.neigh_max),
                    )
                )

            if flow_metrics is not None and ((s + 1) % metrics_cfg.get("every", 10) == 0):
                flow_metrics.log_step(s + 1, t, dt, state)

            if (s == 0) or ((s + 1) % max(1, log_every) == 0):
                print(
                    f"[STEP {diag.step:04d}] dt={diag.dt:.3e} "
                    f"|v|max={diag.v_max:.3e} "
                    f"rho(min/avg/max)={diag.rho_min:.2f}/{diag.rho_mean:.2f}/{diag.rho_max:.2f} "
                    f"err% (avg)={100.0 * diag.rho_rel_err_mean:.2f} "
                    f"p(min/avg/max)={diag.p_min:.2f}/{diag.p_mean:.2f}/{diag.p_max:.2f} "
                    f"neigh(min/avg/max)={diag.neigh_min}/{diag.neigh_mean:.1f}/{diag.neigh_max}"
                )

            # Optional: v_x(y) profile sampling (debug-only).
            vxprof_cfg = solver_cfg.get("debug_vx_profile", {})
            if bool(vxprof_cfg.get("enable", False)):
                every = int(vxprof_cfg.get("every", 10))
                bins = int(vxprof_cfg.get("bins", 8))
                x_window_raw = vxprof_cfg.get("x_window", None)
                x_window = float(x_window_raw) if x_window_raw is not None else 0.0
                if every > 0 and (s + 1) % every == 0 and cfg.domain_min is not None and cfg.domain_max is not None:
                    fluid_ids = state.fluid_indices
                    if fluid_ids.size:
                        pos = state.pos[fluid_ids]
                        vel = state.vel[fluid_ids]
                        if x_window > 0.0:
                            xmid = 0.5 * float(cfg.domain_min[0] + cfg.domain_max[0])
                            xmin = xmid - 0.5 * x_window
                            xmax = xmid + 0.5 * x_window
                            sel = (pos[:, 0] >= xmin) & (pos[:, 0] <= xmax)
                            pos = pos[sel]
                            vel = vel[sel]
                        if pos.size and bins > 0:
                            y0 = float(cfg.domain_min[1])
                            y1 = float(cfg.domain_max[1])
                            edges = np.linspace(y0, y1, bins + 1, dtype=np.float64)
                            means: list[float] = []
                            vmaxs: list[float] = []
                            counts: list[int] = []
                            for b in range(bins):
                                m = (pos[:, 1] >= edges[b]) & (pos[:, 1] < edges[b + 1])
                                if np.any(m):
                                    vx = vel[m, 0]
                                    means.append(float(np.mean(vx)))
                                    vmaxs.append(float(np.max(np.abs(vx))))
                                    counts.append(int(np.count_nonzero(m)))
                                else:
                                    means.append(0.0)
                                    vmaxs.append(0.0)
                                    counts.append(0)
                            parts = []
                            for b in range(bins):
                                parts.append(f"b{b}:n={counts[b]} mean={means[b]:.3e} vmax={vmaxs[b]:.3e}")
                            scope = "all_x" if x_window <= 0.0 else f"xwin={x_window:.3f}"
                            print(f"[VXPROF] step={s+1} bins={bins} {scope} " + " ".join(parts))
                            if results_logger is not None:
                                results_logger.log_vxprof(
                                    VxProfileMetrics(
                                        step=int(s + 1),
                                        bins=int(bins),
                                        x_window=float(x_window),
                                        vx_mean=[float(x) for x in means],
                                    )
                                )

            if csv_enabled and ((s + 1) % max(1, csv_every) == 0):
                export_particles_csv(csv_dir / f"particles_step_{diag.step:04d}.csv", state)

            if vtk_enabled and ((s + 1) % max(1, vtk_every) == 0):
                export_particles_vtk_legacy(vtk_dir / f"particles_step_{diag.step:04d}.vtk", state)

            if results_logger is not None and results_every > 0 and ((s + 1) % results_every == 0):
                results_logger.export(results_out_dir, base_name=results_base_name, formats=results_formats)

        if results_logger is not None:
            results_logger.export(results_out_dir, base_name=results_base_name, formats=results_formats)

    print("[BOOT] done")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())


