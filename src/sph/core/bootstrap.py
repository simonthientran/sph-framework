"""CLI bootstrap entrypoint for SPH scenes."""

from __future__ import annotations

import json
import signal
import sys
from pathlib import Path

import numpy as np

from sph.boundaries import BoundaryManager, InflowBoundary, OutflowBoundary, WallBoundary
from sph.core.diagnostics import compute_step_diagnostics
from sph.core.simulator import SimConfig, step_simulation
from sph.core.state_builder import build_scene_state
from sph.diagnostics.flow_metrics import FlowMetrics
from sph.diagnostics.poiseuille import PoiseuilleDiagnostics, build_poiseuille_config
from sph.diagnostics.vx_profile import VxProfileDiagnostics, build_vx_profile_config
from sph.io.csv_export import export_particles_csv
from sph.io.results_export import ResultsLogger, StepMetrics, VxProfileMetrics
from sph.io.vtk_export import export_particles_vtk_legacy
from sph.neighbors.spatial_hash import SpatialHash
from sph.solver.pcisph import get_last_step_stats as get_pcisph_last_step_stats


def _build_boundary_manager(scene: dict, spacing: float, rho0: float) -> BoundaryManager:
    manager = BoundaryManager()
    for bc in scene.get("boundaries", []):
        kind = str(bc.get("type", "")).lower()
        if kind == "wall":
            manager.add_boundary(
                WallBoundary(
                    domain_min=bc["min"],
                    domain_max=bc["max"],
                    slip_mode=str(bc.get("slip_mode", "no-slip")),
                    restitution=float(bc.get("restitution", 0.0)),
                    eps=float(bc.get("eps", 1e-6)),
                    faces=bc.get("faces"),
                )
            )
        elif kind == "inflow":
            manager.add_boundary(
                InflowBoundary(
                    region_min=bc["min"],
                    region_max=bc["max"],
                    velocity=bc.get("velocity", [0.0, 0.0]),
                    spacing=float(bc.get("spacing", spacing)),
                    refill=bool(bc.get("refill", True)),
                    density_value=float(bc.get("rho", rho0)),
                )
            )
        elif kind == "outflow":
            manager.add_boundary(
                OutflowBoundary(
                    region_min=bc["min"],
                    region_max=bc["max"],
                    sponge_strength=float(bc.get("sponge_strength", 0.15)),
                )
            )
    return manager


def main() -> int:
    try:
        try:
            signal.signal(signal.SIGPIPE, signal.SIG_DFL)
        except Exception:
            pass

        if len(sys.argv) < 2:
            print("Usage: python -m sph.core.bootstrap <scene.json>")
            return 2
        scene_path = Path(sys.argv[1]).resolve()
        if not scene_path.exists():
            print("[ERROR] scene file not found")
            return 1

        scene = json.loads(scene_path.read_text(encoding="utf-8"))
        solver_cfg = scene.get("solver", {"type": "wcsph"})
        solver_type = str(solver_cfg.get("type", "wcsph")).lower()
        print(f"[BOOT] solver={solver_type} cfg={json.dumps(solver_cfg, sort_keys=True)}")

        state = build_scene_state(scene)
        dim = int(state.dim)
        spacing = float(scene["fluid"]["spacing"])
        h = float(scene["neighbors"]["support_radius"])
        rho0 = float(scene["material"]["rho0"])

        forces_cfg = scene.get("forces", {})
        gravity = np.array(forces_cfg.get("gravity", [0.0, -9.81])[:dim], dtype=np.float64)
        body_force = np.array(forces_cfg.get("body_force", [0.0] * dim)[:dim], dtype=np.float64)
        g = gravity + body_force

        time_cfg = scene.get("time", {})
        use_cfl = str(time_cfg.get("mode", "cfl")).lower() == "cfl"
        steps = int(time_cfg.get("steps", 50))
        log_every = int(time_cfg.get("log_every", 10))

        domain_cfg = scene.get("domain", {})
        boundary_cfg = scene.get("boundary", {})
        domain_min = None
        domain_max = None
        if "min" in domain_cfg and "max" in domain_cfg:
            domain_min = np.asarray(domain_cfg["min"], dtype=np.float64)
            domain_max = np.asarray(domain_cfg["max"], dtype=np.float64)

        axes_map = {"x": 0, "y": 1, "z": 2}
        periodic_axes = tuple(axes_map[str(a).lower()] for a in domain_cfg.get("periodic_axes", []))

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

        scene_name = scene_path.stem
        metrics_cfg = scene.get("metrics", {})
        metrics_enabled = bool(metrics_cfg.get("enable", False))
        metrics_every = int(metrics_cfg.get("every", 10))
        metrics_logger: FlowMetrics | None = None
        if metrics_enabled:
            metrics_logger = FlowMetrics(
                output_dir=str(metrics_cfg.get("out_dir", "out")),
                scene_name=scene_name,
                config=metrics_cfg,
            )
        poiseuille_cfg = build_poiseuille_config(scene)
        poiseuille_logger: PoiseuilleDiagnostics | None = None
        poiseuille_every = int(scene.get("poiseuille_diagnostics", {}).get("every", 25))
        if poiseuille_cfg is not None:
            poiseuille_out = Path(scene.get("poiseuille_diagnostics", {}).get("out_file", f"out/{scene_name}/poiseuille_profile.csv"))
            poiseuille_logger = PoiseuilleDiagnostics(out_file=poiseuille_out, cfg=poiseuille_cfg)
        vxprof_cfg = build_vx_profile_config(
            scene=scene,
            solver_cfg=solver_cfg,
            support_radius=h,
            domain_min=domain_min,
            domain_max=domain_max,
            scene_name=scene_name,
        )
        vxprof_logger: VxProfileDiagnostics | None = None
        if vxprof_cfg is not None:
            vxprof_logger = VxProfileDiagnostics(cfg=vxprof_cfg)

        boundary_manager = _build_boundary_manager(scene, spacing=spacing, rho0=rho0)
        use_boundary_manager = len(boundary_manager.boundaries) > 0

        t = 0.0
        for s in range(steps):
            step = s + 1
            if use_boundary_manager:
                # Boundary lifecycle: pre-step -> solver -> wall response -> post-step.
                boundary_manager.pre_step(state, cfg.dt_fixed if not cfg.use_cfl else cfg.dt_max)

            dt = step_simulation(
                state=state,
                cfg=cfg,
                particle_size=spacing,
                solver_cfg_dict=solver_cfg,
                step_idx=step,
                enforce_domain_constraints=not use_boundary_manager,
            )
            pcisph_stats = get_pcisph_last_step_stats(state) if solver_type == "pcisph" else None

            if use_boundary_manager:
                boundary_manager.apply_walls(state, cfg, debug=bool(solver_cfg.get("debug", False)))
                boundary_manager.post_step(state)

            if not np.isfinite(dt) or dt <= 0.0:
                raise RuntimeError(f"non-positive or invalid dt at step {step}: {dt}")
            t += float(dt)

            ns = SpatialHash(
                support_radius=h,
                dim=dim,
                periodic_min=cfg.domain_min,
                periodic_max=cfg.domain_max,
                periodic_axes=cfg.periodic_axes,
            )
            ns.build(state.pos)
            diag = compute_step_diagnostics(step=step, dt=dt, state=state, rho0=rho0, neighbor_search=ns)

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

            if metrics_logger is not None and metrics_every > 0 and step % metrics_every == 0:
                metrics_logger.log_step(step=step, time_value=t, dt=float(dt), state=state)
            if poiseuille_logger is not None and poiseuille_every > 0 and step % poiseuille_every == 0:
                l2_rel = poiseuille_logger.sample_and_log(step=step, time_value=t, state=state)
                print(f"[POISEUILLE] step={step} l2_rel={l2_rel:.3e}")

            vx_sample = vxprof_logger.sample(step=step, state=state) if vxprof_logger is not None else None
            if vx_sample is not None and results_logger is not None:
                x_window = float(vxprof_cfg.x_slice_width) if (vxprof_cfg is not None and vxprof_cfg.use_x_slice) else 0.0
                results_logger.log_vxprof(
                    VxProfileMetrics(
                        step=int(vx_sample.step),
                        bins=int(len(vx_sample.mean_vx)),
                        x_window=x_window,
                        vx_mean=[float(v) for v in vx_sample.mean_vx],
                    )
                )

            if step == 1 or step % max(1, log_every) == 0:
                pcisph_part = ""
                if pcisph_stats is not None:
                    pcisph_part = (
                        f" | pcisph(iters={int(pcisph_stats.get('iters_used', 0))}/"
                        f"{int(pcisph_stats.get('max_iters', 0))}"
                        f", err_avg={float(pcisph_stats.get('rho_err_avg', float('nan'))):.2e}"
                        f", err_max={float(pcisph_stats.get('rho_err_max', float('nan'))):.2e})"
                    )
                print(
                    f"[STEP {diag.step:04d}] dt={diag.dt:.3e} "
                    f"|v|max={diag.v_max:.3e} "
                    f"rho(min/avg/max)={diag.rho_min:.2f}/{diag.rho_mean:.2f}/{diag.rho_max:.2f} "
                    f"err% (avg)={100.0 * diag.rho_rel_err_mean:.2f} "
                    f"p(min/avg/max)={diag.p_min:.2f}/{diag.p_mean:.2f}/{diag.p_max:.2f} "
                    f"neigh(min/avg/max)={diag.neigh_min}/{diag.neigh_mean:.1f}/{diag.neigh_max}"
                    f"{pcisph_part}"
                )

            if csv_enabled and step % max(1, csv_every) == 0:
                export_particles_csv(csv_dir / f"particles_step_{step:04d}.csv", state)
            if vtk_enabled and step % max(1, vtk_every) == 0:
                export_particles_vtk_legacy(vtk_dir / f"particles_step_{step:04d}.vtk", state)
            if results_logger is not None and results_every > 0 and step % results_every == 0:
                results_logger.export(results_out_dir, base_name=results_base_name, formats=results_formats)

        if results_logger is not None:
            results_logger.export(results_out_dir, base_name=results_base_name, formats=results_formats)
        print("[BOOT] done")
        return 0
    except BrokenPipeError:
        return 0
    except Exception as exc:
        print(f"[BOOT][ERROR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())