[1mdiff --git a/src/sph/core/bootstrap.py b/src/sph/core/bootstrap.py[m
[1mindex 5f95c20..bb02c85 100644[m
[1m--- a/src/sph/core/bootstrap.py[m
[1m+++ b/src/sph/core/bootstrap.py[m
[36m@@ -41,147 +41,156 @@[m [mfrom sph.neighbors.spatial_hash import SpatialHash[m
 [m
 [m
 def main() -> int:[m
[31m-    print("[BOOT] NEW BOOTSTRAP ACTIVE ✅")[m
[31m-[m
[31m-    if len(sys.argv) < 2:[m
[31m-        print("Usage: python -m sph.core.bootstrap <scene.json>")[m
[31m-        return 2[m
[31m-[m
[31m-    scene_path = Path(sys.argv[1]).resolve()[m
[31m-    if not scene_path.exists():[m
[31m-        print("[ERROR] scene file not found")[m
[31m-        return 1[m
[31m-[m
[31m-    with scene_path.open("r", encoding="utf-8") as f:[m
[31m-        scene = json.load(f)[m
[31m-[m
[31m-    solver_cfg = scene.get("solver", {"type": "wcsph"})[m
[31m-    solver_type = str(solver_cfg.get("type", "wcsph")).lower()[m
[31m-    # Print solver params for reproducibility/observability (no physics impact).[m
[31m-    print(f"[BOOT] solver={solver_type} cfg={json.dumps(solver_cfg, sort_keys=True)}")[m
[31m-[m
[31m-    # -------------------------------------------------------------------------[m
[31m-    # Build state (fluid + boundary particles)[m
[31m-    # Boundary particles are static; fluid particles are integrated each step.[m
[31m-    # This matches the particle-based boundary handling idea around Eq. (83)/(84).[m
[31m-    # -------------------------------------------------------------------------[m
[31m-    state = build_scene_state(scene)[m
[31m-[m
[31m-    dim = int(state.dim)[m
[31m-    spacing = float(scene["fluid"]["spacing"])[m
[31m-    h = float(scene["neighbors"]["support_radius"])[m
[31m-    rho0 = float(scene["material"]["rho0"])[m
[31m-[m
[31m-    # Gravity from scene (fallback: -9.81 in y for 2D)[m
[31m-    g = np.array(scene.get("forces", {}).get("gravity", [0.0, -9.81])[:dim], dtype=np.float64)[m
[31m-[m
[31m-    # Time settings (dt is selected inside the solver according to Eq. (33) if enabled)[m
[31m-    time_cfg = scene.get("time", {})[m
[31m-    use_cfl = (time_cfg.get("mode", "cfl") == "cfl")[m
[31m-    steps = int(time_cfg.get("steps", 50))[m
[31m-    log_every = int(time_cfg.get("log_every", 10))[m
[31m-[m
[31m-    # Domain / Boundary constraints[m
[31m-    domain_cfg = scene.get("domain", {})[m
[31m-    boundary_cfg = scene.get("boundary", {})  # optional (preferred for collision params)[m
[31m-    domain_min = None[m
[31m-    domain_max = None[m
[31m-    if "min" in domain_cfg and "max" in domain_cfg:[m
[31m-        domain_min = np.array(domain_cfg["min"], dtype=np.float64)[m
[31m-        domain_max = np.array(domain_cfg["max"], dtype=np.float64)[m
[31m-[m
[31m-    cfg = SimConfig([m
[31m-        support_radius=h,[m
[31m-        rho0=rho0,[m
[31m-        eos_k=float(scene.get("material", {}).get("eos", {}).get("k", 500.0)),[m
[31m-        g=g,[m
[31m-        cfl_lambda=float(time_cfg.get("cfl", 0.4)),[m
[31m-        dt_min=float(time_cfg.get("dt_min", 1e-5)),[m
[31m-        dt_max=float(time_cfg.get("dt_max", 5e-4)),[m
[31m-        dt_fixed=float(time_cfg.get("dt_fixed", 5e-4)),[m
[31m-        use_cfl=bool(use_cfl),[m
[31m-        # viscosity fields are optional in SimConfig and default to disabled[m
[31m-        enable_viscosity=False,[m
[31m-        kinematic_viscosity=0.0,[m
[31m-        # domain collision[m
[31m-        domain_min=domain_min,[m
[31m-        domain_max=domain_max,[m
[31m-        boundary_restitution=float(boundary_cfg.get("restitution", domain_cfg.get("restitution", 0.0))),[m
[31m-        boundary_friction=float(boundary_cfg.get("friction", domain_cfg.get("friction", 0.05))),[m
[31m-        boundary_eps=([m
[31m-            float(boundary_cfg.get("eps"))[m
[31m-            if boundary_cfg.get("eps", None) is not None[m
[31m-            else (float(domain_cfg.get("eps")) if domain_cfg.get("eps", None) is not None else None)[m
[31m-        ),[m
[31m-    )[m
[31m-[m
[31m-    # -------------------------------------------------------------------------[m
[31m-    # Optional exports controlled by scene:[m
[31m-    #   export.csv.enable/every/dir[m
[31m-    #   export.vtk.enable/every/dir[m
[31m-    # -------------------------------------------------------------------------[m
[31m-    export_cfg = scene.get("export", {})[m
[31m-[m
[31m-    csv_cfg = export_cfg.get("csv", {})[m
[31m-    csv_enabled = bool(csv_cfg.get("enable", False))[m
[31m-    csv_every = int(csv_cfg.get("every", 10))[m
[31m-    csv_dir = Path(csv_cfg.get("dir", "out/csv"))[m
[31m-[m
[31m-    vtk_cfg = export_cfg.get("vtk", {})[m
[31m-    vtk_enabled = bool(vtk_cfg.get("enable", False))[m
[31m-    vtk_every = int(vtk_cfg.get("every", 10))[m
[31m-    vtk_dir = Path(vtk_cfg.get("dir", "out/vtk"))[m
[31m-[m
[31m-    # Export step 0000 if enabled (pre-step snapshot)[m
[31m-    if csv_enabled:[m
[31m-        export_particles_csv(csv_dir / "particles_step_0000.csv", state)[m
[31m-    if vtk_enabled:[m
[31m-        export_particles_vtk_legacy(vtk_dir / "particles_step_0000.vtk", state)[m
[31m-[m
[31m-    # -------------------------------------------------------------------------[m
[31m-    # Main simulation loop[m
[31m-    #[m
[31m-    # Ordering matches the selected solver:[m
[31m-    # - WCSPH uses Algorithm 1 structure and boundary handling via Eq. (83)/(84).[m
[31m-    # - PCISPH uses Eq. (51),(53),(57),(58),(59),(60) (see sph/solver/pcisph.py).[m
[31m-    #[m
[31m-    # This loop only:[m
[31m-    # - dispatches to the existing solver step[m
[31m-    # - builds a neighbor search for diagnostics[m
[31m-    # - prints/export snapshots for observability[m
[31m-    # -------------------------------------------------------------------------[m
[31m-    for s in range(steps):[m
[31m-        dt = step_simulation([m
[31m-            state=state,[m
[31m-            cfg=cfg,[m
[31m-            particle_size=spacing,[m
[31m-            solver_cfg_dict=solver_cfg,[m
[31m-            step_idx=s + 1,[m
[32m+[m[32m    try:[m
[32m+[m[32m        print("[BOOT] NEW BOOTSTRAP ACTIVE ✅")[m
[32m+[m
[32m+[m[32m        if len(sys.argv) < 2:[m
[32m+[m[32m            print("Usage: python -m sph.core.bootstrap <scene.json>")[m
[32m+[m[32m            return 2[m
[32m+[m
[32m+[m[32m        scene_path = Path(sys.argv[1]).resolve()[m
[32m+[m[32m        if not scene_path.exists():[m
[32m+[m[32m            print("[ERROR] scene file not found")[m
[32m+[m[32m            return 1[m
[32m+[m
[32m+[m[32m        with scene_path.open("r", encoding="utf-8") as f:[m
[32m+[m[32m            scene = json.load(f)[m
[32m+[m
[32m+[m[32m        solver_cfg = scene.get("solver", {"type": "wcsph"})[m
[32m+[m[32m        solver_type = str(solver_cfg.get("type", "wcsph")).lower()[m
[32m+[m[32m        # Print solver params for reproducibility/observability (no physics impact).[m
[32m+[m[32m        print(f"[BOOT] solver={solver_type} cfg={json.dumps(solver_cfg, sort_keys=True)}")[m
[32m+[m
[32m+[m[32m        # -------------------------------------------------------------------------[m
[32m+[m[32m        # Build state (fluid + boundary particles)[m
[32m+[m[32m        # -------------------------------------------------------------------------[m
[32m+[m[32m        state = build_scene_state(scene)[m
[32m+[m
[32m+[m[32m        dim = int(state.dim)[m
[32m+[m[32m        spacing = float(scene["fluid"]["spacing"])[m
[32m+[m[32m        h = float(scene["neighbors"]["support_radius"])[m
[32m+[m[32m        rho0 = float(scene["material"]["rho0"])[m
[32m+[m
[32m+[m[32m        # External body forces (gravity + optional constant body_force).[m
[32m+[m[32m        forces_cfg = scene.get("forces", {})[m
[32m+[m[32m        gravity = np.array(forces_cfg.get("gravity", [0.0, -9.81])[:dim], dtype=np.float64)[m
[32m+[m[32m        body_force = np.array(forces_cfg.get("body_force", [0.0] * dim)[:dim], dtype=np.float64)[m
[32m+[m[32m        g = gravity + body_force[m
[32m+[m
[32m+[m[32m        time_cfg = scene.get("time", {})[m
[32m+[m[32m        use_cfl = (time_cfg.get("mode", "cfl") == "cfl")[m
[32m+[m[32m        steps = int(time_cfg.get("steps", 50))[m
[32m+[m[32m        log_every = int(time_cfg.get("log_every", 10))[m
[32m+[m
[32m+[m[32m        domain_cfg = scene.get("domain", {})[m
[32m+[m[32m        boundary_cfg = scene.get("boundary", {})  # optional (preferred for collision params)[m
[32m+[m[32m        domain_min = None[m
[32m+[m[32m        domain_max = None[m
[32m+[m[32m        if "min" in domain_cfg and "max" in domain_cfg:[m
[32m+[m[32m            domain_min = np.array(domain_cfg["min"], dtype=np.float64)[m
[32m+[m[32m            domain_max = np.array(domain_cfg["max"], dtype=np.float64)[m
[32m+[m
[32m+[m[32m        visc_cfg = scene.get("material", {}).get("viscosity", {})[m
[32m+[m[32m        cfg = SimConfig([m
[32m+[m[32m            support_radius=h,[m
[32m+[m[32m            rho0=rho0,[m
[32m+[m[32m            eos_k=float(scene.get("material", {}).get("eos", {}).get("k", 500.0)),[m
[32m+[m[32m            g=g,[m
[32m+[m[32m            cfl_lambda=float(time_cfg.get("cfl", 0.4)),[m
[32m+[m[32m            dt_min=float(time_cfg.get("dt_min", 1e-5)),[m
[32m+[m[32m            dt_max=float(time_cfg.get("dt_max", 5e-4)),[m
[32m+[m[32m            dt_fixed=float(time_cfg.get("dt_fixed", 5e-4)),[m
[32m+[m[32m            use_cfl=bool(use_cfl),[m
[32m+[m[32m            enable_viscosity=bool(visc_cfg.get("enable", False)),[m
[32m+[m[32m            kinematic_viscosity=float(visc_cfg.get("nu", 0.0)),[m
[32m+[m[32m            domain_min=domain_min,[m
[32m+[m[32m            domain_max=domain_max,[m
[32m+[m[32m            boundary_restitution=float(boundary_cfg.get("restitution", domain_cfg.get("restitution", 0.0))),[m
[32m+[m[32m            boundary_friction=float(boundary_cfg.get("friction", domain_cfg.get("friction", 0.05))),[m
[32m+[m[32m            boundary_eps=([m
[32m+[m[32m                float(boundary_cfg.get("eps"))[m
[32m+[m[32m                if boundary_cfg.get("eps", None) is not None[m
[32m+[m[32m                else (float(domain_cfg.get("eps")) if domain_cfg.get("eps", None) is not None else None)[m
[32m+[m[32m            ),[m
         )[m
 [m
[31m-        # Diagnostics neighbor search on current positions (read-only)[m
[31m-        ns = SpatialHash(support_radius=h, dim=dim)[m
[31m-        ns.build(state.pos)[m
[31m-        diag = compute_step_diagnostics(step=s + 1, dt=dt, state=state, rho0=rho0, neighbor_search=ns)[m
[31m-[m
[31m-        if (s == 0) or ((s + 1) % max(1, log_every) == 0):[m
[31m-            print([m
[31m-                f"[STEP {diag.step:04d}] dt={diag.dt:.3e} "[m
[31m-                f"|v|max={diag.v_max:.3e} "[m
[31m-                f"rho(min/avg/max)={diag.rho_min:.2f}/{diag.rho_mean:.2f}/{diag.rho_max:.2f} "[m
[31m-                f"err% (avg)={100.0 * diag.rho_rel_err_mean:.2f} "[m
[31m-                f"p(min/avg/max)={diag.p_min:.2f}/{diag.p_mean:.2f}/{diag.p_max:.2f} "[m
[31m-                f"neigh(min/avg/max)={diag.neigh_min}/{diag.neigh_mean:.1f}/{diag.neigh_max}"[m
[32m+[m[32m        export_cfg = scene.get("export", {})[m
[32m+[m[32m        csv_cfg = export_cfg.get("csv", {})[m
[32m+[m[32m        csv_enabled = bool(csv_cfg.get("enable", False))[m
[32m+[m[32m        csv_every = int(csv_cfg.get("every", 10))[m
[32m+[m[32m        csv_dir = Path(csv_cfg.get("dir", "out/csv"))[m
[32m+[m
[32m+[m[32m        vtk_cfg = export_cfg.get("vtk", {})[m
[32m+[m[32m        vtk_enabled = bool(vtk_cfg.get("enable", False))[m
[32m+[m[32m        vtk_every = int(vtk_cfg.get("every", 10))[m
[32m+[m[32m        vtk_dir = Path(vtk_cfg.get("dir", "out/vtk"))[m
[32m+[m
[32m+[m[32m        if csv_enabled:[m
[32m+[m[32m            export_particles_csv(csv_dir / "particles_step_0000.csv", state)[m
[32m+[m[32m        if vtk_enabled:[m
[32m+[m[32m            export_particles_vtk_legacy(vtk_dir / "particles_step_0000.vtk", state)[m
[32m+[m
[32m+[m[32m        for s in range(steps):[m
[32m+[m[32m            dt = step_simulation([m
[32m+[m[32m                state=state,[m
[32m+[m[32m                cfg=cfg,[m
[32m+[m[32m                particle_size=spacing,[m
[32m+[m[32m                solver_cfg_dict=solver_cfg,[m
[32m+[m[32m                step_idx=s + 1,[m
             )[m
 [m
[31m-        if csv_enabled and ((s + 1) % max(1, csv_every) == 0):[m
[31m-            export_particles_csv(csv_dir / f"particles_step_{diag.step:04d}.csv", state)[m
[31m-[m
[31m-        if vtk_enabled and ((s + 1) % max(1, vtk_every) == 0):[m
[31m-            export_particles_vtk_legacy(vtk_dir / f"particles_step_{diag.step:04d}.vtk", state)[m
[31m-[m
[31m-    print("[BOOT] done")[m
[31m-    return 0[m
[32m+[m[32m            ns = SpatialHash(support_radius=h, dim=dim)[m
[32m+[m[32m            ns.build(state.pos)[m
[32m+[m[32m            diag = compute_step_diagnostics(step=s + 1, dt=dt, state=state, rho0=rho0, neighbor_search=ns)[m
[32m+[m
[32m+[m[32m            if (s == 0) or ((s + 1) % max(1, log_every) == 0):[m
[32m+[m[32m                print([m
[32m+[m[32m                    f"[STEP {diag.step:04d}] dt={diag.dt:.3e} "[m
[32m+[m[32m                    f"|v|max={diag.v_max:.3e} "[m
[32m+[m[32m                    f"rho(min/avg/max)={diag.rho_min:.2f}/{diag.rho_mean:.2f}/{diag.rho_max:.2f} "[m
[32m+[m[32m                    f"err% (avg)={100.0 * diag.rho_rel_err_mean:.2f} "[m
[32m+[m[32m                    f"p(min/avg/max)={diag.p_min:.2f}/{diag.p_mean:.2f}/{diag.p_max:.2f} "[m
[32m+[m[32m                    f"neigh(min/avg/max)={diag.neigh_min}/{diag.neigh_mean:.1f}/{diag.neigh_max}"[m
[32m+[m[32m                )[m
[32m+[m
[32m+[m[32m            # Optional: v_x(y) profile sampling (debug-only).[m
[32m+[m[32m            vxprof_cfg = solver_cfg.get("debug_vx_profile", {})[m
[32m+[m[32m            if bool(vxprof_cfg.get("enable", False)):[m
[32m+[m[32m                every = int(vxprof_cfg.get("every", 10))[m
[32m+[m[32m                bins = int(vxprof_cfg.get("bins", 8))[m
[32m+[m[32m                x_window = float(vxprof_cfg.get("x_window", 0.1))[m
[32m+[m[32m                if every > 0 and (s + 1) % every == 0 and cfg.domain_min is not None and cfg.domain_max is not None:[m
[32m+[m[32m                    xmid = 0.5 * float(cfg.domain_min[0] + cfg.domain_max[0])[m
[32m+[m[32m                    xmin = xmid - 0.5 * x_window[m
[32m+[m[32m                    xmax = xmid + 0.5 * x_window[m
[32m+[m[32m                    fluid_ids = state.fluid_indices[m
[32m+[m[32m                    if fluid_ids.size:[m
[32m+[m[32m                        pos = state.pos[fluid_ids][m
[32m+[m[32m                        vel = state.vel[fluid_ids][m
[32m+[m[32m                        sel = (pos[:, 0] >= xmin) & (pos[:, 0] <= xmax)[m
[32m+[m[32m                        pos = pos[sel][m
[32m+[m[32m                        vel = vel[sel][m
[32m+[m[32m                        if pos.size and bins > 0:[m
[32m+[m[32m                            y0 = float(cfg.domain_min[1])[m
[32m+[m[32m                            y1 = float(cfg.domain_max[1])[m
[32m+[m[32m                            edges = np.linspace(y0, y1, bins + 1, dtype=np.float64)[m
[32m+[m[32m                            means = [][m
[32m+[m[32m                            for b in range(bins):[m
[32m+[m[32m                                m = (pos[:, 1] >= edges[b]) & (pos[:, 1] < edges[b + 1])[m
[32m+[m[32m                                means.append(float(np.mean(vel[m, 0])) if np.any(m) else float("nan"))[m
[32m+[m[32m                            means_str = ", ".join(f"{v:.3e}" if np.isfinite(v) else "nan" for v in means)[m
[32m+[m[32m                            print(f"[VXPROF] step={s+1} bins={bins} xwin={x_window:.3f} vx_mean={means_str}")[m
[32m+[m
[32m+[m[32m            if csv_enabled and ((s + 1) % max(1, csv_every) == 0):[m
[32m+[m[32m                export_particles_csv(csv_dir / f"particles_step_{diag.step:04d}.csv", state)[m
[32m+[m
[32m+[m[32m            if vtk_enabled and ((s + 1) % max(1, vtk_every) == 0):[m
[32m+[m[32m                export_particles_vtk_legacy(vtk_dir / f"particles_step_{diag.step:04d}.vtk", state)[m
[32m+[m
[32m+[m[32m        print("[BOOT] done")[m
[32m+[m[32m        return 0[m
[32m+[m[32m    except BrokenPipeError:[m
[32m+[m[32m        return 0[m
 [m
 [m
 if __name__ == "__main__":[m
[1mdiff --git a/src/sph/core/state_builder.py b/src/sph/core/state_builder.py[m
[1mindex d228383..4f876c2 100644[m
[1m--- a/src/sph/core/state_builder.py[m
[1m+++ b/src/sph/core/state_builder.py[m
[36m@@ -83,6 +83,100 @@[m [mdef _sample_box_boundary_2d(domain_min: np.ndarray, domain_max: np.ndarray, spac[m
     return all_pts[m
 [m
 [m
[32m+[m[32mdef spawn_boundary_walls_channel([m
[32m+[m[32m    *,[m
[32m+[m[32m    domain_min: np.ndarray,[m
[32m+[m[32m    domain_max: np.ndarray,[m
[32m+[m[32m    spacing: float,[m
[32m+[m[32m    layers: int,[m
[32m+[m[32m    walls: tuple[str, ...] = ("xmin", "xmax", "ymin", "ymax"),[m
[32m+[m[32m) -> np.ndarray:[m
[32m+[m[32m    """[m
[32m+[m[32m    Spawn static boundary particles for a 2D channel/pipe segment.[m
[32m+[m
[32m+[m[32m    This is a *scene builder* utility: it defines geometry only, not physics.[m
[32m+[m[32m    It reuses the same boundary sampling strategy as `_sample_box_boundary_2d`,[m
[32m+[m[32m    but allows selecting which faces are walled.[m
[32m+[m
[32m+[m[32m    Args:[m
[32m+[m[32m        domain_min/domain_max: AABB of the channel domain.[m
[32m+[m[32m        spacing: particle spacing.[m
[32m+[m[32m        layers: number of boundary layers.[m
[32m+[m[32m        walls: subset of {"xmin","xmax","ymin","ymax"} to sample.[m
[32m+[m
[32m+[m[32m    Returns:[m
[32m+[m[32m        (Nb,2) boundary particle positions.[m
[32m+[m[32m    """[m
[32m+[m[32m    walls_set = set(walls)[m
[32m+[m[32m    allowed = {"xmin", "xmax", "ymin", "ymax"}[m
[32m+[m[32m    unknown = walls_set - allowed[m
[32m+[m[32m    if unknown:[m
[32m+[m[32m        raise ValueError(f"unknown channel walls: {sorted(unknown)}")[m
[32m+[m
[32m+[m[32m    pts: list[np.ndarray] = [][m
[32m+[m[32m    for k in range(int(layers)):[m
[32m+[m[32m        off = k * float(spacing)[m
[32m+[m[32m        x0 = float(domain_min[0] + off)[m
[32m+[m[32m        x1 = float(domain_max[0] - off)[m
[32m+[m[32m        y0 = float(domain_min[1] + off)[m
[32m+[m[32m        y1 = float(domain_max[1] - off)[m
[32m+[m
[32m+[m[32m        xs = np.arange(x0, x1 + 1e-12, spacing, dtype=np.float64)[m
[32m+[m[32m        ys = np.arange(y0, y1 + 1e-12, spacing, dtype=np.float64)[m
[32m+[m
[32m+[m[32m        if "ymin" in walls_set:[m
[32m+[m[32m            pts.append(np.stack([xs, np.full_like(xs, y0)], axis=1))[m
[32m+[m[32m        if "ymax" in walls_set:[m
[32m+[m[32m            pts.append(np.stack([xs, np.full_like(xs, y1)], axis=1))[m
[32m+[m
[32m+[m[32m        if len(ys) > 2:[m
[32m+[m[32m            ys_inner = ys[1:-1][m
[32m+[m[32m            if "xmin" in walls_set:[m
[32m+[m[32m                pts.append(np.stack([np.full_like(ys_inner, x0), ys_inner], axis=1))[m
[32m+[m[32m            if "xmax" in walls_set:[m
[32m+[m[32m                pts.append(np.stack([np.full_like(ys_inner, x1), ys_inner], axis=1))[m
[32m+[m
[32m+[m[32m    if not pts:[m
[32m+[m[32m        return np.zeros((0, 2), dtype=np.float64)[m
[32m+[m[32m    all_pts = np.concatenate(pts, axis=0)[m
[32m+[m[32m    all_pts = np.unique(np.round(all_pts / spacing).astype(np.int64), axis=0).astype(np.float64) * spacing[m
[32m+[m[32m    return all_pts[m
[32m+[m
[32m+[m
[32m+[m[32mdef spawn_fluid_block_in_channel([m
[32m+[m[32m    *,[m
[32m+[m[32m    domain_min: np.ndarray,[m
[32m+[m[32m    domain_max: np.ndarray,[m
[32m+[m[32m    spacing: float,[m
[32m+[m[32m    wall_layers: int,[m
[32m+[m[32m    pad_x: float = 0.0,[m
[32m+[m[32m    pad_y: float = 0.0,[m
[32m+[m[32m) -> np.ndarray:[m
[32m+[m[32m    """[m
[32m+[m[32m    Spawn a uniform fluid particle grid inside a 2D channel, leaving clearance[m
[32m+[m[32m    from the walls.[m
[32m+[m
[32m+[m[32m    This is intended for pipe/channel examples where the geometry is "fill the[m
[32m+[m[32m    channel interior". It is a geometry helper only.[m
[32m+[m
[32m+[m[32m    Args:[m
[32m+[m[32m        domain_min/domain_max: AABB of the channel.[m
[32m+[m[32m        spacing: particle spacing.[m
[32m+[m[32m        wall_layers: boundary layers count; we leave at least wall_layers*spacing[m
[32m+[m[32m            clearance from any wall to avoid initial overlap with boundary particles.[m
[32m+[m[32m        pad_x/pad_y: additional clearance in x/y.[m
[32m+[m
[32m+[m[32m    Returns:[m
[32m+[m[32m        (Nf,2) fluid positions.[m
[32m+[m[32m    """[m
[32m+[m[32m    clearance = float(max(0, int(wall_layers))) * float(spacing)[m
[32m+[m[32m    pmin = np.array([domain_min[0] + clearance + pad_x, domain_min[1] + clearance + pad_y], dtype=np.float64)[m
[32m+[m[32m    pmax = np.array([domain_max[0] - clearance - pad_x, domain_max[1] - clearance - pad_y], dtype=np.float64)[m
[32m+[m[32m    if np.any(pmax <= pmin):[m
[32m+[m[32m        raise ValueError("channel too small for requested wall clearance/padding")[m
[32m+[m[32m    return _grid_points_2d(pmin, pmax, spacing)[m
[32m+[m
[32m+[m
 def build_scene_state(scene: dict) -> ParticleState:[m
     # region agent log[m
     _agent_log([m
[36m@@ -110,19 +204,42 @@[m [mdef build_scene_state(scene: dict) -> ParticleState:[m
 [m
     # --- fluid block[m
     fluid = scene["fluid"][m
[31m-    if fluid["type"] != "block":[m
[32m+[m[32m    ftype = str(fluid["type"]).lower()[m
[32m+[m[32m    if ftype == "block":[m
[32m+[m[32m        fmin = np.array(fluid["min"], dtype=np.float64)[m
[32m+[m[32m        fmax = np.array(fluid["max"], dtype=np.float64)[m
[32m+[m[32m        fluid_pos = _grid_points_2d(fmin, fmax, spacing)[m
[32m+[m[32m    elif ftype == "channel_fill":[m
[32m+[m[32m        # Pipe/channel example: fill channel interior with a uniform grid.[m
[32m+[m[32m        pad = float(fluid.get("padding", 0.0))[m
[32m+[m[32m        fluid_pos = spawn_fluid_block_in_channel([m
[32m+[m[32m            domain_min=domain_min,[m
[32m+[m[32m            domain_max=domain_max,[m
[32m+[m[32m            spacing=spacing,[m
[32m+[m[32m            wall_layers=layers,[m
[32m+[m[32m            pad_x=float(fluid.get("pad_x", pad)),[m
[32m+[m[32m            pad_y=float(fluid.get("pad_y", pad)),[m
[32m+[m[32m        )[m
[32m+[m[32m    else:[m
         raise ValueError(f"unsupported fluid type: {fluid['type']}")[m
 [m
[31m-    fmin = np.array(fluid["min"], dtype=np.float64)[m
[31m-    fmax = np.array(fluid["max"], dtype=np.float64)[m
[31m-[m
[31m-    fluid_pos = _grid_points_2d(fmin, fmax, spacing)[m
[31m-[m
     v0 = np.array(fluid.get("initial_velocity", [0.0, 0.0]), dtype=np.float64)[m
     fluid_vel = np.repeat(v0[None, :], fluid_pos.shape[0], axis=0)[m
 [m
     # --- boundary sampling (static)[m
[31m-    boundary_pos = _sample_box_boundary_2d(domain_min, domain_max, spacing, layers=layers)[m
[32m+[m[32m    domain_cfg = scene.get("domain", {})[m
[32m+[m[32m    walls = tuple(str(w).lower() for w in domain_cfg.get("boundary_walls", ["xmin", "xmax", "ymin", "ymax"]))[m
[32m+[m[32m    if str(domain_cfg.get("type", "box")).lower() in {"channel", "pipe", "box"}:[m
[32m+[m[32m        # `box` remains the default; `boundary_walls` lets channel scenes omit end caps.[m
[32m+[m[32m        boundary_pos = spawn_boundary_walls_channel([m
[32m+[m[32m            domain_min=domain_min,[m
[32m+[m[32m            domain_max=domain_max,[m
[32m+[m[32m            spacing=spacing,[m
[32m+[m[32m            layers=layers,[m
[32m+[m[32m            walls=walls,[m
[32m+[m[32m        )[m
[32m+[m[32m    else:[m
[32m+[m[32m        boundary_pos = _sample_box_boundary_2d(domain_min, domain_max, spacing, layers=layers)[m
     boundary_vel = np.zeros((boundary_pos.shape[0], dim), dtype=np.float64)[m
 [m
     # --- combine[m
