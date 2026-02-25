import json
from pathlib import Path

import numpy as np

from sph.core.simulator import SimConfig, step_simulation
from sph.core.state_builder import build_scene_state


def test_scene_pipe_flow_regression_200_steps_stable_and_profile_trend():
    """
    Mini regression for pipe flow:
    - runs 200 steps without NaNs/Infs
    - |rho_mean - rho0| / rho0 < 1%
    - centerline v_x mean > near-wall v_x mean (very coarse "parabolic" trend)
    """
    scene_path = Path("scenes/examples/pipe_flow_2d.json")
    scene = json.loads(scene_path.read_text(encoding="utf-8"))

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
    use_cfl = (time_cfg.get("mode", "cfl") == "cfl")

    domain_cfg = scene.get("domain", {})
    boundary_cfg = scene.get("boundary", {})
    domain_min = np.array(domain_cfg["min"], dtype=np.float64)
    domain_max = np.array(domain_cfg["max"], dtype=np.float64)
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
        boundary_restitution=float(boundary_cfg.get("restitution", 0.0)),
        boundary_friction=float(boundary_cfg.get("friction", 0.05)),
        boundary_eps=float(boundary_cfg.get("eps")) if boundary_cfg.get("eps", None) is not None else None,
        periodic_axes=periodic_axes,
    )

    solver_cfg = scene.get("solver", {"type": "wcsph"})
    for s in range(200):
        dt = step_simulation(
            state=state,
            cfg=cfg,
            particle_size=spacing,
            solver_cfg_dict=solver_cfg,
            step_idx=s + 1,
        )
        assert np.isfinite(dt)
        assert np.all(np.isfinite(state.pos))
        assert np.all(np.isfinite(state.vel))
        assert np.all(np.isfinite(state.rho))
        assert np.all(np.isfinite(state.p))

        # Guard against explosive speeds (very loose; expected velocities are small here).
        vmax = float(np.max(np.linalg.norm(state.vel[state.fluid_indices], axis=1)))
        assert vmax < 20.0

    rho_mean = float(np.mean(state.rho[state.fluid_indices]))
    assert abs(rho_mean - rho0) / rho0 < 0.01

    # Coarse profile trend: bin by y and compare center bin vs near-wall bin.
    fluid_ids = state.fluid_indices
    pos = state.pos[fluid_ids]
    vel = state.vel[fluid_ids]
    y0 = float(domain_min[1])
    y1 = float(domain_max[1])
    bins = 8
    edges = np.linspace(y0, y1, bins + 1, dtype=np.float64)
    means = np.zeros((bins,), dtype=np.float64)
    counts = np.zeros((bins,), dtype=np.int64)
    for b in range(bins):
        m = (pos[:, 1] >= edges[b]) & (pos[:, 1] < edges[b + 1])
        counts[b] = int(np.count_nonzero(m))
        means[b] = float(np.mean(vel[m, 0])) if counts[b] > 0 else 0.0

    # Compare a middle bin to a near-wall bin that has particles.
    mid = bins // 2
    wall = 0 if counts[0] > 0 else 1
    assert means[mid] > means[wall]


