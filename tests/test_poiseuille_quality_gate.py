import json
from pathlib import Path

import numpy as np

from sph.core.simulator import SimConfig, step_simulation
from sph.core.state_builder import build_scene_state
from sph.diagnostics.poiseuille import PoiseuilleDiagnostics, build_poiseuille_config
from sph.neighbors.spatial_hash import SpatialHash


def test_poiseuille_quality_gate_500_steps():
    scene_path = Path("scenes/benchmarks/poiseuille_2d.json")
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    scene["time"]["steps"] = 500

    state = build_scene_state(scene)
    dim = int(state.dim)
    spacing = float(scene["fluid"]["spacing"])
    h = float(scene["neighbors"]["support_radius"])
    rho0 = float(scene["material"]["rho0"])

    forces_cfg = scene.get("forces", {})
    gravity = np.array(forces_cfg.get("gravity", [0.0, 0.0])[:dim], dtype=np.float64)
    body_force = np.array(forces_cfg.get("body_force", [0.0, 0.0])[:dim], dtype=np.float64)
    g = gravity + body_force

    time_cfg = scene["time"]
    domain_cfg = scene["domain"]
    boundary_cfg = scene.get("boundary", {})
    axes_map = {"x": 0, "y": 1, "z": 2}
    periodic_axes = tuple(axes_map[str(a).lower()] for a in domain_cfg.get("periodic_axes", []))

    cfg = SimConfig(
        support_radius=h,
        rho0=rho0,
        eos_k=float(scene["material"]["eos"]["k"]),
        g=g,
        cfl_lambda=float(time_cfg["cfl"]),
        dt_min=float(time_cfg["dt_min"]),
        dt_max=float(time_cfg["dt_max"]),
        dt_fixed=float(time_cfg["dt_fixed"]),
        use_cfl=str(time_cfg.get("mode", "cfl")).lower() == "cfl",
        enable_viscosity=bool(scene["material"]["viscosity"]["enable"]),
        kinematic_viscosity=float(scene["material"]["viscosity"]["nu"]),
        domain_min=np.asarray(domain_cfg["min"], dtype=np.float64),
        domain_max=np.asarray(domain_cfg["max"], dtype=np.float64),
        boundary_restitution=float(boundary_cfg.get("restitution", 0.0)),
        boundary_friction=float(boundary_cfg.get("friction", 0.05)),
        boundary_eps=float(boundary_cfg.get("eps", 1e-6)),
        periodic_axes=periodic_axes,
    )

    solver_cfg = scene["solver"]
    poi_cfg = build_poiseuille_config(scene)
    assert poi_cfg is not None
    poi_logger = PoiseuilleDiagnostics(out_file=Path("out/tests/poiseuille_profile_test.csv"), cfg=poi_cfg)

    t = 0.0
    l2_samples: list[float] = []
    for s in range(500):
        dt = step_simulation(
            state=state,
            cfg=cfg,
            particle_size=spacing,
            solver_cfg_dict=solver_cfg,
            step_idx=s + 1,
            enforce_domain_constraints=True,
        )
        assert dt > 0.0
        t += float(dt)

        if s % 25 == 24:
            l2_samples.append(float(poi_logger.sample_and_log(step=s + 1, time_value=t, state=state)))

        if s % 50 == 49:
            assert np.all(np.isfinite(state.pos))
            assert np.all(np.isfinite(state.vel))
            assert np.all(np.isfinite(state.rho))
            assert np.all(np.isfinite(state.p))

    fluid_ids = state.fluid_indices
    active_mask = state.pos[fluid_ids, 0] < 1e8
    active_ids = fluid_ids[active_mask]
    assert active_ids.size > 0

    rho_avg = float(np.mean(state.rho[active_ids]))
    assert 0.98 * rho0 <= rho_avg <= 1.02 * rho0

    ns = SpatialHash(
        support_radius=h,
        dim=dim,
        periodic_min=cfg.domain_min,
        periodic_max=cfg.domain_max,
        periodic_axes=cfg.periodic_axes,
    )
    ns.build(state.pos)
    neigh_counts = np.array([len(ns.query(int(i), state.pos)) for i in active_ids], dtype=np.float64)
    assert float(np.mean(neigh_counts)) > 8.0

    assert len(l2_samples) >= 4
    # Loose monotonic quality gate: final error should improve against early transient.
    assert float(np.mean(l2_samples[-3:])) <= float(np.mean(l2_samples[:3])) + 1e-6

