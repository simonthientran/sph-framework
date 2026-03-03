import numpy as np

from sph.boundaries import BoundaryManager, InflowBoundary, OutflowBoundary, WallBoundary
from sph.core.simulator import SimConfig, step_simulation
from sph.core.state_builder import build_scene_state


def test_channel_poiseuille_profile():
    """Short-run Poiseuille test: stable dt/fields and near-parabolic profile."""
    spacing = 0.05
    domain_min = [0.0, 0.0]
    domain_max = [1.0, 0.4]
    scene = {
        "meta": {"dimensions": 2},
        "domain": {
            "type": "channel",
            "min": domain_min,
            "max": domain_max,
            "boundary_layers": 3,
            "boundary_walls": ["ymin", "ymax"]
        },
        "fluid": {
            "type": "channel_fill",
            "spacing": spacing,
            "initial_velocity": [1.0, 0.0]
        },
        "material": {
            "rho0": 1000.0,
            "eos": {"k": 10000.0},
            "viscosity": {"enable": True, "nu": 5e-2}
        },
        "neighbors": {
            "support_radius": 0.1
        },
        "time": {
            "mode": "fixed",
            "dt_fixed": 1e-4,
        },
    }
    state = build_scene_state(scene)

    cfg = SimConfig(
        support_radius=0.1,
        rho0=1000.0,
        eos_k=10000.0,
        g=np.array([0.0, 0.0]),
        cfl_lambda=0.4,
        dt_min=1e-5,
        dt_max=2e-4,
        dt_fixed=1e-4,
        use_cfl=False,
        enable_viscosity=True,
        kinematic_viscosity=5e-2,
        domain_min=None,
        domain_max=None,
    )

    manager = BoundaryManager([
        InflowBoundary(region_min=[0.0, 0.0], region_max=[0.1, 0.4], velocity=[1.0, 0.0], spacing=spacing),
        OutflowBoundary(region_min=[0.9, 0.0], region_max=[1.0, 0.4]),
        WallBoundary(
            domain_min=domain_min,
            domain_max=domain_max,
            slip_mode="no-slip",
            restitution=0.0,
            faces=["ymin", "ymax"],
        ),
    ])

    steps = 120
    for _ in range(steps):
        manager.pre_step(state, cfg.dt_fixed)
        dt = step_simulation(
            state=state,
            cfg=cfg,
            particle_size=spacing,
            solver_cfg_dict={"type": "wcsph"},
            enforce_domain_constraints=False,
        )
        assert dt > 0
        manager.apply_walls(state, cfg)
        manager.post_step(state)

        assert np.all(np.isfinite(state.pos))
        assert np.all(np.isfinite(state.vel))

    fluid_ids = state.fluid_indices
    active_mask = state.pos[fluid_ids, 0] < 1e8
    pos = state.pos[fluid_ids][active_mask]
    vel = state.vel[fluid_ids][active_mask]
    assert pos.shape[0] > 20
    assert float(np.mean(vel[:, 0])) > 0.0

    bins = 8
    y_edges = np.linspace(domain_min[1], domain_max[1], bins + 1)
    obs = np.zeros((bins,), dtype=np.float64)
    cnt = np.zeros((bins,), dtype=np.int64)
    for b in range(bins):
        if b + 1 == bins:
            m = (pos[:, 1] >= y_edges[b]) & (pos[:, 1] <= y_edges[b + 1])
        else:
            m = (pos[:, 1] >= y_edges[b]) & (pos[:, 1] < y_edges[b + 1])
        cnt[b] = int(np.count_nonzero(m))
        obs[b] = float(np.mean(vel[m, 0])) if cnt[b] > 0 else 0.0

    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    h = float(domain_max[1] - domain_min[1])
    y0 = float(domain_min[1])
    ref = (y_centers - y0) * (h - (y_centers - y0))
    ref /= max(float(np.max(ref)), 1e-12)
    obs_norm = obs / max(float(np.max(np.abs(obs))), 1e-12)
    l2 = float(np.sqrt(np.mean((obs_norm - ref) ** 2)))
    assert l2 < 0.60, f"normalized profile L2 too high: {l2:.3f}"
