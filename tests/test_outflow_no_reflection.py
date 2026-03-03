import numpy as np

from sph.boundaries import BoundaryManager, OutflowBoundary, WallBoundary
from sph.core.simulator import SimConfig, step_simulation
from sph.core.state_builder import build_scene_state

def test_outflow_no_reflection():
    """Particles crossing outflow should be removed with no x-wall reflection."""
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
            "type": "block",
            "min": [0.8, 0.1],
            "max": [0.94, 0.3],
            "spacing": spacing,
            "initial_velocity": [10.0, 0.0] # Move fast right
        },
        "material": {
            "rho0": 1000.0,
            "eos": {"k": 1000.0},
        },
        "neighbors": {
            "support_radius": 0.1
        },
    }
    state = build_scene_state(scene)
    initial_fluid_count = len(state.fluid_indices)
    assert initial_fluid_count > 0

    cfg = SimConfig(
        support_radius=0.1,
        rho0=1000.0,
        eos_k=0.0,
        g=np.array([0.0, 0.0]),
        cfl_lambda=0.4,
        dt_min=1e-5,
        dt_max=2e-4,
        dt_fixed=1e-3,
        use_cfl=False,
    )

    manager = BoundaryManager([
        OutflowBoundary(region_min=[0.95, 0.0], region_max=[1.05, 0.4]),
        WallBoundary(
            domain_min=domain_min,
            domain_max=domain_max,
            slip_mode="free-slip",
            restitution=0.5,
            faces=["ymin", "ymax"],
        ),
    ])

    max_x_over_time: list[float] = []
    for _ in range(150):
        manager.pre_step(state, cfg.dt_fixed)
        dt = step_simulation(
            state=state,
            cfg=cfg,
            particle_size=spacing,
            solver_cfg_dict={"type": "wcsph"},
            enforce_domain_constraints=False,
        )
        manager.apply_walls(state, cfg)
        manager.post_step(state)
        assert dt > 0.0

        fluid_ids = state.fluid_indices
        active = state.pos[fluid_ids, 0] < 1e8
        if not np.any(active):
            break
        max_x_over_time.append(float(np.max(state.pos[fluid_ids][active, 0])))

    fluid_ids = state.fluid_indices
    active_mask = state.pos[fluid_ids, 0] < 1e8
    active_count = np.count_nonzero(active_mask)
    assert active_count == 0, f"Expected 0 active particles, got {active_count}"

    # If outflow reflected on x-max wall, we'd observe a drop in max-x after reaching outlet.
    if len(max_x_over_time) >= 5:
        tail = np.array(max_x_over_time[-5:], dtype=np.float64)
        assert bool(np.all(np.diff(tail) >= -1e-8)), "Detected x-direction reflection near outflow"
