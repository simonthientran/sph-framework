import pytest
import numpy as np
from sph.core.simulator import SimConfig, step_simulation
from sph.core.state_builder import build_scene_state
from sph.boundaries import BoundaryManager, WallBoundary, OutflowBoundary

def test_outflow_no_reflection():
    """
    Test that particles moving into the outflow region are cleanly removed
    (put into the inactive pool) rather than bouncing off standard walls.
    """
    spacing = 0.05
    domain_min = [0.0, 0.0]
    domain_max = [1.0, 0.4]
    
    # We create a block of fluid directly moving towards the boundary
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
    
    # Notice we have a Wall boundary on the main domain, BUT
    # we also add an outflow right before the wall on the right.
    bm = BoundaryManager([
        # Outflow region [0.95, 1.05] catches particles before they bounce at 1.0
        OutflowBoundary(region_min=[0.95, 0.0], region_max=[1.05, 0.4]),
        WallBoundary(domain_min=domain_min, domain_max=domain_max, slip_mode="free-slip", restitution=0.5)
    ])
    
    # Step forward until they should have exited
    steps = 150
    for s in range(steps):
        bm.pre_step(state, 1e-3)
        dt = step_simulation(
            state=state,
            cfg=cfg,
            particle_size=spacing,
            solver_cfg_dict={"type": "wcsph"}
        )
        bm.apply_walls(state, cfg)
        bm.post_step(state)
        
        fluid_ids = state.fluid_indices
        active_pos = state.pos[fluid_ids][state.pos[fluid_ids, 0] < 1e8]
        if active_pos.size > 0:
            if s % 10 == 0:
                print(f"step {s} max_x = {np.max(active_pos[:, 0]):.3f}")
        else:
            print(f"step {s} all particles exited")
            break
        
    # By step 150 at v=10, they should easily cross x=0.95 and be deleted.
    fluid_ids = state.fluid_indices
    active_mask = state.pos[fluid_ids, 0] < 1e8
    
    # We should have NO active particles remaining, or very few if some lagged
    active_count = np.count_nonzero(active_mask)
    if active_count > 0:
        print("Active pos:", state.pos[fluid_ids[active_mask]])
        print("Active vel:", state.vel[fluid_ids[active_mask]])
    assert active_count == 0, f"Expected 0 active particles, got {active_count}"
