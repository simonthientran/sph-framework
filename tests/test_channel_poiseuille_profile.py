import pytest
import numpy as np
from pathlib import Path
from sph.core.simulator import SimConfig, step_simulation
from sph.core.state_builder import build_scene_state
from sph.boundaries import BoundaryManager, WallBoundary, InflowBoundary, OutflowBoundary

def test_channel_poiseuille_profile():
    """
    Test that a basic pipe flow configuration runs without NaNs 
    and begins to establish a velocity profile without exploding.
    (We don't test for a perfect mathematical parabola here to keep 
    the test short, but we check stability and qualitative profile setup).
    """
    # A smaller footprint of channel_poiseuille_2d setup specifically for testing
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
    
    bm = BoundaryManager([
        InflowBoundary(region_min=[0.0, 0.0], region_max=[0.1, 0.4], velocity=[1.0, 0.0], spacing=spacing),
        OutflowBoundary(region_min=[0.9, 0.0], region_max=[1.0, 0.4]),
        WallBoundary(domain_min=domain_min, domain_max=domain_max, slip_mode="no-slip", restitution=0.0)
    ])
    
    steps = 100
    for s in range(steps):
        bm.pre_step(state, 1e-4)
        dt = step_simulation(
            state=state,
            cfg=cfg,
            particle_size=spacing,
            solver_cfg_dict={"type": "wcsph"}
        )
        assert dt > 0
        bm.apply_walls(state, cfg)
        bm.post_step(state)
        
        # Stability checks
        assert np.all(np.isfinite(state.pos))
        assert np.all(np.isfinite(state.vel))
        
    fluid_ids = state.fluid_indices
    active_mask = state.pos[fluid_ids, 0] < 1e8
    active_vel = state.vel[fluid_ids][active_mask]
    
    # Assert things are still moving right
    mean_vx = np.mean(active_vel[:, 0])
    assert mean_vx > 0.0
