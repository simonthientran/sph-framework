import numpy as np

from sph.core.state_builder import build_scene_state
from sph.core.simulator import SimConfig, step_wcsph_algorithm1_with_boundaries


def test_boundary_particles_remain_static():
    scene = {
        "meta": {"name": "test_boundary", "version": 1, "seed": 0, "dimensions": 2},
        "domain": {"type": "box", "min": [0.0, 0.0], "max": [0.2, 0.2], "boundary_layers": 2},
        "fluid": {"type": "block", "min": [0.05, 0.05], "max": [0.10, 0.10], "spacing": 0.02, "initial_velocity": [0.0, 0.0]},
        "neighbors": {"type": "spatial_hash", "support_radius": 0.04},
        "forces": {"gravity": [0.0, -9.81]},
        "material": {"rho0": 1000.0, "eos": {"k": 2000.0}},
        "time": {"mode": "fixed", "dt_fixed": 0.001, "steps": 1}
    }

    state = build_scene_state(scene)

    b = state.boundary_indices
    pos0 = state.pos[b].copy()

    cfg = SimConfig(
        support_radius=0.04,
        rho0=1000.0,
        eos_k=2000.0,
        g=np.array([0.0, -9.81]),
        cfl_lambda=0.4,
        dt_min=1e-5,
        dt_max=1e-2,
        dt_fixed=0.001,
        use_cfl=False,
    )

    step_wcsph_algorithm1_with_boundaries(state, cfg=cfg, particle_size=0.02)

    assert np.allclose(state.pos[b], pos0)
    assert np.allclose(state.vel[b], 0.0)
