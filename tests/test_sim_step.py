import numpy as np

from sph.core.simulator import SimConfig, step_wc_sph
from sph.core.state_builder import build_fluid_block


def test_one_step_gravity_moves_particles_down():
    scene = {
        "meta": {"name": "test", "version": 1, "seed": 0, "dimensions": 2},
        "fluid": {"type": "block", "min": [0.0, 0.0], "max": [0.1, 0.1], "spacing": 0.02, "initial_velocity": [0.0, 0.0]},
        "neighbors": {"type": "spatial_hash", "support_radius": 0.04},
        "material": {"rho0": 1000.0, "eos": {"k": 2000.0}, "viscosity": {"enable": False, "nu": 0.0}},
        "time": {"mode": "fixed", "dt_fixed": 0.001},
    }

    state = build_fluid_block(scene)
    y0 = state.pos[:, 1].copy()

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
        enable_viscosity=False,
        kinematic_viscosity=0.0,
    )

    step_wc_sph(state, cfg=cfg, particle_size=0.02)
    assert np.min(state.pos[:, 1] - y0) < 0.0
