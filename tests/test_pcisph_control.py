import numpy as np

from sph.core.simulator import SimConfig
from sph.core.state import ParticleState
from sph.solver.pcisph import (
    _apply_negative_pressure_mode_inplace,
    _inactive_mask_with_hold_steps,
    step_pcisph_with_boundaries,
)


def _make_minimal_state(dim: int = 2) -> ParticleState:
    # Two fluid particles far apart => 0 neighbors (SpatialHash excludes self).
    pos = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float64)
    vel = np.array([[1.0, 2.0], [0.0, 0.0]], dtype=np.float64)
    acc = np.zeros_like(pos)
    mass = np.ones((2,), dtype=np.float64)
    rho = np.full((2,), 1000.0, dtype=np.float64)
    p = np.zeros((2,), dtype=np.float64)
    is_boundary = np.zeros((2,), dtype=np.bool_)
    return ParticleState(dim=dim, pos=pos, vel=vel, acc=acc, mass=mass, rho=rho, p=p, is_boundary=is_boundary)


def test_inactive_particles_are_not_frozen_advect_normally():
    """
    Inactive in PCISPH means "pressure solve skip" only.
    Particles must still receive external forces + integrate v/x normally.
    """
    state = _make_minimal_state()
    state.validate()

    dt = 0.01
    g = np.array([0.0, -9.81], dtype=np.float64)
    cfg = SimConfig(
        support_radius=0.001,  # tiny => 0 neighbors
        rho0=1000.0,
        eos_k=500.0,
        g=g,
        cfl_lambda=0.4,
        dt_min=1e-5,
        dt_max=1e-2,
        dt_fixed=dt,
        use_cfl=False,
    )

    pos0 = state.pos.copy()
    vel0 = state.vel.copy()

    # Force all fluid to be inactive by threshold.
    step_pcisph_with_boundaries(
        state=state,
        cfg=cfg,
        particle_size=0.02,
        max_iters=1,
        density_tol=0.0,
        min_neighbors_for_pressure=10,
        adaptive_min_neighbors_for_pressure=False,
        negative_pressure_mode="none",
        debug=False,
    )

    vel_expected = vel0 + dt * g[None, :]
    pos_expected = pos0 + dt * vel_expected

    assert np.allclose(state.vel, vel_expected)
    assert np.allclose(state.pos, pos_expected)


def test_negative_pressure_mode_hard_zero_clamps_to_zero():
    p = np.array([-2.0, 0.5, -0.1], dtype=np.float64)
    ids = np.array([0, 1, 2], dtype=np.int64)
    cap_used = _apply_negative_pressure_mode_inplace(p, ids, mode="hard_zero", cap=None, soft_factor=0.5)
    assert cap_used == 0.0
    assert np.allclose(p, np.array([0.0, 0.5, 0.0], dtype=np.float64))


def test_negative_pressure_mode_soft_cap_limits_negative_pressure():
    p = np.array([-2.0, 0.5, -0.1], dtype=np.float64)
    ids = np.array([0, 1, 2], dtype=np.int64)
    cap_used = _apply_negative_pressure_mode_inplace(p, ids, mode="soft_cap", cap=1.0, soft_factor=0.5)
    assert np.isclose(cap_used, 1.0)
    assert np.allclose(p, np.array([-1.0, 0.5, -0.1], dtype=np.float64))


def test_inactive_hold_steps_requires_consecutive_under_threshold():
    # One-particle "fluid" under threshold for 1 step should NOT become inactive if hold_steps=2.
    key = 123
    n = 5
    fluid_ids = np.array([1], dtype=np.int64)
    under = np.array([True], dtype=np.bool_)

    inactive_1 = _inactive_mask_with_hold_steps(
        state_key=key, n=n, fluid_ids=fluid_ids, under_neighbor_threshold=under, hold_steps=2
    )
    assert inactive_1.shape == (1,)
    assert bool(inactive_1[0]) is False

    inactive_2 = _inactive_mask_with_hold_steps(
        state_key=key, n=n, fluid_ids=fluid_ids, under_neighbor_threshold=under, hold_steps=2
    )
    assert bool(inactive_2[0]) is True


