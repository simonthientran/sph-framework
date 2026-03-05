import numpy as np

from sph.core.simulator import SimConfig, enforce_domain_boundary_constraints
from sph.core.state import ParticleState


def test_boundary_push_out_epsilon_and_velocity_not_killed():
    """
    Regression test: boundary response must not "teleport" particles to exact min/max,
    and must not kill tangential velocity components.
    """
    # One fluid particle slightly outside x_min with outward normal component and tangential component.
    pos = np.array([[-1e-3, 0.2]], dtype=np.float64)
    vel = np.array([[-1.0, 2.0]], dtype=np.float64)  # vx is outward at x_min, vy tangential
    acc = np.zeros_like(pos)
    mass = np.ones((1,), dtype=np.float64)
    rho = np.full((1,), 1000.0, dtype=np.float64)
    p = np.zeros((1,), dtype=np.float64)
    is_boundary = np.zeros((1,), dtype=np.bool_)
    state = ParticleState(dim=2, pos=pos, vel=vel, acc=acc, mass=mass, rho=rho, p=p, is_boundary=is_boundary)

    cfg = SimConfig(
        support_radius=0.045,
        rho0=1000.0,
        eos_k=500.0,
        g=np.array([0.0, -9.81], dtype=np.float64),
        cfl_lambda=0.4,
        dt_min=1e-5,
        dt_max=5e-4,
        dt_fixed=5e-4,
        use_cfl=True,
        domain_min=np.array([0.0, 0.0], dtype=np.float64),
        domain_max=np.array([1.0, 0.6], dtype=np.float64),
        boundary_eps=1e-6,
        boundary_restitution=0.0,  # perfectly inelastic in normal direction
        boundary_friction=0.0,  # no tangential damping
        boundary_tangent_friction=0.0,
    )

    enforce_domain_boundary_constraints(state, cfg, debug=False)

    # Position must be pushed inside by eps (not exactly on boundary).
    assert state.pos[0, 0] > cfg.domain_min[0]
    assert np.isclose(state.pos[0, 0], cfg.domain_min[0] + cfg.boundary_eps)

    # Normal component (vx) should be zeroed due to restitution=0, but tangential vy must remain.
    assert np.isclose(state.vel[0, 0], 0.0)
    assert np.isclose(state.vel[0, 1], 2.0)


def test_boundary_push_is_clamped_by_fraction_of_dx():
    pos = np.array([[-0.02, 0.2]], dtype=np.float64)
    vel = np.array([[-1.0, 0.0]], dtype=np.float64)
    acc = np.zeros_like(pos)
    mass = np.ones((1,), dtype=np.float64)
    rho = np.full((1,), 1000.0, dtype=np.float64)
    p = np.zeros((1,), dtype=np.float64)
    is_boundary = np.zeros((1,), dtype=np.bool_)
    state = ParticleState(dim=2, pos=pos, vel=vel, acc=acc, mass=mass, rho=rho, p=p, is_boundary=is_boundary)

    cfg = SimConfig(
        support_radius=0.045,
        rho0=1000.0,
        eos_k=500.0,
        g=np.array([0.0, -9.81], dtype=np.float64),
        cfl_lambda=0.4,
        dt_min=1e-5,
        dt_max=5e-4,
        dt_fixed=5e-4,
        use_cfl=True,
        domain_min=np.array([0.0, 0.0], dtype=np.float64),
        domain_max=np.array([1.0, 0.6], dtype=np.float64),
        boundary_eps=0.0,
        boundary_restitution=0.0,
        boundary_tangent_friction=0.0,
        max_penetration_push_frac_of_dx=0.25,
    )
    dx = 0.02
    enforce_domain_boundary_constraints(state, cfg, particle_size=dx, debug=False)

    push_applied = float(state.pos[0, 0] - pos[0, 0])
    assert push_applied <= 0.25 * dx + 1e-12


