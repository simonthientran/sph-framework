import numpy as np

from sph.core.simulator import compute_dt_cfl
from sph.core.state import ParticleState
from sph.neighbors.spatial_hash import SpatialHash
from sph.sph.density import compute_density_summation
from sph.sph.kernels import cubic_spline_W
from sph.sph.pressure import pressure_acceleration_symmetric, pressure_state_equation_linear
from sph.sph.viscosity import viscosity_acceleration_laplace_eq23


def _make_state(pos, vel=None, mass=1.0, rho=1000.0, p=0.0):
    pos = np.asarray(pos, dtype=np.float64)
    n, dim = pos.shape

    if vel is None:
        vel = np.zeros((n, dim), dtype=np.float64)
    else:
        vel = np.asarray(vel, dtype=np.float64)

    state = ParticleState(
        dim=dim,
        pos=pos.copy(),
        vel=vel.copy(),
        acc=np.zeros((n, dim), dtype=np.float64),
        mass=np.full((n,), float(mass), dtype=np.float64),
        rho=np.full((n,), float(rho), dtype=np.float64),
        p=np.full((n,), float(p), dtype=np.float64),
        is_boundary=np.zeros((n,), dtype=np.bool_),
    )
    state.validate()
    return state


def test_density_matches_direct_pairwise_sum_for_symmetric_square():
    """
    Setup:
      4 equal-mass particles on a unit square, with h large enough that each
      particle sees both edge-neighbors and the diagonal neighbor.

    Expected physical behavior:
      By symmetry all four particles must reconstruct the same density, and
      that density must equal the direct SPH summation over self + 3 neighbors.

    Failure indicates:
      Broken self-contribution, missed neighbors, asymmetric neighbor search,
      or incorrect density accumulation.
    """
    h = 1.5
    mass = 2.0
    pos = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    state = _make_state(pos=pos, mass=mass)

    ns = SpatialHash(support_radius=h, dim=2)
    ns.build(state.pos)

    rho = compute_density_summation(state=state, neighbor_search=ns, h=h)

    expected = np.zeros_like(rho)
    for i in range(state.n):
        for j in range(state.n):
            rel = ns.relative_vector(state.pos[i], state.pos[j])
            expected[i] += state.mass[j] * cubic_spline_W(rel, h=h, dim=2)

    assert np.allclose(rho, expected, rtol=1e-12, atol=1e-12)
    assert np.allclose(rho, rho[0], rtol=1e-12, atol=1e-12)


def test_pressure_state_equation_is_zero_at_rest_and_monotonic_around_rho0():
    """
    Setup:
      Three densities below, at, and above rho0.

    Expected physical behavior:
      The linear WCSPH equation of state gives negative pressure below rho0,
      zero at rho0, and positive pressure above rho0, with the exact linear map
      p = k (rho - rho0).

    Failure indicates:
      Wrong sign convention, wrong rest-density reference, or broken scaling.
    """
    rho0 = 1000.0
    k = 25.0
    rho = np.array([950.0, 1000.0, 1050.0], dtype=np.float64)

    p = pressure_state_equation_linear(rho=rho, rho0=rho0, k=k)

    assert np.allclose(p, k * (rho - rho0))
    assert p[0] < 0.0
    assert p[1] == 0.0
    assert p[2] > 0.0


def test_pressure_force_is_equal_and_opposite_for_two_matching_particles():
    """
    Setup:
      Two particles with identical density and pressure, mirrored about x = 0.

    Expected physical behavior:
      Pressure forces must satisfy Newton's third law: accelerations are equal
      and opposite, with no transverse component in this 1D-aligned setup.

    Failure indicates:
      A symmetry violation in the pressure gradient or pair force assembly.
    """
    h = 0.2
    state = _make_state(
        pos=[[-0.04, 0.0], [0.04, 0.0]],
        rho=1000.0,
        p=2000.0,
        mass=1.0,
    )

    ns = SpatialHash(support_radius=h, dim=2)
    ns.build(state.pos)

    a = pressure_acceleration_symmetric(state=state, neighbor_search=ns, h=h)

    assert a[0, 0] < 0.0
    assert a[1, 0] > 0.0
    assert np.allclose(a[:, 1], 0.0, atol=1e-12)
    assert np.allclose(a[0], -a[1], rtol=1e-12, atol=1e-12)


def test_viscosity_acceleration_reduces_relative_velocity_symmetrically():
    """
    Setup:
      Two nearby particles moving with equal and opposite x-velocity.

    Expected physical behavior:
      Viscosity should smooth the velocity jump: the fast-right particle gets a
      leftward acceleration, the fast-left particle gets a rightward one, and
      the pair response stays symmetric.

    Failure indicates:
      Anti-diffusive viscosity, sign errors, or broken pair symmetry.
    """
    h = 0.2
    state = _make_state(
        pos=[[-0.04, 0.0], [0.04, 0.0]],
        vel=[[1.0, 0.0], [-1.0, 0.0]],
        rho=1000.0,
        mass=1.0,
    )

    ns = SpatialHash(support_radius=h, dim=2)
    ns.build(state.pos)

    a = viscosity_acceleration_laplace_eq23(state=state, neighbor_search=ns, h=h, nu=0.1)

    assert a[0, 0] < 0.0
    assert a[1, 0] > 0.0
    assert np.allclose(a[:, 1], 0.0, atol=1e-12)
    assert np.allclose(a[0], -a[1], rtol=1e-12, atol=1e-12)
    assert np.dot(a[0], state.vel[0] - state.vel[1]) < 0.0
    assert np.dot(a[1], state.vel[1] - state.vel[0]) < 0.0


def test_cfl_timestep_respects_zero_velocity_inverse_speed_and_clamps():
    """
    Setup:
      Three tiny velocity fields: zero speed, moderate speed, and very high speed.

    Expected physical behavior:
      CFL picks dt_max when nothing moves, scales as lambda * h_tilde / vmax
      for moderate motion, and clamps at dt_min when the admissible step
      becomes too small.

    Failure indicates:
      Incorrect vmax handling, wrong CFL scaling, or missing min/max clamping.
    """
    dt_zero = compute_dt_cfl(
        v=np.zeros((2, 2), dtype=np.float64),
        h_tilde=0.01,
        lam=0.4,
        dt_min=1e-4,
        dt_max=5e-3,
    )
    dt_mid = compute_dt_cfl(
        v=np.array([[2.0, 0.0], [0.0, 0.0]], dtype=np.float64),
        h_tilde=0.01,
        lam=0.4,
        dt_min=1e-4,
        dt_max=5e-3,
    )
    dt_high = compute_dt_cfl(
        v=np.array([[1000.0, 0.0], [0.0, 0.0]], dtype=np.float64),
        h_tilde=0.01,
        lam=0.4,
        dt_min=1e-4,
        dt_max=5e-3,
    )

    assert dt_zero == 5e-3
    assert np.isclose(dt_mid, 0.4 * 0.01 / 2.0)
    assert dt_high == 1e-4
