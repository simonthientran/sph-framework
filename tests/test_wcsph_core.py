"""
Comprehensive WCSPH core block tests.

Covers: Tait EOS, density, pressure acceleration, mass conservation,
determinism, CFL, XSPH toggle, full simulation step coherence.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from sph.core.state import ParticleState
from sph.core.simulator import (
    SimConfig,
    build_neighbor_search,
    step_wcsph_algorithm1_with_boundaries,
    _compute_eos_pressure,
    _compute_sound_speed,
)
from sph.neighbors.spatial_hash import SpatialHash
from sph.sph.density import compute_density_summation, compute_density_with_boundaries_eq83
from sph.sph.pressure import (
    pressure_tait_eos,
    pressure_state_equation_linear,
    compute_tait_B,
    compute_eos_sound_speed,
    pressure_acceleration_with_boundaries_eq84,
)
from sph.sph.viscosity import viscosity_acceleration_laplace_eq23
from sph.sph.xsph import xsph_velocity_correction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_uniform_block_2d(
    nx: int = 10,
    ny: int = 10,
    dx: float = 0.01,
    rho0: float = 1000.0,
) -> ParticleState:
    """Create a 2D uniform particle block (fluid only, no boundaries)."""
    xs = np.arange(nx) * dx + 0.5 * dx
    ys = np.arange(ny) * dx + 0.5 * dx
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    pos = np.column_stack([xx.ravel(), yy.ravel()])
    n = pos.shape[0]
    mass_per_particle = rho0 * dx ** 2
    return ParticleState(
        dim=2,
        pos=pos.astype(np.float64),
        vel=np.zeros((n, 2), dtype=np.float64),
        acc=np.zeros((n, 2), dtype=np.float64),
        mass=np.full(n, mass_per_particle, dtype=np.float64),
        rho=np.full(n, rho0, dtype=np.float64),
        p=np.zeros(n, dtype=np.float64),
        is_boundary=np.zeros(n, dtype=bool),
    )


def _make_cfg(
    dx: float = 0.01,
    rho0: float = 1000.0,
    eos_k: float = 500.0,
    eos_type: str = "linear",
    eos_gamma: float = 7.0,
    enable_xsph: bool = True,
    xsph_epsilon: float = 0.05,
    use_cfl: bool = False,
    dt_fixed: float = 1e-4,
    domain_min: list | None = None,
    domain_max: list | None = None,
) -> SimConfig:
    h = dx * 1.6
    support_radius = 2.0 * h
    return SimConfig(
        support_radius=support_radius,
        smoothing_length=h,
        rho0=rho0,
        eos_k=eos_k,
        g=np.array([0.0, -9.81]),
        cfl_lambda=0.25,
        dt_min=1e-6,
        dt_max=1e-3,
        dt_fixed=dt_fixed,
        use_cfl=use_cfl,
        eos_type=eos_type,
        eos_gamma=eos_gamma,
        enable_xsph=enable_xsph,
        xsph_epsilon=xsph_epsilon,
        domain_min=np.array(domain_min or [0.0, 0.0]),
        domain_max=np.array(domain_max or [0.2, 0.2]),
    )


# ---------------------------------------------------------------------------
# Tait EOS
# ---------------------------------------------------------------------------

class TestTaitEOS:
    def test_pressure_at_rest_density_is_zero(self):
        rho = np.array([1000.0])
        p = pressure_tait_eos(rho, rho0=1000.0, B=100.0, gamma=7.0)
        assert np.isclose(p[0], 0.0, atol=1e-10)

    def test_pressure_above_rho0_is_positive(self):
        rho = np.array([1010.0])
        p = pressure_tait_eos(rho, rho0=1000.0, B=100.0, gamma=7.0)
        assert p[0] > 0.0

    def test_pressure_below_rho0_clamped_to_zero(self):
        """Tait clamps density to rho0 from below to prevent negative pressure."""
        rho = np.array([900.0, 500.0, 0.1])
        p = pressure_tait_eos(rho, rho0=1000.0, B=100.0, gamma=7.0)
        assert np.all(p == 0.0)

    def test_monotonically_increasing(self):
        rho = np.linspace(1000.0, 1100.0, 50)
        p = pressure_tait_eos(rho, rho0=1000.0, B=500.0, gamma=7.0)
        assert np.all(np.diff(p) > 0)

    def test_tait_B_from_sound_speed(self):
        rho0 = 1000.0
        c0 = 20.0
        gamma = 7.0
        B = compute_tait_B(rho0, c0, gamma)
        expected = rho0 * c0**2 / gamma
        assert np.isclose(B, expected)

    def test_sound_speed_linear_vs_tait(self):
        c_lin = compute_eos_sound_speed(500.0, 1000.0, eos_type="linear")
        c_tait = compute_eos_sound_speed(500.0, 1000.0, eos_type="tait", gamma=7.0)
        assert c_lin == pytest.approx(np.sqrt(500.0 / 1000.0))
        assert c_tait == pytest.approx(np.sqrt(7.0 * 500.0 / 1000.0))
        assert c_tait > c_lin

    def test_tait_vs_linear_at_small_deviation(self):
        """For small rho deviation from rho0, Tait ~ gamma*B*(rho/rho0 - 1) ~ linear."""
        rho0 = 1000.0
        B = 500.0
        gamma = 7.0
        rho = np.array([1001.0])
        p_tait = pressure_tait_eos(rho, rho0, B, gamma)
        p_linear_approx = gamma * B * (rho - rho0) / rho0
        assert np.isclose(p_tait[0], p_linear_approx[0], rtol=0.01)


# ---------------------------------------------------------------------------
# EOS selection in SimConfig
# ---------------------------------------------------------------------------

class TestEOSSelection:
    def test_linear_eos_via_config(self):
        cfg = _make_cfg(eos_type="linear", eos_k=500.0)
        rho = np.array([1010.0, 990.0])
        p = _compute_eos_pressure(rho, cfg)
        expected = 500.0 * (rho - 1000.0)
        np.testing.assert_allclose(p, expected)

    def test_tait_eos_via_config(self):
        cfg = _make_cfg(eos_type="tait", eos_k=500.0, eos_gamma=7.0)
        rho = np.array([1010.0])
        p = _compute_eos_pressure(rho, cfg)
        assert p[0] > 0.0
        assert p[0] == pytest.approx(
            pressure_tait_eos(rho, 1000.0, 500.0, 7.0)[0]
        )

    def test_sound_speed_via_config(self):
        cfg_lin = _make_cfg(eos_type="linear", eos_k=500.0)
        cfg_tait = _make_cfg(eos_type="tait", eos_k=500.0, eos_gamma=7.0)
        c_lin = _compute_sound_speed(cfg_lin)
        c_tait = _compute_sound_speed(cfg_tait)
        assert c_tait > c_lin


# ---------------------------------------------------------------------------
# Density summation
# ---------------------------------------------------------------------------

class TestDensitySummation:
    def test_interior_density_close_to_rho0(self):
        state = _make_uniform_block_2d(nx=12, ny=12, dx=0.01, rho0=1000.0)
        h = 0.016
        ns = SpatialHash(support_radius=0.032, dim=2)
        ns.build(state.pos)
        rho = compute_density_summation(state, ns, h)
        center = np.array([0.06, 0.06])
        dists = np.linalg.norm(state.pos - center, axis=1)
        interior = dists < 0.03
        assert interior.sum() > 5
        rho_int = rho[interior]
        assert np.all(np.abs(rho_int / 1000.0 - 1.0) < 0.05)

    def test_all_densities_positive(self):
        state = _make_uniform_block_2d()
        ns = SpatialHash(support_radius=0.032, dim=2)
        ns.build(state.pos)
        rho = compute_density_summation(state, ns, h=0.016)
        assert np.all(rho > 0)

    def test_density_symmetric(self):
        """Two particles equidistant from a third get same density."""
        state = _make_uniform_block_2d(nx=5, ny=5, dx=0.01)
        ns = SpatialHash(support_radius=0.032, dim=2)
        ns.build(state.pos)
        rho = compute_density_summation(state, ns, h=0.016)
        center_idx = 12  # (2,2) in 5x5 grid
        assert rho[center_idx] == rho[center_idx]  # tautology; real check is below
        idx_11 = 6   # (1,1)
        idx_33 = 18  # (3,3)
        assert np.isclose(rho[idx_11], rho[idx_33], rtol=1e-10)


# ---------------------------------------------------------------------------
# Pressure acceleration
# ---------------------------------------------------------------------------

class TestPressureAcceleration:
    def test_net_pressure_force_near_zero_uniform(self):
        """On a uniform grid at rest density, deep interior particles have small pressure accel."""
        state = _make_uniform_block_2d(nx=16, ny=16, dx=0.01, rho0=1000.0)
        h = 0.016
        support = 0.032
        ns = SpatialHash(support_radius=support, dim=2)
        ns.build(state.pos)
        state.rho[:] = compute_density_summation(state, ns, h)
        state.p[:] = pressure_state_equation_linear(state.rho, rho0=1000.0, k=500.0)
        a = pressure_acceleration_with_boundaries_eq84(state, ns, h, rho0=1000.0)
        center = np.array([0.08, 0.08])
        dists = np.linalg.norm(state.pos - center, axis=1)
        interior = dists < 0.02
        assert interior.sum() > 5
        a_int = a[interior]
        assert np.all(np.abs(a_int) < 10.0)


# ---------------------------------------------------------------------------
# Viscosity
# ---------------------------------------------------------------------------

class TestViscosity:
    def test_viscosity_zero_on_uniform_velocity(self):
        """No velocity gradient -> zero viscosity acceleration."""
        state = _make_uniform_block_2d(nx=8, ny=8, dx=0.01)
        state.vel[:] = [1.0, 0.5]
        h = 0.016
        ns = SpatialHash(support_radius=0.032, dim=2)
        ns.build(state.pos)
        state.rho[:] = compute_density_summation(state, ns, h)
        a_visc = viscosity_acceleration_laplace_eq23(state, ns, h, nu=0.001)
        center = np.array([0.04, 0.04])
        dists = np.linalg.norm(state.pos - center, axis=1)
        interior = dists < 0.02
        a_int = a_visc[interior]
        assert np.all(np.abs(a_int) < 1.0)

    def test_viscosity_nonzero_with_gradient(self):
        """Linear velocity gradient -> nonzero viscosity acceleration."""
        state = _make_uniform_block_2d(nx=8, ny=8, dx=0.01)
        state.vel[:, 0] = state.pos[:, 1] * 10.0
        h = 0.016
        ns = SpatialHash(support_radius=0.032, dim=2)
        ns.build(state.pos)
        state.rho[:] = compute_density_summation(state, ns, h)
        a_visc = viscosity_acceleration_laplace_eq23(state, ns, h, nu=0.01)
        assert np.any(np.abs(a_visc) > 0.01)


# ---------------------------------------------------------------------------
# XSPH toggle
# ---------------------------------------------------------------------------

class TestXSPHToggle:
    def test_xsph_on_vs_off_differ_in_position(self):
        """XSPH affects advection, so positions should differ."""
        rng = np.random.default_rng(42)
        state_a = _make_uniform_block_2d(nx=6, ny=6, dx=0.01)
        state_a.vel[:, 0] = rng.uniform(-0.1, 0.1, state_a.n)
        state_b = copy.deepcopy(state_a)
        cfg_on = _make_cfg(enable_xsph=True, xsph_epsilon=0.5, use_cfl=False, dt_fixed=1e-5)
        cfg_off = _make_cfg(enable_xsph=False, use_cfl=False, dt_fixed=1e-5)
        step_wcsph_algorithm1_with_boundaries(state_a, cfg_on, particle_size=0.01)
        step_wcsph_algorithm1_with_boundaries(state_b, cfg_off, particle_size=0.01)
        assert not np.allclose(state_a.pos, state_b.pos, atol=1e-15)

    def test_xsph_does_not_corrupt_stored_velocity(self):
        """XSPH must only affect advection, NOT the stored velocity.

        Monaghan's XSPH modifies the position update equation:
            x_i^{n+1} = x_i^n + dt * (v_i + eps * dv_xsph)
        The stored velocity v_i must remain the physical velocity so that
        CFL, viscosity, and pressure forces in subsequent steps are correct.
        """
        rng = np.random.default_rng(99)
        state_on = _make_uniform_block_2d(nx=6, ny=6, dx=0.01)
        state_on.vel[:, 0] = rng.uniform(-0.5, 0.5, state_on.n)
        state_off = copy.deepcopy(state_on)
        cfg_on = _make_cfg(enable_xsph=True, xsph_epsilon=0.5, use_cfl=False, dt_fixed=1e-5)
        cfg_off = _make_cfg(enable_xsph=False, use_cfl=False, dt_fixed=1e-5)
        step_wcsph_algorithm1_with_boundaries(state_on, cfg_on, particle_size=0.01)
        step_wcsph_algorithm1_with_boundaries(state_off, cfg_off, particle_size=0.01)
        np.testing.assert_allclose(state_on.vel, state_off.vel, atol=1e-12)


# ---------------------------------------------------------------------------
# CFL timestep
# ---------------------------------------------------------------------------

class TestCFLTimestep:
    def test_cfl_returns_reasonable_dt(self):
        state = _make_uniform_block_2d()
        state.vel[:, 0] = 1.0
        cfg = _make_cfg(use_cfl=True, eos_k=500.0)
        dt = step_wcsph_algorithm1_with_boundaries(state, cfg, particle_size=0.01)
        assert cfg.dt_min <= dt <= cfg.dt_max

    def test_higher_velocity_smaller_dt(self):
        state_slow = _make_uniform_block_2d()
        state_fast = _make_uniform_block_2d()
        state_slow.vel[:, 0] = 0.1
        state_fast.vel[:, 0] = 10.0
        cfg = _make_cfg(use_cfl=True)
        dt_slow = step_wcsph_algorithm1_with_boundaries(state_slow, cfg, particle_size=0.01)
        dt_fast = step_wcsph_algorithm1_with_boundaries(state_fast, cfg, particle_size=0.01)
        assert dt_fast <= dt_slow

    def test_tait_cfl_smaller_than_linear(self):
        """Tait has higher effective c0, so CFL should give smaller dt."""
        c_lin = _compute_sound_speed(_make_cfg(eos_type="linear", eos_k=500.0))
        c_tait = _compute_sound_speed(_make_cfg(eos_type="tait", eos_k=500.0, eos_gamma=7.0))
        assert c_tait > c_lin


# ---------------------------------------------------------------------------
# Full simulation step
# ---------------------------------------------------------------------------

class TestSimulationStep:
    def test_mass_conservation(self):
        state = _make_uniform_block_2d()
        total_mass_before = float(state.mass.sum())
        cfg = _make_cfg(use_cfl=False, dt_fixed=1e-4)
        step_wcsph_algorithm1_with_boundaries(state, cfg, particle_size=0.01)
        total_mass_after = float(state.mass.sum())
        assert total_mass_before == pytest.approx(total_mass_after, abs=1e-15)

    def test_deterministic(self):
        """Same initial conditions -> identical result."""
        state_a = _make_uniform_block_2d()
        state_b = _make_uniform_block_2d()
        cfg = _make_cfg(use_cfl=False, dt_fixed=1e-4)
        dt_a = step_wcsph_algorithm1_with_boundaries(state_a, cfg, particle_size=0.01)
        dt_b = step_wcsph_algorithm1_with_boundaries(state_b, cfg, particle_size=0.01)
        assert dt_a == dt_b
        np.testing.assert_array_equal(state_a.pos, state_b.pos)
        np.testing.assert_array_equal(state_a.vel, state_b.vel)

    def test_no_nan_after_step(self):
        state = _make_uniform_block_2d()
        cfg = _make_cfg(use_cfl=False, dt_fixed=1e-4)
        step_wcsph_algorithm1_with_boundaries(state, cfg, particle_size=0.01)
        assert np.all(np.isfinite(state.pos))
        assert np.all(np.isfinite(state.vel))
        assert np.all(np.isfinite(state.rho))
        assert np.all(np.isfinite(state.p))

    def test_gravity_moves_particles_downward(self):
        state = _make_uniform_block_2d()
        y_before = state.pos[:, 1].copy()
        cfg = _make_cfg(use_cfl=False, dt_fixed=1e-4)
        step_wcsph_algorithm1_with_boundaries(state, cfg, particle_size=0.01)
        assert np.mean(state.pos[:, 1]) < np.mean(y_before)

    def test_tait_step_no_crash(self):
        state = _make_uniform_block_2d()
        cfg = _make_cfg(eos_type="tait", eos_k=500.0, eos_gamma=7.0, use_cfl=False, dt_fixed=1e-5)
        dt = step_wcsph_algorithm1_with_boundaries(state, cfg, particle_size=0.01)
        assert dt > 0
        assert np.all(np.isfinite(state.pos))
        assert np.all(np.isfinite(state.vel))

    def test_step_with_viscosity(self):
        state = _make_uniform_block_2d()
        cfg = _make_cfg(use_cfl=False, dt_fixed=1e-4)
        from dataclasses import replace
        cfg_visc = replace(cfg, enable_viscosity=True, kinematic_viscosity=0.001)
        dt = step_wcsph_algorithm1_with_boundaries(state, cfg_visc, particle_size=0.01)
        assert dt > 0
        assert np.all(np.isfinite(state.vel))


# ---------------------------------------------------------------------------
# build_neighbor_search helper
# ---------------------------------------------------------------------------

class TestBuildNeighborSearch:
    def test_returns_spatial_hash_with_periodic(self):
        cfg = _make_cfg()
        from dataclasses import replace
        cfg_p = replace(cfg, periodic_axes=(True, False))
        ns = build_neighbor_search(cfg_p, dim=2)
        assert isinstance(ns, SpatialHash)
        assert ns.periodic_axes == (True, False)

    def test_returns_spatial_hash_no_periodic(self):
        cfg = _make_cfg()
        ns = build_neighbor_search(cfg, dim=2)
        assert isinstance(ns, SpatialHash)
