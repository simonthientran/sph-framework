from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sph.core.state import ParticleState
from sph.neighbors.spatial_hash import SpatialHash
from sph.sph.density import compute_density_summation
from sph.sph.pressure import pressure_state_equation_linear, pressure_acceleration_symmetric
from sph.sph.viscosity import viscosity_acceleration_laplace_eq23


@dataclass(frozen=True)
class SimConfig:
    """
    Minimal simulation configuration for a weakly-compressible SPH (WCSPH) step.

    This mirrors the quantities used in the tutorial's simple WCSPH loop
    (Algorithm 1) and related equations.
    """

    # Kernel / neighborhood
    support_radius: float
    rho0: float

    # State equation parameter: p_i = k (rho_i - rho0)
    eos_k: float

    # External acceleration (e.g. gravity), shape (dim,)
    g: np.ndarray

    # Time stepping (CFL-based or fixed)
    cfl_lambda: float
    dt_min: float
    dt_max: float
    dt_fixed: float
    use_cfl: bool

    # Viscosity (optional, based on Laplacian discretization)
    enable_viscosity: bool
    kinematic_viscosity: float  # nu


def compute_dt_cfl(
    v: np.ndarray,
    h_tilde: float,
    lam: float,
    dt_min: float,
    dt_max: float,
) -> float:
    """
    CFL time step restriction (Eq. (33) in the SPH Tutorial):

        dt <= lambda * h_tilde / ||v_max||

    where:
      - h_tilde is a characteristic particle size,
      - v_max is the maximum particle speed.
    """
    if v.size == 0:
        return dt_max

    vmax = float(np.max(np.linalg.norm(v, axis=1)))
    if vmax <= 1e-12:
        return dt_max

    dt = lam * float(h_tilde) / vmax
    return float(np.clip(dt, dt_min, dt_max))


def step_wc_sph(state: ParticleState, cfg: SimConfig, particle_size: float) -> float:
    """
    Perform one weakly-compressible SPH (WCSPH) step using a simple version
    of Algorithm 1 from the SPH tutorial.

    Steps:
      1) Reconstruct density by summation.
      2) Compute non-pressure accelerations (gravity + optional viscosity)
         and advance to an intermediate velocity v* with symplectic Euler.
      3) Compute pressures from the state equation and corresponding
         pressure accelerations.
      4) Update velocity and positions with symplectic Euler.

    Returns:
        dt used in this step.
    """
    h = float(cfg.support_radius)

    # --- neighbor search
    ns = SpatialHash(support_radius=h, dim=state.dim)
    ns.build(state.pos)

    # --- density reconstruction
    state.rho[:] = compute_density_summation(state=state, neighbor_search=ns, h=h)

    # --- time step (CFL or fixed)
    if cfg.use_cfl:
        dt = compute_dt_cfl(
            state.vel,
            h_tilde=float(particle_size),
            lam=float(cfg.cfl_lambda),
            dt_min=float(cfg.dt_min),
            dt_max=float(cfg.dt_max),
        )
    else:
        dt = float(cfg.dt_fixed)

    # --- non-pressure accelerations (gravity + viscosity)
    # constant body force (e.g. gravity)
    a_nonp = np.tile(cfg.g[None, :], (state.n, 1))

    if cfg.enable_viscosity and cfg.kinematic_viscosity > 0.0:
        a_visc = viscosity_acceleration_laplace_eq23(
            state=state,
            neighbor_search=ns,
            h=h,
            nu=float(cfg.kinematic_viscosity),
        )
        a_nonp += a_visc

    # v* = v + dt * a_nonp
    v_star = state.vel + dt * a_nonp

    # --- pressure via state equation
    state.p[:] = pressure_state_equation_linear(
        state.rho, rho0=float(cfg.rho0), k=float(cfg.eos_k)
    )

    # --- pressure acceleration
    a_p = pressure_acceleration_symmetric(state=state, neighbor_search=ns, h=h)

    # v(t+dt) = v* + dt * a_p
    state.vel[:] = v_star + dt * a_p

    # x(t+dt) = x + dt * v(t+dt)
    state.pos[:] = state.pos + dt * state.vel

    return dt


