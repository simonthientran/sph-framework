from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sph.sph.xsph import xsph_velocity_correction

from sph.core.state import ParticleState
from sph.neighbors.spatial_hash import SpatialHash
from sph.sph.density import compute_density_summation, compute_density_with_boundaries_eq83
from sph.sph.pressure import (
    pressure_state_equation_linear,
    pressure_acceleration_symmetric,
    pressure_state_equation_linear_section44,
    pressure_acceleration_with_boundaries_eq84,
)
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

    # External acceleration (e.g., gravity), shape (dim,)
    g: np.ndarray

    # Time stepping (CFL-based or fixed)
    cfl_lambda: float
    dt_min: float
    dt_max: float
    dt_fixed: float
    use_cfl: bool

    # Viscosity (optional, based on Laplacian discretization).
    # Defaults keep viscosity disabled, matching the behavior used in the
    # original tests; providing defaults is a structural convenience and
    # does not change the underlying physics.
    enable_viscosity: bool = False
    kinematic_viscosity: float = 0.0  # nu


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

    This function is identical in logic to the previously committed version
    used by existing tests; we only share the SimConfig definition with
    the boundary-aware variant below.
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
    # constant body force (e.g., gravity)
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


def compute_dt_cfl_eq33(
    v: np.ndarray,
    h_tilde: float,
    lam: float,
    dt_min: float,
    dt_max: float,
) -> float:
    """
    Alias of compute_dt_cfl, kept for notation consistency with Eq. (33)
    in the tutorial. This does not change the numerical scheme.
    """
    return compute_dt_cfl(v=v, h_tilde=h_tilde, lam=lam, dt_min=dt_min, dt_max=dt_max)


def step_wcsph_algorithm1_with_boundaries(state: ParticleState, cfg: SimConfig, particle_size: float) -> float:
    """
    WCSPH loop with particle-based boundary handling, following Algorithm 1
    and Eqs. (33), (83) and (84) in the SPH tutorial.

    This is an extension of step_wc_sph that:
    - uses density including boundary contributions (Eq. 83),
    - uses pressure acceleration with mirrored boundary pressures (Eq. 84),
    - integrates only fluid particles, keeping boundary particles static.

    The underlying equations and ordering follow the tutorial; we only
    add the explicit separation of fluid vs boundary particles.
    """
    h = float(cfg.support_radius)

    # neighbor search over ALL particles (fluid + boundary)
    ns = SpatialHash(support_radius=h, dim=state.dim)
    ns.build(state.pos)

    # (1) density including boundary contribution (Eq. 83)
    state.rho[:] = compute_density_with_boundaries_eq83(
        state=state,
        neighbor_search=ns,
        h=h,
        rho0=cfg.rho0,
    )

    # (dt) CFL (Eq. 33) or fixed, applied to moving (fluid) particles
    if cfg.use_cfl:
        v_fluid = state.vel[~state.is_boundary]
        dt = compute_dt_cfl_eq33(
            v_fluid,
            h_tilde=float(particle_size),
            lam=float(cfg.cfl_lambda),
            dt_min=float(cfg.dt_min),
            dt_max=float(cfg.dt_max),
        )
    else:
        dt = float(cfg.dt_fixed)

    # (2) non-pressure forces: external only (gravity) on fluid
    fluid_ids = state.fluid_indices
    state.vel[fluid_ids] = state.vel[fluid_ids] + dt * cfg.g[None, :]

    # (3) state equation (Section 4.4 examples)
    # We use the Section 4.4 notation wrapper; numerically equivalent to
    # the linear state equation used in step_wc_sph.
    state.p[:] = pressure_state_equation_linear_section44(
        state.rho,
        rho0=cfg.rho0,
        k=cfg.eos_k,
    )

    # (4) pressure acceleration incl. boundary (Eq. 84 + mirroring)
    a_p = pressure_acceleration_with_boundaries_eq84(
        state=state,
        neighbor_search=ns,
        h=h,
        rho0=cfg.rho0,
    )

    # v(t+dt) = v* + dt * a_p  (Algorithm 1 structure)
    state.vel[fluid_ids] = state.vel[fluid_ids] + dt * a_p[fluid_ids]

    # XSPH smoothing (optional stabilization)
    dv_xsph = xsph_velocity_correction(state, ns, h=h, eps=0.05)
    state.vel[fluid_ids] += dv_xsph[fluid_ids]

    # --- velocity correction
    # x(t+dt) = x + dt * v(t+dt)  for fluid only
    state.pos[fluid_ids] = state.pos[fluid_ids] + dt * state.vel[fluid_ids]

    # boundary particles remain static by construction (not integrated)
    state.vel[state.is_boundary] = 0.0

    return dt


def step_simulation(
    state: ParticleState,
    cfg: SimConfig,
    particle_size: float,
    solver_cfg_dict: dict,
    step_idx: int | None = None,
) -> float:
    """
    Dispatch simulation step based on scene solver configuration.

    This function exists purely for execution wiring / architecture:
    - It does not change any solver math or ordering.
    - It calls into the existing WCSPH step (default) or the new PCISPH step.

    Supported scene config:
      "solver": { "type": "wcsph" }  (default)
      "solver": { "type": "pcisph", "max_iters": 8, "density_tol": 0.01 }
    """
    solver_cfg_dict = solver_cfg_dict or {"type": "wcsph"}
    solver_type = str(solver_cfg_dict.get("type", "wcsph")).lower()

    if solver_type == "wcsph":
        return step_wcsph_algorithm1_with_boundaries(state=state, cfg=cfg, particle_size=particle_size)

    if solver_type == "pcisph":
        # Lazy import avoids circular imports and keeps WCSPH unaffected.
        from sph.solver.pcisph import step_pcisph_with_boundaries

        max_iters = int(solver_cfg_dict.get("max_iters", 8))
        density_tol = float(solver_cfg_dict.get("density_tol", 0.01))
        warm_start_pressure = bool(solver_cfg_dict.get("warm_start_pressure", True))
        debug_fixed_dt = bool(solver_cfg_dict.get("debug_fixed_dt", False))
        debug = bool(solver_cfg_dict.get("debug", False))
        debug_dump_on_step = solver_cfg_dict.get("debug_dump_on_step", None)
        debug_dump_on_step = int(debug_dump_on_step) if debug_dump_on_step is not None else None
        return step_pcisph_with_boundaries(
            state=state,
            cfg=cfg,
            particle_size=particle_size,
            max_iters=max_iters,
            density_tol=density_tol,
            warm_start_pressure=warm_start_pressure,
            debug_fixed_dt=debug_fixed_dt,
            debug=debug,
            debug_dump_on_step=debug_dump_on_step,
            step_idx=step_idx,
        )

    raise ValueError(f"Unknown solver type: {solver_type!r}")

