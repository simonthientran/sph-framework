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
    smoothing_length: float
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
    # Optional acoustic speed used for WCSPH CFL acoustics.
    # If None, we derive c0 ~= sqrt(k/rho0) for linear EOS.
    eos_c0: float | None = None

    # Viscosity (optional, based on Laplacian discretization).
    # Defaults keep viscosity disabled, matching the behavior used in the
    # original tests; providing defaults is a structural convenience and
    # does not change the underlying physics.
    enable_viscosity: bool = False
    kinematic_viscosity: float = 0.0  # nu

    # Domain boundary constraints (axis-aligned bounding box)
    # If domain_min/max are provided, fluid particles are clamped to this box
    # with velocity reflection (restitution) and friction.
    domain_min: np.ndarray | None = None
    domain_max: np.ndarray | None = None
    boundary_restitution: float = 0.0
    boundary_friction: float = 0.05
    boundary_tangent_friction: float = 0.1
    boundary_normal_damping: float = 0.2
    max_penetration_push_frac_of_dx: float = 0.25
    boundary_log_speed_threshold: float = 30.0
    # Collision push-out epsilon. If None, we derive eps = 1e-4 * support_radius.
    boundary_eps: float | None = None
    # Optional clamp for pressure acceleration norm on fluid particles.
    # This is a numerical safety guard for problematic startup configurations.
    # None disables clamping.
    boundary_force_accel_clamp: float | None = None


def enforce_domain_boundary_constraints(
    state: ParticleState,
    cfg: SimConfig,
    *,
    particle_size: float | None = None,
    debug: bool = False,
) -> None:
    """
    Enforce axis-aligned bounding box constraints on FLUID particles.
    """
    if cfg.domain_min is None or cfg.domain_max is None:
        return

    fluid_ids = state.fluid_indices
    # If no fluid particles, nothing to do
    if fluid_ids.size == 0:
        return

    pos = state.pos
    vel = state.vel
    
    dmin = cfg.domain_min
    dmax = cfg.domain_max
    restitution = float(cfg.boundary_restitution)
    tangent_friction = float(np.clip(cfg.boundary_tangent_friction, 0.0, 1.0))
    normal_damping = float(np.clip(cfg.boundary_normal_damping, 0.0, 1.0))
    speed_log_threshold = float(max(cfg.boundary_log_speed_threshold, 0.0))
    eps = cfg.boundary_eps
    if eps is None:
        # Default: a tiny fraction of the kernel support to avoid particles landing
        # exactly on the boundary (which can lead to repeated "teleport-to-wall"
        # artifacts due to floating point round-off).
        eps = 1e-4 * float(cfg.support_radius)
    eps = float(max(eps, 0.0))
    if particle_size is not None and float(particle_size) > 0.0:
        dx_est = float(particle_size)
    else:
        dx_est = 0.5 * float(cfg.support_radius)
    max_pen_push = float(max(0.0, cfg.max_penetration_push_frac_of_dx) * dx_est)
    
    # We iterate per dimension. Vectorized over fluid particles.
    # dim is inferred from dmin/dmax shape.
    dim = state.dim
    
    # Debug throttling: avoid printing too many per step.
    debug_budget = 10

    for d in range(dim):
        # -------------------------
        # x/y/z MIN face (normal +axis)
        # -------------------------
        mask_lo = pos[fluid_ids, d] < dmin[d]
        if np.any(mask_lo):
            idx = fluid_ids[mask_lo]
            old_pos_d = pos[idx, d].copy()
            old_vel = vel[idx].copy()

            depth = (dmin[d] - old_pos_d)
            push = np.minimum(depth, max_pen_push)
            pos_new = old_pos_d + push
            fully_resolved = depth <= max_pen_push
            pos_new[fully_resolved] = dmin[d] + eps
            pos[idx, d] = pos_new

            # Normal vector for min face points inward (+axis)
            n = np.zeros((dim,), dtype=np.float64)
            n[d] = 1.0
            # NOTE: vel[idx] uses advanced indexing (copy), so we must write back to vel[ids_move].
            v_n = (vel[idx] @ n)  # scalar normal component
            moving_out = v_n < 0.0
            if np.any(moving_out):
                ids_move = idx[moving_out]
                vn = v_n[moving_out][:, None] * n[None, :]
                vt = vel[ids_move] - vn
                vn_new = (-restitution * vn) * (1.0 - normal_damping)
                vt_new = (1.0 - tangent_friction) * vt
                vel[ids_move] = vn_new + vt_new

            if debug and debug_budget > 0:
                speed_before = np.linalg.norm(old_vel, axis=1)
                speed_after = np.linalg.norm(vel[idx], axis=1)
                should_log = (depth > 0.5 * max_pen_push) | (speed_before > speed_log_threshold) | (speed_after > speed_log_threshold)
                log_ids = np.where(should_log)[0]
                for k in log_ids[:debug_budget]:
                    i = int(idx[k])
                    print(
                        f"[BOUNDARY] i={i} face={['x','y','z'][d]}_min "
                        f"pen={float(depth[k]):.3e} "
                        f"pos:{float(old_pos_d[k]):.6f}->{float(pos[i, d]):.6f} "
                        f"vel:{old_vel[k]}->{vel[i]}"
                    )
                debug_budget -= int(min(int(log_ids.size), debug_budget))

        # -------------------------
        # x/y/z MAX face (normal -axis)
        # -------------------------
        mask_hi = pos[fluid_ids, d] > dmax[d]
        if np.any(mask_hi):
            idx = fluid_ids[mask_hi]
            old_pos_d = pos[idx, d].copy()
            old_vel = vel[idx].copy()

            depth = (old_pos_d - dmax[d])
            push = np.minimum(depth, max_pen_push)
            pos_new = old_pos_d - push
            fully_resolved = depth <= max_pen_push
            pos_new[fully_resolved] = dmax[d] - eps
            pos[idx, d] = pos_new

            n = np.zeros((dim,), dtype=np.float64)
            n[d] = -1.0
            v_n = (vel[idx] @ n)
            moving_out = v_n < 0.0
            if np.any(moving_out):
                ids_move = idx[moving_out]
                vn = v_n[moving_out][:, None] * n[None, :]
                vt = vel[ids_move] - vn
                vn_new = (-restitution * vn) * (1.0 - normal_damping)
                vt_new = (1.0 - tangent_friction) * vt
                vel[ids_move] = vn_new + vt_new

            if debug and debug_budget > 0:
                speed_before = np.linalg.norm(old_vel, axis=1)
                speed_after = np.linalg.norm(vel[idx], axis=1)
                should_log = (depth > 0.5 * max_pen_push) | (speed_before > speed_log_threshold) | (speed_after > speed_log_threshold)
                log_ids = np.where(should_log)[0]
                for k in log_ids[:debug_budget]:
                    i = int(idx[k])
                    print(
                        f"[BOUNDARY] i={i} face={['x','y','z'][d]}_max "
                        f"pen={float(depth[k]):.3e} "
                        f"pos:{float(old_pos_d[k]):.6f}->{float(pos[i, d]):.6f} "
                        f"vel:{old_vel[k]}->{vel[i]}"
                    )
                debug_budget -= int(min(int(log_ids.size), debug_budget))


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


def compute_dt_wcsph_constraints(
    *,
    v: np.ndarray,
    h_tilde: float,
    lam: float,
    dt_min: float,
    dt_max: float,
    c0: float,
    nu: float,
) -> float:
    """
    WCSPH timestep with convective, acoustic, and viscous constraints:

      dt_conv ~= lam * h / max(|v|)
      dt_acou ~= lam * h / (c0 + max(|v|))
      dt_visc ~= 0.125 * h^2 / nu

    We use the minimum active constraint and clip to [dt_min, dt_max].
    """
    vmax = float(np.max(np.linalg.norm(v, axis=1))) if v.size else 0.0
    h = float(h_tilde)
    lam = float(lam)
    c0 = float(max(c0, 0.0))
    nu = float(max(nu, 0.0))
    eps = 1e-12

    dt_candidates = [float(dt_max)]
    if vmax > eps:
        dt_candidates.append(lam * h / vmax)
    dt_candidates.append(lam * h / (c0 + vmax + eps))
    if nu > eps:
        dt_candidates.append(0.125 * h * h / nu)
    dt = min(dt_candidates)
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
    h = float(cfg.smoothing_length)

    # --- neighbor search
    ns = SpatialHash(support_radius=float(cfg.support_radius), dim=state.dim)
    ns.build(state.pos)

    # --- density reconstruction
    state.rho[:] = compute_density_summation(state=state, neighbor_search=ns, h=h)

    # --- time step (CFL or fixed)
    if cfg.use_cfl:
        c0 = float(cfg.eos_c0) if cfg.eos_c0 is not None else float(np.sqrt(max(cfg.eos_k, 0.0) / max(cfg.rho0, 1e-12)))
        dt = compute_dt_wcsph_constraints(
            v=state.vel,
            h_tilde=float(particle_size),
            lam=float(cfg.cfl_lambda),
            dt_min=float(cfg.dt_min),
            dt_max=float(cfg.dt_max),
            c0=c0,
            nu=float(cfg.kinematic_viscosity) if cfg.enable_viscosity else 0.0,
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
    h = float(cfg.smoothing_length)

    # neighbor search over ALL particles (fluid + boundary)
    ns = SpatialHash(support_radius=float(cfg.support_radius), dim=state.dim)
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
        c0 = float(cfg.eos_c0) if cfg.eos_c0 is not None else float(np.sqrt(max(cfg.eos_k, 0.0) / max(cfg.rho0, 1e-12)))
        dt = compute_dt_wcsph_constraints(
            v=v_fluid,
            h_tilde=float(particle_size),
            lam=float(cfg.cfl_lambda),
            dt_min=float(cfg.dt_min),
            dt_max=float(cfg.dt_max),
            c0=c0,
            nu=float(cfg.kinematic_viscosity) if cfg.enable_viscosity else 0.0,
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
    if cfg.boundary_force_accel_clamp is not None and cfg.boundary_force_accel_clamp > 0.0:
        clamp = float(cfg.boundary_force_accel_clamp)
        a_pf = a_p[fluid_ids]
        an = np.linalg.norm(a_pf, axis=1)
        over = an > clamp
        if np.any(over):
            # Scale vectors to exactly clamp norm.
            a_pf[over] = a_pf[over] * (clamp / an[over])[:, None]
            a_p[fluid_ids] = a_pf

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

    # Enforce domain boundaries (collision)
    enforce_domain_boundary_constraints(state, cfg, particle_size=float(particle_size))

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
        # ------------------------------------------------------------------
        # Negative pressure handling (control logic only; equations unchanged)
        #
        # New config:
        #   negative_pressure_mode: one of ["none", "hard_zero", "soft_cap"]
        #   negative_pressure_cap: float | null
        #   negative_pressure_soft_factor: float (default 0.2)
        #
        # Backwards compatibility:
        # - legacy "clamp_negative_pressure" / "clamp_negative_pressure_final" map to:
        #     True  -> "soft_cap"
        #     False -> "none"
        # - legacy "clamp_negative_pressure_iter" is still supported.
        # ------------------------------------------------------------------
        legacy_clamp = bool(solver_cfg_dict.get("clamp_negative_pressure", True))
        legacy_final = bool(solver_cfg_dict.get("clamp_negative_pressure_final", legacy_clamp))
        if "negative_pressure_mode" in solver_cfg_dict:
            negative_pressure_mode = str(solver_cfg_dict.get("negative_pressure_mode", "soft_cap")).lower()
        else:
            negative_pressure_mode = "soft_cap" if legacy_final else "none"
        negative_pressure_cap = solver_cfg_dict.get("negative_pressure_cap", None)
        negative_pressure_cap = float(negative_pressure_cap) if negative_pressure_cap is not None else None
        negative_pressure_soft_factor = float(solver_cfg_dict.get("negative_pressure_soft_factor", 0.2))
        clamp_negative_pressure_iter = bool(solver_cfg_dict.get("clamp_negative_pressure_iter", False))
        # Less aggressive default near free-surface (control logic only; no equation changes).
        min_neighbors_for_pressure = int(solver_cfg_dict.get("min_neighbors_for_pressure", 7))
        adaptive_min_neighbors_for_pressure = bool(solver_cfg_dict.get("adaptive_min_neighbors_for_pressure", True))
        active_neighbor_ratio = float(solver_cfg_dict.get("active_neighbor_ratio", 0.7))
        min_neighbors_floor = int(solver_cfg_dict.get("min_neighbors_floor", 5))
        inactive_hold_steps = int(solver_cfg_dict.get("inactive_hold_steps", 0))
        force_active_if_density_low = bool(solver_cfg_dict.get("force_active_if_density_low", True))
        force_active_rho_min = solver_cfg_dict.get("force_active_rho_min", None)
        force_active_rho_min = float(force_active_rho_min) if force_active_rho_min is not None else None
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
            negative_pressure_mode=negative_pressure_mode,
            negative_pressure_cap=negative_pressure_cap,
            negative_pressure_soft_factor=negative_pressure_soft_factor,
            clamp_negative_pressure_iter=clamp_negative_pressure_iter,
            min_neighbors_for_pressure=min_neighbors_for_pressure,
            adaptive_min_neighbors_for_pressure=adaptive_min_neighbors_for_pressure,
            active_neighbor_ratio=active_neighbor_ratio,
            min_neighbors_floor=min_neighbors_floor,
            inactive_hold_steps=inactive_hold_steps,
            force_active_if_density_low=force_active_if_density_low,
            force_active_rho_min=force_active_rho_min,
            debug_fixed_dt=debug_fixed_dt,
            debug=debug,
            debug_dump_on_step=debug_dump_on_step,
            step_idx=step_idx,
        )

    raise ValueError(f"Unknown solver type: {solver_type!r}")

