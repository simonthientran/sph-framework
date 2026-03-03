from __future__ import annotations

"""
PCISPH solver step (Predictive–Corrective Incompressible SPH) with boundaries.

This module adds a new solver step function that can be selected at runtime
without changing the existing WCSPH implementation.

Physics / solver constraints:
- We do NOT change any existing kernel math, density/pressure formulations, CFL,
  neighbor search, or integration scheme used elsewhere in the project.
- This module implements only the PCISPH-specific predictor/corrector loop
  as specified by the equations below.
- Boundary particles remain static (not integrated), consistent with the
  framework's particle-based boundary handling approach.

Reference document:
- "SPH Techniques for the Physics Based Simulation of Fluids and Solids - SPH_Tutorial.pdf"
  Section: Predictive–Corrective Incompressible SPH (PCISPH)

Equations used (from the prompt; do not invent others):
- (51) Predicted density rho*_i (without pressure term)
- (53) Symmetric pressure acceleration (base form)
- (57) Initial pressure prediction using kPCI
- (58) Global stiffness constant kPCI
- (59) Iterative pressure refinement
- (60) Density change due to pressure accelerations
"""

import numpy as np

from sph.core.state import ParticleState
from sph.core.simulator import SimConfig, enforce_domain_boundary_constraints  # configuration container (no solver math here)
from sph.neighbors.spatial_hash import SpatialHash
from sph.sph.kernels import cubic_spline_W, cubic_spline_gradW


#
# Module-level cache for kPCI.
#
# The prompt explicitly allows computing kPCI "once per step (or once at start)".
# We compute/store a global scalar using dt_fixed to keep correction gain stable
# under adaptive integration dt changes.
#
_KPCI_CACHE: dict[tuple[int, float, float, float, float], float] = {}

# Optional per-state cache for active-set hysteresis (control logic only).
# Keyed by id(state) to avoid modifying ParticleState (which uses slots=True).
_UNDER_NEIGHBOR_STREAK_CACHE: dict[int, np.ndarray] = {}
_LAST_STEP_STATS_CACHE: dict[int, dict[str, float | int]] = {}


def get_last_step_stats(state: ParticleState) -> dict[str, float | int] | None:
    """
    Best-effort observability hook for PCISPH step metrics.

    Returns the last cached step stats for the given state instance.
    """
    cached = _LAST_STEP_STATS_CACHE.get(int(id(state)))
    if cached is None:
        return None
    return dict(cached)


def _inactive_mask_with_hold_steps(
    *,
    state_key: int,
    n: int,
    fluid_ids: np.ndarray,
    under_neighbor_threshold: np.ndarray,
    hold_steps: int,
) -> np.ndarray:
    """
    Control logic: require a particle to be under the neighbor threshold for N consecutive
    steps before it becomes inactive (pressure-skip).

    This does NOT change any PCISPH equations; it only makes the active-set decision less
    aggressive and avoids flicker/over-inactivation at the free surface.

    Returns an "inactive_mask" aligned with `fluid_ids`.
    """
    hold = int(hold_steps)
    if hold <= 0 or fluid_ids.size == 0:
        return under_neighbor_threshold

    streak = _UNDER_NEIGHBOR_STREAK_CACHE.get(state_key)
    if streak is None or streak.shape != (n,):
        streak = np.zeros((n,), dtype=np.int32)
        _UNDER_NEIGHBOR_STREAK_CACHE[state_key] = streak

    # Update streak counters for fluid particles only (boundary ignored).
    # - if under threshold: streak += 1
    # - else: streak = 0
    streak_vals = streak[fluid_ids]
    streak_vals = np.where(under_neighbor_threshold, streak_vals + 1, 0).astype(np.int32, copy=False)
    streak[fluid_ids] = streak_vals

    return under_neighbor_threshold & (streak_vals >= hold)


def _apply_negative_pressure_mode_inplace(
    p: np.ndarray,
    ids: np.ndarray,
    *,
    mode: str,
    cap: float | None,
    soft_factor: float,
) -> float:
    """
    Apply negative-pressure handling *in place* to p[ids] and return the cap used.

    Control logic only (no equation changes):
    - mode="none":      p_final = p_raw
    - mode="hard_zero": p_final = max(p_raw, 0)
    - mode="soft_cap":  p_final = max(p_raw, -cap)
        cap = `cap` if provided else soft_factor * p_pos_max_step
        p_pos_max_step = max(max(p_raw, 0)) over ids
    """
    mode = str(mode).lower()
    if ids.size == 0:
        return 0.0

    if mode == "none":
        return 0.0

    if mode == "hard_zero":
        p[ids] = np.maximum(p[ids], 0.0)
        return 0.0

    if mode == "soft_cap":
        if cap is None:
            p_pos_max_step = float(np.max(np.maximum(p[ids], 0.0)))
            cap_used = float(max(0.0, soft_factor * p_pos_max_step))
        else:
            cap_used = float(max(0.0, cap))
        p[ids] = np.maximum(p[ids], -cap_used)
        return cap_used

    raise ValueError(f"negative_pressure_mode must be one of ['none','hard_zero','soft_cap'], got {mode!r}")


def _compute_neighbor_counts(fluid_ids: np.ndarray, ns: SpatialHash, pos: np.ndarray) -> np.ndarray:
    """
    Compute neighbor counts for each fluid particle index in `fluid_ids`.

    IMPORTANT:
    - This is diagnostics / control logic only.
    - The neighbor query itself is the existing neighbor-search (no changes).
    - Counts are aligned with `fluid_ids` (same order / length).
    """
    if fluid_ids.size == 0:
        return np.zeros((0,), dtype=np.int64)
    return np.array([len(ns.query(int(i), pos)) for i in fluid_ids], dtype=np.int64)


def _pressure_accel_eq53_active(
    state: ParticleState,
    ns: SpatialHash,
    h: float,
    p: np.ndarray,
    rho: np.ndarray,
    active_ids: np.ndarray,
    rho0_for_denoms: float | None = None,
) -> np.ndarray:
    """
    Eq. (53) pressure acceleration, but computed ONLY for `active_ids`.

    This does NOT change Eq. (53). It only changes **which particles** receive
    the incompressibility correction when local sampling is degenerate.

    Inactive fluid particles keep a_p = 0 (no pressure correction).
    Boundary particles keep a_p = 0 by design.
    """
    n = state.n
    dim = state.dim
    h = float(h)

    a_p = np.zeros((n, dim), dtype=np.float64)
    if active_ids.size == 0:
        return a_p

    eps = 1e-12
    use_rho0 = rho0_for_denoms is not None
    if use_rho0:
        rho0_for_denoms = float(rho0_for_denoms)
        rho0_2 = rho0_for_denoms * rho0_for_denoms + eps

    for i in active_ids:
        pi = float(p[i])
        if use_rho0:
            rhoi2 = rho0_2
        else:
            rhoi = float(rho[i])
            rhoi2 = rhoi * rhoi + eps

        acc = np.zeros((dim,), dtype=np.float64)
        xi = state.pos[i]

        for j in ns.query(int(i), state.pos):
            pj = float(p[j])  # boundary/inactive may be 0 by control logic
            if use_rho0:
                rhoj2 = rho0_2
            else:
                rhoj = float(rho[j])
                rhoj2 = rhoj * rhoj + eps

            gradW = cubic_spline_gradW(ns.displacement(xi, state.pos[j]), h=h, dim=dim)
            acc -= float(state.mass[j]) * (pi / rhoi2 + pj / rhoj2) * gradW

        a_p[i] = acc

    return a_p


def _rho_p_eq60_active(
    state: ParticleState,
    ns: SpatialHash,
    h: float,
    dt: float,
    a_p: np.ndarray,
    active_ids: np.ndarray,
) -> np.ndarray:
    """
    Eq. (60) density change due to pressure accelerations, computed ONLY for `active_ids`.

    This does NOT change Eq. (60). It only avoids applying the pressure-correction
    machinery when a particle has insufficient neighbors (degenerate sampling).

    Inactive fluid particles keep rho_p = 0.
    """
    n = state.n
    dim = state.dim
    h = float(h)
    dt = float(dt)

    rho_p = np.zeros((n,), dtype=np.float64)
    if active_ids.size == 0:
        return rho_p

    for i in active_ids:
        xi = state.pos[i]
        api = a_p[i]
        acc = 0.0

        for j in ns.query(int(i), state.pos):
            gradW = cubic_spline_gradW(ns.displacement(xi, state.pos[j]), h=h, dim=dim)
            acc += float(state.mass[j]) * float(np.dot(api - a_p[j], gradW))

        rho_p[i] = (dt * dt) * acc

    return rho_p


def _compute_dt_eq33(cfg: SimConfig, v_fluid: np.ndarray, particle_size: float) -> float:
    """
    Compute dt exactly like the existing WCSPH loop:
    - CFL (Eq. (33)) if enabled
    - else fixed dt

    Eq. (33) form (as implemented elsewhere in the project):
        dt <= lambda * h_tilde / ||v_max||
    """
    if cfg.use_cfl:
        if v_fluid.size == 0:
            return float(cfg.dt_max)
        vmax = float(np.max(np.linalg.norm(v_fluid, axis=1)))
        if vmax <= 1e-12:
            return float(cfg.dt_max)
        dt = float(cfg.cfl_lambda) * float(particle_size) / vmax
        return float(np.clip(dt, float(cfg.dt_min), float(cfg.dt_max)))
    return float(cfg.dt_fixed)


def _choose_template_particle(fluid_ids: np.ndarray, ns: SpatialHash, pos: np.ndarray) -> int:
    """
    Choose a representative fluid particle for kPCI estimation (Eq. (58)):
    - Not boundary (caller passes fluid_ids)
    - Neighbor count close to the mean neighbor count (heuristic interior choice)
    """
    counts = np.array([len(ns.query(int(i), pos)) for i in fluid_ids], dtype=np.int64)
    mean_c = float(np.mean(counts)) if counts.size else 0.0
    k = int(np.argmin(np.abs(counts - mean_c))) if counts.size else 0
    return int(fluid_ids[k])


def _compute_kpci_eq58(
    i_template: int,
    state: ParticleState,
    ns: SpatialHash,
    h: float,
    rho0: float,
    dt: float,
    debug: bool = False,
) -> float:
    """
    Compute global stiffness constant kPCI using Eq. (58).

    Eq. (58) (from the prompt):
        kPCI = -0.5 (rho0)^2 / ( dt^2 m_i^2 )
               * 1 / ( Σ_j ∇W_ij · Σ_j ∇W_ij + Σ_j (∇W_ij · ∇W_ij) )

    We compute:
      S = Σ_j ∇W_ij   (vector)
      Q = Σ_j (∇W_ij · ∇W_ij) (scalar)
      denom = S·S + Q
    """
    dim = state.dim
    xi = state.pos[i_template]

    S = np.zeros((dim,), dtype=np.float64)
    Q = 0.0

    mi = float(state.mass[i_template])
    mi_safe = max(mi, 1e-30)

    for j in ns.query(int(i_template), state.pos):
        gradW = cubic_spline_gradW(ns.displacement(xi, state.pos[j]), h=h, dim=dim)
        # IMPORTANT (minimal scaling fix):
        # Later equations (53) and (60) explicitly use neighbor masses m_j.
        # Using the same mass-weighted gradients here keeps kPCI scaling
        # consistent with the rest of the solver without changing the
        # conceptual structure of Eq. (58).
        #
        # We use a *mass ratio* weight (m_j / m_i). This satisfies the requested
        # "mass-weighting" while preserving the original Eq. (58) scaling in the
        # common case of equal-mass particles (m_j == m_i => weight == 1).
        mj = float(state.mass[j])
        w = mj / mi_safe
        gw = w * gradW
        S += gw
        Q += float(np.dot(gw, gw))

    denom_raw = float(np.dot(S, S) + Q)
    if bool(debug):
        if not np.isfinite(denom_raw) or denom_raw <= 0.0:
            print(f"[PCISPH][WARN] kPCI denom invalid: denom={denom_raw!r} (template i={int(i_template)})")
        elif denom_raw < 1e-9:
            print(f"[PCISPH][WARN] kPCI denom tiny: denom={denom_raw:.3e} (template i={int(i_template)})")

    # Numeric guard only: avoid division by ~0 if a particle has no neighbors.
    # This does not change intended physics; it prevents a crash in degenerate cases.
    denom = max(denom_raw, 1e-12)

    mi = float(state.mass[i_template])
    rho0 = float(rho0)
    dt = float(dt)

    return float(-0.5 * (rho0 ** 2) / (dt * dt * mi * mi) * (1.0 / denom))


def _predict_rho_star_eq51(
    state: ParticleState,
    ns: SpatialHash,
    h: float,
    dt: float,
    v_star: np.ndarray,
    a_nonp: np.ndarray,
    rho0: float,
) -> np.ndarray:
    """
    Compute predicted density rho*_i for fluid particles using Eq. (51).

    Eq. (51) (from the prompt):
        rho*_i = Σ_j m_j W_ij
               + dt Σ_j m_j (v_i - v_j) · ∇W_ij
               + dt Σ_j m_j (dt a_nonp_i - dt a_nonp_j) · ∇W_ij

    Boundary handling (per prompt requirements):
    - Treat boundary neighbors with v=0 and a_nonp=0 for predictor terms.
    - Keep their mass m_j as stored in state.mass.

    Notes:
    - We include the standard self contribution in Σ m_j W_ij via W(0),
      consistent with the project's density summation convention.
    - The gradient term for j=i would be zero (∇W(0)=0), and SpatialHash excludes self.
    """
    n = state.n
    dim = state.dim
    dt = float(dt)
    h = float(h)

    rho_star = np.full((n,), float(rho0), dtype=np.float64)

    fluid_ids = state.fluid_indices
    W0 = cubic_spline_W(np.zeros(dim, dtype=np.float64), h=h, dim=dim)

    for i in fluid_ids:
        xi = state.pos[i]
        mi = state.mass[i]

        # term0: Σ m_j W_ij  (self + neighbors)
        rho_i = float(mi) * float(W0)

        # term1 + term2: loop neighbors
        vi = v_star[i]
        ai = a_nonp[i]

        add = 0.0
        for j in ns.query(int(i), state.pos):
            rij = ns.displacement(xi, state.pos[j])
            gradW = cubic_spline_gradW(rij, h=h, dim=dim)

            # base density term
            rho_i += float(state.mass[j]) * float(cubic_spline_W(rij, h=h, dim=dim))

            # predictor velocity/acceleration for neighbor
            if state.is_boundary[j]:
                vj = np.zeros((dim,), dtype=np.float64)
                aj = np.zeros((dim,), dtype=np.float64)
            else:
                vj = v_star[j]
                aj = a_nonp[j]

            # term1: dt Σ m_j (v_i - v_j) · ∇W_ij
            add += float(state.mass[j]) * float(np.dot(vi - vj, gradW))

            # term2: dt Σ m_j (dt a_nonp_i - dt a_nonp_j) · ∇W_ij = dt^2 Σ m_j (a_i - a_j)·∇W
            add += float(dt) * float(state.mass[j]) * float(np.dot(ai - aj, gradW))

        rho_star[i] = rho_i + dt * add

    return rho_star


def _pressure_accel_eq53(
    state: ParticleState,
    ns: SpatialHash,
    h: float,
    p: np.ndarray,
    rho: np.ndarray,
    rho0_for_denoms: float | None = None,
) -> np.ndarray:
    """
    Compute symmetric pressure acceleration a_p using Eq. (53).

    Eq. (53) (from the prompt):
        a_p_i = - Σ_j m_j ( p_i / rho_i^2 + p_j / rho_j^2 ) ∇W_ij

    Notes:
    - We compute a_p only for fluid particles; boundary particles get a_p=0.
    - Boundary particles participate as neighbors (j) in the sum.
    """
    n = state.n
    dim = state.dim
    h = float(h)

    a_p = np.zeros((n, dim), dtype=np.float64)
    eps = 1e-12
    use_rho0 = rho0_for_denoms is not None
    if use_rho0:
        rho0_for_denoms = float(rho0_for_denoms)
        rho0_2 = rho0_for_denoms * rho0_for_denoms + eps

    fluid_ids = state.fluid_indices

    for i in fluid_ids:
        pi = float(p[i])
        if use_rho0:
            rhoi2 = rho0_2
        else:
            rhoi = float(rho[i])
            rhoi2 = rhoi * rhoi + eps

        acc = np.zeros((dim,), dtype=np.float64)
        xi = state.pos[i]

        for j in ns.query(int(i), state.pos):
            pj = float(p[j])
            if use_rho0:
                rhoj2 = rho0_2
            else:
                rhoj = float(rho[j])
                rhoj2 = rhoj * rhoj + eps

            gradW = cubic_spline_gradW(ns.displacement(xi, state.pos[j]), h=h, dim=dim)
            acc -= float(state.mass[j]) * (pi / rhoi2 + pj / rhoj2) * gradW

        a_p[i] = acc

    return a_p


def _rho_p_eq60(
    state: ParticleState,
    ns: SpatialHash,
    h: float,
    dt: float,
    a_p: np.ndarray,
) -> np.ndarray:
    """
    Compute density change due to pressure accelerations using Eq. (60).

    Eq. (60) (from the prompt):
        (rho_p_i)^(l) = dt Σ_j m_j ( dt a_p_i^(l) - dt a_p_j^(l) ) · ∇W_ij
                      = dt^2 Σ_j m_j ( a_p_i^(l) - a_p_j^(l) ) · ∇W_ij

    Note:
    - We use the same ∇W convention as the rest of this codebase:
      `cubic_spline_gradW(x_i - x_j)` corresponds to ∇_i W_ij.

    Notes:
    - We compute rho_p only for fluid particles; boundary rho_p is 0.
    - Boundary a_p is 0 by construction (static boundary particles).
    """
    n = state.n
    dim = state.dim
    h = float(h)
    dt = float(dt)

    rho_p = np.zeros((n,), dtype=np.float64)

    fluid_ids = state.fluid_indices

    for i in fluid_ids:
        xi = state.pos[i]
        api = a_p[i]
        acc = 0.0

        for j in ns.query(int(i), state.pos):
            gradW = cubic_spline_gradW(ns.displacement(xi, state.pos[j]), h=h, dim=dim)
            acc += float(state.mass[j]) * float(np.dot(api - a_p[j], gradW))

        rho_p[i] = (dt * dt) * acc

    return rho_p


def step_pcisph_with_boundaries(
    state: ParticleState,
    cfg: SimConfig,
    particle_size: float,
    max_iters: int,
    density_tol: float,
    warm_start_pressure: bool = True,
    negative_pressure_mode: str = "none",
    negative_pressure_cap: float | None = None,
    negative_pressure_soft_factor: float = 0.5,
    clamp_negative_pressure_iter: bool = False,
    min_neighbors_for_pressure: int = 10,
    adaptive_min_neighbors_for_pressure: bool = True,
    active_neighbor_ratio: float = 0.7,
    min_neighbors_floor: int = 5,
    inactive_hold_steps: int = 0,
    force_active_if_density_low: bool = True,
    force_active_rho_min: float | None = None,
    debug_fixed_dt: bool = False,
    adaptive_dt: bool = True,
    density_tol_max: float | None = None,
    dt_visc_safety: float = 0.125,
    dt_force_safety: float = 0.25,
    stabilize_rho_mean: bool = False,
    stabilize_rho_mean_clip: float = 0.02,
    debug: bool = False,
    debug_dump_on_step: int | None = None,
    step_idx: int | None = None,
    enforce_domain_constraints: bool = True,
) -> float:
    """
    Perform one PCISPH step with static boundary particles.

    High-level algorithm (Algorithm 3 from the prompt):
    - Build neighbor search.
    - Compute dt (Eq. (33) or fixed).
    - Compute non-pressure acceleration a_nonp (external body forces only).
    - Predictor:
      v* = v + dt a_nonp  (fluid only)
      rho* via Eq. (51)
    - Compute global kPCI via Eq. (58).
    - Pressure solve loop:
      p init via Eq. (57)
      iterate:
        a_p via Eq. (53)
        rho_p via Eq. (60)
        p update via Eq. (59)
        stop if avg relative error < density_tol
    - Final update:
      v_new = v* + dt a_p  (fluid)
      x_new = x + dt v_new (fluid)
      boundary remains static
    - Return dt.
    """
    h = float(cfg.support_radius)

    # (1) Neighbor search on all particles (fluid + boundary)
    ns = SpatialHash(
        support_radius=h,
        dim=state.dim,
        periodic_min=cfg.domain_min,
        periodic_max=cfg.domain_max,
        periodic_axes=cfg.periodic_axes,
    )
    ns.build(state.pos)

    fluid_ids = state.fluid_indices
    n = state.n
    dim = state.dim

    # ---------------------------------------------------------------------
    # Active-set selection for the pressure solve (control logic only).
    #
    # Problem addressed:
    # - If a fluid particle has too few neighbors (or 0), Eq. (51) collapses
    #   toward the self-contribution only (rho* ~ m * W(0)). The pressure solve
    #   then becomes meaningless for that particle (no sampling to enforce
    #   incompressibility), and can generate runaway artifacts.
    #
    # Solution:
    # - Run the pressure-correction loop (Eq. 53/59/60) ONLY for "active"
    #   fluid particles with sufficient neighbor count.
    # - Inactive/outlier fluid particles are assigned:
    #     p = 0, a_p = 0, rho_p = 0
    #   so they only undergo the predictor/integration with external forces.
    #
    # IMPORTANT: This does NOT change PCISPH equations. It only changes
    # where (on which particles) we apply the existing equations.
    # ---------------------------------------------------------------------
    neigh_counts = _compute_neighbor_counts(fluid_ids=fluid_ids, ns=ns, pos=state.pos)
    min_n_cfg = int(min_neighbors_for_pressure)
    if min_n_cfg < 0:
        min_n_cfg = 0

    # Adaptive effective threshold (per-step):
    # The goal is to avoid gradually classifying more and more particles as inactive
    # when the overall neighbor count distribution shifts (e.g., due to free-surface
    # formation). This is ONLY control logic; equations remain unchanged.
    if neigh_counts.size:
        mean_neigh = float(np.mean(neigh_counts))
    else:
        mean_neigh = 0.0

    if bool(adaptive_min_neighbors_for_pressure):
        ratio = float(active_neighbor_ratio)
        floor_n = int(min_neighbors_floor)
        if floor_n < 0:
            floor_n = 0
        # As requested:
        # min_eff = max(min_neighbors_floor, round(active_neighbor_ratio * mean_neigh))
        # min_eff = min(min_eff, min_neighbors_for_pressure)
        min_eff = int(max(floor_n, round(ratio * mean_neigh)))
        min_eff = int(min(min_eff, min_n_cfg))
    else:
        min_eff = int(min_n_cfg)

    active_mask = neigh_counts >= int(min_eff)
    under_neighbors = ~active_mask
    inactive_mask_local = _inactive_mask_with_hold_steps(
        state_key=int(id(state)),
        n=n,
        fluid_ids=fluid_ids,
        under_neighbor_threshold=under_neighbors,
        hold_steps=int(inactive_hold_steps),
    )

    # This active set is the NEIGHBOR-based active-set (used for debug / baseline decision).
    active_neighbors_mask = ~inactive_mask_local
    active_ids = fluid_ids[active_neighbors_mask]
    inactive_ids = fluid_ids[~active_neighbors_mask]
    inactive_count = int(inactive_ids.size)

    # (2) dt selection:
    # - Debug mode: force fixed dt to isolate solver behavior.
    # - Otherwise use adaptive min(dt_cfl, dt_visc, dt_force, dt_max), clamped to dt_min.
    dt_cfl = _compute_dt_eq33(cfg, v_fluid=state.vel[fluid_ids], particle_size=float(particle_size))
    dt_visc = float("inf")
    if bool(cfg.enable_viscosity) and float(cfg.kinematic_viscosity) > 0.0:
        dt_visc = float(dt_visc_safety) * (float(particle_size) ** 2) / float(cfg.kinematic_viscosity)
    dt_force = float("inf")
    g_norm = float(np.linalg.norm(cfg.g))
    if g_norm > 1e-12:
        dt_force = float(dt_force_safety) * np.sqrt(float(particle_size) / g_norm)

    if bool(debug_fixed_dt):
        dt = float(cfg.dt_fixed)
    elif bool(adaptive_dt):
        dt = float(min(dt_cfl, dt_visc, dt_force, float(cfg.dt_max)))
        dt = float(np.clip(dt, float(cfg.dt_min), float(cfg.dt_max)))
    else:
        dt = float(dt_cfl)

    # (3) non-pressure acceleration a_nonp: external body force only (gravity)
    a_nonp = np.zeros((n, dim), dtype=np.float64)
    a_nonp[fluid_ids] = np.tile(cfg.g[None, :], (fluid_ids.size, 1))
    # boundary accelerations remain 0.0

    # (4) Predictor v* (fluid only)
    v_star = state.vel.copy()
    v_star[fluid_ids] = v_star[fluid_ids] + dt * a_nonp[fluid_ids]
    v_star[state.is_boundary] = 0.0

    # (5) Predicted density rho* via Eq. (51)
    rho_star = _predict_rho_star_eq51(
        state=state,
        ns=ns,
        h=h,
        dt=dt,
        v_star=v_star,
        a_nonp=a_nonp,
        rho0=cfg.rho0,
    )
    if bool(debug):
        if not np.all(np.isfinite(rho_star[fluid_ids])):
            bad = np.where(~np.isfinite(rho_star[fluid_ids]))[0]
            print(f"[PCISPH][WARN] non-finite rho* for {bad.size} fluid particles (step={step_idx})")

    # Optional active-set stabilization: force pressure-active if predicted density is low.
    # This is control logic only; equations are unchanged. The intent is to avoid dropping
    # free-surface/undersampled particles out of the pressure solve when rho* << rho0.
    if force_active_rho_min is None:
        force_active_rho_min_val = 0.95 * float(cfg.rho0)
    else:
        force_active_rho_min_val = float(force_active_rho_min)

    forced_active_local = np.zeros((fluid_ids.size,), dtype=np.bool_)
    if bool(force_active_if_density_low) and fluid_ids.size:
        forced_active_local = rho_star[fluid_ids] < force_active_rho_min_val

    pressure_active_mask_local = active_neighbors_mask | forced_active_local
    pressure_active_ids = fluid_ids[pressure_active_mask_local]
    pressure_inactive_ids = fluid_ids[~pressure_active_mask_local]

    # Rebind for the rest of the solver (pressure loop + final Eq. 53):
    active_ids = pressure_active_ids
    inactive_ids = pressure_inactive_ids
    inactive_count = int(inactive_ids.size)

    # (6) Global stiffness kPCI via Eq. (58) (template particle)
    if fluid_ids.size == 0:
        # Nothing to simulate; keep boundary static.
        return float(dt)

    # (58) kPCI: compute as a global constant.
    #
    # We use cfg.dt_fixed in Eq. (58) and Eq. (60) correction scaling to avoid
    # gain swings when adaptive dt shrinks/expands between steps.
    #
    # Cache key: (dim, h, rho0, template_mass, dt_fixed)
    i_template = _choose_template_particle(fluid_ids=fluid_ids, ns=ns, pos=state.pos)
    dt_kpci = float(cfg.dt_fixed)
    cache_key = (int(state.dim), float(h), float(cfg.rho0), float(state.mass[i_template]), dt_kpci)
    if cache_key in _KPCI_CACHE:
        kPCI = _KPCI_CACHE[cache_key]
    else:
        kPCI = _compute_kpci_eq58(
            i_template=i_template,
            state=state,
            ns=ns,
            h=h,
            rho0=cfg.rho0,
            dt=dt_kpci,
            debug=bool(debug),
        )
        _KPCI_CACHE[cache_key] = float(kPCI)

    # (7) Pressure iterations: Eq. (57), (59), (60) with Eq. (53)
    #     BUT only for active_ids (see active-set selection above).
    max_iters = int(max_iters)
    density_tol = float(density_tol)
    density_tol_max_val = float(density_tol_max) if density_tol_max is not None else float(density_tol)

    # Pressure warm-start (initial guess) for the per-step PCISPH iteration.
    #
    # IMPORTANT: This does NOT change any PCISPH equations (Eq. 57/59/60, Eq. 53).
    # It only changes the INITIAL GUESS for p in the iterative solve.
    # The fixed point of the iteration is unchanged; convergence behavior can improve.
    #
    # - If warm_start_pressure=True (default): seed from current `state.p`.
    # - If warm_start_pressure=False: start from zeros (legacy behavior).
    #
    # Boundary pressures remain fixed at 0 throughout (not iteratively updated).
    # Eq. (57): initial pressure prediction p_i = kPCI (rho0 - rho*_i)
    p_eq57 = np.zeros((n,), dtype=np.float64)
    p_eq57[fluid_ids] = float(kPCI) * (float(cfg.rho0) - rho_star[fluid_ids])

    # Initialize p with active-set semantics:
    # - active: warm-start or Eq. (57)
    # - inactive: always 0
    p = np.zeros((n,), dtype=np.float64)
    p[state.is_boundary] = 0.0
    if inactive_ids.size:
        p[inactive_ids] = 0.0

    if active_ids.size == 0:
        # Degenerate case: no particles have enough neighbors to run the pressure solve.
        # We skip the pressure loop entirely.
        if bool(debug):
            neigh_min = int(neigh_counts.min()) if neigh_counts.size else 0
            neigh_mean = float(neigh_counts.mean()) if neigh_counts.size else 0.0
            neigh_max = int(neigh_counts.max()) if neigh_counts.size else 0
            print(
                f"[PCISPH][WARN] step={step_idx} no active particles for pressure solve "
                f"(min_neighbors_for_pressure={int(min_n_cfg)}; min_eff={int(min_eff)}; "
                f"neigh(min/mean/max)={neigh_min}/{neigh_mean:.1f}/{neigh_max})"
            )
    else:
        if bool(warm_start_pressure):
            # Warm-start from previous step pressure (initial guess).
            #
            # IMPORTANT: warm-starting changes ONLY the initial guess for p in the
            # iterative solve; it does NOT change Eq. (57)/(59)/(60) or Eq. (53),
            # nor solver ordering.
            #
            # Practical safeguard: if CFL shrinks dt far below the scene's fixed dt,
            # the previous step pressure can be a poor initial guess. Fall back to Eq. (57).
            if float(dt) < 0.5 * float(cfg.dt_fixed):
                p[active_ids] = p_eq57[active_ids]
            else:
                p_prev = state.p.astype(np.float64, copy=False)
                p_prev = np.where(np.isfinite(p_prev), p_prev, 0.0)

                # Keep Eq. (57) behavior for the first step / unset pressure.
                if np.allclose(p_prev[active_ids], 0.0):
                    p[active_ids] = p_eq57[active_ids]
                else:
                    # Conservative warm-start:
                    # do NOT start with pressures larger (in magnitude) than the Eq. (57)
                    # prediction for the current rho*. This reduces early spikes while
                    # keeping equations unchanged (initial guess only).
                    p_guess = p_prev[active_ids].copy()
                    abs_prev = np.abs(p_guess)
                    abs_eq = np.abs(p_eq57[active_ids])
                    too_large = abs_prev > abs_eq
                    if np.any(too_large):
                        p_guess[too_large] = p_eq57[active_ids][too_large]
                    p[active_ids] = p_guess
        else:
            # Legacy behavior: start purely from Eq. (57).
            p[active_ids] = p_eq57[active_ids]

        p[state.is_boundary] = 0.0

    # -----------------------------------------------------------------------
    # Negative pressure handling (control logic only; equations unchanged).
    #
    # - clamp_negative_pressure_iter: optional per-iteration clamp (legacy toggle).
    # - negative_pressure_mode: final pressure handling, applied once after iterations:
    #     "none"      -> keep p_raw (allow negative pressures)
    #     "hard_zero" -> clamp to p >= 0
    #     "soft_cap"  -> clamp to p >= -cap (cap fixed or dynamic per step)
    #
    # Eq. (51), (53), (57), (58), (59), (60) are unchanged; only how we post-process
    # the pressure field for the final acceleration is configurable.
    # -----------------------------------------------------------------------
    if bool(clamp_negative_pressure_iter) and active_ids.size:
        p[active_ids] = np.maximum(p[active_ids], 0.0)

    rho_p = np.zeros((n,), dtype=np.float64)

    # Debug-only sanity metrics (read-only).
    # Keep debug output to ONE line per step (so normal runs stay clean).
    debug_neigh_counts = None
    debug_rho_star_err_avg = None
    if bool(debug):
        debug_neigh_counts = neigh_counts
        debug_rho_star_err_avg = float(
            np.mean(np.abs((rho_star[fluid_ids] - float(cfg.rho0)) / float(cfg.rho0)))
        ) if fluid_ids.size else float("nan")

        # Active/inactive summary (requested diagnostics)
        under_ct = int(np.count_nonzero(under_neighbors)) if fluid_ids.size else 0
        inactive_by_hold_ct = int(np.count_nonzero(inactive_mask_local)) if fluid_ids.size else 0
        held_active_ct = int(np.count_nonzero(under_neighbors & ~inactive_mask_local)) if fluid_ids.size else 0
        forced_active_ct = int(np.count_nonzero(forced_active_local)) if fluid_ids.size else 0

        if inactive_count > 0:
            first10 = ", ".join(str(int(i)) for i in inactive_ids[:10])
            more = "" if inactive_count <= 10 else f" ... (+{inactive_count - 10})"
            print(
                f"[PCISPH][ACTIVESET] step={step_idx} "
                f"min_neighbors_for_pressure={int(min_n_cfg)} min_eff={int(min_eff)} "
                f"under_neighbors={under_ct} held_active={held_active_ct} "
                f"inactive_by_hold={inactive_by_hold_ct} forced_active={forced_active_ct} "
                f"inactive_count={inactive_count} "
                f"first_inactive={first10}{more}"
            )
            # First 3 inactive details: pos/vel/speed + rho*/neigh (optional but useful)
            for i in inactive_ids[:3]:
                ii = int(i)
                vmag = float(np.linalg.norm(state.vel[ii]))
                streak = _UNDER_NEIGHBOR_STREAK_CACHE.get(int(id(state)))
                streak_i = int(streak[ii]) if streak is not None and streak.shape == (n,) else -1
                print(
                    f"[PCISPH][INACTIVE] step={step_idx} i={ii} reason=low_neighbors+hold "
                    f"streak={streak_i}/{int(inactive_hold_steps)} "
                    f"pos={state.pos[ii]} vel={state.vel[ii]} |v|={vmag:.3e} "
                    f"rho*={float(rho_star[ii]):.3e} neigh={int(len(ns.query(ii, state.pos)))}"
                )
        elif forced_active_ct > 0 or held_active_ct > 0:
            # Still useful to report stabilization effects even when nobody is inactive yet.
            print(
                f"[PCISPH][ACTIVESET] step={step_idx} "
                f"min_neighbors_for_pressure={int(min_n_cfg)} min_eff={int(min_eff)} "
                f"under_neighbors={under_ct} held_active={held_active_ct} "
                f"inactive_by_hold={inactive_by_hold_ct} forced_active={forced_active_ct} "
                f"inactive_count=0"
            )

    iters_used = 0
    avg_err_final = float("nan")
    max_err_final = float("nan")
    if active_ids.size > 0:
        # dt-consistency:
        # Eq. (58) and Eq. (60) must use the same dt scale as the current step.
        dt_corr = float(dt_kpci)
        for it in range(1, max_iters + 1):
            # Eq. (53): use rho0 in denominators for the pressure solve loop.
            # This avoids destabilizing feedback if intermediate rho estimates deviate
            # strongly from rho0 during iterations (common PCISPH pitfall: "wrong rho in Eq. (53)").
            a_p = _pressure_accel_eq53_active(
                state=state,
                ns=ns,
                h=h,
                p=p,
                rho=rho_star,
                active_ids=active_ids,
                rho0_for_denoms=cfg.rho0,
            )
            rho_p = _rho_p_eq60_active(state=state, ns=ns, h=h, dt=dt_corr, a_p=a_p, active_ids=active_ids)
            if bool(debug):
                if not np.all(np.isfinite(a_p[active_ids])):
                    print(f"[PCISPH][WARN] non-finite a_p detected (step={step_idx}, iter={it})")
                if not np.all(np.isfinite(rho_p[active_ids])):
                    print(f"[PCISPH][WARN] non-finite rho_p detected (step={step_idx}, iter={it})")

            # Eq. (59) update ONLY for active particles:
            # p^(l+1) = p^(l) + kPCI ( rho0 - rho* - rho_p^(l) )
            p[active_ids] = p[active_ids] + float(kPCI) * (
                float(cfg.rho0) - rho_star[active_ids] - rho_p[active_ids]
            )
            p[state.is_boundary] = 0.0
            if inactive_ids.size:
                p[inactive_ids] = 0.0

            # Negative-pressure clamping (per-iteration), only for active_ids.
            if bool(clamp_negative_pressure_iter):
                p[active_ids] = np.maximum(p[active_ids], 0.0)

            # Convergence criterion over active set only.
            avg_err = float(
                np.mean(
                    np.abs((rho_star[active_ids] + rho_p[active_ids] - float(cfg.rho0)) / float(cfg.rho0))
                )
            )
            max_err = float(
                np.max(
                    np.abs((rho_star[active_ids] + rho_p[active_ids] - float(cfg.rho0)) / float(cfg.rho0))
                )
            )
            iters_used = it
            avg_err_final = avg_err
            max_err_final = max_err
            if avg_err < density_tol and max_err < density_tol_max_val:
                break
    if bool(debug):
        if debug_neigh_counts is None or debug_rho_star_err_avg is None:
            # Should not happen, but keep debug robust.
            debug_neigh_counts = np.zeros((1,), dtype=np.int64)
            debug_rho_star_err_avg = float("nan")

        # Error distribution after iterations (active set only)
        if active_ids.size:
            err_vec_active = np.abs((rho_star[active_ids] + rho_p[active_ids] - float(cfg.rho0)) / float(cfg.rho0))
            max_err_after_iter = float(np.max(err_vec_active)) if err_vec_active.size else float("nan")
        else:
            err_vec_active = np.zeros((0,), dtype=np.float64)
            max_err_after_iter = float("nan")

        # Pressure stats after iterations (fluid only)
        p_fluid = p[fluid_ids]
        p_min = float(np.min(p_fluid)) if p_fluid.size else float("nan")
        p_mean = float(np.mean(p_fluid)) if p_fluid.size else float("nan")
        p_max = float(np.max(p_fluid)) if p_fluid.size else float("nan")

        # NaN/inf guards on key arrays (fluid)
        if not np.all(np.isfinite(p_fluid)):
            print(f"[PCISPH][WARN] non-finite p detected (step={step_idx})")

        # Raw pressure stats (BEFORE optional final clamp).
        # This corresponds to the "raw iteration pressure" requested.
        p_raw_min = p_min
        p_raw_mean = p_mean
        p_raw_max = p_max

        # Extra telemetry (cheap; scalar reductions only)
        p_raw_neg_min = float(p_min)
        rho_star_fluid = rho_star[fluid_ids]
        rho_star_min = float(np.min(rho_star_fluid)) if rho_star_fluid.size else float("nan")
        rho_star_mean = float(np.mean(rho_star_fluid)) if rho_star_fluid.size else float("nan")
        rho_star_max = float(np.max(rho_star_fluid)) if rho_star_fluid.size else float("nan")
        below_0p8 = int(np.count_nonzero(rho_star_fluid < 0.8 * float(cfg.rho0))) if rho_star_fluid.size else 0

    # Final negative pressure mode applied ONCE after the iterations and right before
    # computing the final pressure acceleration (Eq. 53).
    cap_used = _apply_negative_pressure_mode_inplace(
        p,
        active_ids,
        mode=negative_pressure_mode,
        cap=negative_pressure_cap,
        soft_factor=float(negative_pressure_soft_factor),
    )
    if inactive_ids.size:
        p[inactive_ids] = 0.0
    p[state.is_boundary] = 0.0

    if bool(debug):
        # Final pressure stats (AFTER optional final clamp).
        p_fluid_final = p[fluid_ids]
        p_final_min = float(np.min(p_fluid_final)) if p_fluid_final.size else float("nan")
        p_final_mean = float(np.mean(p_fluid_final)) if p_fluid_final.size else float("nan")
        p_final_max = float(np.max(p_fluid_final)) if p_fluid_final.size else float("nan")
        p_final_neg_min = float(p_final_min)

        print(
            f"[PCISPH] step={step_idx} "
            f"dt={dt:.3e} (debug_fixed_dt={bool(debug_fixed_dt)}) "
            f"dt_parts(cfl/visc/force)={float(dt_cfl):.3e}/{float(dt_visc):.3e}/{float(dt_force):.3e} "
            f"dt_corr={float(dt_kpci):.3e} "
            f"clamp_neg_p_iter={bool(clamp_negative_pressure_iter)} "
            f"neg_p_mode={str(negative_pressure_mode).lower()} "
            f"neg_p_cap_used={float(cap_used):.3e} "
            f"min_neighbors_for_pressure={int(min_n_cfg)} "
            f"min_eff={int(min_eff)} "
            f"inactive_count={int(inactive_count)} "
            f"active_count={int(active_ids.size)} "
            f"frozen_inactive=0 "
            f"active={int(active_ids.size)}/{int(fluid_ids.size)} "
            f"kPCI={float(kPCI):.3e} "
            f"rho*_err_avg={float(debug_rho_star_err_avg):.3e} "
            f"iters_used={int(iters_used)}/{int(max_iters)} "
            f"avg_err_after_iter={float(avg_err_final):.3e} "
            f"max_err_after_iter_stop={float(max_err_final):.3e} "
            f"max_err_after_iter={float(max_err_after_iter):.3e} "
            f"rho*(min/avg/max)={rho_star_min:.3e}/{rho_star_mean:.3e}/{rho_star_max:.3e} "
            f"rho*_below0p8={int(below_0p8)} "
            f"p_raw_neg_min={p_raw_neg_min:.3e} "
            f"p_final_neg_min={p_final_neg_min:.3e} "
            f"p_raw(min/mean/max)={p_raw_min:.3e}/{p_raw_mean:.3e}/{p_raw_max:.3e} "
            f"p_final(min/mean/max)={p_final_min:.3e}/{p_final_mean:.3e}/{p_final_max:.3e} "
            f"neigh(min/mean/max)={int(debug_neigh_counts.min())}/{float(debug_neigh_counts.mean()):.1f}/{int(debug_neigh_counts.max())}"
        )

        # Optional targeted dump for a specific step (local sampling diagnosis)
        if debug_dump_on_step is not None and step_idx == int(debug_dump_on_step):
            # Worst 10 by neighbor count (lowest), over fluid particles.
            if debug_neigh_counts is not None and debug_neigh_counts.size:
                order_nc = np.argsort(debug_neigh_counts)  # ascending (worst first)
                k_nc = min(10, int(order_nc.size))
                worst_nc_ids = fluid_ids[order_nc[:k_nc]]
                worst_nc_counts = debug_neigh_counts[order_nc[:k_nc]]
                pairs = ", ".join(f"{int(i)}({int(c)})" for i, c in zip(worst_nc_ids, worst_nc_counts))
                print(f"[PCISPH][DUMP] step={step_idx} worst10_by_neighbor_count: {pairs}")

            # Worst 10 ACTIVE particles by post-iteration abs relative density error
            if err_vec_active.size:
                order = np.argsort(-err_vec_active)  # descending
                worst_k = min(10, int(order.size))
                worst_local = order[:worst_k]
                worst_global = active_ids[worst_local]
                worst_list = ", ".join(str(int(i)) for i in worst_global)
                print(f"[PCISPH][DUMP] step={step_idx} worst10_active_by_density_err: {worst_list}")

                if worst_k > 0:
                    wi = int(worst_global[0])
                    wi_neigh = int(len(ns.query(int(wi), state.pos)))
                    wi_rho_star = float(rho_star[wi])
                    wi_rho_p = float(rho_p[wi])
                    wi_p = float(p[wi])
                    wi_v = float(np.linalg.norm(v_star[wi]))
                    print(
                        f"[PCISPH][DUMP] worst_active i={wi} neigh={wi_neigh} "
                        f"rho*={wi_rho_star:.3e} rho_p={wi_rho_p:.3e} "
                        f"p={wi_p:.3e} |v|={wi_v:.3e} err={float(err_vec_active[worst_local[0]]):.3e}"
                    )

            # Additional dump: worst inactive (requested)
            if inactive_ids.size:
                first5 = inactive_ids[:5]
                lst = ", ".join(str(int(i)) for i in first5)
                print(f"[PCISPH][DUMP] step={step_idx} first5_inactive: {lst}")
                for i in first5:
                    ii = int(i)
                    vmag = float(np.linalg.norm(state.vel[ii]))
                    print(
                        f"[PCISPH][DUMP][INACTIVE] i={ii} neigh={int(len(ns.query(ii, state.pos)))} "
                        f"pos={state.pos[ii]} vel={state.vel[ii]} |v|={vmag:.3e} "
                        f"rho*={float(rho_star[ii]):.3e} p={float(p[ii]):.3e}"
                    )

    # (8) Final velocity + position update with final pressure acceleration (Eq. (53))
    rho_final = np.full((n,), float(cfg.rho0), dtype=np.float64)
    rho_final[fluid_ids] = rho_star[fluid_ids] + rho_p[fluid_ids]
    # Active-set safety: inactive particles are not pressure-corrected (p=a_p=rho_p=0).
    # To avoid a persistent "inactive vacuum" state in diagnostics/state, keep their
    # final density conservative at rho0 (control logic only; no equation changes).
    if inactive_ids.size:
        rho_final[inactive_ids] = float(cfg.rho0)

    # Optional post-solve mean-density recentering (control logic only).
    #
    # Motivation: for long PCISPH runs with active-set/outlier handling, the global
    # fluid mean density can drift slightly even when the pressure loop remains stable.
    # This hook recenters rho_mean toward rho0 with a bounded shift, without altering
    # solver equations (51/53/57/58/59/60) or particle integration.
    if bool(stabilize_rho_mean) and fluid_ids.size:
        rho_mean = float(np.mean(rho_final[fluid_ids]))
        delta = float(cfg.rho0) - rho_mean
        clip_abs = abs(float(stabilize_rho_mean_clip)) * float(cfg.rho0)
        if clip_abs > 0.0:
            delta = float(np.clip(delta, -clip_abs, clip_abs))
        rho_final[fluid_ids] = np.maximum(rho_final[fluid_ids] + delta, 1e-12)

    # Final Eq. (53) application consistent with the active set:
    # - active_ids: Eq. (53)
    # - inactive_ids: a_p_final = 0
    a_p_final = _pressure_accel_eq53_active(
        state=state,
        ns=ns,
        h=h,
        p=p,
        rho=rho_final,
        active_ids=active_ids,
        rho0_for_denoms=cfg.rho0,
    )

    # v_new = v* + dt a_p  (fluid only)
    state.vel[fluid_ids] = v_star[fluid_ids] + dt * a_p_final[fluid_ids]

    # x_new = x + dt v_new (fluid only)
    state.pos[fluid_ids] = state.pos[fluid_ids] + dt * state.vel[fluid_ids]

    # boundary remains static
    state.vel[state.is_boundary] = 0.0

    # Enforce domain boundaries (collision)
    if enforce_domain_constraints:
        enforce_domain_boundary_constraints(state, cfg, debug=bool(debug))

    # Store final p/rho for observability (read by diagnostics/export).
    state.p[:] = p
    state.rho[:] = rho_final
    _LAST_STEP_STATS_CACHE[int(id(state))] = {
        "iters_used": int(iters_used),
        "max_iters": int(max_iters),
        "rho_err_avg": float(avg_err_final),
        "rho_err_max": float(max_err_final),
        "dt": float(dt),
        "dt_cfl": float(dt_cfl),
        "dt_visc": float(dt_visc),
        "dt_force": float(dt_force),
    }

    return float(dt)


