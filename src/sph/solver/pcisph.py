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
from sph.core.simulator import SimConfig  # configuration container (no solver math here)
from sph.neighbors.spatial_hash import SpatialHash
from sph.sph.kernels import cubic_spline_W, cubic_spline_gradW


#
# Module-level cache for kPCI.
#
# The prompt explicitly allows computing kPCI "once per step (or once at start)".
# We compute it once at the first PCISPH step and reuse it to improve numerical
# robustness when dt changes due to CFL (Eq. (33)). This does not affect WCSPH.
#
_KPCI_CACHE: dict[tuple[int, float, float, float, float], float] = {}


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

    for j in ns.query(int(i_template), state.pos):
        gradW = cubic_spline_gradW(xi - state.pos[j], h=h, dim=dim)
        S += gradW
        Q += float(np.dot(gradW, gradW))

    denom = float(np.dot(S, S) + Q)

    # Numeric guard only: avoid division by ~0 if a particle has no neighbors.
    # This does not change intended physics; it prevents a crash in degenerate cases.
    denom = max(denom, 1e-12)

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
            rij = xi - state.pos[j]
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

            gradW = cubic_spline_gradW(xi - state.pos[j], h=h, dim=dim)
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
            gradW = cubic_spline_gradW(xi - state.pos[j], h=h, dim=dim)
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
    debug_fixed_dt: bool = False,
    debug: bool = False,
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
    ns = SpatialHash(support_radius=h, dim=state.dim)
    ns.build(state.pos)

    fluid_ids = state.fluid_indices

    # (2) dt selection:
    # - Normally: same policy as WCSPH (CFL Eq. (33) or fixed)
    # - Debug mode: force fixed dt to isolate PCISPH correctness from CFL feedback
    if bool(debug_fixed_dt):
        dt = float(cfg.dt_fixed)
    else:
        dt = _compute_dt_eq33(cfg, v_fluid=state.vel[fluid_ids], particle_size=float(particle_size))

    # (3) non-pressure acceleration a_nonp: external body force only (gravity)
    n = state.n
    dim = state.dim
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

    # (6) Global stiffness kPCI via Eq. (58) (template particle)
    if fluid_ids.size == 0:
        # Nothing to simulate; keep boundary static.
        return float(dt)

    # (58) kPCI: compute as a global constant.
    #
    # The prompt notes kPCI is commonly computed once using a template particle
    # with perfect sampling, and for our implementation allows "once per step
    # (or once at start)" with a single global scalar.
    #
    # IMPORTANT for stability under variable dt (CFL Eq. (33)):
    # Eq. (58) includes dt^2 in the denominator. If dt shrinks due to CFL,
    # recomputing kPCI with the smaller dt makes kPCI grow ~ 1/dt^2 and can
    # create a runaway feedback loop (large pressure -> large v -> smaller dt -> larger kPCI).
    #
    # To follow the "global constant" intent and avoid dt-feedback, we compute
    # kPCI using the scene's fixed dt (cfg.dt_fixed) and cache it.
    # This keeps the PCISPH pressure solver scale consistent across steps.
    #
    # Cache key: (dim, h, rho0, template_mass, dt_kpci)
    i_template = _choose_template_particle(fluid_ids=fluid_ids, ns=ns, pos=state.pos)
    dt_kpci = float(cfg.dt_fixed)
    cache_key = (int(state.dim), float(h), float(cfg.rho0), float(state.mass[i_template]), dt_kpci)
    if cache_key in _KPCI_CACHE:
        kPCI = _KPCI_CACHE[cache_key]
    else:
        kPCI = _compute_kpci_eq58(i_template=i_template, state=state, ns=ns, h=h, rho0=cfg.rho0, dt=dt_kpci)
        _KPCI_CACHE[cache_key] = float(kPCI)

    # (7) Pressure iterations: Eq. (57), (59), (60) with Eq. (53)
    max_iters = int(max_iters)
    density_tol = float(density_tol)

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

    if bool(warm_start_pressure):
        # Warm-start from previous step pressure (initial guess).
        #
        # IMPORTANT: warm-starting changes ONLY the initialization of the iterative
        # solve; it does NOT change Eq. (57)/(59)/(60) or Eq. (53), nor solver ordering.
        #
        # Practical safeguard: if CFL shrinks dt far below the scene's fixed dt,
        # the previous step pressure can be a poor initial guess. Fall back to Eq. (57).
        if float(dt) < 0.5 * float(cfg.dt_fixed):
            p = p_eq57
        else:
            p = state.p.astype(np.float64, copy=True)
            p[~np.isfinite(p)] = 0.0
            p[state.is_boundary] = 0.0

            # Keep Eq. (57) behavior for the first step / unset pressure.
            if np.allclose(p[fluid_ids], 0.0):
                p[fluid_ids] = p_eq57[fluid_ids]
            else:
                # Conservative warm-start:
                # do NOT start with pressures larger (in magnitude) than the Eq. (57)
                # prediction for the current rho*. This can reduce early local spikes
                # without changing any equations (only the initial guess).
                abs_prev = np.abs(p[fluid_ids])
                abs_eq = np.abs(p_eq57[fluid_ids])
                too_large = abs_prev > abs_eq
                if np.any(too_large):
                    p[fluid_ids[too_large]] = p_eq57[fluid_ids[too_large]]
    else:
        # Legacy behavior: start purely from Eq. (57) (equivalent to starting at 0 then applying Eq. 57).
        p = p_eq57

    p[state.is_boundary] = 0.0

    rho_p = np.zeros((n,), dtype=np.float64)

    # Debug-only sanity metrics (read-only).
    # Keep debug output to ONE line per step (so normal runs stay clean).
    debug_neigh_counts = None
    debug_rho_star_err_avg = None
    if bool(debug):
        debug_neigh_counts = np.array([len(ns.query(int(i), state.pos)) for i in fluid_ids], dtype=np.int64)
        debug_rho_star_err_avg = float(np.mean(np.abs((rho_star[fluid_ids] - float(cfg.rho0)) / float(cfg.rho0))))

    iters_used = 0
    avg_err_final = float("nan")
    for it in range(1, max_iters + 1):
        # Eq. (53): use rho0 in denominators for the pressure solve loop.
        # This avoids destabilizing feedback if intermediate rho estimates deviate
        # strongly from rho0 during iterations (common PCISPH pitfall: "wrong rho in Eq. (53)").
        a_p = _pressure_accel_eq53(state=state, ns=ns, h=h, p=p, rho=rho_star, rho0_for_denoms=cfg.rho0)
        rho_p = _rho_p_eq60(state=state, ns=ns, h=h, dt=dt, a_p=a_p)

        # Eq. (59): p^(l+1) = p^(l) + kPCI ( rho0 - rho* - rho_p^(l) )
        p[fluid_ids] = p[fluid_ids] + float(kPCI) * (float(cfg.rho0) - rho_star[fluid_ids] - rho_p[fluid_ids])
        p[state.is_boundary] = 0.0

        # avg_err = mean( abs((rho* + rho_p - rho0)/rho0) ) over fluid
        avg_err = float(np.mean(np.abs((rho_star[fluid_ids] + rho_p[fluid_ids] - float(cfg.rho0)) / float(cfg.rho0))))
        iters_used = it
        avg_err_final = avg_err
        if avg_err < density_tol:
            break
    if bool(debug):
        if debug_neigh_counts is None or debug_rho_star_err_avg is None:
            # Should not happen, but keep debug robust.
            debug_neigh_counts = np.zeros((1,), dtype=np.int64)
            debug_rho_star_err_avg = float("nan")
        print(
            f"[PCISPH] dt={dt:.3e} (debug_fixed_dt={bool(debug_fixed_dt)}) "
            f"kPCI={float(kPCI):.3e} "
            f"rho*_err_avg={float(debug_rho_star_err_avg):.3e} "
            f"iters_used={int(iters_used)}/{int(max_iters)} "
            f"avg_err_final={float(avg_err_final):.3e} "
            f"neigh(min/mean/max)={int(debug_neigh_counts.min())}/{float(debug_neigh_counts.mean()):.1f}/{int(debug_neigh_counts.max())}"
        )

    # (8) Final velocity + position update with final pressure acceleration (Eq. (53))
    rho_final = np.full((n,), float(cfg.rho0), dtype=np.float64)
    rho_final[fluid_ids] = rho_star[fluid_ids] + rho_p[fluid_ids]

    a_p_final = _pressure_accel_eq53(state=state, ns=ns, h=h, p=p, rho=rho_final, rho0_for_denoms=cfg.rho0)

    # v_new = v* + dt a_p  (fluid only)
    state.vel[fluid_ids] = v_star[fluid_ids] + dt * a_p_final[fluid_ids]

    # x_new = x + dt v_new (fluid only)
    state.pos[fluid_ids] = state.pos[fluid_ids] + dt * state.vel[fluid_ids]

    # boundary remains static
    state.vel[state.is_boundary] = 0.0

    # Store final p/rho for observability (read by diagnostics/export).
    state.p[:] = p
    state.rho[:] = rho_final

    return float(dt)


