from __future__ import annotations

import numpy as np

from sph.core.state import ParticleState


def l2_error(measured: np.ndarray, reference: np.ndarray) -> float:
    m = np.asarray(measured, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    mask = np.isfinite(m) & np.isfinite(r)
    if not np.any(mask):
        return float("nan")
    d = m[mask] - r[mask]
    return float(np.sqrt(np.mean(d * d)))


def linf_error(measured: np.ndarray, reference: np.ndarray) -> float:
    m = np.asarray(measured, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    mask = np.isfinite(m) & np.isfinite(r)
    if not np.any(mask):
        return float("nan")
    return float(np.max(np.abs(m[mask] - r[mask])))


def density_error_stats(rho: np.ndarray, rho0: float) -> dict[str, float]:
    rel = np.abs((np.asarray(rho, dtype=np.float64) - float(rho0)) / float(rho0))
    if rel.size == 0:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": float(np.min(rel)),
        "mean": float(np.mean(rel)),
        "max": float(np.max(rel)),
    }


def mass_total(state: ParticleState) -> float:
    # With fixed particle masses in this framework, this should be invariant.
    return float(np.sum(state.mass))


def momentum_total(state: ParticleState) -> np.ndarray:
    p = state.mass[:, None] * state.vel
    return np.sum(p, axis=0).astype(np.float64)


def kinetic_energy_total(state: ParticleState) -> float:
    v2 = np.sum(state.vel * state.vel, axis=1)
    return float(0.5 * np.sum(state.mass * v2))


def has_non_finite_state(state: ParticleState) -> bool:
    arrays = [state.pos, state.vel, state.acc, state.mass, state.rho, state.p]
    return any((not np.isfinite(a).all()) for a in arrays)

