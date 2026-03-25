"""
Numba-JIT accelerated SPH kernel and scatter operations.

Matches the CubicSplineKernel in kernel.py exactly:
  q = dist / h, support q < 2.0 (dist < 2h)
  W   = alpha * { 1 - 1.5 q² + 0.75 q³   for q < 1
                  0.25 (2 - q)³          for 1 <= q < 2
                  0                      otherwise }
  alpha = 10 / (7 π h²)  in 2D

All functions accept/return plain numpy arrays — no Python object
overhead. First call triggers JIT compilation (~1-2 s); subsequent
calls run at near-C speed.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange


# ──────────────────────────────────────────────────────── kernel evaluation

@njit(cache=True)
def cubic_spline_W_batch(dist: np.ndarray, h: float) -> np.ndarray:
    """
    Evaluate W(dist, h) for all pairs.

    Returns array of shape (N,), matching kernel.py W() exactly.
    """
    alpha = 10.0 / (7.0 * np.pi * h * h)
    n = len(dist)
    w = np.zeros(n)
    for k in range(n):
        q = dist[k] / h
        if q < 1.0:
            w[k] = alpha * (1.0 - 1.5 * q * q + 0.75 * q * q * q)
        elif q < 2.0:
            t = 2.0 - q
            w[k] = alpha * 0.25 * t * t * t
    return w


@njit(cache=True)
def cubic_spline_gradW_batch(r_ij: np.ndarray,
                              dist: np.ndarray,
                              h: float) -> np.ndarray:
    """
    Evaluate ∇W(r_ij, h) for all pairs.

    r_ij : (N, 2)  displacement vectors r_i - r_j
    dist : (N,)    distances ||r_ij||
    Returns grad_W : (N, 2), matching kernel.py grad_W() exactly.
    """
    alpha = 10.0 / (7.0 * np.pi * h * h)
    n = len(dist)
    gw = np.zeros((n, 2))
    for k in range(n):
        d = dist[k]
        if d < 1e-12:
            continue
        q = d / h
        if q < 1.0:
            dw_dq = alpha * (-3.0 * q + 2.25 * q * q)
        elif q < 2.0:
            t = 2.0 - q
            dw_dq = alpha * (-0.75 * t * t)
        else:
            continue
        # grad_W = (dW/dq / h) * (r / |r|)
        scale = dw_dq / (h * d)
        gw[k, 0] = scale * r_ij[k, 0]
        gw[k, 1] = scale * r_ij[k, 1]
    return gw


# ──────────────────────────────────────────────────────── scatter operations

@njit(parallel=True, cache=True)
def accumulate_density(densities: np.ndarray,
                       idx_i: np.ndarray,
                       idx_j: np.ndarray,
                       mass: float,
                       weights: np.ndarray) -> None:
    """Symmetric accumulation used for density sums."""
    for k in prange(len(idx_i)):
        val = mass * weights[k]
        densities[idx_i[k]] += val
        densities[idx_j[k]] += val


@njit(parallel=True, cache=True)
def accumulate_force(arr: np.ndarray,
                     idx_i: np.ndarray,
                     idx_j: np.ndarray,
                     vectors: np.ndarray) -> None:
    """Add vector to i and subtract from j (Newton's 3rd law)."""
    n_comp = vectors.shape[1]
    for k in prange(len(idx_i)):
        i = idx_i[k]
        j = idx_j[k]
        for d in range(n_comp):
            val = vectors[k, d]
            arr[i, d] += val
            arr[j, d] -= val


@njit(parallel=True, cache=True)
def accumulate_difference(arr: np.ndarray,
                          idx_i: np.ndarray,
                          idx_j: np.ndarray,
                          values: np.ndarray) -> None:
    """Add value to i and subtract from j for scalar arrays."""
    for k in prange(len(idx_i)):
        val = values[k]
        i = idx_i[k]
        j = idx_j[k]
        arr[i] += val
        arr[j] -= val

@njit(cache=True)
def scatter_add_1d(arr: np.ndarray,
                   idx: np.ndarray,
                   values: np.ndarray) -> None:
    """
    arr[idx[k]] += values[k]  for all k.
    Replaces: np.add.at(arr, idx, values)
    Single-sided (idx contains all directions already).
    """
    for k in range(len(idx)):
        arr[idx[k]] += values[k]


@njit(cache=True)
def scatter_add_2d(arr: np.ndarray,
                   idx: np.ndarray,
                   vectors: np.ndarray) -> None:
    """
    arr[idx[k], :] += vectors[k, :]  for all k.
    Replaces: np.add.at(arr, idx, vectors)
    Single-sided (idx contains all directions already).
    """
    for k in range(len(idx)):
        i = idx[k]
        arr[i, 0] += vectors[k, 0]
        arr[i, 1] += vectors[k, 1]


@njit(cache=True)
def scatter_add_2d_symm(arr: np.ndarray,
                        idx_i: np.ndarray,
                        idx_j: np.ndarray,
                        vectors: np.ndarray) -> None:
    """
    arr[i] += vectors[k], arr[j] -= vectors[k]  for all k.
    For use when only one direction per pair is stored (e.g. DFSPH
    pressure velocity updates where antisymmetric correction is needed).
    """
    for k in range(len(idx_i)):
        i = idx_i[k]
        j = idx_j[k]
        arr[i, 0] += vectors[k, 0]
        arr[i, 1] += vectors[k, 1]
        arr[j, 0] -= vectors[k, 0]
        arr[j, 1] -= vectors[k, 1]


# ──────────────────────────────────────────────────────── full force loops

@njit(cache=True)
def viscosity_accum(acc: np.ndarray,
                    idx_i: np.ndarray,
                    idx_j: np.ndarray,
                    r_ij: np.ndarray,
                    dist: np.ndarray,
                    vel_i: np.ndarray,
                    vel_j: np.ndarray,
                    h: float,
                    nu: float,
                    mass: float,
                    rho0: float) -> None:
    """
    Compute and accumulate Morris (1997) viscosity forces in one JIT pass.

    Avoids allocating grad_W, v_ij, and rdotgw arrays and the numpy.sum
    reduction over pairs.  r_ij · ∇W is computed analytically as dW/dq * q.

    acc[idx_i[k]] += coeff * (vel_i[k] - vel_j[k])  for each pair k.
    """
    alpha  = 40.0 / (7.0 * np.pi * h * h)
    h2eps  = 0.01 * h * h
    base   = 2.0 * nu * mass / rho0

    for k in range(len(idx_i)):
        d = dist[k]
        if d < 1e-12:
            continue
        q = d / h
        if q <= 0.5:
            dw_dq = alpha * (18.0 * q * q - 12.0 * q)
        elif q <= 1.0:
            t = 1.0 - q
            dw_dq = alpha * (-6.0 * t * t)
        else:
            continue
        # r_ij · grad_W = dW/dq * q  (analytical scalar)
        rdotgw = dw_dq * q
        denom  = d * d + h2eps
        c = base * rdotgw / denom

        i = idx_i[k]
        acc[i, 0] += c * (vel_i[k, 0] - vel_j[k, 0])
        acc[i, 1] += c * (vel_i[k, 1] - vel_j[k, 1])


# ──────────────────────────────────────────────────────── JIT warm-up

def warmup_numba(h: float = 0.024) -> None:
    """
    Trigger JIT compilation for all kernels.
    Call once at startup to avoid cold-start delay on first sim step.
    Takes ~1-2 s but makes subsequent calls near-instant.
    """
    dummy_dist = np.array([0.1 * h, 0.4 * h, 0.8 * h], dtype=np.float64)
    dummy_r    = np.array([[0.1 * h, 0.0],
                            [0.0, 0.4 * h],
                            [0.6 * h, 0.5 * h]], dtype=np.float64)
    dummy_arr1 = np.zeros(4, dtype=np.float64)
    dummy_arr2 = np.zeros((4, 2), dtype=np.float64)
    dummy_idx  = np.array([0, 1, 2], dtype=np.int32)
    dummy_idx2 = np.array([1, 2, 3], dtype=np.int32)
    dummy_vec  = np.zeros((3, 2), dtype=np.float64)
    dummy_val  = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    cubic_spline_W_batch(dummy_dist, h)
    cubic_spline_gradW_batch(dummy_r, dummy_dist, h)
    accumulate_density(dummy_arr1, dummy_idx, dummy_idx2, 1.0, dummy_val)
    accumulate_force(dummy_arr2, dummy_idx, dummy_idx2, dummy_vec)
    accumulate_difference(dummy_arr1, dummy_idx, dummy_idx2, dummy_val)
    scatter_add_1d(dummy_arr1, dummy_idx, dummy_val)
    scatter_add_2d(dummy_arr2, dummy_idx, dummy_vec)
    scatter_add_2d_symm(dummy_arr2, dummy_idx, dummy_idx2, dummy_vec)
