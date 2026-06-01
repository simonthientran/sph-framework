"""
Z-sort (Morton code) particle reordering.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _spread_bits_3d(v: int) -> int:
    v &= 0x1FFFFF
    v = (v | (v << 32)) & 0x1F00000000FFFF
    v = (v | (v << 16)) & 0x1F0000FF0000FF
    v = (v | (v << 8)) & 0x100F00F00F00F00F
    v = (v | (v << 4)) & 0x10C30C30C30C30C3
    v = (v | (v << 2)) & 0x1249249249249249
    return v


@njit(cache=True)
def _morton_encode_3d(x: int, y: int, z: int) -> int:
    return _spread_bits_3d(x) | (_spread_bits_3d(y) << 1) | (_spread_bits_3d(z) << 2)


def compute_morton_order(
    positions: np.ndarray,
    domain_min: np.ndarray,
    domain_max: np.ndarray,
    grid_res: int = 1024,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2:
        raise ValueError(f"positions must have shape (N, dim), got {positions.shape}")
    if positions.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)

    domain_min = np.asarray(domain_min, dtype=np.float64)
    domain_max = np.asarray(domain_max, dtype=np.float64)
    domain_size = np.where((domain_max - domain_min) > 1.0e-10, domain_max - domain_min, 1.0)
    norm = ((positions - domain_min[None, :]) / domain_size[None, :]) * float(grid_res - 1)
    norm = np.clip(norm, 0.0, float(grid_res - 1)).astype(np.int64)

    codes = np.zeros(positions.shape[0], dtype=np.int64)
    if positions.shape[1] == 2:
        for i in range(positions.shape[0]):
            codes[i] = _morton_encode_3d(int(norm[i, 0]), int(norm[i, 1]), 0)
    else:
        for i in range(positions.shape[0]):
            codes[i] = _morton_encode_3d(int(norm[i, 0]), int(norm[i, 1]), int(norm[i, 2]))
    return np.argsort(codes, kind="mergesort")


def apply_zsort(
    fluid,
    domain_min: np.ndarray,
    domain_max: np.ndarray,
    sort_every: int = 10,
    step: int = 0,
) -> bool:
    if fluid.n <= 1:
        return False
    if sort_every > 0 and step % sort_every != 0:
        return False

    idx = compute_morton_order(fluid.positions, domain_min, domain_max)
    if idx.shape[0] != fluid.n:
        return False

    fluid.positions = fluid.positions[idx]
    fluid.velocities = fluid.velocities[idx]
    fluid.densities = fluid.densities[idx]
    fluid.pressures = fluid.pressures[idx]
    fluid.accelerations = fluid.accelerations[idx]
    fluid.k_dfsph = fluid.k_dfsph[idx]
    fluid.p_cd_prev = fluid.p_cd_prev[idx]
    fluid.p_df_prev = fluid.p_df_prev[idx]
    fluid.rho_self = fluid.rho_self[idx]
    fluid.rho_ff = fluid.rho_ff[idx]
    fluid.rho_fb = fluid.rho_fb[idx]
    fluid.neighbor_pairs = None
    return True
