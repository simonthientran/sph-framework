from __future__ import annotations

import numpy as np

from sph.fluid_model import FluidModel
from sph.utils.zsort import apply_zsort, compute_morton_order


def test_compute_morton_order_returns_permutation():
    rng = np.random.default_rng(42)
    pos = rng.random((1000, 3))
    idx = compute_morton_order(pos, pos.min(axis=0), pos.max(axis=0))
    assert idx.shape == (1000,)
    assert len(set(idx.tolist())) == 1000


def test_apply_zsort_reorders_all_fluid_state_arrays():
    fluid = FluidModel(4, rho0=1000.0, mass=1.0, h=0.1, dim=3)
    fluid.positions[:] = np.array(
        [
            [0.8, 0.8, 0.8],
            [0.1, 0.1, 0.1],
            [0.6, 0.6, 0.6],
            [0.3, 0.3, 0.3],
        ],
        dtype=np.float64,
    )
    fluid.velocities[:, 0] = np.array([8.0, 1.0, 6.0, 3.0], dtype=np.float64)
    fluid.densities[:] = np.array([80.0, 10.0, 60.0, 30.0], dtype=np.float64)
    fluid.k_dfsph[:] = np.array([800.0, 100.0, 600.0, 300.0], dtype=np.float64)

    applied = apply_zsort(fluid, np.zeros(3), np.ones(3), sort_every=1, step=1)

    assert applied is True
    assert np.all(np.diff(fluid.positions[:, 0]) >= 0.0)
    assert fluid.velocities[0, 0] == 1.0
    assert fluid.densities[0] == 10.0
    assert fluid.k_dfsph[0] == 100.0
