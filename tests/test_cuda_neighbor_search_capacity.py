from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from sph.core.backends.cuda_neighbor_search import CUDANeighborSearch


def test_next_pair_capacity_grows_with_headroom() -> None:
    new_capacity = CUDANeighborSearch._next_pair_capacity(500_000, 2_859_230)

    assert new_capacity > 2_859_230
    assert new_capacity >= 4_000_000


def test_host_fallback_grows_capacity_instead_of_raising(monkeypatch) -> None:
    pairs = SimpleNamespace(
        ff_i=np.arange(12, dtype=np.int32),
        ff_j=np.arange(12, dtype=np.int32),
        fb_i=np.arange(3, dtype=np.int32),
        fb_j=np.arange(3, dtype=np.int32),
    )

    class _DummySearch:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def build_neighbor_pairs(self, fluid_positions, boundary_positions):
            return pairs

    monkeypatch.setattr(
        "sph.core.backends.cuda_neighbor_search.KDTreeNeighborSearch",
        _DummySearch,
    )

    search = object.__new__(CUDANeighborSearch)
    search.support_radius = 0.1
    search.dim = 2
    search.periodic_x = None
    search.max_pairs = 4
    search._host_fallback = True
    search._last_build_timings_ms = {}

    ff_i, ff_j, n_ff, fb_i, fb_j, n_fb = search._build_host_fallback(
        np.zeros((8, 2), dtype=np.float64),
        np.zeros((4, 2), dtype=np.float64),
    )

    assert n_ff == 12
    assert n_fb == 3
    assert search.max_pairs > 12
    np.testing.assert_array_equal(ff_i.copy_to_host(), pairs.ff_i)
    np.testing.assert_array_equal(ff_j.copy_to_host(), pairs.ff_j)
    np.testing.assert_array_equal(fb_i.copy_to_host(), pairs.fb_i)
    np.testing.assert_array_equal(fb_j.copy_to_host(), pairs.fb_j)
