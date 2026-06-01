from __future__ import annotations

from sph.core.backends.numba_cuda_backend import NumbaCUDABackend


class _DummyHostBackend:
    def __init__(self, sim) -> None:
        self.sim = sim


def test_numba_cuda_backend_exposes_host_sim_compat_alias() -> None:
    sentinel = object()
    backend = object.__new__(NumbaCUDABackend)
    backend._host_backend = _DummyHostBackend(sentinel)

    assert backend.sim is sentinel
