import os

os.environ.setdefault("NUMBA_ENABLE_CUDASIM", "1")

import numpy as np
from numba import cuda

from sph.core.backends.cuda_neighbor_search import CUDANeighborSearch
from sph.neighbor_search_kdtree import KDTreeNeighborSearch


def _sorted_pair_array(idx_i: np.ndarray, idx_j: np.ndarray) -> np.ndarray:
    if idx_i.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    order = np.lexsort((idx_j, idx_i))
    return np.column_stack((idx_i[order], idx_j[order])).astype(np.int32, copy=False)


def _ff_distances(positions: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    if pairs.size == 0:
        return np.zeros(0, dtype=np.float64)
    delta = positions[pairs[:, 0]] - positions[pairs[:, 1]]
    return np.sqrt(np.einsum("ij,ij->i", delta, delta))


def _fb_distances(fluid_positions: np.ndarray, boundary_positions: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    if pairs.size == 0:
        return np.zeros(0, dtype=np.float64)
    delta = fluid_positions[pairs[:, 0]] - boundary_positions[pairs[:, 1]]
    return np.sqrt(np.einsum("ij,ij->i", delta, delta))


def _assert_pair_match(
    fluid_positions: np.ndarray,
    boundary_positions: np.ndarray,
    support_radius: float,
    domain_min: np.ndarray,
    domain_max: np.ndarray,
    periodic_x: tuple[float, float] | None = None,
) -> None:
    cpu_search = KDTreeNeighborSearch(
        support_radius=support_radius,
        dim=2,
        periodic_x=periodic_x,
    )
    cpu_pairs = cpu_search.build_neighbor_pairs(fluid_positions, boundary_positions)

    gpu_search = CUDANeighborSearch(
        support_radius=support_radius,
        domain_min=domain_min,
        domain_max=domain_max,
        n_fluid_max=int(fluid_positions.shape[0]),
        n_boundary=int(boundary_positions.shape[0]),
        max_pairs=256,
        periodic_x=periodic_x,
    )

    fluid_gpu = cuda.to_device(np.ascontiguousarray(fluid_positions))
    boundary_gpu = cuda.to_device(np.ascontiguousarray(boundary_positions))
    ff_i_gpu, ff_j_gpu, n_ff, fb_i_gpu, fb_j_gpu, n_fb = gpu_search.build(fluid_gpu, boundary_gpu)

    assert n_ff == int(cpu_pairs.ff_i.size)
    assert n_fb == int(cpu_pairs.fb_i.size)

    cpu_ff_pairs = _sorted_pair_array(cpu_pairs.ff_i, cpu_pairs.ff_j)
    cpu_fb_pairs = _sorted_pair_array(cpu_pairs.fb_i, cpu_pairs.fb_j)
    gpu_ff_pairs = _sorted_pair_array(ff_i_gpu.copy_to_host(), ff_j_gpu.copy_to_host())
    gpu_fb_pairs = _sorted_pair_array(fb_i_gpu.copy_to_host(), fb_j_gpu.copy_to_host())

    assert np.array_equal(gpu_ff_pairs, cpu_ff_pairs)
    assert np.array_equal(gpu_fb_pairs, cpu_fb_pairs)

    np.testing.assert_allclose(
        _ff_distances(fluid_positions, gpu_ff_pairs),
        _ff_distances(fluid_positions, cpu_ff_pairs),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        _fb_distances(fluid_positions, boundary_positions, gpu_fb_pairs),
        _fb_distances(fluid_positions, boundary_positions, cpu_fb_pairs),
        atol=1.0e-12,
    )


def test_cuda_neighbor_search_matches_cpu_pairs_non_periodic() -> None:
    fluid_positions = np.array([
        [0.10, 0.10],
        [0.28, 0.12],
        [0.46, 0.11],
        [0.22, 0.31],
        [0.49, 0.34],
    ], dtype=np.float64)
    boundary_positions = np.array([
        [0.02, 0.08],
        [0.55, 0.30],
        [0.24, 0.52],
    ], dtype=np.float64)

    _assert_pair_match(
        fluid_positions=fluid_positions,
        boundary_positions=boundary_positions,
        support_radius=0.26,
        domain_min=np.array([0.0, 0.0], dtype=np.float64),
        domain_max=np.array([0.8, 0.8], dtype=np.float64),
    )


def test_cuda_neighbor_search_matches_cpu_pairs_periodic_x() -> None:
    fluid_positions = np.array([
        [0.05, 0.10],
        [0.94, 0.11],
        [0.50, 0.10],
        [0.52, 0.28],
    ], dtype=np.float64)
    boundary_positions = np.array([
        [0.97, 0.09],
        [0.02, 0.29],
        [0.49, 0.42],
    ], dtype=np.float64)

    _assert_pair_match(
        fluid_positions=fluid_positions,
        boundary_positions=boundary_positions,
        support_radius=0.18,
        domain_min=np.array([0.0, 0.0], dtype=np.float64),
        domain_max=np.array([1.0, 0.6], dtype=np.float64),
        periodic_x=(0.0, 1.0),
    )
