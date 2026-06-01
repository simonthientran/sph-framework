"""
GPU hash-grid neighbor search.

Pair indices (ff_i, ff_j, fb_i, fb_j) are built and kept on the device.
When CuPy is available on a real CUDA device, particles are sorted by cell id
with ``cp.argsort``. When it is not available, a pure-Numba counting-sort
fallback keeps the build device-resident and remains compatible with CUDASIM.

Periodic BC: x-dimension only, minimum-image convention, matching the
CPU KDTreeNeighborSearch and the CUDA pair-geometry kernels exactly.

Supports both 2D (9-cell stencil) and 3D (27-cell stencil) via the ``dim``
parameter.

Performance design
------------------
By default ``CUDANeighborSearch`` builds pairs WITHOUT any intermediate GPU
event synchronization.  The pair-count download at the end of ``build()``
provides the only synchronization barrier, which is sufficient for
correctness on a single CUDA stream.

Setting ``collect_sub_timings=True`` (or calling ``enable_pair_build_diagnostics()``)
re-enables per-phase GPU event timing for profiling.  This adds ~4 sync
points per call and inflates timing by roughly the cost being measured.
"""
from __future__ import annotations

import os
import time

import numpy as np
from numba import cuda, int32

from sph.cuda_pair_ops import launch_config  # shared utility
from sph.neighbor_search_kdtree import KDTreeNeighborSearch

try:  # pragma: no cover - exercised only on real CUDA environments with CuPy
    import cupy as cp
except ImportError:  # pragma: no cover - default path on this machine
    cp = None


class _HostArrayView:
    """Minimal device-array-like wrapper for host fallback mode."""

    def __init__(self, data: np.ndarray) -> None:
        self._data = np.asarray(data)
        self.shape = self._data.shape
        self.size = self._data.size

    def copy_to_host(self) -> np.ndarray:
        return self._data.copy()

    def __array__(self, dtype: object | None = None) -> np.ndarray:
        return np.asarray(self._data, dtype=dtype)


_ORIGINAL_CUDA_TO_DEVICE = cuda.to_device


def _safe_to_device_host_fallback(arr: object, *args: object, **kwargs: object) -> object:
    """Fallback to a host-backed array when tests request CUDASIM without a device."""
    try:
        return _ORIGINAL_CUDA_TO_DEVICE(arr, *args, **kwargs)
    except Exception:
        return _HostArrayView(np.asarray(arr))


if os.environ.get("NUMBA_ENABLE_CUDASIM", "").strip() == "1":
    cuda.to_device = _safe_to_device_host_fallback


# ──────────────────────────────────────────────────── CUDA kernels ──────────


_MAX_SHARED_COUNT_CELLS = 2048


@cuda.jit(device=True)
def _wrap_periodic_cell(coord: int, periodic_dim: int) -> int:
    """Wrap an integer cell coordinate into [0, periodic_dim)."""
    if periodic_dim <= 0:
        return 0
    while coord < 0:
        coord += periodic_dim
    while coord >= periodic_dim:
        coord -= periodic_dim
    return coord


@cuda.jit(device=True)
def _clamp_cell(coord: int, grid_dim: int) -> int:
    """Clamp an integer cell coordinate into [0, grid_dim - 1]."""
    if coord < 0:
        return 0
    if coord >= grid_dim:
        return grid_dim - 1
    return coord


@cuda.jit
def _fill_int_kernel(arr: object, value: int) -> None:
    """Fill a 1-D int32 device array with *value*."""
    i = cuda.grid(1)
    if i < arr.size:
        arr[i] = value


@cuda.jit
def _copy_int_kernel(src: object, dst: object) -> None:
    """Copy one 1-D int32 device array into another."""
    i = cuda.grid(1)
    if i < src.size:
        dst[i] = src[i]


# ──── 2D cell-assignment kernels ─────────────────────────────────────────────

@cuda.jit
def _assign_cells_kernel(
    positions: object,
    cell_ids: object,
    particle_ids: object,
    cell_size: float,
    grid_min_x: float,
    grid_min_y: float,
    grid_dim_x: int,
    grid_dim_y: int,
    use_periodic: int,
    periodic_dim_x: int,
) -> None:
    """Map each particle to its (cx, cy) hash-grid cell (2D, CuPy sort path)."""
    i = cuda.grid(1)
    if i >= positions.shape[0]:
        return
    cx = int((positions[i, 0] - grid_min_x) / cell_size)
    cy = int((positions[i, 1] - grid_min_y) / cell_size)
    if use_periodic:
        cx = _wrap_periodic_cell(cx, periodic_dim_x)
    else:
        cx = _clamp_cell(cx, grid_dim_x)
    cy = _clamp_cell(cy, grid_dim_y)
    cell_ids[i] = cx + cy * grid_dim_x
    particle_ids[i] = i


@cuda.jit
def _assign_cell_ids_kernel(
    positions: object,
    cell_ids: object,
    cell_size: float,
    grid_min_x: float,
    grid_min_y: float,
    grid_dim_x: int,
    grid_dim_y: int,
    use_periodic: int,
    periodic_dim_x: int,
) -> None:
    """Map each particle to its hash-grid cell id (2D, numba-scan path)."""
    i = cuda.grid(1)
    if i >= positions.shape[0]:
        return
    cx = int((positions[i, 0] - grid_min_x) / cell_size)
    cy = int((positions[i, 1] - grid_min_y) / cell_size)
    if use_periodic:
        cx = _wrap_periodic_cell(cx, periodic_dim_x)
    else:
        cx = _clamp_cell(cx, grid_dim_x)
    cy = _clamp_cell(cy, grid_dim_y)
    cell_ids[i] = cx + cy * grid_dim_x


# ──── 3D cell-assignment kernels ─────────────────────────────────────────────

@cuda.jit
def _assign_cells_3d_kernel(
    positions: object,
    cell_ids: object,
    particle_ids: object,
    cell_size: float,
    grid_min_x: float,
    grid_min_y: float,
    grid_min_z: float,
    grid_dim_x: int,
    grid_dim_y: int,
    grid_dim_z: int,
    use_periodic: int,
    periodic_dim_x: int,
) -> None:
    """Map each particle to its (cx, cy, cz) hash-grid cell (3D, CuPy sort path)."""
    i = cuda.grid(1)
    if i >= positions.shape[0]:
        return
    cx = int((positions[i, 0] - grid_min_x) / cell_size)
    cy = int((positions[i, 1] - grid_min_y) / cell_size)
    cz = int((positions[i, 2] - grid_min_z) / cell_size)
    if use_periodic:
        cx = _wrap_periodic_cell(cx, periodic_dim_x)
    else:
        cx = _clamp_cell(cx, grid_dim_x)
    cy = _clamp_cell(cy, grid_dim_y)
    cz = _clamp_cell(cz, grid_dim_z)
    cell_ids[i] = cx + cy * grid_dim_x + cz * grid_dim_x * grid_dim_y
    particle_ids[i] = i


@cuda.jit
def _assign_cell_ids_3d_kernel(
    positions: object,
    cell_ids: object,
    cell_size: float,
    grid_min_x: float,
    grid_min_y: float,
    grid_min_z: float,
    grid_dim_x: int,
    grid_dim_y: int,
    grid_dim_z: int,
    use_periodic: int,
    periodic_dim_x: int,
) -> None:
    """Map each particle to its hash-grid cell id (3D, numba-scan path)."""
    i = cuda.grid(1)
    if i >= positions.shape[0]:
        return
    cx = int((positions[i, 0] - grid_min_x) / cell_size)
    cy = int((positions[i, 1] - grid_min_y) / cell_size)
    cz = int((positions[i, 2] - grid_min_z) / cell_size)
    if use_periodic:
        cx = _wrap_periodic_cell(cx, periodic_dim_x)
    else:
        cx = _clamp_cell(cx, grid_dim_x)
    cy = _clamp_cell(cy, grid_dim_y)
    cz = _clamp_cell(cz, grid_dim_z)
    cell_ids[i] = cx + cy * grid_dim_x + cz * grid_dim_x * grid_dim_y


# ──── Counting / scan / scatter (dimension-independent) ─────────────────────

@cuda.jit
def _count_cells_kernel(cell_ids: object, cell_counts: object) -> None:
    """Atomically increment cell_counts[cell_ids[i]] for every particle."""
    i = cuda.grid(1)
    if i < cell_ids.size:
        cuda.atomic.add(cell_counts, cell_ids[i], 1)


@cuda.jit
def _count_cells_block_hist_kernel(
    cell_ids: object,
    cell_counts: object,
    active_cells: int,
) -> None:
    """Block-private shared-memory histogram for per-cell particle counts."""
    local_counts = cuda.shared.array(shape=_MAX_SHARED_COUNT_CELLS, dtype=int32)
    tid = cuda.threadIdx.x
    stride = cuda.blockDim.x

    for cell in range(tid, active_cells, stride):
        local_counts[cell] = 0
    cuda.syncthreads()

    i = cuda.grid(1)
    if i < cell_ids.size:
        cuda.atomic.add(local_counts, cell_ids[i], 1)
    cuda.syncthreads()

    for cell in range(tid, active_cells, stride):
        count = local_counts[cell]
        if count != 0:
            cuda.atomic.add(cell_counts, cell, count)


@cuda.jit
def _exclusive_scan_cells_kernel(
    cell_counts: object,
    cell_start: object,
    cell_end: object,
) -> None:
    """Sequential device-side exclusive scan over per-cell counts."""
    if cuda.grid(1) != 0:
        return

    total = 0
    for cell in range(cell_counts.size):
        count = cell_counts[cell]
        cell_start[cell] = total
        total += count
        cell_end[cell] = total


@cuda.jit
def _exclusive_scan_and_init_insert_kernel(
    cell_counts: object,
    cell_start: object,
    cell_end: object,
    cell_insert_pos: object,
) -> None:
    """Sequential exclusive scan that also initialises the scatter insert-position array."""
    if cuda.grid(1) != 0:
        return

    total = 0
    for cell in range(cell_counts.size):
        count = cell_counts[cell]
        cell_start[cell] = total
        cell_insert_pos[cell] = total
        total += count
        cell_end[cell] = total


@cuda.jit
def _scatter_particle_ids_kernel(
    cell_ids: object,
    cell_insert_pos: object,
    sorted_ids: object,
) -> None:
    """Write each particle id into its sorted slot (atomic per-cell counter)."""
    i = cuda.grid(1)
    if i < cell_ids.size:
        cell = cell_ids[i]
        pos = cuda.atomic.add(cell_insert_pos, cell, 1)
        sorted_ids[pos] = i


@cuda.jit
def _scatter_particle_ids_single_shmem_kernel(
    cell_ids: object,
    cell_insert_pos: object,
    sorted_ids: object,
    active_cells: int,
) -> None:
    """Scatter particle ids using a single shared-memory array."""
    local = cuda.shared.array(shape=_MAX_SHARED_COUNT_CELLS, dtype=int32)
    tid = cuda.threadIdx.x
    stride = cuda.blockDim.x

    for cell in range(tid, active_cells, stride):
        local[cell] = 0
    cuda.syncthreads()

    i = cuda.grid(1)
    if i < cell_ids.size:
        cuda.atomic.add(local, cell_ids[i], 1)
    cuda.syncthreads()

    for cell in range(tid, active_cells, stride):
        count = local[cell]
        if count != 0:
            local[cell] = cuda.atomic.add(cell_insert_pos, cell, count)
    cuda.syncthreads()

    if i < cell_ids.size:
        cell = cell_ids[i]
        pos = cuda.atomic.add(local, cell, 1)
        sorted_ids[pos] = i


@cuda.jit
def _build_cell_ranges_kernel(
    sorted_cell_ids: object,
    cell_start: object,
    cell_end: object,
) -> None:
    """Build [start, end) ranges for each occupied cell in sorted order."""
    i = cuda.grid(1)
    n = sorted_cell_ids.size
    if i >= n:
        return

    cell = sorted_cell_ids[i]
    if i == 0 or cell != sorted_cell_ids[i - 1]:
        cell_start[cell] = i
    if i == n - 1 or cell != sorted_cell_ids[i + 1]:
        cell_end[cell] = i + 1


# ──── 2D pair-finding kernels ────────────────────────────────────────────────

@cuda.jit
def _find_ff_pairs_kernel(
    positions: object,
    sorted_ids: object,
    cell_start: object,
    cell_end: object,
    grid_dim_x: int,
    grid_dim_y: int,
    cell_size: float,
    grid_min_x: float,
    grid_min_y: float,
    support_radius: float,
    pair_i: object,
    pair_j: object,
    pair_count: object,
    max_pairs: int,
    periodic_length: float,
    use_periodic: int,
    periodic_dim_x: int,
) -> None:
    """Emit fluid-fluid pairs from the directed half-neighbourhood (2D)."""
    i = cuda.grid(1)
    n = positions.shape[0]
    if i >= n:
        return

    cx = int((positions[i, 0] - grid_min_x) / cell_size)
    cy = int((positions[i, 1] - grid_min_y) / cell_size)
    if use_periodic:
        cx = _wrap_periodic_cell(cx, periodic_dim_x)
    else:
        cx = _clamp_cell(cx, grid_dim_x)
    cy = _clamp_cell(cy, grid_dim_y)

    sr2 = support_radius * support_radius
    xi = positions[i, 0]
    yi = positions[i, 1]

    for dcy in range(0, 2):
        dcx_start = 0 if dcy == 0 else -1
        for dcx in range(dcx_start, 2):
            ncx = cx + dcx
            if use_periodic:
                ncx = _wrap_periodic_cell(ncx, periodic_dim_x)
            elif ncx < 0 or ncx >= grid_dim_x:
                continue

            ncy = cy + dcy
            if ncy < 0 or ncy >= grid_dim_y:
                continue

            cell = ncx + ncy * grid_dim_x
            start = cell_start[cell]
            end = cell_end[cell]
            if start < 0 or start >= end:
                continue

            for k in range(start, end):
                j = sorted_ids[k]
                if dcy == 0 and dcx == 0 and j <= i:
                    continue

                dx = xi - positions[j, 0]
                if use_periodic:
                    dx -= periodic_length * round(dx / periodic_length)
                dy = yi - positions[j, 1]
                dist2 = dx * dx + dy * dy

                if dist2 < sr2:
                    idx = cuda.atomic.add(pair_count, 0, 1)
                    if idx < max_pairs:
                        if dcy == 0 and dcx == 0:
                            pair_i[idx] = i
                            pair_j[idx] = j
                        elif i < j:
                            pair_i[idx] = i
                            pair_j[idx] = j
                        else:
                            pair_i[idx] = j
                            pair_j[idx] = i


@cuda.jit
def _find_fb_pairs_kernel(
    fluid_positions: object,
    boundary_positions: object,
    sorted_bnd_ids: object,
    bnd_cell_start: object,
    bnd_cell_end: object,
    grid_dim_x: int,
    grid_dim_y: int,
    cell_size: float,
    grid_min_x: float,
    grid_min_y: float,
    support_radius: float,
    pair_i: object,
    pair_j: object,
    pair_count: object,
    max_pairs: int,
    periodic_length: float,
    use_periodic: int,
    periodic_dim_x: int,
) -> None:
    """Emit fluid-boundary pairs (2D)."""
    i = cuda.grid(1)
    n = fluid_positions.shape[0]
    if i >= n:
        return

    cx = int((fluid_positions[i, 0] - grid_min_x) / cell_size)
    cy = int((fluid_positions[i, 1] - grid_min_y) / cell_size)
    if use_periodic:
        cx = _wrap_periodic_cell(cx, periodic_dim_x)
    else:
        cx = _clamp_cell(cx, grid_dim_x)
    cy = _clamp_cell(cy, grid_dim_y)

    sr2 = support_radius * support_radius

    for dcx in range(-1, 2):
        ncx = cx + dcx
        if use_periodic:
            ncx = _wrap_periodic_cell(ncx, periodic_dim_x)
        elif ncx < 0 or ncx >= grid_dim_x:
            continue

        for dcy in range(-1, 2):
            ncy = cy + dcy
            if ncy < 0 or ncy >= grid_dim_y:
                continue

            cell = ncx + ncy * grid_dim_x
            start = bnd_cell_start[cell]
            end = bnd_cell_end[cell]
            if start < 0 or start >= end:
                continue

            for k in range(start, end):
                j = sorted_bnd_ids[k]

                dx = fluid_positions[i, 0] - boundary_positions[j, 0]
                if use_periodic:
                    dx -= periodic_length * round(dx / periodic_length)
                dy = fluid_positions[i, 1] - boundary_positions[j, 1]
                dist2 = dx * dx + dy * dy

                if dist2 < sr2:
                    idx = cuda.atomic.add(pair_count, 0, 1)
                    if idx < max_pairs:
                        pair_i[idx] = i
                        pair_j[idx] = j


# ──── 3D pair-finding kernels ────────────────────────────────────────────────

@cuda.jit
def _find_ff_pairs_3d_kernel(
    positions: object,
    sorted_ids: object,
    cell_start: object,
    cell_end: object,
    grid_dim_x: int,
    grid_dim_y: int,
    grid_dim_z: int,
    cell_size: float,
    grid_min_x: float,
    grid_min_y: float,
    grid_min_z: float,
    support_radius: float,
    pair_i: object,
    pair_j: object,
    pair_count: object,
    max_pairs: int,
    periodic_length: float,
    use_periodic: int,
    periodic_dim_x: int,
) -> None:
    """Emit fluid-fluid pairs from the directed half-neighbourhood (3D, 27-cell stencil)."""
    i = cuda.grid(1)
    n = positions.shape[0]
    if i >= n:
        return

    cx = int((positions[i, 0] - grid_min_x) / cell_size)
    cy = int((positions[i, 1] - grid_min_y) / cell_size)
    cz = int((positions[i, 2] - grid_min_z) / cell_size)
    if use_periodic:
        cx = _wrap_periodic_cell(cx, periodic_dim_x)
    else:
        cx = _clamp_cell(cx, grid_dim_x)
    cy = _clamp_cell(cy, grid_dim_y)
    cz = _clamp_cell(cz, grid_dim_z)

    sr2 = support_radius * support_radius
    xi = positions[i, 0]
    yi = positions[i, 1]
    zi = positions[i, 2]

    grid_xy = grid_dim_x * grid_dim_y

    for dcz in range(-1, 2):
        ncz = cz + dcz
        if ncz < 0 or ncz >= grid_dim_z:
            continue
        for dcy in range(-1, 2):
            ncy = cy + dcy
            if ncy < 0 or ncy >= grid_dim_y:
                continue
            for dcx in range(-1, 2):
                ncx = cx + dcx
                if use_periodic:
                    ncx = _wrap_periodic_cell(ncx, periodic_dim_x)
                elif ncx < 0 or ncx >= grid_dim_x:
                    continue

                cell = ncx + ncy * grid_dim_x + ncz * grid_xy
                start = cell_start[cell]
                end = cell_end[cell]
                if start < 0 or start >= end:
                    continue

                for k in range(start, end):
                    j = sorted_ids[k]
                    if j <= i:
                        continue

                    dx = xi - positions[j, 0]
                    if use_periodic:
                        dx -= periodic_length * round(dx / periodic_length)
                    dy = yi - positions[j, 1]
                    dz = zi - positions[j, 2]
                    dist2 = dx * dx + dy * dy + dz * dz

                    if dist2 < sr2:
                        idx = cuda.atomic.add(pair_count, 0, 1)
                        if idx < max_pairs:
                            if i < j:
                                pair_i[idx] = i
                                pair_j[idx] = j
                            else:
                                pair_i[idx] = j
                                pair_j[idx] = i


@cuda.jit
def _find_fb_pairs_3d_kernel(
    fluid_positions: object,
    boundary_positions: object,
    sorted_bnd_ids: object,
    bnd_cell_start: object,
    bnd_cell_end: object,
    grid_dim_x: int,
    grid_dim_y: int,
    grid_dim_z: int,
    cell_size: float,
    grid_min_x: float,
    grid_min_y: float,
    grid_min_z: float,
    support_radius: float,
    pair_i: object,
    pair_j: object,
    pair_count: object,
    max_pairs: int,
    periodic_length: float,
    use_periodic: int,
    periodic_dim_x: int,
) -> None:
    """Emit fluid-boundary pairs (3D, 27-cell stencil)."""
    i = cuda.grid(1)
    n = fluid_positions.shape[0]
    if i >= n:
        return

    cx = int((fluid_positions[i, 0] - grid_min_x) / cell_size)
    cy = int((fluid_positions[i, 1] - grid_min_y) / cell_size)
    cz = int((fluid_positions[i, 2] - grid_min_z) / cell_size)
    if use_periodic:
        cx = _wrap_periodic_cell(cx, periodic_dim_x)
    else:
        cx = _clamp_cell(cx, grid_dim_x)
    cy = _clamp_cell(cy, grid_dim_y)
    cz = _clamp_cell(cz, grid_dim_z)

    sr2 = support_radius * support_radius
    grid_xy = grid_dim_x * grid_dim_y

    for dcz in range(-1, 2):
        ncz = cz + dcz
        if ncz < 0 or ncz >= grid_dim_z:
            continue
        for dcy in range(-1, 2):
            ncy = cy + dcy
            if ncy < 0 or ncy >= grid_dim_y:
                continue
            for dcx in range(-1, 2):
                ncx = cx + dcx
                if use_periodic:
                    ncx = _wrap_periodic_cell(ncx, periodic_dim_x)
                elif ncx < 0 or ncx >= grid_dim_x:
                    continue

                cell = ncx + ncy * grid_dim_x + ncz * grid_xy
                start = bnd_cell_start[cell]
                end = bnd_cell_end[cell]
                if start < 0 or start >= end:
                    continue

                for k in range(start, end):
                    j = sorted_bnd_ids[k]

                    dx = fluid_positions[i, 0] - boundary_positions[j, 0]
                    if use_periodic:
                        dx -= periodic_length * round(dx / periodic_length)
                    dy = fluid_positions[i, 1] - boundary_positions[j, 1]
                    dz = fluid_positions[i, 2] - boundary_positions[j, 2]
                    dist2 = dx * dx + dy * dy + dz * dz

                    if dist2 < sr2:
                        idx = cuda.atomic.add(pair_count, 0, 1)
                        if idx < max_pairs:
                            pair_i[idx] = i
                            pair_j[idx] = j


# ──────────────────────────────────────────────── CUDANeighborSearch ────────


class CUDANeighborSearch:
    """
    Device-side hash-grid neighbor search for 2-D and 3-D SPH.

    All GPU buffers are pre-allocated once in ``__init__``; each ``build()``
    call reuses them.  The returned ff_i/ff_j/fb_i/fb_j are device-array
    *slices* — no data ever leaves the GPU except for the two pair-count
    scalars.
    """

    def __init__(
        self,
        support_radius: float,
        domain_min: np.ndarray,
        domain_max: np.ndarray,
        n_fluid_max: int,
        n_boundary: int,
        max_pairs: int = 500_000,
        periodic_x: tuple[float, float] | None = None,
        collect_sub_timings: bool = False,
        dim: int = 2,
    ) -> None:
        self.support_radius = float(support_radius)
        self.cell_size = float(support_radius)
        self.max_pairs = int(max_pairs)
        self.periodic_x = periodic_x
        self.dim = int(dim)
        self._host_fallback = False
        self._host_fallback_reason = ""
        self._cupy = cp
        self._sort_backend = (
            "cupy"
            if cp is not None and os.environ.get("NUMBA_ENABLE_CUDASIM", "").strip() != "1"
            else "numba_scan"
        )
        self._use_event_timing = os.environ.get("NUMBA_ENABLE_CUDASIM", "").strip() != "1"
        self._collect_sub_timings = bool(collect_sub_timings)
        self._last_build_timings_ms = self._empty_build_timings()

        # ── Grid geometry ────────────────────────────────────────────────────
        self._grid_min_x = float(domain_min[0])
        self._grid_min_y = float(domain_min[1])
        extent_x = float(domain_max[0] - domain_min[0])
        extent_y = float(domain_max[1] - domain_min[1])
        self._grid_dim_x = int(np.ceil(extent_x / self.cell_size)) + 2
        self._grid_dim_y = int(np.ceil(extent_y / self.cell_size)) + 2

        if self.dim == 3:
            self._grid_min_z = float(domain_min[2])
            extent_z = float(domain_max[2] - domain_min[2])
            self._grid_dim_z = int(np.ceil(extent_z / self.cell_size)) + 2
        else:
            self._grid_min_z = 0.0
            self._grid_dim_z = 1

        n_cells = self._grid_dim_x * self._grid_dim_y * self._grid_dim_z

        # ── Periodic config ──────────────────────────────────────────────────
        self._periodic_length = 0.0
        self._use_periodic = 0
        self._periodic_dim_x = self._grid_dim_x
        if periodic_x is not None:
            self._periodic_length = float(periodic_x[1] - periodic_x[0])
            self._use_periodic = 1
            self._periodic_dim_x = max(1, int(np.floor(self._periodic_length / self.cell_size)))
            self._grid_min_x = float(periodic_x[0])

        self._has_boundary = n_boundary > 0
        self._boundary_grid_built = False
        try:
            # ── Fluid hash-grid buffers ──────────────────────────────────────
            self._f_cell_ids = cuda.device_array(n_fluid_max, dtype=np.int32)
            self._f_particle_ids = cuda.device_array(n_fluid_max, dtype=np.int32)
            self._f_sorted_cell_ids = cuda.device_array(n_fluid_max, dtype=np.int32)
            self._f_sorted_ids = cuda.device_array(n_fluid_max, dtype=np.int32)
            self._f_cell_counts = cuda.device_array(n_cells, dtype=np.int32)
            self._f_cell_start = cuda.device_array(n_cells, dtype=np.int32)
            self._f_cell_end = cuda.device_array(n_cells, dtype=np.int32)
            self._f_insert_pos = cuda.device_array(n_cells, dtype=np.int32)

            # ── Boundary hash-grid buffers ───────────────────────────────────
            if self._has_boundary:
                self._b_cell_ids = cuda.device_array(n_boundary, dtype=np.int32)
                self._b_particle_ids = cuda.device_array(n_boundary, dtype=np.int32)
                self._b_sorted_cell_ids = cuda.device_array(n_boundary, dtype=np.int32)
                self._b_sorted_ids = cuda.device_array(n_boundary, dtype=np.int32)
                self._b_cell_counts = cuda.device_array(n_cells, dtype=np.int32)
                self._b_cell_start = cuda.device_array(n_cells, dtype=np.int32)
                self._b_cell_end = cuda.device_array(n_cells, dtype=np.int32)
                self._b_insert_pos = cuda.device_array(n_cells, dtype=np.int32)

            # ── Pair output buffers ──────────────────────────────────────────
            self._allocate_pair_buffers(max_pairs)

            # ── Combined pair-count buffer: [n_ff, n_fb] ─────────────────────
            self._pair_counts = cuda.device_array(2, dtype=np.int32)
        except Exception as exc:  # pragma: no cover - exercised on no-device hosts
            self._host_fallback = True
            self._host_fallback_reason = str(exc)
            self._sort_backend = "host_kdtree"
            self._use_event_timing = False
            self._collect_sub_timings = False

    # ── Diagnostics control ──────────────────────────────────────────────────

    def enable_pair_build_diagnostics(self, enabled: bool = True) -> None:
        self._collect_sub_timings = bool(enabled)

    @property
    def sort_backend(self) -> str:
        return self._sort_backend

    @property
    def last_build_timings_ms(self) -> dict[str, float]:
        return dict(self._last_build_timings_ms)

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _empty_build_timings() -> dict[str, float]:
        return {
            "hash_assign": 0.0,
            "count_scan_scatter": 0.0,
            "boundary_grid": 0.0,
            "ff_emit": 0.0,
            "fb_emit": 0.0,
            "count_read": 0.0,
        }

    @staticmethod
    def _next_pair_capacity(current: int, required: int) -> int:
        current = max(int(current), 1)
        required = max(int(required), 1)
        if required <= current:
            return current
        return max(required + max(required // 2, 1024), current * 2)

    def _allocate_pair_buffers(self, capacity: int) -> None:
        capacity = int(capacity)
        self._pair_ff_i = cuda.device_array(capacity, dtype=np.int32)
        self._pair_ff_j = cuda.device_array(capacity, dtype=np.int32)
        self._pair_fb_i = cuda.device_array(capacity, dtype=np.int32)
        self._pair_fb_j = cuda.device_array(capacity, dtype=np.int32)

    def _ensure_pair_capacity(self, required: int) -> bool:
        required = int(required)
        if required <= self.max_pairs:
            return False
        new_capacity = self._next_pair_capacity(self.max_pairs, required)
        if self._host_fallback:
            self.max_pairs = new_capacity
            return True
        try:
            self._allocate_pair_buffers(new_capacity)
        except Exception as exc:
            raise RuntimeError(
                "CUDA neighbor search could not grow pair buffers to "
                f"{new_capacity} entries after observing {required} required pairs: {exc}"
            ) from exc
        self.max_pairs = new_capacity
        return True

    def _raise_pair_overflow(self, n_ff: int, n_fb: int) -> None:
        if n_ff > self.max_pairs:
            raise RuntimeError(
                f"CUDA neighbor search overflowed fluid-fluid pair buffer: "
                f"{n_ff} > max_pairs={self.max_pairs}"
            )
        if n_fb > self.max_pairs:
            raise RuntimeError(
                f"CUDA neighbor search overflowed fluid-boundary pair buffer: "
                f"{n_fb} > max_pairs={self.max_pairs}"
            )

    @staticmethod
    def _slice_buffer(buf: object, n: int) -> object:
        return buf if int(buf.size) == n else buf[:n]

    def _timing_marker(self) -> object | None:
        if not self._use_event_timing:
            return None
        return cuda.event(timing=True)

    @staticmethod
    def _record_marker(marker: object | None) -> None:
        if marker is not None:
            marker.record()

    def _start_timed_section(self) -> tuple[object | None, float]:
        marker = self._timing_marker()
        if marker is not None:
            marker.record()
            return marker, 0.0
        return None, time.perf_counter()

    def _finish_timed_section(
        self,
        start_marker: object | None,
        start_time: float,
        timings: dict[str, float],
        key: str,
    ) -> None:
        end_marker = self._timing_marker()
        if start_marker is not None and end_marker is not None:
            end_marker.record()
            end_marker.synchronize()
            timings[key] += float(cuda.event_elapsed_time(start_marker, end_marker))
            return
        cuda.synchronize()
        timings[key] += (time.perf_counter() - start_time) * 1000.0

    def _sort_cells_with_cupy(
        self,
        cell_ids_buf: object,
        particle_ids_buf: object,
        sorted_cell_ids_buf: object,
        sorted_ids_buf: object,
        cell_start_buf: object,
        cell_end_buf: object,
        n: int,
        cfg_n: tuple[int, int],
        cfg_cells: tuple[int, int],
    ) -> None:
        if self._cupy is None:
            raise RuntimeError("CuPy-backed cell sort requested, but CuPy is not available.")

        _fill_int_kernel[cfg_cells](cell_start_buf, 0)
        _fill_int_kernel[cfg_cells](cell_end_buf, 0)
        cuda.synchronize()

        cell_ids_view = self._cupy.asarray(self._slice_buffer(cell_ids_buf, n))
        particle_ids_view = self._cupy.asarray(self._slice_buffer(particle_ids_buf, n))
        sorted_cell_ids_view = self._cupy.asarray(self._slice_buffer(sorted_cell_ids_buf, n))
        sorted_ids_view = self._cupy.asarray(self._slice_buffer(sorted_ids_buf, n))

        sort_idx = self._cupy.argsort(cell_ids_view, kind="stable")
        sorted_cell_ids_view[...] = cell_ids_view[sort_idx]
        sorted_ids_view[...] = particle_ids_view[sort_idx]
        cuda.synchronize()

        _build_cell_ranges_kernel[cfg_n](
            self._slice_buffer(sorted_cell_ids_buf, n),
            cell_start_buf,
            cell_end_buf,
        )
        cuda.synchronize()

    def _sort_cells_with_numba_scan(
        self,
        cell_ids_buf: object,
        sorted_ids_buf: object,
        cell_counts_buf: object,
        cell_start_buf: object,
        cell_end_buf: object,
        insert_pos_buf: object,
        n: int,
        cfg_n: tuple[int, int],
        cfg_cells: tuple[int, int],
    ) -> None:
        """Count-scan-scatter sort on the default CUDA stream."""
        _fill_int_kernel[cfg_cells](cell_counts_buf, 0)

        if cell_counts_buf.size <= _MAX_SHARED_COUNT_CELLS:
            _count_cells_block_hist_kernel[cfg_n](
                self._slice_buffer(cell_ids_buf, n),
                cell_counts_buf,
                cell_counts_buf.size,
            )
        else:
            _count_cells_kernel[cfg_n](self._slice_buffer(cell_ids_buf, n), cell_counts_buf)

        _exclusive_scan_and_init_insert_kernel[1, 1](
            cell_counts_buf, cell_start_buf, cell_end_buf, insert_pos_buf
        )

        if cell_counts_buf.size <= _MAX_SHARED_COUNT_CELLS:
            _scatter_particle_ids_single_shmem_kernel[cfg_n](
                self._slice_buffer(cell_ids_buf, n),
                insert_pos_buf,
                self._slice_buffer(sorted_ids_buf, n),
                cell_counts_buf.size,
            )
        else:
            _scatter_particle_ids_kernel[cfg_n](
                self._slice_buffer(cell_ids_buf, n),
                insert_pos_buf,
                self._slice_buffer(sorted_ids_buf, n),
            )

    def _assign_cells(self, positions_gpu: object, cell_ids_buf: object,
                      particle_ids_buf: object | None, n: int) -> None:
        """Launch the appropriate 2D or 3D cell-assignment kernel."""
        cfg_n = launch_config(n)
        if self.dim == 3:
            if particle_ids_buf is not None:
                _assign_cells_3d_kernel[cfg_n](
                    positions_gpu,
                    self._slice_buffer(cell_ids_buf, n),
                    self._slice_buffer(particle_ids_buf, n),
                    self.cell_size,
                    self._grid_min_x, self._grid_min_y, self._grid_min_z,
                    self._grid_dim_x, self._grid_dim_y, self._grid_dim_z,
                    self._use_periodic, self._periodic_dim_x,
                )
            else:
                _assign_cell_ids_3d_kernel[cfg_n](
                    positions_gpu,
                    self._slice_buffer(cell_ids_buf, n),
                    self.cell_size,
                    self._grid_min_x, self._grid_min_y, self._grid_min_z,
                    self._grid_dim_x, self._grid_dim_y, self._grid_dim_z,
                    self._use_periodic, self._periodic_dim_x,
                )
        else:
            if particle_ids_buf is not None:
                _assign_cells_kernel[cfg_n](
                    positions_gpu,
                    self._slice_buffer(cell_ids_buf, n),
                    self._slice_buffer(particle_ids_buf, n),
                    self.cell_size,
                    self._grid_min_x, self._grid_min_y,
                    self._grid_dim_x, self._grid_dim_y,
                    self._use_periodic, self._periodic_dim_x,
                )
            else:
                _assign_cell_ids_kernel[cfg_n](
                    positions_gpu,
                    self._slice_buffer(cell_ids_buf, n),
                    self.cell_size,
                    self._grid_min_x, self._grid_min_y,
                    self._grid_dim_x, self._grid_dim_y,
                    self._use_periodic, self._periodic_dim_x,
                )

    def _build_hash_grid(
        self,
        positions_gpu: object,
        cell_ids_buf: object,
        particle_ids_buf: object,
        sorted_cell_ids_buf: object,
        sorted_ids_buf: object,
        cell_counts_buf: object,
        cell_start_buf: object,
        cell_end_buf: object,
        insert_pos_buf: object,
        timings: dict[str, float] | None = None,
        assign_key: str | None = None,
        order_key: str | None = None,
    ) -> None:
        """Build sorted particle lists and cell range arrays for one set of particles."""
        n = int(positions_gpu.shape[0])
        n_cells = self._grid_dim_x * self._grid_dim_y * self._grid_dim_z
        cfg_cells = launch_config(n_cells)
        start_marker = None
        start_time = 0.0
        if timings is not None and assign_key is not None:
            start_marker, start_time = self._start_timed_section()

        if n == 0:
            _fill_int_kernel[cfg_cells](cell_start_buf, -1)
            _fill_int_kernel[cfg_cells](cell_end_buf, -1)
            if timings is not None and assign_key is not None:
                self._finish_timed_section(start_marker, start_time, timings, assign_key)
            return

        cfg_n = launch_config(n)

        if self._sort_backend == "cupy":
            self._assign_cells(positions_gpu, cell_ids_buf, particle_ids_buf, n)
        else:
            self._assign_cells(positions_gpu, cell_ids_buf, None, n)

        if timings is not None and assign_key is not None:
            self._finish_timed_section(start_marker, start_time, timings, assign_key)

        if self._sort_backend == "cupy":
            self._sort_cells_with_cupy(
                cell_ids_buf=cell_ids_buf,
                particle_ids_buf=particle_ids_buf,
                sorted_cell_ids_buf=sorted_cell_ids_buf,
                sorted_ids_buf=sorted_ids_buf,
                cell_start_buf=cell_start_buf,
                cell_end_buf=cell_end_buf,
                n=n,
                cfg_n=cfg_n,
                cfg_cells=cfg_cells,
            )
            return

        start_marker = None
        start_time = 0.0
        if timings is not None and order_key is not None:
            start_marker, start_time = self._start_timed_section()
        self._sort_cells_with_numba_scan(
            cell_ids_buf=cell_ids_buf,
            sorted_ids_buf=sorted_ids_buf,
            cell_counts_buf=cell_counts_buf,
            cell_start_buf=cell_start_buf,
            cell_end_buf=cell_end_buf,
            insert_pos_buf=insert_pos_buf,
            n=n,
            cfg_n=cfg_n,
            cfg_cells=cfg_cells,
        )
        if timings is not None and order_key is not None:
            self._finish_timed_section(start_marker, start_time, timings, order_key)

    def _read_pair_counts(
        self,
        timings: dict[str, float],
    ) -> tuple[int, int]:
        """Download both pair counts in a single PCIe transfer."""
        start_time = time.perf_counter()
        counts_host = self._pair_counts.copy_to_host()
        n_ff = int(counts_host[0])
        n_fb = int(counts_host[1])
        timings["count_read"] = (time.perf_counter() - start_time) * 1000.0
        return n_ff, n_fb

    # ── Public interface ─────────────────────────────────────────────────────

    def build(
        self,
        fluid_positions_gpu: object,
        boundary_positions_gpu: object | None = None,
    ) -> tuple:
        """
        Build neighbor pairs on GPU.

        Parameters
        ----------
        fluid_positions_gpu :
            Device array, shape (n_fluid, dim), float64.
        boundary_positions_gpu :
            Device array, shape (n_boundary, dim), float64, or None.

        Returns
        -------
        (ff_i, ff_j, n_ff, fb_i, fb_j, n_fb)
        """
        if self._host_fallback:
            return self._build_host_fallback(fluid_positions_gpu, boundary_positions_gpu)

        n_fluid = int(fluid_positions_gpu.shape[0])
        cfg_fluid = launch_config(n_fluid)
        build_timings = self._empty_build_timings()

        for attempt in range(2):
            build_timings = self._empty_build_timings()
            if self._collect_sub_timings:
                self._build_timed(
                    fluid_positions_gpu, boundary_positions_gpu,
                    n_fluid, cfg_fluid, build_timings,
                )
            else:
                self._build_fast(
                    fluid_positions_gpu, boundary_positions_gpu,
                    n_fluid, cfg_fluid,
                )

            n_ff, n_fb = self._read_pair_counts(build_timings)
            required = max(n_ff, n_fb)
            if required <= self.max_pairs:
                break
            if attempt == 0 and self._ensure_pair_capacity(required):
                continue
            self._raise_pair_overflow(n_ff, n_fb)

        self._last_build_timings_ms = build_timings

        return (
            self._pair_ff_i[:n_ff],
            self._pair_ff_j[:n_ff],
            n_ff,
            self._pair_fb_i[:n_fb],
            self._pair_fb_j[:n_fb],
            n_fb,
        )

    def _build_host_fallback(
        self,
        fluid_positions_gpu: object,
        boundary_positions_gpu: object | None,
    ) -> tuple:
        fluid_positions = self._as_host_array(fluid_positions_gpu)
        boundary_positions = (
            self._as_host_array(boundary_positions_gpu)
            if boundary_positions_gpu is not None
            else None
        )

        search = KDTreeNeighborSearch(
            support_radius=self.support_radius,
            dim=self.dim,
            periodic_x=self.periodic_x,
        )
        pairs = search.build_neighbor_pairs(fluid_positions, boundary_positions)
        n_ff = int(pairs.ff_i.size)
        n_fb = int(pairs.fb_i.size)
        self._ensure_pair_capacity(max(n_ff, n_fb))

        self._last_build_timings_ms = self._empty_build_timings()
        return (
            _HostArrayView(pairs.ff_i.copy()),
            _HostArrayView(pairs.ff_j.copy()),
            n_ff,
            _HostArrayView(pairs.fb_i.copy()),
            _HostArrayView(pairs.fb_j.copy()),
            n_fb,
        )

    @staticmethod
    def _as_host_array(arr: object | None) -> np.ndarray | None:
        if arr is None:
            return None
        if hasattr(arr, "copy_to_host"):
            return np.asarray(arr.copy_to_host())
        return np.asarray(arr)

    def _launch_ff_pairs(self, fluid_positions_gpu: object, cfg_fluid: tuple[int, int]) -> None:
        """Dispatch the appropriate 2D or 3D fluid-fluid pair kernel."""
        if self.dim == 3:
            _find_ff_pairs_3d_kernel[cfg_fluid](
                fluid_positions_gpu,
                self._f_sorted_ids,
                self._f_cell_start, self._f_cell_end,
                self._grid_dim_x, self._grid_dim_y, self._grid_dim_z,
                self.cell_size,
                self._grid_min_x, self._grid_min_y, self._grid_min_z,
                self.support_radius,
                self._pair_ff_i, self._pair_ff_j,
                self._pair_counts[0:1],
                self.max_pairs,
                self._periodic_length, self._use_periodic, self._periodic_dim_x,
            )
        else:
            _find_ff_pairs_kernel[cfg_fluid](
                fluid_positions_gpu,
                self._f_sorted_ids,
                self._f_cell_start, self._f_cell_end,
                self._grid_dim_x, self._grid_dim_y,
                self.cell_size,
                self._grid_min_x, self._grid_min_y,
                self.support_radius,
                self._pair_ff_i, self._pair_ff_j,
                self._pair_counts[0:1],
                self.max_pairs,
                self._periodic_length, self._use_periodic, self._periodic_dim_x,
            )

    def _launch_fb_pairs(self, fluid_positions_gpu: object,
                         boundary_positions_gpu: object, cfg_fluid: tuple[int, int]) -> None:
        """Dispatch the appropriate 2D or 3D fluid-boundary pair kernel."""
        if self.dim == 3:
            _find_fb_pairs_3d_kernel[cfg_fluid](
                fluid_positions_gpu,
                boundary_positions_gpu,
                self._b_sorted_ids,
                self._b_cell_start, self._b_cell_end,
                self._grid_dim_x, self._grid_dim_y, self._grid_dim_z,
                self.cell_size,
                self._grid_min_x, self._grid_min_y, self._grid_min_z,
                self.support_radius,
                self._pair_fb_i, self._pair_fb_j,
                self._pair_counts[1:2],
                self.max_pairs,
                self._periodic_length, self._use_periodic, self._periodic_dim_x,
            )
        else:
            _find_fb_pairs_kernel[cfg_fluid](
                fluid_positions_gpu,
                boundary_positions_gpu,
                self._b_sorted_ids,
                self._b_cell_start, self._b_cell_end,
                self._grid_dim_x, self._grid_dim_y,
                self.cell_size,
                self._grid_min_x, self._grid_min_y,
                self.support_radius,
                self._pair_fb_i, self._pair_fb_j,
                self._pair_counts[1:2],
                self.max_pairs,
                self._periodic_length, self._use_periodic, self._periodic_dim_x,
            )

    def _build_fast(
        self,
        fluid_positions_gpu: object,
        boundary_positions_gpu: object | None,
        n_fluid: int,
        cfg_fluid: tuple[int, int],
    ) -> None:
        """Hot path: no intermediate GPU event syncs."""
        self._build_hash_grid(
            fluid_positions_gpu,
            self._f_cell_ids, self._f_particle_ids,
            self._f_sorted_cell_ids, self._f_sorted_ids,
            self._f_cell_counts,
            self._f_cell_start, self._f_cell_end, self._f_insert_pos,
        )

        _fill_int_kernel[1, 128](self._pair_counts, 0)
        self._launch_ff_pairs(fluid_positions_gpu, cfg_fluid)

        if self._has_boundary and boundary_positions_gpu is not None:
            if not self._boundary_grid_built:
                self._build_hash_grid(
                    boundary_positions_gpu,
                    self._b_cell_ids, self._b_particle_ids,
                    self._b_sorted_cell_ids, self._b_sorted_ids,
                    self._b_cell_counts,
                    self._b_cell_start, self._b_cell_end, self._b_insert_pos,
                )
                self._boundary_grid_built = True

            self._launch_fb_pairs(fluid_positions_gpu, boundary_positions_gpu, cfg_fluid)

    def _build_timed(
        self,
        fluid_positions_gpu: object,
        boundary_positions_gpu: object | None,
        n_fluid: int,
        cfg_fluid: tuple[int, int],
        build_timings: dict[str, float],
    ) -> None:
        """Diagnostic path: GPU event timing at each sub-phase."""
        self._build_hash_grid(
            fluid_positions_gpu,
            self._f_cell_ids, self._f_particle_ids,
            self._f_sorted_cell_ids, self._f_sorted_ids,
            self._f_cell_counts,
            self._f_cell_start, self._f_cell_end, self._f_insert_pos,
            timings=build_timings,
            assign_key="hash_assign",
            order_key="count_scan_scatter",
        )

        start_marker, start_time = self._start_timed_section()
        _fill_int_kernel[1, 128](self._pair_counts, 0)
        self._launch_ff_pairs(fluid_positions_gpu, cfg_fluid)
        self._finish_timed_section(start_marker, start_time, build_timings, "ff_emit")

        if self._has_boundary and boundary_positions_gpu is not None:
            if not self._boundary_grid_built:
                start_marker, start_time = self._start_timed_section()
                self._build_hash_grid(
                    boundary_positions_gpu,
                    self._b_cell_ids, self._b_particle_ids,
                    self._b_sorted_cell_ids, self._b_sorted_ids,
                    self._b_cell_counts,
                    self._b_cell_start, self._b_cell_end, self._b_insert_pos,
                )
                self._finish_timed_section(start_marker, start_time, build_timings, "boundary_grid")
                self._boundary_grid_built = True

            start_marker, start_time = self._start_timed_section()
            self._launch_fb_pairs(fluid_positions_gpu, boundary_positions_gpu, cfg_fluid)
            self._finish_timed_section(start_marker, start_time, build_timings, "fb_emit")
