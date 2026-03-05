from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sph.geometry.mesh import Mesh


@dataclass(frozen=True)
class DistanceStats:
    min_distance: float
    mean_min_distance: float
    close_fraction: float
    close_count: int
    fluid_count: int


def sample_mesh_surface_uniform(mesh: Mesh, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Uniform-ish sampling on each triangle using a barycentric lattice.

    For a triangle, choose n ~ ceil(max_edge/spacing), then generate all
    barycentric pairs (i/n, j/n, 1-i/n-j/n) with i+j<=n. This ensures edge
    coverage and approximately spacing-sized nearest-neighbor distance.
    """
    spacing = float(spacing)
    if spacing <= 0.0:
        raise ValueError(f"spacing must be > 0, got {spacing}")
    if mesh.triangle_count == 0:
        return (np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64))

    all_points: list[np.ndarray] = []
    all_normals: list[np.ndarray] = []
    tri_xyz = mesh.vertices[mesh.triangles]
    for ti, tri in enumerate(tri_xyz):
        v0, v1, v2 = tri
        e01 = np.linalg.norm(v1 - v0)
        e12 = np.linalg.norm(v2 - v1)
        e20 = np.linalg.norm(v0 - v2)
        max_edge = float(max(e01, e12, e20))
        n = max(1, int(np.ceil(max_edge / spacing)))

        pts: list[np.ndarray] = []
        for i in range(n + 1):
            for j in range(n + 1 - i):
                u = i / n
                v = j / n
                w = 1.0 - u - v
                p = w * v0 + u * v1 + v * v2
                pts.append(p)
        tri_pts = np.asarray(pts, dtype=np.float64)
        tri_n = np.repeat(mesh.triangle_normals[ti][None, :], tri_pts.shape[0], axis=0)
        all_points.append(tri_pts)
        all_normals.append(tri_n)

    pts = np.concatenate(all_points, axis=0)
    nrm = np.concatenate(all_normals, axis=0)
    return _deduplicate_points_with_normals(pts, nrm, tol=max(spacing * 0.25, 1e-9))


def generate_boundary_layers(
    points: np.ndarray,
    normals: np.ndarray,
    *,
    layers: int,
    layer_spacing: float,
    direction: str = "outward",
) -> tuple[np.ndarray, np.ndarray]:
    if layers <= 0:
        raise ValueError("layers must be >= 1")
    layer_spacing = float(layer_spacing)
    if layer_spacing <= 0.0:
        raise ValueError("layer_spacing must be > 0")
    sign = 1.0 if direction.lower() == "outward" else -1.0

    all_p = []
    all_n = []
    for k in range(layers):
        off = sign * float(k) * layer_spacing
        all_p.append(points + off * normals)
        all_n.append(normals)
    p = np.concatenate(all_p, axis=0)
    n = np.concatenate(all_n, axis=0)
    return (p, n)


def estimate_point_spacing(points: np.ndarray, *, max_points: int = 400) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 2:
        return float("nan")
    if pts.shape[0] > max_points:
        idx = np.linspace(0, pts.shape[0] - 1, max_points).astype(np.int64)
        pts = pts[idx]
    d2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    nearest = np.sqrt(np.min(d2, axis=1))
    return float(np.median(nearest))


def compute_fluid_boundary_distance_stats(
    fluid_positions: np.ndarray,
    boundary_positions: np.ndarray,
    *,
    threshold: float,
    max_fluid_points: int = 2000,
    max_boundary_points: int = 4000,
) -> DistanceStats:
    f = np.asarray(fluid_positions, dtype=np.float64)
    b = np.asarray(boundary_positions, dtype=np.float64)
    if f.size == 0 or b.size == 0:
        return DistanceStats(min_distance=float("nan"), mean_min_distance=float("nan"), close_fraction=0.0, close_count=0, fluid_count=int(f.shape[0]))
    if f.shape[0] > max_fluid_points:
        idxf = np.linspace(0, f.shape[0] - 1, max_fluid_points).astype(np.int64)
        f = f[idxf]
    if b.shape[0] > max_boundary_points:
        idxb = np.linspace(0, b.shape[0] - 1, max_boundary_points).astype(np.int64)
        b = b[idxb]

    min_d = np.full((f.shape[0],), np.inf, dtype=np.float64)
    chunk = 512
    for i0 in range(0, b.shape[0], chunk):
        bc = b[i0 : i0 + chunk]
        d2 = np.sum((f[:, None, :] - bc[None, :, :]) ** 2, axis=2)
        min_d = np.minimum(min_d, np.sqrt(np.min(d2, axis=1)))
    close = min_d < float(threshold)
    return DistanceStats(
        min_distance=float(np.min(min_d)),
        mean_min_distance=float(np.mean(min_d)),
        close_fraction=float(np.mean(close)),
        close_count=int(np.count_nonzero(close)),
        fluid_count=int(min_d.shape[0]),
    )


def write_boundary_cloud_csv(path: str | Path, points: np.ndarray, normals: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as f:
        f.write("x,y,z,nx,ny,nz\n")
        for i in range(points.shape[0]):
            f.write(
                f"{points[i,0]:.17g},{points[i,1]:.17g},{points[i,2]:.17g},"
                f"{normals[i,0]:.17g},{normals[i,1]:.17g},{normals[i,2]:.17g}\n"
            )


def _deduplicate_points_with_normals(points: np.ndarray, normals: np.ndarray, *, tol: float) -> tuple[np.ndarray, np.ndarray]:
    if points.shape[0] == 0:
        return (points, normals)
    q = np.round(points / tol).astype(np.int64)
    uniq, inv = np.unique(q, axis=0, return_inverse=True)
    out_p = np.zeros((uniq.shape[0], 3), dtype=np.float64)
    out_n = np.zeros((uniq.shape[0], 3), dtype=np.float64)
    cnt = np.zeros((uniq.shape[0],), dtype=np.int64)
    for i, g in enumerate(inv):
        out_p[g] += points[i]
        out_n[g] += normals[i]
        cnt[g] += 1
    out_p /= cnt[:, None]
    nz = np.linalg.norm(out_n, axis=1)
    ok = nz > 0.0
    out_n[ok] /= nz[ok][:, None]
    out_n[~ok] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return (out_p, out_n)

