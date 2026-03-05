from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_UNITS_TO_METERS = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "mm": 1e-3,
    "millimeter": 1e-3,
    "millimeters": 1e-3,
    "cm": 1e-2,
    "centimeter": 1e-2,
    "centimeters": 1e-2,
}


def _rotation_matrix_xyz_deg(euler_deg: list[float] | tuple[float, float, float] | np.ndarray) -> np.ndarray:
    ex, ey, ez = [float(v) for v in euler_deg]
    rx, ry, rz = np.deg2rad([ex, ey, ez])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rx_m = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    ry_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rz_m = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    # XYZ intrinsic equivalent matrix.
    return rz_m @ ry_m @ rx_m


@dataclass(frozen=True)
class Mesh:
    """
    Triangle mesh container suitable for SPH boundary generation.

    - vertices: shape (N,3)
    - triangles: shape (M,3), integer indices into vertices
    """

    vertices: np.ndarray
    triangles: np.ndarray
    triangle_normals: np.ndarray
    triangle_areas: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    surface_area: float
    characteristic_length: float
    degenerate_triangles_dropped: int
    non_manifold_edge_ratio: float
    open_edge_ratio: float
    normal_consistency_ratio: float | None

    @property
    def triangle_count(self) -> int:
        return int(self.triangles.shape[0])

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @classmethod
    def from_triangle_vertices(
        cls,
        triangles_xyz: np.ndarray,
        *,
        area_epsilon: float = 1e-16,
    ) -> "Mesh":
        tris = np.asarray(triangles_xyz, dtype=np.float64)
        if tris.ndim != 3 or tris.shape[1:] != (3, 3):
            raise ValueError(f"triangles_xyz must have shape (M,3,3), got {tris.shape}")
        if tris.shape[0] == 0:
            raise ValueError("Mesh has zero triangles.")

        v0 = tris[:, 0, :]
        v1 = tris[:, 1, :]
        v2 = tris[:, 2, :]
        n_raw = np.cross(v1 - v0, v2 - v0)
        area2 = np.linalg.norm(n_raw, axis=1)
        areas = 0.5 * area2

        keep = areas > float(area_epsilon)
        dropped = int(np.count_nonzero(~keep))
        tris = tris[keep]
        n_raw = n_raw[keep]
        areas = areas[keep]
        if tris.shape[0] == 0:
            raise ValueError("All triangles are degenerate (area ~ 0).")

        n_norm = np.linalg.norm(n_raw, axis=1)
        normals = np.zeros_like(n_raw)
        nz = n_norm > 0.0
        normals[nz] = n_raw[nz] / n_norm[nz][:, None]

        tri_flat = tris.reshape(-1, 3)
        vertices, inverse = np.unique(tri_flat, axis=0, return_inverse=True)
        triangles = inverse.reshape(-1, 3).astype(np.int64)

        bbox_min = np.min(vertices, axis=0)
        bbox_max = np.max(vertices, axis=0)
        diag = float(np.linalg.norm(bbox_max - bbox_min))
        surface_area = float(np.sum(areas))

        non_manifold_ratio, open_edge_ratio, normal_consistency_ratio = _mesh_edge_quality(triangles)
        return cls(
            vertices=vertices.astype(np.float64),
            triangles=triangles,
            triangle_normals=normals.astype(np.float64),
            triangle_areas=areas.astype(np.float64),
            bbox_min=bbox_min.astype(np.float64),
            bbox_max=bbox_max.astype(np.float64),
            surface_area=surface_area,
            characteristic_length=diag,
            degenerate_triangles_dropped=dropped,
            non_manifold_edge_ratio=non_manifold_ratio,
            open_edge_ratio=open_edge_ratio,
            normal_consistency_ratio=normal_consistency_ratio,
        )

    def transformed(
        self,
        *,
        scale: float | list[float] | tuple[float, float, float] | np.ndarray = 1.0,
        translate: list[float] | tuple[float, float, float] | np.ndarray | None = None,
        rotate_euler_deg: list[float] | tuple[float, float, float] | np.ndarray | None = None,
        units_hint: str | None = None,
    ) -> "Mesh":
        """Return a transformed copy of the mesh (scale -> rotate -> translate)."""
        if np.isscalar(scale):
            svec = np.array([float(scale), float(scale), float(scale)], dtype=np.float64)
        else:
            sarr = np.asarray(scale, dtype=np.float64)
            if sarr.shape != (3,):
                raise ValueError("mesh scale must be scalar or length-3")
            svec = sarr

        if units_hint is not None:
            key = str(units_hint).strip().lower()
            if key not in _UNITS_TO_METERS:
                raise ValueError(f"Unsupported units_hint={units_hint!r}. Use one of: {sorted(_UNITS_TO_METERS.keys())}")
            svec = svec * _UNITS_TO_METERS[key]

        r = np.eye(3, dtype=np.float64)
        if rotate_euler_deg is not None:
            e = np.asarray(rotate_euler_deg, dtype=np.float64)
            if e.shape != (3,):
                raise ValueError("rotate_euler_deg must be length-3")
            r = _rotation_matrix_xyz_deg(e)

        t = np.zeros((3,), dtype=np.float64)
        if translate is not None:
            t_arr = np.asarray(translate, dtype=np.float64)
            if t_arr.shape != (3,):
                raise ValueError("translate must be length-3")
            t = t_arr

        v = (self.vertices * svec[None, :]) @ r.T + t[None, :]
        tri_xyz = v[self.triangles]
        return Mesh.from_triangle_vertices(tri_xyz)


def _mesh_edge_quality(triangles: np.ndarray) -> tuple[float, float, float | None]:
    """
    Compute watertight-ish and normal-orientation consistency diagnostics.

    - open_edge_ratio: fraction of unique edges with only one adjacent triangle.
    - non_manifold_edge_ratio: fraction with >2 adjacent triangles.
    - normal_consistency_ratio: for edges shared by exactly 2 triangles, fraction
      where the two triangles use opposite directed edge orientation.
    """
    if triangles.size == 0:
        return (0.0, 0.0, None)

    edge_count: dict[tuple[int, int], int] = {}
    directed_map: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for tri in triangles:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        edges = [(i0, i1), (i1, i2), (i2, i0)]
        for a, b in edges:
            key = (a, b) if a < b else (b, a)
            edge_count[key] = edge_count.get(key, 0) + 1
            directed_map.setdefault(key, []).append((a, b))

    keys = list(edge_count.keys())
    if not keys:
        return (0.0, 0.0, None)
    counts = np.array([edge_count[k] for k in keys], dtype=np.int64)
    open_ratio = float(np.count_nonzero(counts == 1) / len(keys))
    nonmanifold_ratio = float(np.count_nonzero(counts > 2) / len(keys))

    consistent = 0
    total_shared = 0
    for key, c in edge_count.items():
        if c != 2:
            continue
        d = directed_map[key]
        total_shared += 1
        # For consistent orientation, the two half-edges should be opposite.
        if len(d) == 2 and d[0] == (d[1][1], d[1][0]):
            consistent += 1
    consistency = (float(consistent / total_shared) if total_shared > 0 else None)
    return (nonmanifold_ratio, open_ratio, consistency)

