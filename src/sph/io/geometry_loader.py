"""
STL/OBJ → boundary particle positions + normals.
Inspired by SPlisHSPlasH's surface sampling pipeline.

open3d is optional: Poisson-disk sampling and SDF queries fall back to
trimesh-only implementations when open3d is not installed.
"""
from __future__ import annotations

import numpy as np
import trimesh


def _contains_points_fallback(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    chunk_size: int = 128,
) -> np.ndarray:
    """
    Determine inside/outside without ``open3d`` or ``rtree``.

    This uses parity ray casting against the triangle soup with a skewed ray
    direction to avoid axis-aligned degeneracies on box-like meshes.
    """
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    if triangles.size == 0:
        return np.zeros(points.shape[0], dtype=bool)

    v0 = triangles[:, 0]
    edge1 = triangles[:, 1] - v0
    edge2 = triangles[:, 2] - v0
    direction = np.array([1.0, 0.3713906764, 0.1732050808], dtype=np.float64)
    direction /= np.linalg.norm(direction)

    h = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
    a = np.einsum("ij,ij->i", edge1, h)
    eps = 1.0e-9
    valid_triangles = np.abs(a) > eps
    a_safe = np.where(valid_triangles, a, 1.0)

    inside = np.zeros(points.shape[0], dtype=bool)
    for start in range(0, points.shape[0], chunk_size):
        chunk = np.asarray(points[start : start + chunk_size], dtype=np.float64)
        if chunk.size == 0:
            continue
        s = chunk[:, None, :] - v0[None, :, :]
        u = np.einsum("mti,ti->mt", s, h) / a_safe[None, :]
        q = np.cross(s, edge1[None, :, :])
        v = np.einsum("mti,i->mt", q, direction) / a_safe[None, :]
        t = np.einsum("mti,ti->mt", q, edge2) / a_safe[None, :]

        hits = (
            valid_triangles[None, :]
            & (u >= -eps)
            & (u <= 1.0 + eps)
            & (v >= -eps)
            & ((u + v) <= 1.0 + eps)
            & (t > eps)
        )
        inside[start : start + chunk.shape[0]] = (hits.sum(axis=1) % 2) == 1
    return inside


def _signed_distance_fallback(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    chunk_size: int = 128,
) -> np.ndarray:
    """
    Signed-distance fallback for environments without ``open3d``.

    ``trimesh.proximity.signed_distance`` currently pulls in ``rtree``.  This
    helper keeps the repo self-contained by combining:
    - exact closest-point distance via ``closest_point_naive`` in small chunks
    - custom parity ray casting for the sign
    """
    from trimesh.proximity import closest_point_naive

    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return np.zeros(0, dtype=np.float32)

    signed = np.zeros(points.shape[0], dtype=np.float32)
    inside = _contains_points_fallback(mesh, points, chunk_size=chunk_size)

    for start in range(0, points.shape[0], chunk_size):
        chunk = points[start : start + chunk_size]
        if chunk.size == 0:
            continue
        _, distance, _ = closest_point_naive(mesh, chunk)
        distance = np.asarray(distance, dtype=np.float32)
        mask = inside[start : start + chunk.shape[0]]
        distance[mask] *= -1.0
        signed[start : start + chunk.shape[0]] = distance

    return signed


def load_mesh(path: str) -> trimesh.Trimesh:
    """Load STL, OBJ, PLY, GLTF — trimesh handles all."""
    mesh = trimesh.load(path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load mesh from {path}")
    print(
        f"Loaded mesh: {len(mesh.vertices)} vertices, "
        f"{len(mesh.faces)} faces, "
        f"watertight={mesh.is_watertight}"
    )
    return mesh


def transform_mesh(
    mesh: trimesh.Trimesh,
    translation: np.ndarray | None = None,
    rotation_axis: np.ndarray | None = None,
    rotation_angle_degrees: float = 0.0,
    scale: np.ndarray | None = None,
) -> trimesh.Trimesh:
    """
    Apply a scene-authored rigid transform about the mesh centroid.

    The translation is interpreted as the target centroid position. This makes
    unit test assets like ``box.stl`` usable with SPlisHSPlasH-style scene
    transforms even though the mesh itself is not centered at the origin.
    """
    translation_arr = np.asarray(translation if translation is not None else [0.0, 0.0, 0.0], dtype=np.float64)
    axis_arr = np.asarray(rotation_axis if rotation_axis is not None else [0.0, 1.0, 0.0], dtype=np.float64)
    scale_arr = np.asarray(scale if scale is not None else [1.0, 1.0, 1.0], dtype=np.float64)

    if (
        np.allclose(translation_arr, 0.0)
        and np.allclose(scale_arr, 1.0)
        and abs(float(rotation_angle_degrees)) <= 1.0e-12
    ):
        return mesh

    result = mesh.copy()
    centroid = np.asarray(result.bounds.mean(axis=0), dtype=np.float64)
    result.apply_translation(-centroid)

    scale_matrix = np.eye(4, dtype=np.float64)
    scale_matrix[0, 0] = float(scale_arr[0])
    scale_matrix[1, 1] = float(scale_arr[1])
    scale_matrix[2, 2] = float(scale_arr[2])
    result.apply_transform(scale_matrix)

    axis_norm = np.linalg.norm(axis_arr)
    if axis_norm > 1.0e-12 and abs(float(rotation_angle_degrees)) > 1.0e-12:
        rotation = trimesh.transformations.rotation_matrix(
            np.deg2rad(float(rotation_angle_degrees)),
            axis_arr / axis_norm,
            point=[0.0, 0.0, 0.0],
        )
        result.apply_transform(rotation)

    result.apply_translation(translation_arr)
    return result


def sample_surface_particles(
    mesh: trimesh.Trimesh,
    dx: float,
    n_layers: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample boundary particles on mesh surface + inward layers.

    Inspired by SPlisHSPlasH RegularTriangleSampling:
    - Sample surface at spacing dx
    - Extrude inward by n_layers * dx along surface normals

    Returns:
        positions: (N, 3) float64
        normals:   (N, 3) float64  (pointing INTO fluid)
    """
    n_samples = max(int(mesh.area / (dx * dx)), 100)
    # Keep the fallback deterministic so STL/OBJ boundary counts are stable
    # across validation runs when Open3D is unavailable.
    points, face_ids = trimesh.sample.sample_surface(mesh, n_samples, seed=0)

    face_normals = mesh.face_normals[face_ids]  # outward normals (into solid)
    inward_normals = -face_normals  # point into fluid (stored as particle normals)

    all_positions: list[np.ndarray] = []
    all_normals: list[np.ndarray] = []

    for layer in range(n_layers):
        # Extrude into solid at (layer+0.5)*dx — matching the flat-wall convention used
        # by the domain boundary_layers code, which places layers at 0.5, 1.5, 2.5 * dx.
        # Starting at 0.5*dx (not 1*dx) ensures the innermost boundary layer is close
        # enough to the fluid surface that all n_layers fall within the support radius.
        layer_pos = points + (layer + 0.5) * dx * face_normals
        all_positions.append(layer_pos)
        all_normals.append(inward_normals)

    positions = np.vstack(all_positions)
    normals = np.vstack(all_normals)

    # Remove duplicate points (corner/edge overlap)
    from scipy.spatial import cKDTree

    tree = cKDTree(positions)
    pairs = tree.query_pairs(dx * 0.5)
    keep = np.ones(len(positions), dtype=bool)
    for i, j in pairs:
        keep[max(i, j)] = False

    return positions[keep].astype(np.float64), normals[keep].astype(np.float64)


def sample_surface_poisson(
    mesh: trimesh.Trimesh,
    dx: float,
    n_layers: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Higher quality: Poisson disk sampling via Open3D.
    Falls back to trimesh if Open3D not available.
    """
    try:
        import open3d as o3d

        o3d_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
            o3d.utility.Vector3iVector(np.asarray(mesh.faces)),
        )
        o3d_mesh.compute_vertex_normals()

        n_points = max(int(mesh.area / (dx * dx * 0.8)), 100)
        pcd = o3d_mesh.sample_points_poisson_disk(n_points)

        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        outward_normals = normals  # Open3D normals point outward (into solid)
        inward_normals = -normals   # flip to point into fluid

        all_positions: list[np.ndarray] = []
        all_normals: list[np.ndarray] = []
        for layer in range(n_layers):
            # Extrude at (layer+0.5)*dx — matches flat-wall convention (0.5, 1.5, 2.5 * dx)
            all_positions.append(points + (layer + 0.5) * dx * outward_normals)
            all_normals.append(inward_normals)

        return (
            np.vstack(all_positions).astype(np.float64),
            np.vstack(all_normals).astype(np.float64),
        )

    except ImportError:
        return sample_surface_particles(mesh, dx, n_layers)


def compute_sdf_grid(
    mesh: trimesh.Trimesh,
    resolution: int = 50,
    padding_ratio: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute SDF on a regular grid.

    Tries Open3D (Embree-based) first; falls back to trimesh proximity
    queries when Open3D is not installed.

    Returns:
        sdf_values: (res, res, res) float32  — negative inside the mesh
        grid_min:   (3,) float64
        grid_max:   (3,) float64
        cell_size:  float
    """
    bb = mesh.bounds
    pad = (bb[1] - bb[0]).max() * float(padding_ratio)
    grid_min = (bb[0] - pad).astype(np.float64)
    grid_max = (bb[1] + pad).astype(np.float64)

    xs = np.linspace(grid_min[0], grid_max[0], resolution)
    ys = np.linspace(grid_min[1], grid_max[1], resolution)
    zs = np.linspace(grid_min[2], grid_max[2], resolution)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    query_pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)

    try:
        import open3d as o3d

        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(
            o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float32)),
                o3d.utility.Vector3iVector(np.asarray(mesh.faces)),
            )
        )
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh_t)
        result = scene.compute_signed_distance(o3d.core.Tensor(query_pts))
        sdf_values = result.numpy().reshape(resolution, resolution, resolution)

    except ImportError:
        signed = _signed_distance_fallback(mesh, query_pts.astype(np.float64))
        sdf_values = signed.reshape(resolution, resolution, resolution).astype(np.float32)

    cell_size = float((grid_max - grid_min).max() / resolution)
    return sdf_values, grid_min, grid_max, cell_size


def fill_volume_from_sdf(
    sdf_values: np.ndarray,
    grid_min: np.ndarray,
    grid_max: np.ndarray,
    dx: float,
    inside_threshold: float = 0.0,
) -> np.ndarray:
    """
    Generate fluid particle positions inside a mesh using the precomputed SDF.

    SDF < inside_threshold → inside mesh → place fluid particle.

    Returns:
        positions: (N, 3) float64
    """
    from scipy.interpolate import RegularGridInterpolator

    res = sdf_values.shape[0]
    xs = np.linspace(grid_min[0], grid_max[0], res)
    ys = np.linspace(grid_min[1], grid_max[1], res)
    zs = np.linspace(grid_min[2], grid_max[2], res)

    interp = RegularGridInterpolator(
        (xs, ys, zs),
        sdf_values,
        method="linear",
        bounds_error=False,
        fill_value=1.0,
    )

    cx = np.arange(grid_min[0] + dx / 2, grid_max[0], dx)
    cy = np.arange(grid_min[1] + dx / 2, grid_max[1], dx)
    cz = np.arange(grid_min[2] + dx / 2, grid_max[2], dx)
    gx, gy, gz = np.meshgrid(cx, cy, cz, indexing="ij")
    candidates = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    sdf_at_candidates = interp(candidates)
    inside = sdf_at_candidates < inside_threshold

    return candidates[inside].astype(np.float64)
