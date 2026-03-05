from __future__ import annotations

import numpy as np


def sample_triangle_surface(triangles: np.ndarray, spacing: float) -> np.ndarray:
    """
    Sample points on triangle surfaces using random barycentric sampling.

    For each triangle:
      area = 0.5 * || (v1-v0) x (v2-v0) ||
      n = ceil(area / spacing^2)
    """
    tris = np.asarray(triangles, dtype=np.float64)
    if tris.ndim != 3 or tris.shape[1:] != (3, 3):
        raise ValueError(f"triangles must have shape (N,3,3), got {tris.shape}")
    spacing = float(spacing)
    if spacing <= 0.0:
        raise ValueError(f"spacing must be > 0, got {spacing}")
    if tris.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)

    points: list[np.ndarray] = []
    for tri in tris:
        v0, v1, v2 = tri
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        n = max(1, int(np.ceil(area / (spacing * spacing))))

        uv = np.random.rand(n, 2)
        u = uv[:, 0]
        v = uv[:, 1]
        mask = (u + v) > 1.0
        u[mask] = 1.0 - u[mask]
        v[mask] = 1.0 - v[mask]

        p = v0[None, :] + u[:, None] * (v1 - v0)[None, :] + v[:, None] * (v2 - v0)[None, :]
        points.append(p)

    return np.concatenate(points, axis=0) if points else np.zeros((0, 3), dtype=np.float64)

