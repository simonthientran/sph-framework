from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from sph.geometry.mesh import Mesh


def load_stl_mesh(path: str | Path, *, area_epsilon: float = 1e-16) -> Mesh:
    """
    Load STL (ASCII or binary) and return validated Mesh.
    """
    stl_path = Path(path)
    if not stl_path.exists():
        raise FileNotFoundError(f"STL file not found: {stl_path}")

    raw = stl_path.read_bytes()
    triangles = _parse_binary_stl(raw)
    if triangles is None:
        triangles = _parse_ascii_stl(raw.decode("utf-8", errors="ignore"))
    if triangles is None or triangles.shape[0] == 0:
        raise ValueError(f"Failed to parse STL triangles from {stl_path}")
    return Mesh.from_triangle_vertices(triangles, area_epsilon=area_epsilon)


def _parse_binary_stl(raw: bytes) -> np.ndarray | None:
    if len(raw) < 84:
        return None
    tri_count = struct.unpack("<I", raw[80:84])[0]
    expected = 84 + tri_count * 50
    if tri_count == 0 or len(raw) != expected:
        return None

    tris = np.empty((tri_count, 3, 3), dtype=np.float64)
    off = 84
    for i in range(tri_count):
        # normal(12 bytes) + vertices(36 bytes) + attr(2 bytes)
        rec = raw[off : off + 50]
        if len(rec) != 50:
            return None
        vals = struct.unpack("<12fH", rec)
        tris[i, 0, :] = vals[3:6]
        tris[i, 1, :] = vals[6:9]
        tris[i, 2, :] = vals[9:12]
        off += 50
    return tris


def _parse_ascii_stl(text: str) -> np.ndarray | None:
    verts: list[list[float]] = []
    for line in text.splitlines():
        s = line.strip().lower()
        if not s.startswith("vertex"):
            continue
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        try:
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
        except ValueError:
            continue
        verts.append([x, y, z])
    if not verts or (len(verts) % 3) != 0:
        return None
    arr = np.asarray(verts, dtype=np.float64).reshape(-1, 3, 3)
    return arr

