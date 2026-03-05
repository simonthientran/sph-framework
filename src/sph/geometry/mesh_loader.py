from __future__ import annotations

from pathlib import Path

import numpy as np

from sph.geometry.stl import load_stl_mesh as _load_stl_mesh


def load_stl_mesh(path: str | Path) -> np.ndarray:
    """
    Backward-compatible loader returning triangles as shape (N, 3, 3).
    """
    mesh = _load_stl_mesh(path)
    return mesh.vertices[mesh.triangles].astype(np.float64)

