"""
Create pipe geometries as STL files for SPH benchmarks.

Outputs (into assets/pipes/):
  straight_pipe.stl   — hollow straight tube, inner R=0.07 m, length L=1.0 m
  elbow_pipe.stl      — two cylinder sections joined at 90 deg, R=0.05 m
  expanding_pipe.stl  — frustum expanding R=0.05 -> R=0.10 m, length L=0.5 m

Design notes
------------
Boolean CSG (trimesh.boolean.difference) needs an external backend (manifold/
blender/openscad). When none is available it raises, so the straight pipe falls
back to a manually-built watertight hollow tube — the same lateral-surface
approach used by scripts/create_cylinder_pipe.py.

Paths are resolved relative to this file so the script writes to the repo's
assets/pipes/ regardless of the current working directory.

Usage
-----
  python scripts/create_geometries.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

ASSETS = Path(__file__).parent.parent / "assets"
PIPES = ASSETS / "pipes"
PIPES.mkdir(parents=True, exist_ok=True)


def make_hollow_tube(
    r_inner: float,
    r_outer: float,
    length: float,
    sections: int = 64,
) -> trimesh.Trimesh:
    """
    Build a watertight hollow tube aligned with the z-axis (centered at origin).

    Manual fallback for when boolean CSG has no backend. Four vertex rings
    (inner/outer x bottom/top) connected by an inner surface, an outer surface,
    and two annular end caps. fix_normals() orients everything outward.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, sections, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    n = sections
    z0 = -0.5 * length
    z1 = 0.5 * length

    verts = np.zeros((4 * n, 3), dtype=np.float64)
    for k, (r, z) in enumerate(
        [(r_inner, z0), (r_inner, z1), (r_outer, z0), (r_outer, z1)]
    ):
        verts[k * n : (k + 1) * n, 0] = r * cos_t
        verts[k * n : (k + 1) * n, 1] = r * sin_t
        verts[k * n : (k + 1) * n, 2] = z

    faces: list[list[int]] = []
    # inner lateral surface
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + j])
        faces.append([i, n + j, n + i])
    # outer lateral surface
    for i in range(n):
        j = (i + 1) % n
        faces.append([2 * n + i, 3 * n + i, 3 * n + j])
        faces.append([2 * n + i, 3 * n + j, 2 * n + j])
    # cap at z0
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, 2 * n + i, 2 * n + j])
        faces.append([i, 2 * n + j, j])
    # cap at z1
    for i in range(n):
        j = (i + 1) % n
        faces.append([n + i, n + j, 3 * n + j])
        faces.append([n + i, 3 * n + j, 3 * n + i])

    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=np.array(faces, dtype=np.int64),
        process=False,
    )
    mesh.fix_normals()
    return mesh


def create_straight_pipe(out_path: Path) -> trimesh.Trimesh:
    """Hollow straight pipe: inner R=0.07, outer R=0.08, L=1.0.

    Try boolean difference first; fall back to a manual hollow tube if no CSG
    backend is available.
    """
    r_inner = 0.07
    r_outer = 0.08
    length = 1.0
    sections = 64
    try:
        outer = trimesh.creation.cylinder(radius=r_outer, height=length, sections=sections)
        inner = trimesh.creation.cylinder(radius=r_inner, height=length * 1.1, sections=sections)
        straight = trimesh.boolean.difference([outer, inner])
        if straight is None or straight.is_empty or len(straight.faces) == 0:
            raise ValueError("boolean difference returned an empty mesh")
        print("  straight_pipe: built via trimesh.boolean.difference")
    except Exception as exc:  # noqa: BLE001 - any CSG backend issue -> fallback
        print(f"  straight_pipe: boolean CSG unavailable ({exc}); using manual hollow tube")
        straight = make_hollow_tube(r_inner, r_outer, length, sections=sections)
    straight.export(str(out_path))
    return straight


def create_elbow_pipe(out_path: Path) -> trimesh.Trimesh:
    """Two cylinder sections joined at 90 degrees, R=0.05."""
    radius = 0.05
    seg_len = 0.5
    sections = 64

    # Horizontal segment along x: cylinder defaults to z-axis, rotate to x.
    h_pipe = trimesh.creation.cylinder(radius=radius, height=seg_len, sections=sections)
    Rz_to_x = trimesh.transformations.rotation_matrix(np.pi / 2.0, [0, 1, 0])
    h_pipe.apply_transform(Rz_to_x)
    h_pipe.apply_translation([seg_len / 2.0, 0.0, 0.0])

    # Vertical segment along y: rotate z-axis cylinder to y, join at the elbow.
    v_pipe = trimesh.creation.cylinder(radius=radius, height=seg_len, sections=sections)
    Rz_to_y = trimesh.transformations.rotation_matrix(np.pi / 2.0, [1, 0, 0])
    v_pipe.apply_transform(Rz_to_y)
    v_pipe.apply_translation([seg_len, seg_len / 2.0, 0.0])

    try:
        elbow = trimesh.boolean.union([h_pipe, v_pipe])
        if elbow is None or elbow.is_empty or len(elbow.faces) == 0:
            raise ValueError("boolean union returned an empty mesh")
        print("  elbow_pipe: built via trimesh.boolean.union")
    except Exception as exc:  # noqa: BLE001
        print(f"  elbow_pipe: boolean CSG unavailable ({exc}); using concatenated sections")
        elbow = trimesh.util.concatenate([h_pipe, v_pipe])
    elbow.export(str(out_path))
    return elbow


def create_expanding_pipe(out_path: Path) -> trimesh.Trimesh:
    """Frustum expanding from R=0.05 to R=0.10 over length L=0.5 (z-axis)."""
    r_small = 0.05
    r_large = 0.10
    length = 0.5
    sections = 64

    theta = np.linspace(0.0, 2.0 * np.pi, sections, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    n = sections
    z0 = 0.0
    z1 = length

    # bottom ring (r_small) + top ring (r_large) + 2 cap centers
    verts = np.zeros((2 * n + 2, 3), dtype=np.float64)
    verts[:n, 0] = r_small * cos_t
    verts[:n, 1] = r_small * sin_t
    verts[:n, 2] = z0
    verts[n : 2 * n, 0] = r_large * cos_t
    verts[n : 2 * n, 1] = r_large * sin_t
    verts[n : 2 * n, 2] = z1
    c0 = 2 * n        # bottom center
    c1 = 2 * n + 1    # top center
    verts[c0] = [0.0, 0.0, z0]
    verts[c1] = [0.0, 0.0, z1]

    faces: list[list[int]] = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + j])
        faces.append([i, n + j, n + i])
        # bottom cap
        faces.append([c0, j, i])
        # top cap
        faces.append([c1, n + i, n + j])

    cone = trimesh.Trimesh(
        vertices=verts,
        faces=np.array(faces, dtype=np.int64),
        process=False,
    )
    cone.fix_normals()
    print("  expanding_pipe: built frustum (R=0.05 -> R=0.10)")
    cone.export(str(out_path))
    return cone


def _report(name: str, path: Path, mesh: trimesh.Trimesh) -> None:
    size = path.stat().st_size if path.exists() else 0
    print(
        f"Created {path.name}  "
        f"(vertices={len(mesh.vertices)}, faces={len(mesh.faces)}, "
        f"watertight={mesh.is_watertight}, {size} bytes)"
    )


def main() -> None:
    print(f"Writing pipe geometries to {PIPES}")

    print("1. Straight pipe (inner R=0.07, L=1.0):")
    straight_path = PIPES / "straight_pipe.stl"
    straight = create_straight_pipe(straight_path)
    _report("straight_pipe", straight_path, straight)

    print("2. Elbow pipe (R=0.05, 90 deg):")
    elbow_path = PIPES / "elbow_pipe.stl"
    elbow = create_elbow_pipe(elbow_path)
    _report("elbow_pipe", elbow_path, elbow)

    print("3. Expanding pipe (R=0.05 -> R=0.10, L=0.5):")
    expanding_path = PIPES / "expanding_pipe.stl"
    expanding = create_expanding_pipe(expanding_path)
    _report("expanding_pipe", expanding_path, expanding)

    print("\nSummary:")
    for p in (straight_path, elbow_path, expanding_path):
        ok = p.exists() and p.stat().st_size > 0
        print(f"  {'OK ' if ok else 'FAIL'} {p}")
    print("Done.")


if __name__ == "__main__":
    main()
