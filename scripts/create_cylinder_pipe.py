"""
Create hollow cylinder pipe STL for SPH pipe flow benchmarks.

Two outputs:
  assets/cylinder_pipe.stl       — open lateral surface, for periodic-x flow
  assets/cylinder_pipe_closed.stl — closed (watertight) tube, for inlet/outlet demo

Geometry
--------
  R_inner  = 0.10 m   → 10 × particle_spacing (spacing=0.01), so 10 particle
                          layers fit across the radius (professor requirement)
  R_outer  = 0.13 m   → 3 boundary-particle layers at spacing=0.01
  L_open   = 0.20 m   → short periodic pipe
  L_closed = 0.50 m   → longer inlet/outlet pipe
  sections = 64        → round, not faceted

The open cylinder (no end caps) is correct for periodic-x Poiseuille flow.
is_watertight will be False for the open cylinder — this is expected and does
not affect Poisson boundary-particle sampling.

Usage
-----
  python scripts/create_cylinder_pipe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh

ASSETS = Path(__file__).parent.parent / "assets"
ASSETS.mkdir(exist_ok=True)


def make_open_cylinder(
    R: float, L: float, sections: int = 64, center_y: float = 0.0, center_z: float = 0.0
) -> trimesh.Trimesh:
    """
    Lateral surface only (no end caps), aligned with the x-axis.

    The outward surface normal points RADIALLY OUTWARD from the x-axis.
    This is correct for Akinci 2012 boundary sampling: fluid is inside
    (r < R) and boundary particles are layered radially outward (r > R).

    Args:
        center_y: y-coordinate of cylinder axis (default 0)
        center_z: z-coordinate of cylinder axis (default 0)
    """
    theta = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    n = sections

    # Two rings: x=0 (start) and x=L (end)
    verts = np.zeros((2 * n, 3), dtype=np.float64)
    verts[:n, 0] = 0.0
    verts[:n, 1] = center_y + R * cos_t
    verts[:n, 2] = center_z + R * sin_t
    verts[n:, 0] = L
    verts[n:, 1] = center_y + R * cos_t
    verts[n:, 2] = center_z + R * sin_t

    # Winding: [i, j, n+j] gives normal via cross product:
    #   v1 = j - i   (circumferential, CCW when viewed from outside = +θ direction)
    #   v2 = n+j - i (diagonal, has +x component)
    # n = v1 × v2 points radially OUTWARD (+r direction) for CCW vertex ordering.
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + j])
        faces.append([i, n + j, n + i])

    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=np.array(faces, dtype=np.int64),
        process=False,
    )
    # Verify outward normals: face normals should have positive dot with radial direction.
    # If not, invert the mesh.
    sample_normal = mesh.face_normals[0]
    mid_vert = verts[0]
    radial_dir = np.array([0.0, mid_vert[1] - center_y, mid_vert[2] - center_z])
    if np.dot(sample_normal, radial_dir) < 0:
        mesh.invert()
    return mesh


def make_closed_cylinder(
    R_inner: float, R_outer: float, L: float, sections: int = 64,
    center_y: float = 0.0, center_z: float = 0.0
) -> trimesh.Trimesh:
    """
    Hollow closed tube (watertight): inner surface, outer surface, two annular caps.

    The inner surface has inward normals (toward fluid = toward x-axis) so
    that Akinci sampling layers boundary particles radially inward from r=R_inner.
    The outer surface and caps close the mesh for watertightness.

    Args:
        center_y: y-coordinate of cylinder axis (default 0)
        center_z: z-coordinate of cylinder axis (default 0)
    """
    theta = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    n = sections

    # Vertex groups (4 rings):
    # 0..n-1   : inner ring at x=0
    # n..2n-1  : inner ring at x=L
    # 2n..3n-1 : outer ring at x=0
    # 3n..4n-1 : outer ring at x=L
    verts = np.zeros((4 * n, 3), dtype=np.float64)
    for k, (r, x) in enumerate(
        [(R_inner, 0.0), (R_inner, L), (R_outer, 0.0), (R_outer, L)]
    ):
        verts[k * n : (k + 1) * n, 0] = x
        verts[k * n : (k + 1) * n, 1] = center_y + r * cos_t
        verts[k * n : (k + 1) * n, 2] = center_z + r * sin_t

    faces = []

    # Inner lateral surface (normals pointing inward = toward axis)
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + j])
        faces.append([i, n + j, n + i])

    # Outer lateral surface (normals pointing outward = away from axis)
    for i in range(n):
        j = (i + 1) % n
        faces.append([2 * n + i, 3 * n + i, 3 * n + j])
        faces.append([2 * n + i, 3 * n + j, 2 * n + j])

    # Annular caps: connect inner and outer rings at x=0 and x=L
    # Cap at x=0 (normal pointing in -x direction)
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, 2 * n + i, 2 * n + j])
        faces.append([i, 2 * n + j, j])

    # Cap at x=L (normal pointing in +x direction)
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


def main() -> None:
    R_INNER = 0.10
    R_OUTER = 0.13
    L_OPEN = 0.50   # periodic-x benchmark pipe length
    L_CLOSED = 0.50  # longer pipe for inlet/outlet demo
    SECTIONS = 64
    CENTER_Y = 0.0
    CENTER_Z = 0.0

    # --- Open cylinder (lateral surface only, for periodic-x flow) -----------
    open_cyl = make_open_cylinder(
        R=R_INNER, L=L_OPEN, sections=SECTIONS,
        center_y=CENTER_Y, center_z=CENTER_Z,
    )
    out_open = ASSETS / "cylinder_pipe.stl"
    open_cyl.export(str(out_open))
    print(f"Created {out_open}")
    print(f"  R={R_INNER} m  L={L_OPEN} m  center=({CENTER_Y},{CENTER_Z})  sections={SECTIONS}")
    print(f"  vertices={len(open_cyl.vertices)}  faces={len(open_cyl.faces)}")
    print(f"  is_watertight={open_cyl.is_watertight}  "
          f"(False is expected for open-ended tube)")

    # --- Closed hollow cylinder (watertight, for inlet/outlet demo) ----------
    closed_cyl = make_closed_cylinder(
        R_inner=R_INNER, R_outer=R_OUTER, L=L_CLOSED, sections=SECTIONS,
        center_y=CENTER_Y, center_z=CENTER_Z,
    )
    out_closed = ASSETS / "cylinder_pipe_closed.stl"
    closed_cyl.export(str(out_closed))
    print(f"\nCreated {out_closed}")
    print(f"  R_inner={R_INNER} m  R_outer={R_OUTER} m  L={L_CLOSED} m")
    print(f"  vertices={len(closed_cyl.vertices)}  faces={len(closed_cyl.faces)}")
    print(f"  is_watertight={closed_cyl.is_watertight}")

    # Sanity check
    print()
    print("Analytical Hagen-Poiseuille (cylinder):")
    print(f"  vmax = f * R² / (4 * nu)")
    print(f"  With f=0.002 m/s², R={R_INNER} m, nu=0.001 m²/s:")
    vmax = 0.002 * R_INNER**2 / (4 * 0.001)
    print(f"  vmax = {vmax:.4f} m/s")
    print()
    print("Particle resolution (spacing=0.01 m):")
    print(f"  Layers across radius: {int(R_INNER / 0.01)} (professor requires >= 10)")


if __name__ == "__main__":
    main()
