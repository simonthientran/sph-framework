"""
Create an L-shaped elbow pipe STL for the 3D elbow benchmark scene.

Geometry: vertical section (y-axis) joined to a horizontal section (x-axis).
Both sections are hollow square tubes (easier to mesh than cylinders without
boolean operations). The inner channel is 0.16m wide; outer walls add 0.04m
each side giving 0.24m total cross-section.

Usage:
    cd /path/to/sph-framework
    python scripts/create_elbow.py
Output:
    assets/elbow.stl
"""
from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

try:
    import trimesh
except ImportError:
    print("ERROR: trimesh not installed. Run: pip install trimesh")
    sys.exit(1)


def make_box_mesh(x0, y0, z0, x1, y1, z1) -> trimesh.Trimesh:
    """Create a solid box mesh from corner to corner."""
    vertices = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ], dtype=float)
    faces = np.array([
        [0, 2, 1], [0, 3, 2],  # bottom
        [4, 5, 6], [4, 6, 7],  # top
        [0, 1, 5], [0, 5, 4],  # front
        [2, 3, 7], [2, 7, 6],  # back
        [0, 4, 7], [0, 7, 3],  # left
        [1, 2, 6], [1, 6, 5],  # right
    ])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def create_hollow_square_tube(
    axis: str,
    start: float,
    end: float,
    center_y: float,
    center_z: float,
    inner_half: float = 0.08,
    wall_thick: float = 0.04,
) -> trimesh.Trimesh:
    """
    Create a hollow square tube aligned with `axis` (x or y).

    The tube runs from `start` to `end` along the axis,
    centered at (center_y, center_z) in the perpendicular plane.

    Returns the shell (6 faces: 4 walls + 2 caps with holes) as a
    set of solid outer and inner boxes combined via subtraction if
    trimesh supports it, otherwise returns just the outer box for
    boundary particle generation (the inner fluid space is empty
    in particle-based simulation).
    """
    iy0 = center_y - inner_half
    iy1 = center_y + inner_half
    iz0 = center_z - inner_half
    iz1 = center_z + inner_half

    oy0 = center_y - inner_half - wall_thick
    oy1 = center_y + inner_half + wall_thick
    oz0 = center_z - inner_half - wall_thick
    oz1 = center_z + inner_half + wall_thick

    parts = []

    if axis == "y":
        # 4 wall slabs running along y
        parts.append(make_box_mesh(start, oy0, oz0,  end, iy0, oz1))   # -z wall
        parts.append(make_box_mesh(start, iy1, oz0,  end, oy1, oz1))   # +z wall (typo: actually y-wall)
        parts.append(make_box_mesh(start, iy0, oz0,  end, iy1, iz0))   # -z wall
        parts.append(make_box_mesh(start, iy0, iz1,  end, iy1, oz1))   # +z wall
    else:  # axis == "x"
        parts.append(make_box_mesh(start, oy0, oz0,  end, iy0, oz1))
        parts.append(make_box_mesh(start, iy1, oz0,  end, oy1, oz1))
        parts.append(make_box_mesh(start, iy0, oz0,  end, iy1, iz0))
        parts.append(make_box_mesh(start, iy0, iz1,  end, iy1, oz1))

    return trimesh.util.concatenate(parts)


def main():
    out_path = Path(__file__).parent.parent / "assets" / "elbow.stl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Parameters
    inner_half = 0.08   # half inner width (inner channel = 0.16m)
    wall_thick = 0.04   # wall thickness
    cy = cz = 0.10      # pipe center in y and z

    # Vertical section: y from 0.0 to 0.5 (going down with gravity)
    vert = create_hollow_square_tube(
        axis="y", start=0.0, end=0.5,
        center_y=cy, center_z=cz,
        inner_half=inner_half, wall_thick=wall_thick,
    )

    # Horizontal section: x from 0.0 to 0.5 (turning into x direction)
    # Connect at y=0.0 (bottom of vertical section)
    horiz = create_hollow_square_tube(
        axis="x", start=-0.5, end=0.0,
        center_y=0.0, center_z=cz,
        inner_half=inner_half, wall_thick=wall_thick,
    )

    # Combine (no boolean subtraction needed — particle boundary uses outer surface)
    elbow = trimesh.util.concatenate([vert, horiz])

    # Export
    elbow.export(str(out_path))
    print(f"Created: {out_path}")
    print(f"  Vertices: {len(elbow.vertices)}")
    print(f"  Faces: {len(elbow.faces)}")
    print(f"  Bounds: {elbow.bounds}")


if __name__ == "__main__":
    main()
