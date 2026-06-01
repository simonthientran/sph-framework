"""
Render 5 evenly-spaced snapshots of the 3D dam break simulation from VTK files.
Uses PyVista offscreen rendering with a consistent camera angle.
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

VTK_DIR = Path("out/dam_break_3d")
OUT_DIR  = Path("out/dam_break_3d/screenshots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── collect all fluid VTK files (exclude boundary-only exports) ──────────────
vtk_files = sorted(VTK_DIR.glob("particles_step_*.vtk"))
if not vtk_files:
    print(f"ERROR: no VTK files found in {VTK_DIR}")
    sys.exit(1)

print(f"Found {len(vtk_files)} VTK files  ({vtk_files[0].name} … {vtk_files[-1].name})")

# pick 5 evenly-spaced frames
indices = np.linspace(0, len(vtk_files) - 1, 5, dtype=int)
frames  = [vtk_files[i] for i in indices]
print(f"Rendering frames: {[f.name for f in frames]}")

# ── render each frame ─────────────────────────────────────────────────────────
pv.global_theme.background = "white"
pv.global_theme.window_size = [1200, 800]

for idx, vtk_path in enumerate(frames):
    mesh = pv.read(vtk_path)

    # read speed from velocity VECTORS field
    if "v" in mesh.point_data.keys():
        vel = mesh.point_data["v"]
        speed = np.linalg.norm(vel, axis=1)
        mesh.point_data["speed"] = speed
        scalar = "speed"
        cmap   = "viridis"
    elif "rho" in mesh.point_data.keys():
        scalar = "rho"
        cmap   = "coolwarm"
    else:
        scalar = mesh.point_data.active_scalars_name
        cmap   = "viridis"

    # extract step number from filename
    stem  = vtk_path.stem  # e.g. particles_step_0500
    parts = stem.split("_")
    step_num = parts[-1]   # "0500"

    pl = pv.Plotter(off_screen=True, window_size=[1200, 800])
    pl.add_mesh(
        mesh,
        scalars=scalar,
        cmap=cmap,
        point_size=6,
        render_points_as_spheres=True,
        clim=None,
    )
    pl.add_axes()
    pl.add_title(f"Dam break – step {int(step_num)}", font_size=14)

    # isometric-ish camera for 3D box  [0,1.0] x [0,0.6] x [0,0.4]
    pl.camera_position = [
        (1.8, 1.2, 1.5),   # camera position
        (0.5, 0.2, 0.2),   # focal point (centre of container)
        (0.0, 1.0, 0.0),   # up vector
    ]
    pl.camera.zoom(1.1)

    out_path = OUT_DIR / f"frame_{idx:02d}_step_{step_num}.png"
    pl.screenshot(str(out_path))
    pl.close()
    print(f"  Saved {out_path}  (scalar={scalar})")

print(f"\nAll {len(frames)} screenshots saved to {OUT_DIR}")
