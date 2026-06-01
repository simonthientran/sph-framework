#!/usr/bin/env python3
from __future__ import annotations

"""Render key screenshots from the STL-container VTK sequence."""

from pathlib import Path

import numpy as np
import pyvista as pv


def main() -> int:
    vtk_dir = Path("out/stl_container_3d")
    out_dir = vtk_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    vtk_files = sorted(vtk_dir.glob("*.vtk"))
    if not vtk_files:
        print("No VTK files found. Run the STL-container simulation first.")
        return 1

    n = len(vtk_files)
    key_frames = [
        vtk_files[0],
        vtk_files[n // 4],
        vtk_files[n // 2],
        vtk_files[(3 * n) // 4],
        vtk_files[-1],
    ]

    for index, vtk_file in enumerate(key_frames):
        mesh = pv.read(vtk_file)

        plotter = pv.Plotter(off_screen=True, window_size=(1200, 800))
        plotter.set_background("white")

        scalar_name = None
        if "debug_velocity" in mesh.point_data:
            velocity = np.asarray(mesh.point_data["debug_velocity"], dtype=np.float64)
            mesh.point_data["speed"] = np.linalg.norm(velocity, axis=1)
            scalar_name = "speed"
        elif "v" in mesh.point_data:
            velocity = np.asarray(mesh.point_data["v"], dtype=np.float64)
            mesh.point_data["speed"] = np.linalg.norm(velocity, axis=1)
            scalar_name = "speed"

        if "is_boundary" in mesh.point_data:
            is_boundary = np.asarray(mesh.point_data["is_boundary"], dtype=np.float64) > 0.5
            fluid = mesh.extract_points(~is_boundary, adjacent_cells=False)
            boundary = mesh.extract_points(is_boundary, adjacent_cells=False)
        else:
            fluid = mesh
            boundary = None

        if boundary is not None and boundary.n_points:
            plotter.add_mesh(
                boundary,
                color="black",
                point_size=7,
                opacity=0.18,
                render_points_as_spheres=True,
            )

        plotter.add_mesh(
            fluid,
            scalars=scalar_name,
            cmap="plasma",
            point_size=11,
            render_points_as_spheres=True,
            clim=[0.0, 2.0] if scalar_name else None,
        )

        plotter.add_mesh(
            pv.Box(bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0)),
            style="wireframe",
            color="black",
            line_width=2,
        )

        plotter.camera_position = [
            (1.7, 1.2, 1.5),
            (0.5, 0.45, 0.5),
            (0.0, 1.0, 0.0),
        ]
        step_label = vtk_file.stem
        plotter.add_title(f"STL Container 3D - {step_label}", font_size=14)

        out_path = out_dir / f"frame_{index:03d}_{step_label}.png"
        plotter.screenshot(out_path)
        plotter.close()
        print(f"Saved {out_path}")

    print(f"\n{len(key_frames)} frames saved to {out_dir}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
