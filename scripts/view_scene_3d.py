#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from sph.visualization import PyVistaSceneViewer
from sph.visualization.pyvista_viewer import list_viewer_scalars


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive PyVista viewer for SPH runtime scenes.",
    )
    parser.add_argument("--scene", required=True, help="Path to a scene JSON file.")
    parser.add_argument("--backend", default="numba_cpu", help="Backend name to use.")
    parser.add_argument(
        "--scalar",
        default="speed",
        choices=list_viewer_scalars(),
        help="Fluid scalar used for coloring.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Advance this many steps before showing or capturing output.",
    )
    parser.add_argument(
        "--off-screen",
        action="store_true",
        help="Render without opening an interactive window.",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        help="Save a screenshot to this path after rendering.",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        help="Write a numbered PNG frame sequence while stepping the simulation.",
    )
    parser.add_argument(
        "--frame-steps",
        type=int,
        default=12,
        help="Number of runtime steps to record when --frames-dir is used.",
    )
    parser.add_argument(
        "--hide-boundary-mesh",
        action="store_true",
        help="Hide source boundary meshes and show only runtime particles.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    viewer = PyVistaSceneViewer(
        args.scene,
        backend_name=args.backend,
        scalar=args.scalar,
        off_screen=(
            args.off_screen
            or args.screenshot is not None
            or args.frames_dir is not None
        ),
        show_boundary_mesh=not args.hide_boundary_mesh,
    )
    try:
        if args.steps > 0:
            viewer.advance(args.steps)

        if args.frames_dir is not None:
            outputs = viewer.record_frames(args.frames_dir, steps=args.frame_steps)
            print(f"Recorded frames: {len(outputs)} files in {args.frames_dir}")
            return 0

        if args.screenshot is not None:
            output = viewer.save_screenshot(args.screenshot)
            print(f"Saved screenshot: {output}")
            return 0

        viewer.show(auto_close=False)
        return 0
    finally:
        viewer.close()


if __name__ == "__main__":
    raise SystemExit(main())
