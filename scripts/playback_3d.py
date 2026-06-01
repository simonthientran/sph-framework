#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from sph.visualization import PyVistaPlaybackViewer
from sph.visualization.pyvista_playback import collect_vtk_frame_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive PyVista playback viewer for exported VTK sequences.",
    )
    parser.add_argument("--vtk-dir", required=True, help="Directory containing exported VTK frames.")
    parser.add_argument("--scene", help="Optional scene JSON for boundary mesh overlay.")
    parser.add_argument("--scalar", default="particle_speed", help="Scalar field to color fluid particles.")
    parser.add_argument("--frame", type=int, default=0, help="Initial frame index.")
    parser.add_argument("--interval-ms", type=int, default=180, help="Playback timer interval in milliseconds.")
    parser.add_argument("--off-screen", action="store_true", help="Render without opening an interactive window.")
    parser.add_argument("--screenshot", type=Path, help="Save a screenshot of the current frame.")
    parser.add_argument(
        "--frames-dir",
        type=Path,
        help="Write a numbered PNG sequence from the loaded VTK frames.",
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="Maximum number of frames to export when --frames-dir is used.",
    )
    parser.add_argument("--hide-boundary-mesh", action="store_true", help="Hide source boundary mesh overlay.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    frame_count = len(collect_vtk_frame_paths(args.vtk_dir))
    if frame_count == 0:
        parser.error(f"No VTK frames found in {args.vtk_dir}")

    viewer = PyVistaPlaybackViewer(
        args.vtk_dir,
        scene_path=args.scene,
        scalar=args.scalar,
        off_screen=args.off_screen or args.screenshot is not None or args.frames_dir is not None,
        show_boundary_mesh=not args.hide_boundary_mesh,
        playback_interval_ms=args.interval_ms,
    )
    try:
        viewer.set_frame(args.frame)

        if args.frames_dir is not None:
            outputs = viewer.record_frames(args.frames_dir, frame_limit=args.frame_limit)
            print(f"Recorded playback frames: {len(outputs)} files in {args.frames_dir}")
            return 0

        if args.screenshot is not None:
            output = viewer.save_screenshot(args.screenshot)
            print(f"Saved playback screenshot: {output}")
            return 0

        viewer.show(auto_close=False)
        return 0
    finally:
        viewer.close()


if __name__ == "__main__":
    raise SystemExit(main())
