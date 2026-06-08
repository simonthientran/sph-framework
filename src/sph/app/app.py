"""Entry point for the interactive SPH application."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PyQt6 import QtWidgets
import pyqtgraph as pg

from sph.app.main_window import MainWindow
from sph.core.simulation import list_scene_files


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_scene_dir() -> Path:
    return _repo_root() / "scenes" / "examples"


def _default_scene_file() -> Path | None:
    """Default scene shown on launch: the clean 90° elbow (round L-shape)."""
    elbow = _repo_root() / "scenes" / "pipe_elbow_industrial.json"
    return elbow if elbow.exists() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SPH Live App")
    parser.add_argument(
        "--scenes",
        type=Path,
        default=_default_scene_dir(),
        help="Directory containing scene JSON files.",
    )
    parser.add_argument(
        "--scene",
        type=Path,
        default=None,
        help="Optional default scene file (defaults to the 90° elbow).",
    )
    args = parser.parse_args(argv)

    scene_dir = args.scenes.resolve()
    scene_paths = list_scene_files(scene_dir)
    if not scene_paths:
        raise SystemExit(f"No scenes found in {scene_dir}")

    # Default to the clean 90° elbow when no --scene is given.
    initial_scene = args.scene.resolve() if args.scene else _default_scene_file()
    if initial_scene and initial_scene not in scene_paths:
        scene_paths.insert(0, initial_scene)

    pg.setConfigOptions(antialias=True, background="k", foreground="w")
    qt_app = QtWidgets.QApplication(sys.argv[:1])
    window = MainWindow(scene_paths=scene_paths, initial_scene=initial_scene)
    window.show()
    return qt_app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
