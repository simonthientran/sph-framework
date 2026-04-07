from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pyvista")

from sph.visualization import PyVistaSceneViewer


def _write_temp_scene(scene_name: str, tmp_path: Path) -> Path:
    scene_path = Path("scenes/examples") / scene_name
    scene = json.loads(scene_path.read_text())
    for boundary in scene.get("boundaries", []):
        if "file" in boundary:
            boundary["file"] = str((Path("assets") / Path(boundary["file"]).name).resolve())
    temp_scene_path = tmp_path / scene_path.name
    temp_scene_path.write_text(json.dumps(scene), encoding="utf-8")
    return temp_scene_path


def test_pyvista_viewer_screenshot_and_step(tmp_path: Path) -> None:
    scene_path = _write_temp_scene("box_fill_3d.json", tmp_path)
    screenshot_path = tmp_path / "viewer.png"

    viewer = PyVistaSceneViewer(
        scene_path,
        backend_name="numba_cpu",
        scalar="neighbor_count",
        off_screen=True,
    )
    try:
        summary0 = viewer.last_summary
        assert summary0 is not None
        assert summary0.fluid_count > 0
        assert summary0.boundary_count > 0
        viewer.advance(1)
        summary1 = viewer.last_summary
        assert summary1 is not None
        assert summary1.step == 1
        viewer.save_screenshot(screenshot_path)
    finally:
        viewer.close()

    assert screenshot_path.exists()
    assert screenshot_path.stat().st_size > 0


def test_pyvista_viewer_records_frame_sequence_for_emitter_scene(tmp_path: Path) -> None:
    scene_path = _write_temp_scene("box_inflow_3d.json", tmp_path)
    frames_dir = tmp_path / "frames"

    viewer = PyVistaSceneViewer(
        scene_path,
        backend_name="numba_cpu",
        scalar="speed",
        off_screen=True,
    )
    try:
        outputs = viewer.record_frames(frames_dir, steps=2)
        summary = viewer.last_summary
        assert summary is not None
        assert summary.step >= 2
    finally:
        viewer.close()

    assert len(outputs) == 3
    for output in outputs:
        assert output.exists()
        assert output.stat().st_size > 0
