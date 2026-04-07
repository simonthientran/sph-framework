from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pyvista")

from sph.core.simulation import SimulationRunner
from sph.visualization import PyVistaPlaybackViewer


def _write_export_scene(scene_name: str, tmp_path: Path) -> Path:
    scene_path = Path("scenes/examples") / scene_name
    scene = json.loads(scene_path.read_text())
    for boundary in scene.get("boundaries", []):
        if "file" in boundary:
            boundary["file"] = str((Path("assets") / Path(boundary["file"]).name).resolve())
    scene["export"] = {
        "csv": {"enable": False},
        "vtk": {"enable": True, "every": 1, "directory": str((tmp_path / "vtk").resolve())},
    }
    temp_scene_path = tmp_path / scene_path.name
    temp_scene_path.write_text(json.dumps(scene), encoding="utf-8")
    return temp_scene_path


def _export_vtk_sequence(scene_path: Path, steps: int) -> Path:
    runner = SimulationRunner(scene_path, backend_name="numba_cpu")
    paths = runner.export_snapshot(csv=False, vtk=True)
    last_vtk = paths["vtk"]
    for _ in range(steps):
        runner.step()
        paths = runner.export_snapshot(csv=False, vtk=True)
        last_vtk = paths["vtk"]
    return last_vtk.parent


def test_pyvista_playback_viewer_loads_multiple_frames(tmp_path: Path) -> None:
    scene_path = _write_export_scene("box_fill_3d.json", tmp_path)
    vtk_dir = _export_vtk_sequence(scene_path, steps=2)
    screenshot_path = tmp_path / "playback.png"

    viewer = PyVistaPlaybackViewer(
        vtk_dir,
        scene_path=scene_path,
        scalar="density_deviation",
        off_screen=True,
    )
    try:
        summary0 = viewer.current_summary()
        assert summary0.frame_count >= 3
        assert summary0.fluid_count > 0
        assert summary0.boundary_count > 0
        assert summary0.diagnostics is None
        viewer.next_frame(loop=False)
        summary1 = viewer.current_summary()
        assert summary1.frame_index == 1
        assert summary1.diagnostics is not None
        assert "rho_mean" in summary1.diagnostics
        assert "solver_metrics" in summary1.diagnostics
        viewer.cycle_scalar()
        viewer.save_screenshot(screenshot_path)
    finally:
        viewer.close()

    assert screenshot_path.exists()
    assert screenshot_path.stat().st_size > 0


def test_pyvista_playback_records_frame_sequence(tmp_path: Path) -> None:
    scene_path = _write_export_scene("box_inflow_3d.json", tmp_path)
    vtk_dir = _export_vtk_sequence(scene_path, steps=2)
    frames_dir = tmp_path / "playback_frames"

    viewer = PyVistaPlaybackViewer(
        vtk_dir,
        scene_path=scene_path,
        scalar="particle_speed",
        off_screen=True,
    )
    try:
        outputs = viewer.record_frames(frames_dir, frame_limit=3)
    finally:
        viewer.close()

    assert len(outputs) == 3
    for output in outputs:
        assert output.exists()
        assert output.stat().st_size > 0
