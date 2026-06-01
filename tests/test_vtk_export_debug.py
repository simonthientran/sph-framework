from __future__ import annotations

import json
import tempfile
from pathlib import Path

from sph.core.simulation import SimulationRunner


SCENE_BOX = Path("scenes/examples/box_fill_3d.json").resolve()


def test_manual_vtk_export_includes_debug_fields():
    scene = json.loads(SCENE_BOX.read_text())
    scene["boundaries"][0]["file"] = str(Path("assets/box.stl").resolve())

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        scene["export"] = {
            "csv": {"enable": False, "dir": str(out_dir / "csv")},
            "vtk": {"enable": False, "dir": str(out_dir / "vtk")},
        }

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
            json.dump(scene, tmp)
            tmp_path = Path(tmp.name)

        runner = SimulationRunner(tmp_path, backend_name="numba_cpu")
        runner.step()
        paths = runner.export_snapshot(csv=False, vtk=True)

        vtk_path = paths["vtk"]
        assert vtk_path.exists()

        text = vtk_path.read_text(encoding="utf-8")
        assert "SCALARS density_deviation float 1" in text
        assert "SCALARS neighbor_count float 1" in text
        assert "SCALARS low_density_flag float 1" in text
        assert "SCALARS low_neighbor_flag float 1" in text
        assert "SCALARS free_surface_score float 1" in text
        assert "SCALARS fluid_mask float 1" in text
        assert "SCALARS boundary_speed float 1" in text
        assert "VECTORS debug_velocity float" in text
