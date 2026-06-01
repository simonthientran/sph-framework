from __future__ import annotations

import json
import tempfile
from pathlib import Path

from sph.scene.fluid_sources import load_fluid_particle_layout_from_scene
from sph.simulator_new import Simulator


SCENE_BOX = Path("scenes/examples/box_fill_3d.json").resolve()


def test_fluid_layout_supports_multiple_block_sources():
    scene = json.loads(SCENE_BOX.read_text())
    scene.pop("fluid", None)
    scene["fluids"] = [
        {
            "type": "block",
            "name": "left_pool",
            "spacing": 0.05,
            "min": [0.10, 0.10, 0.10],
            "max": [0.20, 0.20, 0.20],
            "initial_velocity": [0.0, 0.0, 0.0],
        },
        {
            "type": "block",
            "name": "right_pool",
            "spacing": 0.05,
            "min": [0.70, 0.10, 0.10],
            "max": [0.80, 0.20, 0.20],
            "initial_velocity": [1.0, 0.0, 0.0],
        },
    ]

    layout = load_fluid_particle_layout_from_scene(scene, SCENE_BOX, dim=3)

    assert layout.spacing == 0.05
    assert len(layout.sources) == 2
    assert layout.sources[0].source_name == "left_pool"
    assert layout.sources[1].source_name == "right_pool"
    assert layout.positions.shape[1] == 3
    assert layout.positions.shape[0] == sum(source.count for source in layout.sources)
    assert layout.positions.shape[0] > 0
    assert float(layout.velocities[:, 0].max()) == 1.0


def test_fluid_layout_supports_mesh_fill_sources():
    scene = json.loads(SCENE_BOX.read_text())
    scene.pop("fluid", None)
    scene["fluids"] = [
        {
            "type": "mesh_fill",
            "name": "box_fill",
            "spacing": 0.05,
            "file": str(Path("assets/box.stl").resolve()),
            "sdf_resolution": 20,
            "inside_threshold": -0.10,
            "initial_velocity": [0.0, 0.0, 0.0],
        }
    ]

    layout = load_fluid_particle_layout_from_scene(scene, SCENE_BOX, dim=3)

    assert len(layout.sources) == 1
    assert layout.sources[0].source_type == "mesh_fill"
    assert layout.positions.shape[0] > 1000
    assert float(layout.bounds_min.min()) >= 0.10
    assert float(layout.bounds_max.max()) <= 0.90


def test_simulator_runs_scene_with_mesh_fill_source():
    scene = json.loads(SCENE_BOX.read_text())
    scene.pop("fluid", None)
    scene["fluids"] = [
        {
            "type": "mesh_fill",
            "name": "box_fill",
            "spacing": 0.05,
            "file": str(Path("assets/box.stl").resolve()),
            "sdf_resolution": 20,
            "inside_threshold": -0.10,
            "initial_velocity": [0.0, 0.0, 0.0],
        }
    ]
    scene["boundaries"][0]["file"] = str(Path("assets/box.stl").resolve())
    scene["time"] = {
        "mode": "fixed",
        "dt_fixed": 1.0e-4,
        "dt_min": 1.0e-4,
        "dt_max": 1.0e-4,
        "steps": 1,
    }

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump(scene, tmp)
        tmp_path = Path(tmp.name)

    sim = Simulator(tmp_path)
    assert sim.fluid.n > 1000
    sim.step()
    assert sim.current_step == 1
