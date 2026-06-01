from __future__ import annotations

import json
import tempfile
from pathlib import Path

from sph.core.simulation import SimulationRunner
from sph.scene.outlets import load_box_outlets_from_scene
from sph.simulator_new import Simulator


SCENE_OUTFLOW = Path("scenes/examples/box_outflow_3d.json").resolve()


def test_box_outlet_scene_parses():
    scene = json.loads(SCENE_OUTFLOW.read_text())
    outlets = load_box_outlets_from_scene(scene, SCENE_OUTFLOW, dim=3)

    assert len(outlets) == 1
    outlet = outlets[0]
    assert outlet.name == "right_outflow"
    assert outlet.direction is not None
    assert outlet.pmin.tolist() == [0.85, 0.15, 0.15]
    assert outlet.pmax.tolist() == [0.98, 0.4, 0.4]


def test_simulator_removes_particles_in_outlet_region():
    sim = Simulator(SCENE_OUTFLOW)
    initial_count = sim.fluid.n

    sim.step()

    assert sim.fluid.n < initial_count
    assert sim.last_solver_stats["removed_particles"] > 0.0
    assert sim.last_solver_stats["outlet_active_count"] == 1.0


def test_simulation_runner_reports_outlet_metrics():
    runner = SimulationRunner(SCENE_OUTFLOW, backend_name="numba_cpu")
    result = runner.step()

    assert result.runtime.fluid_count > 0
    assert result.runtime.solver_metrics["removed_particles"] > 0.0
    assert result.runtime.solver_metrics["outlet_active_count"] == 1.0


def test_outlet_direction_gate_prevents_reverse_flow_removal():
    scene = json.loads(SCENE_OUTFLOW.read_text())
    scene["fluid"]["initial_velocity"] = [-2.0, 0.0, 0.0]
    scene["boundaries"][0]["file"] = str(Path("assets/box.stl").resolve())

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump(scene, tmp)
        tmp_path = Path(tmp.name)

    sim = Simulator(tmp_path)
    initial_count = sim.fluid.n
    sim.step()

    assert sim.fluid.n == initial_count
    assert sim.last_solver_stats["removed_particles"] == 0.0
