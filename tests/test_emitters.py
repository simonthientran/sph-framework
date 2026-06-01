from __future__ import annotations

import json
import tempfile
from pathlib import Path

from sph.core.simulation import SimulationRunner
from sph.scene.emitters import build_emitter_slab_positions, load_box_emitters_from_scene
from sph.simulator_new import Simulator


SCENE_INFLOW = Path("scenes/examples/box_inflow_3d.json").resolve()


def test_box_emitter_scene_parses():
    scene = json.loads(SCENE_INFLOW.read_text())
    emitters = load_box_emitters_from_scene(scene, SCENE_INFLOW, dim=3)

    assert len(emitters) == 1
    spec = emitters[0].spec
    slab = build_emitter_slab_positions(spec, spacing=0.05)
    assert spec.name == "left_inlet"
    assert slab.shape == (9, 3)
    assert float(slab[:, 0].min()) == 0.10
    assert float(slab[:, 0].max()) == 0.10
    assert spec.max_active_fluid_particles is None


def test_simulator_emits_particles_during_step():
    sim = Simulator(SCENE_INFLOW)
    initial_count = sim.fluid.n

    sim.step()

    assert sim.fluid.n > initial_count
    assert sim.last_solver_stats["emitted_particles"] > 0.0
    assert sim.last_solver_stats["emitter_active_count"] == 1.0


def test_simulation_runner_reports_emitter_metrics():
    runner = SimulationRunner(SCENE_INFLOW, backend_name="numba_cpu")
    result = runner.step()

    assert result.runtime.fluid_count > 0
    assert result.runtime.solver_metrics["emitted_particles"] > 0.0
    assert result.runtime.solver_metrics["emitter_active_count"] == 1.0


def test_emitter_respects_end_step_gate():
    scene = json.loads(SCENE_INFLOW.read_text())
    scene["emitters"][0]["start_step"] = 1
    scene["emitters"][0]["end_step"] = 1
    scene["boundaries"][0]["file"] = str(Path("assets/box.stl").resolve())

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump(scene, tmp)
        tmp_path = Path(tmp.name)

    sim = Simulator(tmp_path)
    count0 = sim.fluid.n
    sim.step()
    count1 = sim.fluid.n
    sim.step()
    count2 = sim.fluid.n
    sim.step()
    count3 = sim.fluid.n

    assert count1 == count0
    assert count2 > count1
    assert count3 == count2


def test_emitter_respects_max_active_fluid_budget():
    scene = json.loads(SCENE_INFLOW.read_text())
    scene["emitters"][0]["max_active_fluid_particles"] = 18
    scene["emitters"][0]["start_step"] = 0
    scene["emitters"][0]["end_step"] = 2
    scene["boundaries"][0]["file"] = str(Path("assets/box.stl").resolve())

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump(scene, tmp)
        tmp_path = Path(tmp.name)

    sim = Simulator(tmp_path)
    initial_count = sim.fluid.n
    sim.step()

    assert sim.fluid.n == initial_count
    assert sim.last_solver_stats["emitted_particles"] == 0.0
    assert sim.last_solver_stats["emitter_budget_blocked_count"] == 1.0
