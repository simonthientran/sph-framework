from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np

from sph.simulator_new import Simulator
from sph.validation.boundary import validate_boundary_representations


SCENE_BOX = Path("scenes/examples/box_fill_3d.json").resolve()


def test_boundary_validation_particle_only_scene_passes():
    result = validate_boundary_representations(SCENE_BOX)

    assert result.passed
    assert result.status == "pass"
    assert result.built_representation_kinds == ("mesh_particles",)
    assert result.particle is not None
    assert result.particle.total_count == 5154
    assert result.sdf is None
    assert len(result.source_checks) == 1
    assert result.source_checks[0].source_name == "box.stl"


def test_boundary_validation_with_sdf_builds_both_representations():
    result = validate_boundary_representations(
        SCENE_BOX,
        requested_representations=("particles", "sdf"),
    )

    assert result.passed
    assert result.status == "pass"
    assert "mesh_particles" in result.built_representation_kinds
    assert "mesh_sdf" in result.built_representation_kinds
    assert result.particle is not None
    assert result.sdf is not None
    assert result.sdf.all_grids_usable
    assert len(result.sdf.grid_summaries) == 1
    assert result.sdf_startup_geometry is not None
    assert result.sdf_startup_geometry.status == "pass"
    grid = result.sdf.grid_summaries[0]
    assert grid.source_name == "box.stl"
    assert grid.is_usable
    assert grid.sdf_min < 0.0
    assert grid.sdf_max > 0.0
    assert result.sdf_startup_geometry.inside_fraction > 0.99
    assert result.sdf_startup_geometry.wall_distance_min > 0.0
    assert result.sdf_startup_geometry.normal_finite_fraction > 0.99
    assert abs(result.sdf_startup_geometry.normal_norm_mean - 1.0) < 0.05


def test_sdf_startup_velocity_projection_removes_outward_wall_motion():
    scene = json.loads(SCENE_BOX.read_text())
    scene["fluid"]["min"] = [0.01, 0.10, 0.10]
    scene["fluid"]["max"] = [0.11, 0.20, 0.20]
    scene["fluid"]["initial_velocity"] = [-1.0, 0.0, 0.0]
    scene["boundaries"][0]["file"] = str((Path("assets/box.stl").resolve()))
    scene["boundaries"][0]["representations"] = ["particles", "sdf"]
    scene["boundaries"][0]["sdf"] = {
        "enable": True,
        "startup_warn_gap": 0.05,
        "startup_correction": {
            "enable": False,
            "project_outward_velocity": True,
            "velocity_projection_gap": 0.05,
        },
    }

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump(scene, tmp)
        tmp_path = Path(tmp.name)

    sim = Simulator(tmp_path)
    leftmost = np.argmin(sim.fluid.positions[:, 0])
    assert abs(sim.fluid.velocities[leftmost, 0]) < 1.0e-8


def test_boundary_validation_fails_for_bad_near_wall_startup_geometry():
    scene = json.loads(SCENE_BOX.read_text())
    scene["fluid"]["min"] = [0.01, 0.10, 0.10]
    scene["fluid"]["max"] = [0.11, 0.20, 0.20]
    scene["boundaries"][0]["file"] = str((Path("assets/box.stl").resolve()))
    scene["boundaries"][0]["representations"] = ["particles", "sdf"]
    scene["boundaries"][0]["sdf"] = {
        "enable": True,
        "startup_warn_gap": 0.05,
        "startup_validity": {
            "fail_too_close_fraction": 0.10,
        },
    }

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump(scene, tmp)
        tmp_path = Path(tmp.name)

    result = validate_boundary_representations(
        tmp_path,
        requested_representations=("particles", "sdf"),
    )

    assert not result.passed
    assert result.status == "fail"
    assert result.sdf_startup_geometry is not None
    assert result.sdf_startup_geometry.status == "fail"
    assert result.sdf_startup_geometry.too_close_count > 0


def test_sdf_runtime_wall_guard_corrects_near_wall_particles_after_step():
    scene = json.loads(SCENE_BOX.read_text())
    scene["fluid"]["min"] = [0.01, 0.10, 0.10]
    scene["fluid"]["max"] = [0.11, 0.20, 0.20]
    scene["fluid"]["initial_velocity"] = [-1.0, 0.0, 0.0]
    scene["forces"]["gravity"] = [0.0, 0.0, 0.0]
    scene["time"] = {
        "mode": "fixed",
        "dt_fixed": 1.0e-4,
        "dt_min": 1.0e-4,
        "dt_max": 1.0e-4,
        "steps": 1,
    }
    scene["boundaries"][0]["file"] = str((Path("assets/box.stl").resolve()))
    scene["boundaries"][0]["representations"] = ["particles", "sdf"]
    scene["boundaries"][0]["sdf"] = {
        "enable": True,
        "resolution": 16,
        "startup_warn_gap": 0.05,
        "startup_correction": {
            "enable": False,
            "project_outward_velocity": False,
        },
        "runtime_wall_guard": {
            "enable": True,
            "min_gap": 0.05,
            "target_gap": 0.05,
            "max_shift": 0.05,
            "project_outward_velocity": True,
        },
    }

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump(scene, tmp)
        tmp_path = Path(tmp.name)

    sim = Simulator(tmp_path)
    leftmost = np.argmin(sim.fluid.positions[:, 0])
    assert sim.fluid.positions[leftmost, 0] < 0.02
    sim.step()

    sdf_rep = sim._mesh_sdf_representation
    assert sdf_rep is not None
    wall_distance = sdf_rep.sample_wall_distance(sim.fluid.positions)
    assert float(np.nanmin(wall_distance)) >= 0.049
    assert abs(sim.fluid.velocities[leftmost, 0]) < 1.0e-6
    assert sim.last_solver_stats["sdf_wall_guard_corrected"] > 0.0
