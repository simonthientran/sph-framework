from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sph.boundaries import load_boundary_representations_from_scene


def _box_scene_with_sdf() -> tuple[dict, Path]:
    scene_path = Path("scenes/examples/box_fill_3d.json").resolve()
    scene = json.loads(scene_path.read_text())
    scene["boundaries"][0]["representations"] = ["particles", "sdf"]
    scene["boundaries"][0]["sdf"] = {
        "enable": True,
        "resolution": 16,
        "padding_ratio": 0.05,
    }
    return scene, scene_path


def test_mesh_sdf_representation_builds_with_quality_metadata():
    scene, scene_path = _box_scene_with_sdf()

    reps = load_boundary_representations_from_scene(
        scene,
        scene_path,
        spacing=scene["fluid"]["spacing"],
    )
    by_kind = {rep.kind: rep for rep in reps}

    assert "mesh_particles" in by_kind
    assert "mesh_sdf" in by_kind

    sdf_rep = by_kind["mesh_sdf"]
    assert sdf_rep.all_grids_usable
    assert len(sdf_rep.grids) == 1

    grid = sdf_rep.grids[0]
    assert grid.sdf_values.shape == (16, 16, 16)
    assert grid.source.policy.resolution == 16
    assert grid.source.policy.resolution_source == "explicit"
    assert grid.source.policy.padding_ratio == 0.05
    assert grid.cell_size > 0.0
    assert grid.quality.sdf_min < 0.0
    assert grid.quality.sdf_max > 0.0
    assert grid.quality.is_usable


def test_mesh_sdf_default_policy_uses_particle_spacing():
    scene, scene_path = _box_scene_with_sdf()
    scene["boundaries"][0]["sdf"] = {"enable": True}

    reps = load_boundary_representations_from_scene(
        scene,
        scene_path,
        spacing=scene["fluid"]["spacing"],
    )
    by_kind = {rep.kind: rep for rep in reps}
    grid = by_kind["mesh_sdf"].grids[0]

    assert grid.source.policy.resolution_source == "spacing_default"
    assert grid.source.policy.resolution >= 16
    assert grid.source.policy.cells_per_particle_spacing > 1.0


def test_mesh_sdf_can_query_signed_distance_for_interior_points():
    scene, scene_path = _box_scene_with_sdf()

    reps = load_boundary_representations_from_scene(
        scene,
        scene_path,
        spacing=scene["fluid"]["spacing"],
    )
    by_kind = {rep.kind: rep for rep in reps}
    sdf_rep = by_kind["mesh_sdf"]

    points = np.array(
        [
            [0.5, 0.5, 0.5],
            [0.1, 0.1, 0.1],
            [0.9, 0.6, 0.9],
        ],
        dtype=float,
    )
    signed = sdf_rep.sample_signed_distance(points)
    wall_distance = sdf_rep.sample_wall_distance(points)

    assert signed.shape == (3,)
    assert wall_distance.shape == (3,)
    assert np.all(np.isfinite(signed))
    assert np.all(np.isfinite(wall_distance))
    assert np.all(signed < 0.0)
    assert np.all(wall_distance > 0.0)


def test_mesh_sdf_can_query_wall_normals():
    scene, scene_path = _box_scene_with_sdf()

    reps = load_boundary_representations_from_scene(
        scene,
        scene_path,
        spacing=scene["fluid"]["spacing"],
    )
    by_kind = {rep.kind: rep for rep in reps}
    sdf_rep = by_kind["mesh_sdf"]

    points = np.array(
        [
            [0.10, 0.50, 0.50],
            [0.90, 0.50, 0.50],
            [0.50, 0.10, 0.50],
            [0.50, 0.50, 0.10],
        ],
        dtype=float,
    )
    normals = sdf_rep.sample_wall_normal(points)
    norm = np.linalg.norm(normals, axis=1)

    assert normals.shape == (4, 3)
    assert np.all(np.isfinite(normals))
    assert np.allclose(norm, 1.0, atol=0.15)
    assert normals[0, 0] < -0.5
    assert normals[1, 0] > 0.5
    assert normals[2, 1] < -0.5
    assert normals[3, 2] < -0.5


def test_mesh_sdf_startup_policy_can_enable_correction():
    scene, scene_path = _box_scene_with_sdf()
    scene["boundaries"][0]["sdf"] = {
        "enable": True,
        "startup_warn_gap": 0.04,
        "startup_correction": {
            "enable": True,
            "target_gap": 0.05,
            "max_shift": 0.02,
        },
    }

    reps = load_boundary_representations_from_scene(
        scene,
        scene_path,
        spacing=scene["fluid"]["spacing"],
    )
    by_kind = {rep.kind: rep for rep in reps}
    policy = by_kind["mesh_sdf"].grids[0].source.policy.startup

    assert policy.warn_gap == 0.04
    assert policy.correction_enabled
    assert policy.correction_target_gap == 0.05
    assert policy.correction_max_shift == 0.02


def test_mesh_sdf_startup_policy_can_enable_velocity_projection():
    scene, scene_path = _box_scene_with_sdf()
    scene["boundaries"][0]["sdf"] = {
        "enable": True,
        "startup_warn_gap": 0.04,
        "startup_correction": {
            "enable": False,
            "project_outward_velocity": True,
            "velocity_projection_gap": 0.03,
        },
    }

    reps = load_boundary_representations_from_scene(
        scene,
        scene_path,
        spacing=scene["fluid"]["spacing"],
    )
    by_kind = {rep.kind: rep for rep in reps}
    policy = by_kind["mesh_sdf"].grids[0].source.policy.startup

    assert policy.velocity_projection_enabled
    assert policy.velocity_projection_gap == 0.03


def test_mesh_sdf_runtime_policy_can_enable_wall_guard():
    scene, scene_path = _box_scene_with_sdf()
    scene["boundaries"][0]["sdf"] = {
        "enable": True,
        "runtime_wall_guard": {
            "enable": True,
            "min_gap": 0.03,
            "target_gap": 0.05,
            "max_shift": 0.02,
            "project_outward_velocity": False,
        },
    }

    reps = load_boundary_representations_from_scene(
        scene,
        scene_path,
        spacing=scene["fluid"]["spacing"],
    )
    by_kind = {rep.kind: rep for rep in reps}
    policy = by_kind["mesh_sdf"].grids[0].source.policy.runtime

    assert policy.wall_guard_enabled
    assert policy.wall_guard_min_gap == 0.03
    assert policy.wall_guard_target_gap == 0.05
    assert policy.wall_guard_max_shift == 0.02
    assert not policy.wall_guard_project_outward_velocity
