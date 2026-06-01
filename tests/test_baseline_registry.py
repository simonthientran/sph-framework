from __future__ import annotations

from pathlib import Path

from sph.validation.baseline_registry import (
    BASELINE_RUN_PRESETS,
    FROZEN_BASELINE_OVERALL_BOTTLENECKS,
    FROZEN_BASELINE_SCENES,
    REJECTED_OPTIMIZATION_DIRECTIONS,
    SUCCESSFUL_OPTIMIZATION_DIRECTIONS,
    build_preset_command,
)


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_scene_registry_matches_canonical_benchmark_scenes() -> None:
    base = FROZEN_BASELINE_SCENES["base"]
    dense = FROZEN_BASELINE_SCENES["dense"]

    assert base.scene_path(ROOT).exists()
    assert dense.scene_path(ROOT).exists()

    assert base.fluid_particles == 2250
    assert base.boundary_particles == 756
    assert dense.fluid_particles == 5800
    assert dense.boundary_particles == 1206

    assert base.dt_expected == 1.0e-04
    assert dense.dt_expected == 1.0e-04
    assert base.iter_cd_expected == 1
    assert dense.iter_df_expected == 1


def test_benchmark_presets_use_registered_scenes_and_expected_commands() -> None:
    assert set(BASELINE_RUN_PRESETS) == {
        "cpu-base",
        "cuda-base",
        "cuda-dense",
        "cuda-profile-base",
        "cuda-profile-dense",
    }

    cmd, env, preset = build_preset_command(
        repo_root=ROOT,
        python_bin="python",
        preset_name="cuda-profile-base",
        base_env={},
    )
    assert preset.name == "cuda-profile-base"
    assert "profile_cuda_stages.py" in cmd[1]
    assert str(FROZEN_BASELINE_SCENES["base"].scene_path(ROOT)) in cmd
    assert env["NUMBA_ENABLE_CUDASIM"] == "0"

    cmd, env, preset = build_preset_command(
        repo_root=ROOT,
        python_bin="python",
        preset_name="cpu-base",
        base_env={},
    )
    assert preset.name == "cpu-base"
    assert "bench_density.py" in cmd[1]
    assert cmd[-1] == "numba_cpu"
    assert "NUMBA_ENABLE_CUDASIM" not in env


def test_frozen_baseline_history_and_bottleneck_ranking_are_not_empty() -> None:
    assert FROZEN_BASELINE_OVERALL_BOTTLENECKS[:2] == ("pair_build", "solve")
    assert any("neighbor build replaced" in item.lower() for item in SUCCESSFUL_OPTIMIZATION_DIRECTIONS)
    assert any("mini-buffers" in item.lower() for item in REJECTED_OPTIMIZATION_DIRECTIONS)
    assert any("cell-centric" in item.lower() for item in REJECTED_OPTIMIZATION_DIRECTIONS)
