from __future__ import annotations

from pathlib import Path

from sph.scene.schema import (
    COLORMAP_TYPE_MAP,
    _extract_splishsplash_viscosity,
    load_scene_dict,
    normalize_own_scene_dict,
    normalize_scene_dict,
)


def test_native_scene_normalizes_correctly():
    normalized = load_scene_dict(Path("scenes/dam_break_splishsplash.json"))
    assert normalized["meta"]["dimensions"] == 3
    assert normalized["solver"]["type"] == "dfsph"
    assert isinstance(normalized["fluids"], list)
    assert len(normalized["fluids"]) >= 1
    assert "prefer_iterative" not in normalized["solver"]


def test_native_scene_fluid_singular_promoted_to_fluids():
    raw = {
        "meta": {"dimensions": 2},
        "fluid": {"type": "block", "min": [0, 0], "max": [1, 1], "spacing": 0.05},
        "material": {"rho0": 1000.0},
    }
    out = normalize_own_scene_dict(raw)
    assert "fluid" not in out
    assert isinstance(out["fluids"], list)
    assert len(out["fluids"]) == 1
    assert out["fluids"][0]["type"] == "block"


def test_normalize_solver_defaults_by_scene_type():
    free_surface = normalize_own_scene_dict({
        "meta": {"dimensions": 2, "scene_type": "free_surface"},
        "material": {"rho0": 1000.0},
    })
    assert free_surface["solver"]["type"] == "wcsph"

    internal = normalize_own_scene_dict({
        "meta": {"dimensions": 2, "scene_type": "internal_flow"},
        "material": {"rho0": 1000.0},
    })
    assert internal["solver"]["type"] == "dfsph"


def test_normalize_eos_k_promotion():
    out = normalize_own_scene_dict({
        "meta": {"dimensions": 2},
        "material": {"rho0": 1000.0, "eos": {"type": "linear", "k": 5000.0}},
    })
    assert out["solver"]["eos_k"] == 5000.0


def test_normalize_prefer_iterative_stripped():
    out = normalize_own_scene_dict({
        "meta": {"dimensions": 2},
        "solver": {"type": "wcsph", "prefer_iterative": True},
        "material": {"rho0": 1000.0},
    })
    assert "prefer_iterative" not in out["solver"]
    assert out["solver"]["type"] == "wcsph"


def test_normalize_gravity_defaults():
    out_2d = normalize_own_scene_dict({
        "meta": {"dimensions": 2},
        "material": {"rho0": 1000.0},
    })
    assert out_2d["forces"]["gravity"] == [0.0, -9.81]

    out_3d = normalize_own_scene_dict({
        "meta": {"dimensions": 3},
        "material": {"rho0": 1000.0},
    })
    assert out_3d["forces"]["gravity"] == [0.0, -9.81, 0.0]


def test_normalize_viscosity_consolidation():
    out = normalize_own_scene_dict({
        "meta": {"dimensions": 2},
        "material": {"rho0": 1000.0, "viscosity": {"enable": True, "nu": 0.1}},
    })
    assert out["forces"]["viscosity"]["nu"] == 0.1


# --- New tests for SPlisHSPlasH-style solver sub-block support --------


def test_solver_subblock_dfsph_flattened():
    """DFSPH sub-block params should be flattened into top-level solver dict."""
    out = normalize_own_scene_dict({
        "meta": {"dimensions": 2},
        "solver": {
            "type": "dfsph",
            "eos_k": 8000.0,
            "dfsph": {
                "eta_cd": 0.05,
                "max_iter_cd": 50,
                "density_diffusion": {"enable": False},
            },
        },
        "material": {"rho0": 1000.0},
    })
    assert out["solver"]["type"] == "dfsph"
    assert out["solver"]["eta_cd"] == 0.05
    assert out["solver"]["max_iter_cd"] == 50
    assert out["solver"]["density_diffusion"]["enable"] is False
    assert out["solver"]["eos_k"] == 8000.0
    assert "dfsph" not in out["solver"]


def test_solver_subblock_wcsph_flattened():
    """WCSPH sub-block params should be flattened and stiffness mapped to eos_k."""
    out = normalize_own_scene_dict({
        "meta": {"dimensions": 2, "scene_type": "free_surface"},
        "solver": {
            "type": "wcsph",
            "wcsph": {
                "stiffness": 25000,
                "exponent": 1,
            },
        },
        "material": {"rho0": 1000.0},
    })
    assert out["solver"]["type"] == "wcsph"
    assert out["solver"]["eos_k"] == 25000
    assert out["solver"]["exponent"] == 1
    assert "wcsph" not in out["solver"]
    assert "stiffness" not in out["solver"]


def test_solver_subblock_does_not_override_explicit_top_level():
    """Explicit top-level keys should not be overridden by sub-block values."""
    out = normalize_own_scene_dict({
        "meta": {"dimensions": 2},
        "solver": {
            "type": "dfsph",
            "eta_cd": 0.01,
            "dfsph": {"eta_cd": 0.99},
        },
        "material": {"rho0": 1000.0},
    })
    assert out["solver"]["eta_cd"] == 0.01


def test_non_active_solver_subblocks_removed():
    """Sub-blocks for non-active solver types should be removed."""
    out = normalize_own_scene_dict({
        "meta": {"dimensions": 2},
        "solver": {
            "type": "wcsph",
            "eos_k": 10000.0,
            "dfsph": {"eta_cd": 0.05},
        },
        "material": {"rho0": 1000.0},
    })
    assert "dfsph" not in out["solver"]
    assert out["solver"]["type"] == "wcsph"


def test_boundary_handling_method_defaults():
    """Domain and boundary entries should get boundary_handling_method defaults."""
    out = normalize_own_scene_dict({
        "meta": {"dimensions": 2},
        "domain": {"type": "box", "boundary_layers": 3},
        "boundaries": [{"type": "stl", "file": "box.stl"}],
        "material": {"rho0": 1000.0},
    })
    assert out["domain"]["boundary_handling_method"] == "particle"
    assert out["boundaries"][0]["boundary_handling_method"] == "particle"
    assert out["boundaries"][0]["is_wall"] is True


def test_wcsph_stiffness_to_eos_k_mapping():
    """WCSPH 'stiffness' (SPlisHSPlasH naming) should map to 'eos_k'."""
    out = normalize_own_scene_dict({
        "meta": {"dimensions": 3},
        "solver": {"type": "wcsph", "stiffness": 50000},
        "material": {"rho0": 1000.0},
    })
    assert out["solver"]["eos_k"] == 50000
    assert "stiffness" not in out["solver"]


def test_curated_scenes_all_load():
    curated = [
        "scenes/examples/dam_break_2d.json",
        "scenes/examples/dam_break_3d.json",
        "scenes/examples/00_showcase_basin.json",
        "scenes/examples/basin_3d.json",
        "scenes/examples/pipe_flow_2d.json",
        "scenes/examples/pipe_flow_3d.json",
    ]
    for scene_path in curated:
        d = load_scene_dict(scene_path)
        assert isinstance(d["fluids"], list), f"{scene_path}: fluids not a list"
        assert "type" in d["solver"], f"{scene_path}: no solver.type"
        assert "prefer_iterative" not in d["solver"], f"{scene_path}: prefer_iterative not stripped"
        # Solver sub-blocks should be flattened after normalization
        solver_type = d["solver"]["type"]
        assert solver_type not in d["solver"], (
            f"{scene_path}: solver sub-block '{solver_type}' not flattened"
        )


def test_splishsplash_dfsph_import_maps_solver_config():
    """SPlisHSPlasH DFSPH scene should produce correct solver config."""
    from sph.scene.schema import normalize_scene_dict
    raw = {
        "Configuration": {
            "particleRadius": 0.025,
            "simulationMethod": 4,
            "gravitation": [0, -9.81, 0],
            "boundaryHandlingMethod": 2,
            "DFSPH": {
                "maxIterations": 100,
                "maxError": 0.05,
                "maxIterationsV": 100,
                "maxErrorV": 0.1,
                "enableDivergenceSolver": True,
            },
        },
        "Materials": [{"id": "Fluid", "density0": 1000, "viscosity": 0.01}],
        "RigidBodies": [{"geometryFile": "box.obj", "isWall": True}],
        "FluidBlocks": [{"start": [-0.5, 0, -0.5], "end": [0.5, 1, 0.5]}],
    }
    out = normalize_scene_dict(raw)
    assert out["solver"]["type"] == "dfsph"
    assert out["solver"]["eta_cd"] == 0.05
    assert out["solver"]["max_iter_cd"] == 100
    assert "prefer_iterative" not in out["solver"]
    assert "dfsph" not in out["solver"]
    assert out["boundaries"][0]["is_wall"] is True


def test_splishsplash_wcsph_import_maps_stiffness_to_eos_k():
    """SPlisHSPlasH WCSPH scene should map stiffness/exponent correctly."""
    from sph.scene.schema import normalize_scene_dict
    raw = {
        "Configuration": {
            "particleRadius": 0.025,
            "simulationMethod": 0,
            "gravitation": [0, -9.81, 0],
            "WCSPH": {"stiffness": 25000, "exponent": 1},
        },
        "Materials": [{"id": "Fluid", "density0": 1000}],
        "RigidBodies": [{"geometryFile": "box.obj"}],
        "FluidBlocks": [{"start": [-0.4, -0.4, -0.4], "end": [0.4, 0.4, 0.4]}],
    }
    out = normalize_scene_dict(raw)
    assert out["solver"]["type"] == "wcsph"
    assert out["solver"]["eos_k"] == 25000.0
    assert out["solver"]["exponent"] == 1.0
    assert "stiffness" not in out["solver"]
    assert "wcsph" not in out["solver"]


def test_curated_scenes_solver_params_present():
    """Verify solver-specific params are present after normalization for DFSPH scenes."""
    for scene_path in [
        "scenes/examples/pipe_flow_2d.json",
        "scenes/examples/pipe_flow_3d.json",
    ]:
        d = load_scene_dict(scene_path)
        assert d["solver"]["type"] == "dfsph", f"{scene_path}: expected dfsph"
        assert "density_diffusion" in d["solver"], (
            f"{scene_path}: DFSPH density_diffusion missing after normalization"
        )


# --- SPlisHSPlasH Materials system tests (§4.6) ---------------------------


def test_materials_synthesized_from_legacy_config():
    """When no materials block is present, one should be built from material/forces."""
    out = normalize_own_scene_dict({
        "meta": {"dimensions": 2},
        "material": {"rho0": 800.0},
        "forces": {
            "viscosity": {"enable": True, "nu": 0.05},
            "xsph": {"enable": True, "eps": 0.02},
        },
    })
    assert "materials" in out
    assert isinstance(out["materials"], list)
    assert len(out["materials"]) >= 1
    m = out["materials"][0]
    assert m["id"] == "Fluid"
    assert m["density0"] == 800.0
    assert m["viscosity"] == 0.05
    assert m["xsph"] == 0.02
    assert "colorField" in m
    assert "colorMapType" in m


def test_materials_explicit_block_preserved():
    """Explicit materials blocks should be preserved and populated with defaults."""
    out = normalize_own_scene_dict({
        "meta": {"dimensions": 2},
        "materials": [
            {"id": "Water", "density0": 998.0, "viscosity": 0.001}
        ],
        "material": {"rho0": 998.0},
    })
    assert len(out["materials"]) == 1
    m = out["materials"][0]
    assert m["id"] == "Water"
    assert m["density0"] == 998.0
    assert m["viscosity"] == 0.001
    assert m["colorField"] == "velocity"
    assert m["colorMapType"] == "jet"


def test_materials_sync_backward_compat():
    """materials[0] should sync back to material.rho0 and forces for compat."""
    out = normalize_own_scene_dict({
        "meta": {"dimensions": 2},
        "materials": [
            {"id": "Fluid", "density0": 1200.0, "viscosity": 0.1, "xsph": 0.03}
        ],
    })
    assert out["material"]["rho0"] == 1200.0
    assert out["forces"]["viscosity"]["enable"] is True
    assert out["forces"]["viscosity"]["nu"] == 0.1
    assert out["forces"]["xsph"]["enable"] is True
    assert out["forces"]["xsph"]["eps"] == 0.03


def test_materials_colorfield_from_preferred_overlay():
    """Legacy preferred_overlay should map to colorField in synthesized materials."""
    out = normalize_own_scene_dict({
        "meta": {"dimensions": 2, "preferred_overlay": "density"},
        "material": {"rho0": 1000.0},
    })
    assert out["materials"][0]["colorField"] == "density"


def test_splishsplash_materials_import_full():
    """SPlisHSPlasH Materials with nested viscosity + colorMapType should import."""
    raw = {
        "Configuration": {
            "particleRadius": 0.025,
            "simulationMethod": 4,
            "gravitation": [0, -9.81, 0],
        },
        "Materials": [{
            "id": "Fluid",
            "density0": 1000,
            "viscosityMethod": 1,
            "colorMapType": 1,
            "colorField": "velocity",
            "renderMinValue": 0.0,
            "renderMaxValue": 5.0,
            "Standard viscosity": {"viscosity": 0.01},
            "xsph": 0.05,
        }],
        "FluidBlocks": [{"start": [0, 0, 0], "end": [1, 1, 1]}],
    }
    out = normalize_scene_dict(raw)
    assert "materials" in out
    m = out["materials"][0]
    assert m["id"] == "Fluid"
    assert m["density0"] == 1000.0
    assert m["viscosity"] == 0.01
    assert m["xsph"] == 0.05
    assert m["colorField"] == "velocity"
    assert m["colorMapType"] == "jet"
    assert m["renderMaxValue"] == 5.0


def test_extract_splishsplash_viscosity_nested():
    """Should find viscosity inside method-specific sub-blocks."""
    mat = {
        "id": "Fluid",
        "density0": 1000,
        "viscosityMethod": 1,
        "Standard viscosity": {"viscosity": 0.01},
    }
    assert _extract_splishsplash_viscosity(mat) == 0.01


def test_extract_splishsplash_viscosity_top_level():
    """Should find top-level viscosity if present."""
    mat = {"id": "Fluid", "viscosity": 0.05}
    assert _extract_splishsplash_viscosity(mat) == 0.05


def test_extract_splishsplash_viscosity_missing():
    """Should return 0.0 when no viscosity is specified."""
    mat = {"id": "Fluid", "density0": 1000}
    assert _extract_splishsplash_viscosity(mat) == 0.0


def test_colormap_type_map():
    """COLORMAP_TYPE_MAP should contain expected SPlisHSPlasH entries."""
    assert COLORMAP_TYPE_MAP[0] == "none"
    assert COLORMAP_TYPE_MAP[1] == "jet"
    assert COLORMAP_TYPE_MAP[2] == "plasma"
    assert COLORMAP_TYPE_MAP[4] == "blue_white_red"


def test_curated_scenes_have_materials():
    """All curated scenes should have a materials array after normalization."""
    curated = [
        "scenes/examples/dam_break_2d.json",
        "scenes/examples/dam_break_3d.json",
        "scenes/examples/00_showcase_basin.json",
        "scenes/examples/basin_3d.json",
        "scenes/examples/pipe_flow_2d.json",
        "scenes/examples/pipe_flow_3d.json",
    ]
    for scene_path in curated:
        d = load_scene_dict(scene_path)
        assert "materials" in d, f"{scene_path}: no materials"
        assert isinstance(d["materials"], list), f"{scene_path}: materials not list"
        assert len(d["materials"]) >= 1, f"{scene_path}: empty materials"
        m = d["materials"][0]
        assert "id" in m, f"{scene_path}: no material id"
        assert "density0" in m, f"{scene_path}: no density0"
        assert "colorField" in m, f"{scene_path}: no colorField"
        assert "colorMapType" in m, f"{scene_path}: no colorMapType"


def test_curated_scenes_materials_match_material():
    """materials[0].density0 should match material.rho0 after normalization."""
    curated = [
        "scenes/examples/dam_break_2d.json",
        "scenes/examples/pipe_flow_2d.json",
        "scenes/examples/dam_break_3d.json",
    ]
    for scene_path in curated:
        d = load_scene_dict(scene_path)
        mat_rho0 = d["material"]["rho0"]
        mat0_rho0 = d["materials"][0]["density0"]
        assert mat_rho0 == mat0_rho0, (
            f"{scene_path}: material.rho0={mat_rho0} != materials[0].density0={mat0_rho0}"
        )
