import numpy as np

from sph.core.startup_sanity import evaluate_startup_sanity
from sph.core.state_builder import build_scene_state


def test_startup_sanity_flags_bad_h_over_dx_and_suggests_autotune():
    scene = {
        "meta": {"name": "startup_bad_h", "version": 1, "seed": 0, "dimensions": 2},
        "domain": {"type": "box", "min": [0.0, 0.0], "max": [0.2, 0.2], "boundary_layers": 2},
        "fluid": {
            "type": "block",
            "min": [0.05, 0.05],
            "max": [0.10, 0.10],
            "spacing": 0.02,
            "initial_velocity": [0.0, 0.0],
        },
        "neighbors": {"type": "spatial_hash", "support_radius": 0.005},
        "forces": {"gravity": [0.0, -9.81]},
        "material": {"rho0": 1000.0, "eos": {"k": 2000.0}},
        "time": {"mode": "fixed", "dt_fixed": 0.001, "steps": 1},
    }
    state = build_scene_state(scene)

    report = evaluate_startup_sanity(
        scene=scene,
        state=state,
        h=float(scene["neighbors"]["support_radius"]),
        spacing=float(scene["fluid"]["spacing"]),
        rho0=float(scene["material"]["rho0"]),
        startup_cfg={
            "auto_tune_support_radius": True,
            "support_radius_target_h_over_dx_2d": 1.3,
            "density_error_warn_rel": 0.05,
            "density_error_abort_rel": 0.05,
        },
    )

    assert np.isfinite(report.h_over_dx)
    assert report.h_over_dx < 1.0
    assert report.auto_tuned_support_radius is not None
    assert report.auto_tuned_support_radius > float(scene["neighbors"]["support_radius"])
    assert any("h/dx" in r for r in report.recommendations)


def test_startup_sanity_detects_support_mismatch_and_boundary_layer_overlap():
    scene = {
        "meta": {"name": "startup_overlap", "version": 1, "seed": 0, "dimensions": 2},
        "domain": {"type": "box", "min": [0.0, 0.0], "max": [1.0, 0.6], "boundary_layers": 4},
        "fluid": {
            "type": "block",
            "min": [0.05, 0.05],  # inside boundary layer thickness (=4*0.02=0.08)
            "max": [0.35, 0.55],
            "spacing": 0.02,
            "initial_velocity": [0.0, 0.0],
        },
        "neighbors": {
            "type": "spatial_hash",
            "support_radius": 0.045,
            "smoothing_length": 0.045,
        },
        "forces": {"gravity": [0.0, -9.81]},
        "material": {"rho0": 1000.0, "eos": {"k": 2000.0}},
        "time": {"mode": "fixed", "dt_fixed": 0.001, "steps": 1},
    }
    state = build_scene_state(scene)
    report = evaluate_startup_sanity(
        scene=scene,
        state=state,
        h=float(scene["neighbors"]["support_radius"]),
        spacing=float(scene["fluid"]["spacing"]),
        rho0=float(scene["material"]["rho0"]),
        startup_cfg={"auto_tune_support_radius": False},
    )
    assert report.support_radius_mismatch is True
    assert np.isclose(report.required_support_radius, 0.09)
    assert report.boundary_layer_overlap is True
