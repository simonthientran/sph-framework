"""Unit tests for vx_profile: y_extent_mode walls and walls_inner."""

import json
from pathlib import Path

import numpy as np
import pytest

from sph.core.state import ParticleState
from sph.core.state_builder import build_scene_state
from sph.core.vx_profile import get_y_extent, compute_vx_profile


def test_get_y_extent_walls():
    """walls mode uses full domain height."""
    scene = {
        "domain": {"min": [0.0, 0.0], "max": [1.0, 0.2]},
        "fluid": {"spacing": 0.02},
    }
    y0, H = get_y_extent(scene, "walls")
    assert y0 == 0.0
    assert H == 0.2


def test_get_y_extent_walls_inner():
    """walls_inner: y0_eff = y0_wall + t, H_eff = H_wall - 2*t, t = boundary_layers * spacing."""
    scene = {
        "domain": {"min": [0.0, 0.0], "max": [1.0, 0.2], "boundary_layers": 3},
        "fluid": {"spacing": 0.02},
    }
    y0, H = get_y_extent(scene, "walls_inner")
    t = 3 * 0.02  # 0.06
    assert y0 == pytest.approx(0.0 + t)
    assert y0 == pytest.approx(0.06)
    assert H == pytest.approx(0.2 - 2 * t)
    assert H == pytest.approx(0.08)


def test_get_y_extent_walls_inner_different_spacing():
    """walls_inner with boundary_layers=4, spacing=0.02 -> t=0.08, H_eff=0.2-0.16=0.04."""
    scene = {
        "domain": {"min": [0.0, 0.0], "max": [0.5, 0.2], "boundary_layers": 4},
        "fluid": {"spacing": 0.02},
    }
    y0, H = get_y_extent(scene, "walls_inner")
    assert y0 == pytest.approx(0.08)
    assert H == pytest.approx(0.04)


def test_get_y_extent_walls_inner_H_eff_negative_raises():
    """walls_inner with 2*t >= H_wall raises ValueError."""
    scene = {
        "domain": {"min": [0.0, 0.0], "max": [1.0, 0.1], "boundary_layers": 3},
        "fluid": {"spacing": 0.02},
    }
    with pytest.raises(ValueError, match="H_eff must be > 0"):
        get_y_extent(scene, "walls_inner")


def test_get_y_extent_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown y_extent_mode"):
        get_y_extent({"domain": {"min": [0, 0], "max": [1, 1]}}, "invalid")


def test_compute_vx_profile_walls_inner_used_bins():
    """With fluid only in y in [0.06, 0.14], walls_inner (y0_eff=0.06, H_eff=0.08) gives 8/8 used bins."""
    scene = {
        "domain": {"min": [0.0, 0.0], "max": [1.0, 0.2], "boundary_layers": 3},
        "fluid": {"spacing": 0.02},
    }
    # Build minimal state: a few fluid particles in y in [0.06, 0.14]
    n = 20
    np.random.seed(42)
    y = np.linspace(0.061, 0.139, n)
    x = np.linspace(0.1, 0.9, n)
    pos = np.column_stack([x, y])
    vel = np.zeros((n, 2))
    vel[:, 0] = 1.0  # vx
    state = ParticleState(
        dim=2,
        pos=pos,
        vel=vel,
        acc=np.zeros_like(pos),
        mass=np.ones(n) * 1000.0 * 0.02**2,
        rho=np.ones(n) * 1000.0,
        p=np.zeros(n),
        is_boundary=np.zeros(n, dtype=bool),
    )
    result = compute_vx_profile(
        step=1,
        state=state,
        scene=scene,
        y_extent_mode="walls_inner",
        n_bins=8,
    )
    assert result.mode == "walls_inner"
    assert result.y0_eff == pytest.approx(0.06)
    assert result.H_eff == pytest.approx(0.08)
    assert result.used_bins == 8
    assert result.empty_bins == 0
    assert result.n_bins == 8


def test_pipe_flow_1phase_2d_walls_inner_acceptance():
    """For pipe_flow_1phase_2d (domain y [0,0.2], boundary_layers=3), walls_inner gives
    y0_eff = y0_wall + t, H_eff = H_wall - 2*t (t = boundary_layers * spacing).
    With fluid filling the interior, used_bins=8/8 and empty_bins=0."""
    scene_path = Path(__file__).resolve().parent.parent / "scenes" / "examples" / "pipe_flow_1phase_2d.json"
    if not scene_path.exists():
        pytest.skip("pipe_flow_1phase_2d.json not found")
    with scene_path.open("r", encoding="utf-8") as f:
        scene = json.load(f)
    state = build_scene_state(scene)
    result = compute_vx_profile(
        step=1,
        state=state,
        scene=scene,
        y_extent_mode="walls_inner",
        n_bins=8,
        gx=0.1,
        nu=0.001,
    )
    assert result.mode == "walls_inner"
    # pipe_flow_1phase_2d: spacing 0.01, boundary_layers=3 -> t=0.03, y0_eff=0.03, H_eff=0.14
    assert result.y0_eff == pytest.approx(0.03)
    assert result.H_eff == pytest.approx(0.14)
    assert result.used_bins == 8
    assert result.empty_bins == 0
    assert result.n_bins == 8
    # Analytic vmax = gx * H_eff^2 / (8*nu)
    assert result.vmax_analytic is not None
    assert result.vmax_analytic == pytest.approx(0.1 * (0.14**2) / (8.0 * 0.001))
