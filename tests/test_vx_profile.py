from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sph.diagnostics.vx_profile import VxProfileConfig, VxProfileDiagnostics


@dataclass
class _MockState:
    pos: np.ndarray
    vel: np.ndarray
    fluid_indices: np.ndarray


def test_vx_profile_slice_auto_extent_and_non_empty_bins(tmp_path: Path) -> None:
    # Synthetic channel samples spanning y in [2, 4].
    y = np.linspace(2.0, 4.0, 17, dtype=np.float64)
    x = np.full_like(y, 0.25)
    pos = np.stack([x, y], axis=1)
    vel = np.zeros_like(pos)
    vel[:, 0] = np.linspace(0.1, 0.9, y.size, dtype=np.float64)
    state = _MockState(pos=pos, vel=vel, fluid_indices=np.arange(y.size, dtype=np.int64))

    cfg = VxProfileConfig(
        enable=True,
        every=1,
        bins=8,
        axis=1,
        component=0,
        use_x_slice=False,
        x_mid=None,
        x_slice_width=0.0,
        y_min=None,
        y_max=None,
        y0=None,
        y_extent_mode="slice_auto",
        wall_y0=None,
        wall_h=None,
        wall_padding=0.0,
        avg_window_steps=1,
        y_margin=0.0,
        channel_height=None,
        gx=1.0,
        nu=0.01,
        out_file=tmp_path / "vx_profile_bins.csv",
    )
    diag = VxProfileDiagnostics(cfg=cfg)
    sample = diag.sample(step=1, state=state)

    assert sample is not None
    assert sample.y0_eff == 2.0
    assert sample.h_eff == 2.0
    assert sample.empty_bins == 0
    assert sum(1 for n in sample.counts if n > 0) == cfg.bins


def test_vx_profile_walls_mode_uses_wall_extent(tmp_path: Path) -> None:
    y = np.linspace(2.0, 4.0, 17, dtype=np.float64)
    x = np.full_like(y, 0.25)
    pos = np.stack([x, y], axis=1)
    vel = np.zeros_like(pos)
    vel[:, 0] = np.linspace(0.2, 0.8, y.size, dtype=np.float64)
    state = _MockState(pos=pos, vel=vel, fluid_indices=np.arange(y.size, dtype=np.int64))

    cfg = VxProfileConfig(
        enable=True,
        every=1,
        bins=8,
        axis=1,
        component=0,
        use_x_slice=False,
        x_mid=None,
        x_slice_width=0.0,
        y_min=None,
        y_max=None,
        y0=None,
        y_extent_mode="walls",
        wall_y0=2.0,
        wall_h=2.0,
        wall_padding=0.0,
        avg_window_steps=4,
        y_margin=0.0,
        channel_height=None,
        gx=1.0,
        nu=0.01,
        out_file=tmp_path / "vx_profile_bins_walls.csv",
    )
    diag = VxProfileDiagnostics(cfg=cfg)
    sample = diag.sample(step=1, state=state)

    assert sample is not None
    assert sample.y0_eff == 2.0
    assert sample.h_eff == 2.0
    assert sample.empty_bins == 0
    assert sum(1 for n in sample.counts if n > 0) == cfg.bins

