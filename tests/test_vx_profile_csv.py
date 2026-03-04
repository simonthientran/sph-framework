from __future__ import annotations

from pathlib import Path

import numpy as np

from sph.core.state import ParticleState
from sph.core.vx_profile import compute_vx_profile, export_vx_profile_csv


def _build_synthetic_state(n: int = 16) -> ParticleState:
    y = np.linspace(0.03, 0.17, n)
    x = np.linspace(0.2, 1.8, n)
    pos = np.column_stack([x, y])
    vel = np.zeros((n, 2), dtype=np.float64)
    vel[:, 0] = 0.25
    return ParticleState(
        dim=2,
        pos=pos,
        vel=vel,
        acc=np.zeros_like(pos),
        mass=np.ones(n, dtype=np.float64),
        rho=np.ones(n, dtype=np.float64) * 1000.0,
        p=np.zeros(n, dtype=np.float64),
        is_boundary=np.zeros(n, dtype=np.bool_),
    )


def test_vx_profile_csv_has_metadata_and_required_columns(tmp_path: Path) -> None:
    scene = {
        "meta": {"name": "pipe_flow_1phase_2d"},
        "domain": {"min": [0.0, 0.0], "max": [2.0, 0.2], "boundary_layers": 3},
        "fluid": {"spacing": 0.01, "min": [0.1, 0.03], "max": [1.9, 0.17]},
        "material": {"viscosity": {"enable": True, "nu": 0.001}},
    }
    bins = 8
    result = compute_vx_profile(
        step=25,
        state=_build_synthetic_state(),
        scene=scene,
        y_extent_mode="walls_inner",
        n_bins=bins,
        gx=0.1,
        nu=0.001,
    )

    out_csv = tmp_path / "vx_profile_bins.csv"
    export_vx_profile_csv(
        out_csv,
        result,
        scene_name="pipe_flow_1phase_2d",
        sim_time=0.125,
        timestamp="2026-03-04T00:00:00Z",
    )

    text = out_csv.read_text(encoding="utf-8")
    assert "# y0_eff=" in text
    assert "# H_eff=" in text
    assert "# gx=" in text
    assert "# nu=" in text
    assert "# bins=" in text
    assert "y_center,vx_mean,vx_count,vx_analytic" in text

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    table_idx = next(i for i, ln in enumerate(lines) if not ln.startswith("# "))
    header = lines[table_idx]
    rows = lines[table_idx + 1 :]
    assert header == "y_center,vx_mean,vx_count,vx_analytic"
    assert len(rows) == bins
