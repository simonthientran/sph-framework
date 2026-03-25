"""Minimal benchmark harness for the periodic 2D pipe-flow scene."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from sph.core.simulation import SimulationRunner


def run_pipe_flow_benchmark(scene_path: Path, steps: int) -> List[Dict]:
    runner = SimulationRunner(scene_path)
    reports: List[Dict] = []
    for _ in range(steps):
        result = runner.step()
        metrics = result.metrics
        reports.append(
            {
                "step": metrics.step,
                "dt": metrics.dt,
                "rho_mean": metrics.rho_mean,
                "rho_rel_err_mean": metrics.rho_error_mean,
                "neighbor_mean": metrics.neighbor_mean,
                "velocity_max": metrics.velocity_max,
                "stability": metrics.stability,
            }
        )
    return reports


def summarize_reports(reports: List[Dict]) -> str:
    if not reports:
        return "No reports generated"
    last = reports[-1]
    return (
        f"dt={last.get('dt', 0.0):.3e}, "
        f"rho mean={last.get('rho_mean', 0.0):.2f}, err%={100.0 * last.get('rho_rel_err_mean', 0.0):.2f}, "
        f"neighbors mean={last.get('neighbor_mean', 0.0):.1f}, "
        f"|v|max={last.get('velocity_max', 0.0):.3f}"
    )
