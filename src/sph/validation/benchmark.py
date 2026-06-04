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
        metrics = result.runtime
        reports.append(
            {
                "step": getattr(metrics, "step", 0),
                "dt": getattr(metrics, "dt", 0.0),
                "rho_mean": getattr(metrics, "rho_mean", 0.0),
                "rho_rel_err_mean": getattr(metrics, "rho_error_mean", 0.0),
                "rho_error_mean": getattr(metrics, "rho_error_mean", 0.0),
                "neighbor_mean": getattr(metrics, "neighbor_mean", 0.0),
                "velocity_max": getattr(metrics, "velocity_max", 0.0),
                "stability": getattr(metrics, "stability", "unknown"),
                "fluid_count": getattr(metrics, "fluid_count", 0),
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
