from __future__ import annotations

"""Sidecar metadata for playback-oriented diagnostics."""

from pathlib import Path
import json

from sph.core.backend import RuntimeStats


def runtime_stats_to_playback_dict(stats: RuntimeStats) -> dict:
    density_summary = stats.density_summary
    density_payload = None
    if density_summary is not None:
        density_payload = {
            "interior_count": int(density_summary.interior_count),
            "wall_count": int(density_summary.wall_count),
            "free_surface_count": int(density_summary.free_surface_count),
            "splash_count": int(density_summary.splash_count),
            "overcompressed_count": int(density_summary.overcompressed_count),
            "under_supported_count": int(density_summary.under_supported_count),
            "rho_mean_interior": float(density_summary.rho_mean_interior),
            "rho_min_interior": float(density_summary.rho_min_interior),
            "rho_max_interior": float(density_summary.rho_max_interior),
            "rho_mean_wall": float(density_summary.rho_mean_wall),
            "rho_min_wall": float(density_summary.rho_min_wall),
            "rho_max_wall": float(density_summary.rho_max_wall),
            "rho_mean_free_surface": float(density_summary.rho_mean_free_surface),
            "rho_min_free_surface": float(density_summary.rho_min_free_surface),
        }

    return {
        "step": int(stats.step),
        "dt": float(stats.dt),
        "backend_name": getattr(stats, "backend_name", ""),
        "solver": stats.solver,
        "scene_name": stats.scene_name,
        "velocity_max": float(stats.velocity_max),
        "rho_min": float(stats.rho_min),
        "rho_mean": float(stats.rho_mean),
        "rho_max": float(stats.rho_max),
        "rho_error_mean": float(stats.rho_error_mean),
        "pressure_min": float(stats.pressure_min),
        "pressure_mean": float(stats.pressure_mean),
        "pressure_max": float(stats.pressure_max),
        "neighbor_min": int(stats.neighbor_min),
        "neighbor_mean": float(stats.neighbor_mean),
        "neighbor_max": int(stats.neighbor_max),
        "stability": stats.stability,
        "fluid_count": int(stats.fluid_count),
        "boundary_count": int(stats.boundary_count),
        "wall_time_ms": float(stats.wall_time_ms),
        "stage_timings_ms": {key: float(value) for key, value in stats.stage_timings_ms.items()},
        "solver_metrics": {key: float(value) for key, value in stats.solver_metrics.items()},
        "solver_health_summary": stats.solver_health_summary,
        "solver_health_notes": list(stats.solver_health_notes),
        "density_summary": density_payload,
    }


def export_playback_diagnostics(path: str | Path, stats: RuntimeStats) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = runtime_stats_to_playback_dict(stats)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_playback_diagnostics(path: str | Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
