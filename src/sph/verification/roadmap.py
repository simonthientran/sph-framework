from __future__ import annotations

"""
Projection-solver verification roadmap (planning artifact used by developers).

This module documents expected verification strictness per solver family:
  - WCSPH: baseline compressible solver, moderate density tolerance.
  - PCISPH / IISPH / DFSPH: incompressibility-focused solvers expected to
    meet tighter density-error thresholds and report iteration residuals.
"""

SOLVER_VERIFICATION_TARGETS = {
    "wcsph": {
        "density_rel_err_mean_max": 0.05,
        "requires_iteration_metrics": False,
    },
    "pcisph": {
        "density_rel_err_mean_max": 0.02,
        "requires_iteration_metrics": True,
    },
    "iisph": {
        "density_rel_err_mean_max": 0.015,
        "requires_iteration_metrics": True,
    },
    "dfsph": {
        "density_rel_err_mean_max": 0.01,
        "requires_iteration_metrics": True,
    },
}

