"""SPH validation package.

Public API
----------
``compare``
    CPU-vs-CUDA numerical comparison engine.
``benchmark``
    Lightweight headless benchmark harness.
``baseline_registry``
    Frozen benchmark scene metadata and preset commands.
``contracts``
    Typed snapshot contracts for CUDA stage replay validation.
``profile``
    Velocity-profile utilities for Poiseuille-style physics checks.
``report``
    Per-step pass/warn/fail report for interactive diagnostics.
``boundary``
    Boundary-representation validation and preprocessing inspection helpers.
"""
from __future__ import annotations

from sph.validation.boundary import (
    BoundarySourceCheck,
    BoundaryValidationResult,
    ParticleRepresentationSummary,
    SDFStartupGeometryCheck,
    SDFGridSummary,
    SDFRepresentationSummary,
    validate_boundary_representations,
)
from sph.validation.compare import (
    ArrayCheck,
    ComparisonResult,
    RegimeCheck,
    ScalarCheck,
    cpu_only_validate,
    run_comparison,
)

__all__ = [
    "ArrayCheck",
    "BoundarySourceCheck",
    "BoundaryValidationResult",
    "ComparisonResult",
    "ParticleRepresentationSummary",
    "RegimeCheck",
    "SDFGridSummary",
    "SDFStartupGeometryCheck",
    "SDFRepresentationSummary",
    "ScalarCheck",
    "cpu_only_validate",
    "run_comparison",
    "validate_boundary_representations",
]
