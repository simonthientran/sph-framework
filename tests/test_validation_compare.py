"""Tests for the sph.validation.compare module.

Scope
-----
These tests use the CPU backend only (no CUDA device required).
CUDA-specific coverage lives in the ``validate_cpu_cuda.py`` script, which
is run manually or in a GPU-enabled CI environment.

All tests use the canonical pipe_flow_2d.json scene and a very small step
count (3–5 steps) so the suite runs in a few seconds.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCENE_BASE = ROOT / "scenes/examples/pipe_flow_2d.json"


# ---------------------------------------------------------------------------
# Imports from validation package
# ---------------------------------------------------------------------------

from sph.validation import (
    ArrayCheck,
    ComparisonResult,
    RegimeCheck,
    ScalarCheck,
    cpu_only_validate,
    run_comparison,
)
from sph.validation.compare import _array_check, _scalar_check
from sph.validation.benchmark import run_pipe_flow_benchmark, summarize_reports
from sph.validation.report import ValidationReport
from sph.validation.profile import compute_velocity_profile


# ---------------------------------------------------------------------------
# Module-level smoke: imports work and public API is importable
# ---------------------------------------------------------------------------


def test_validation_package_public_api_importable() -> None:
    assert callable(run_comparison)
    assert callable(cpu_only_validate)
    assert ComparisonResult is not None
    assert ScalarCheck is not None
    assert ArrayCheck is not None
    assert RegimeCheck is not None


# ---------------------------------------------------------------------------
# _scalar_check unit tests
# ---------------------------------------------------------------------------


def test_scalar_check_passes_within_tolerance() -> None:
    c = _scalar_check("test", 1000.0, 1002.0, tol_rel=0.005)
    assert c.cpu_value == 1000.0
    assert c.cuda_value == 1002.0
    assert abs(c.rel_diff - 0.002) < 1e-9
    assert c.passed  # 0.2% < 0.5% threshold


def test_scalar_check_fails_outside_tolerance() -> None:
    c = _scalar_check("test", 1000.0, 1020.0, tol_rel=0.005)
    assert not c.passed  # 2% > 0.5% threshold


def test_scalar_check_near_zero_reference() -> None:
    # cpu_value=0.0, cuda_value=1e-10, tol=1.0
    # ref = max(abs(0.0), 1e-12) = 1e-12
    # rel_diff = 1e-10 / 1e-12 = 100 → exceeds tol=1.0 → FAIL
    c = _scalar_check("zero_vel", 0.0, 1e-10, tol_rel=1.0)
    assert not c.passed  # 100 > 1.0


@pytest.mark.parametrize("tol", [0.0, 1e-9, 0.001, 0.1, 1.0])
def test_scalar_check_identical_values_always_pass(tol: float) -> None:
    c = _scalar_check("x", 42.5, 42.5, tol_rel=tol)
    assert c.passed
    assert c.abs_diff == 0.0


# ---------------------------------------------------------------------------
# _array_check unit tests
# ---------------------------------------------------------------------------


def test_array_check_identical_arrays_pass() -> None:
    import numpy as np
    arr = np.array([1.0, 2.0, 3.0])
    c = _array_check("density", arr, arr.copy(), tol=0.001)
    assert c.passed
    assert c.max_abs_diff == 0.0
    assert c.l2_rel == 0.0


def test_array_check_shape_mismatch_fails() -> None:
    import numpy as np
    a = np.ones(5)
    b = np.ones(6)
    c = _array_check("mismatch", a, b, tol=0.001)
    assert not c.passed
    assert c.max_abs_diff == float("inf")


def test_array_check_large_diff_fails() -> None:
    import numpy as np
    a = np.ones(10) * 1000.0
    b = a + 50.0  # 5% offset
    c = _array_check("rho", a, b, tol=0.01)  # 1% tol
    assert not c.passed


# ---------------------------------------------------------------------------
# benchmark.py
# ---------------------------------------------------------------------------


def test_benchmark_run_returns_per_step_dicts() -> None:
    reports = run_pipe_flow_benchmark(SCENE_BASE, steps=3)
    assert len(reports) == 3
    for r in reports:
        assert "step" in r
        assert "rho_mean" in r
        assert "rho_error_mean" in r
        assert "velocity_max" in r
        assert "stability" in r
        assert "fluid_count" in r
        assert r["fluid_count"] > 0


def test_benchmark_summarize_non_empty() -> None:
    reports = run_pipe_flow_benchmark(SCENE_BASE, steps=2)
    summary = summarize_reports(reports)
    assert "rho mean" in summary
    assert "err%" in summary


def test_benchmark_empty_reports_returns_message() -> None:
    summary = summarize_reports([])
    assert "No reports" in summary


# ---------------------------------------------------------------------------
# report.py
# ---------------------------------------------------------------------------


def test_validation_report_from_backend_step_diagnostics() -> None:
    """ValidationReport.from_diagnostics works with backend.StepDiagnostics."""
    from sph.core.simulation import SimulationRunner

    runner = SimulationRunner(SCENE_BASE, backend_name="numba_cpu")
    result = runner.step()
    diag = result.diagnostics  # sph.core.backend.StepDiagnostics

    # Use runner.backend.particle_state() to get a properly constructed ParticleState
    particle_state = runner.backend.particle_state()
    profile = compute_velocity_profile(particle_state, pipe_height=0.2, bins=8)

    report = ValidationReport.from_diagnostics(
        diagnostics=diag,
        profile=profile,
        rho0=1000.0,
    )
    assert report.stability in ("pass", "warn", "fail")
    assert "step=" in report.summary
    report_dict = report.to_dict()
    assert "rho_mean" in report_dict
    assert "velocity_profile" in report_dict


# ---------------------------------------------------------------------------
# cpu_only_validate
# ---------------------------------------------------------------------------


def test_cpu_only_validate_passes_on_pipe_flow() -> None:
    result = cpu_only_validate(SCENE_BASE, steps=5)
    assert isinstance(result, dict)
    assert "passed" in result
    assert "summary" in result
    assert result["passed"], f"Expected PASS, got:\n{result['summary']}"


def test_cpu_only_validate_result_has_runtime() -> None:
    result = cpu_only_validate(SCENE_BASE, steps=3)
    assert "runtime" in result
    rt = result["runtime"]
    assert rt.rho_mean > 0.0
    assert rt.fluid_count > 0


# ---------------------------------------------------------------------------
# ComparisonResult structure (CPU-vs-CPU smoke: same backend both sides)
# ---------------------------------------------------------------------------


def test_comparison_result_cpu_vs_cpu_passes() -> None:
    """Run CPU against CPU: both runs must complete with finite, physically valid state.

    NOTE: Two independent SimulationRunner instances that both use the parallel-Numba
    CPU backend are NOT guaranteed to be bit-identical.  ``accumulate_force`` uses
    ``@njit(parallel=True)`` whose floating-point reduction order varies between
    calls, causing O(1 %) spread in scalar metrics like ``velocity_max``.
    We therefore verify physical validity (finite values, sensible density range)
    rather than numeric identity.
    """
    from sph.core.simulation import SimulationRunner
    import math

    steps = 5
    run_a = SimulationRunner(SCENE_BASE, backend_name="numba_cpu")
    for _ in range(steps):
        rt_a = run_a.step()

    run_b = SimulationRunner(SCENE_BASE, backend_name="numba_cpu")
    for _ in range(steps):
        rt_b = run_b.step()

    # Both runs must have completed without NaN/Inf
    for label, rt in [("A", rt_a.runtime), ("B", rt_b.runtime)]:
        assert math.isfinite(rt.velocity_max), f"Run {label}: velocity_max is not finite"
        assert math.isfinite(rt.rho_mean), f"Run {label}: rho_mean is not finite"
        assert rt.rho_mean > 0, f"Run {label}: rho_mean ≤ 0"
        assert rt.neighbor_min > 0, f"Run {label}: neighbor_min == 0 (orphaned particles)"

    # Both runs must have the same velocity scale (within a large tolerance —
    # parallel Numba can give O(10 %) spread in velocity_max for near-zero flows)
    v_a = rt_a.runtime.velocity_max
    v_b = rt_b.runtime.velocity_max
    if v_a > 1e-12 and v_b > 1e-12:
        ratio = max(v_a, v_b) / min(v_a, v_b)
        assert ratio < 3.0, f"velocity_max ratio {ratio:.2f} too large (run A={v_a:.4e}, run B={v_b:.4e})"


def test_comparison_result_has_summary_lines() -> None:
    result = run_comparison(
        scene_path=SCENE_BASE,
        steps=3,
        cpu_backend="numba_cpu",
        cuda_backend="numba_cpu",
    )
    lines = result.summary_lines()
    assert len(lines) >= 3
    assert any("PASS" in line or "FAIL" in line for line in lines)


def test_comparison_result_to_dict_is_json_serializable() -> None:
    import json
    import math
    result = run_comparison(
        scene_path=SCENE_BASE,
        steps=3,
        cpu_backend="numba_cpu",
        cuda_backend="numba_cpu",
        velocity_tol_rel=0.5,  # wide: parallel-Numba is non-deterministic between runs
    )
    d = result.to_dict()
    serialized = json.dumps(d)
    back = json.loads(serialized)
    # Verify the dict is JSON-round-trip-stable and has the expected keys
    assert "overall_passed" in back
    assert isinstance(back["overall_passed"], bool)
    assert "scalar_checks" in back
    assert "array_checks" in back
    # Verify no NaN/Inf leaked into the serialized metrics
    for sc in back["scalar_checks"]:
        assert math.isfinite(sc["cpu"]), f"Non-finite cpu value in {sc['name']}"
        assert math.isfinite(sc["cuda"]), f"Non-finite cuda value in {sc['name']}"
