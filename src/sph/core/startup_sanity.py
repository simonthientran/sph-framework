from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sph.boundary.mesh_sampling import compute_fluid_boundary_distance_stats
from sph.core.state import ParticleState
from sph.neighbors.spatial_hash import SpatialHash
from sph.sph.density import compute_density_with_boundaries_eq83


@dataclass(frozen=True)
class StartupSanityReport:
    dx_min: float
    dx_mean: float
    dx_max: float
    h: float
    h_over_dx: float
    support_radius_expected: float
    expected_neighbors_2d: float
    rho_rel_err_min: float
    rho_rel_err_mean: float
    rho_rel_err_max: float
    overlap_close_fraction: float
    overlap_min_dist: float
    outside_domain_count: int
    units_mismatch_detected: bool
    recommendations: tuple[str, ...]
    auto_tuned_support_radius: float | None
    should_abort: bool
    abort_reason: str | None


def _nearest_neighbor_stats(fluid_pos: np.ndarray, sample_size: int) -> tuple[float, float, float]:
    n = int(fluid_pos.shape[0])
    if n <= 1:
        return (float("nan"), float("nan"), float("nan"))
    sample_size = int(max(1, min(sample_size, n)))
    sample_ids = np.linspace(0, n - 1, num=sample_size, dtype=np.int64)
    nn_dists = np.empty((sample_size,), dtype=np.float64)
    for k, i in enumerate(sample_ids):
        d = np.linalg.norm(fluid_pos - fluid_pos[i], axis=1)
        d[i] = np.inf
        nn_dists[k] = float(np.min(d))
    return (float(np.min(nn_dists)), float(np.mean(nn_dists)), float(np.max(nn_dists)))


def evaluate_startup_sanity(
    *,
    scene: dict,
    state: ParticleState,
    h: float,
    spacing: float,
    rho0: float,
    startup_cfg: dict | None = None,
) -> StartupSanityReport:
    startup_cfg = startup_cfg or {}
    sample_size = int(startup_cfg.get("dx_sample_size", 128))
    density_warn_rel = float(startup_cfg.get("density_error_warn_rel", 0.05))
    density_abort_rel = float(startup_cfg.get("density_error_abort_rel", density_warn_rel))
    auto_tune = bool(startup_cfg.get("auto_tune_support_radius", True))
    target_h_over_dx = float(startup_cfg.get("support_radius_target_h_over_dx_2d", 1.3))
    abort_on_units_mismatch = bool(startup_cfg.get("abort_on_units_mismatch", True))

    fluid_ids = state.fluid_indices
    boundary_ids = state.boundary_indices
    fluid_pos = state.pos[fluid_ids]
    boundary_pos = state.pos[boundary_ids]

    dx_min, dx_mean, dx_max = _nearest_neighbor_stats(fluid_pos, sample_size=sample_size)
    h = float(h)
    spacing = float(spacing)
    rho0 = float(rho0)
    h_over_dx = float(h / dx_mean) if np.isfinite(dx_mean) and dx_mean > 0.0 else float("nan")

    # Reference convention requested by verification notes: cubic spline support radius = 2h.
    support_radius_expected = float(2.0 * h)
    if np.isfinite(dx_mean) and dx_mean > 0.0 and state.dim == 2:
        expected_neighbors_2d = float(np.pi * (support_radius_expected / dx_mean) ** 2)
    else:
        expected_neighbors_2d = float("nan")

    # Keep density reconstruction aligned with current solver neighborhood behavior.
    ns = SpatialHash(support_radius=float(h), dim=state.dim)
    ns.build(state.pos)
    rho_init = compute_density_with_boundaries_eq83(state=state, neighbor_search=ns, h=h, rho0=rho0)
    if fluid_ids.size:
        rel = np.abs((rho_init[fluid_ids] - rho0) / rho0)
        rho_rel_err_min = float(np.min(rel))
        rho_rel_err_mean = float(np.mean(rel))
        rho_rel_err_max = float(np.max(rel))
    else:
        rho_rel_err_min = 0.0
        rho_rel_err_mean = 0.0
        rho_rel_err_max = 0.0

    overlap = compute_fluid_boundary_distance_stats(
        fluid_positions=fluid_pos,
        boundary_positions=boundary_pos,
        threshold=0.5 * spacing,
    )
    outside_count = 0
    domain_cfg = scene.get("domain", {})
    if "min" in domain_cfg and "max" in domain_cfg and fluid_ids.size:
        dmin = np.asarray(domain_cfg["min"], dtype=np.float64)
        dmax = np.asarray(domain_cfg["max"], dtype=np.float64)
        outside = np.any((fluid_pos < dmin[None, :]) | (fluid_pos > dmax[None, :]), axis=1)
        outside_count = int(np.count_nonzero(outside))

    units_mismatch_detected = False
    geom_report = scene.get("__geom_report__", {})
    meshes = geom_report.get("meshes", [])
    domain_diag = None
    if "min" in domain_cfg and "max" in domain_cfg:
        dmin = np.asarray(domain_cfg["min"], dtype=np.float64)
        dmax = np.asarray(domain_cfg["max"], dtype=np.float64)
        domain_diag = float(np.linalg.norm(dmax - dmin))
    if domain_diag is not None and domain_diag > 0.0:
        for m in meshes:
            diag = float(m.get("diag", 0.0))
            ratio = diag / domain_diag if domain_diag > 0.0 else 1.0
            if ratio < 1e-3 or ratio > 1e3:
                units_mismatch_detected = True
                break

    recommendations: list[str] = []
    if np.isfinite(h_over_dx) and h_over_dx < target_h_over_dx:
        recommendations.append(
            f"h/dx={h_over_dx:.2f} is low; increase support_radius to >= {target_h_over_dx:.2f}*dx."
        )
    if np.isfinite(expected_neighbors_2d) and expected_neighbors_2d < 15.0:
        recommendations.append("Expected 2D neighbor count is below target (15-30); increase h or reduce dx.")
    if overlap.close_fraction > 0.01:
        recommendations.append("Fluid is close to boundaries at t=0; check overlap and boundary layers/sampling.")
    if outside_count > 0:
        recommendations.append("Some fluid particles start outside domain; adjust fluid block/domain bounds.")
    if rho_rel_err_mean > density_warn_rel:
        recommendations.append("Initial density error is high; check units/scale, overlap, h/dx, and boundary resolution.")
    if units_mismatch_detected:
        recommendations.append("Geometry/domain scale mismatch detected; verify units_hint and mesh transform scale.")

    auto_tuned_support_radius = None
    if auto_tune and np.isfinite(dx_mean) and dx_mean > 0.0:
        tuned = float(target_h_over_dx * dx_mean)
        if tuned > h:
            auto_tuned_support_radius = tuned

    should_abort = False
    abort_reason = None
    if units_mismatch_detected and abort_on_units_mismatch:
        should_abort = True
        abort_reason = "units_mismatch"
    elif rho_rel_err_mean > density_abort_rel:
        should_abort = True
        abort_reason = "high_initial_density_error"

    return StartupSanityReport(
        dx_min=dx_min,
        dx_mean=dx_mean,
        dx_max=dx_max,
        h=h,
        h_over_dx=h_over_dx,
        support_radius_expected=support_radius_expected,
        expected_neighbors_2d=expected_neighbors_2d,
        rho_rel_err_min=rho_rel_err_min,
        rho_rel_err_mean=rho_rel_err_mean,
        rho_rel_err_max=rho_rel_err_max,
        overlap_close_fraction=float(overlap.close_fraction),
        overlap_min_dist=float(overlap.min_distance),
        outside_domain_count=outside_count,
        units_mismatch_detected=units_mismatch_detected,
        recommendations=tuple(recommendations),
        auto_tuned_support_radius=auto_tuned_support_radius,
        should_abort=should_abort,
        abort_reason=abort_reason,
    )


def format_startup_sanity_block(report: StartupSanityReport) -> str:
    lines = [
        "[STARTUP][SANITY]",
        (
            f"  dx(nn) min/mean/max = {report.dx_min:.6e}/{report.dx_mean:.6e}/{report.dx_max:.6e} "
            f"| h={report.h:.6e} | h/dx={report.h_over_dx:.3f}"
        ),
        (
            f"  support_radius(cubic-spline ref)=2h={report.support_radius_expected:.6e} "
            f"| expected_neighbors_2d~{report.expected_neighbors_2d:.2f}"
        ),
        (
            f"  rho_init rel_err min/avg/max = {report.rho_rel_err_min:.2%}/{report.rho_rel_err_mean:.2%}/{report.rho_rel_err_max:.2%}"
        ),
        (
            f"  overlap close_fraction={report.overlap_close_fraction:.2%} "
            f"min_dist={report.overlap_min_dist:.6e} | outside_domain={report.outside_domain_count}"
        ),
    ]
    if report.auto_tuned_support_radius is not None:
        lines.append(
            f"  fix: auto_tune_support_radius -> {report.auto_tuned_support_radius:.6e} (from {report.h:.6e})"
        )
    if report.recommendations:
        lines.append("  recommendations:")
        for r in report.recommendations:
            lines.append(f"    - {r}")
    if report.should_abort:
        lines.append(f"  abort: {report.abort_reason}")
    return "\n".join(lines)

