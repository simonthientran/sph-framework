"""Boundary-representation validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from sph.boundaries import load_boundary_representations_from_scene
from sph.boundaries.mesh_particles import MeshParticleBoundaryRepresentation
from sph.boundaries.mesh_sdf import MeshSDFBoundaryRepresentation
from sph.boundaries.volume_map import VolumeMapBoundaryRepresentation


Classification = Literal["pass", "warn", "fail"]


@dataclass(slots=True, frozen=True)
class BoundarySourceCheck:
    """Consistency summary for one boundary source across representations."""

    source_name: str
    boundary_type: str
    representation_kinds: tuple[str, ...]
    has_particle_representation: bool
    has_sdf_representation: bool
    passed: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "boundary_type": self.boundary_type,
            "representation_kinds": list(self.representation_kinds),
            "has_particle_representation": self.has_particle_representation,
            "has_sdf_representation": self.has_sdf_representation,
            "passed": self.passed,
            "message": self.message,
        }


@dataclass(slots=True, frozen=True)
class ParticleRepresentationSummary:
    """Structured summary for the particle boundary representation."""

    kind: str
    source_names: tuple[str, ...]
    source_types: tuple[str, ...]
    sample_counts: tuple[int, ...]
    total_count: int
    bounds_min: tuple[float, ...]
    bounds_max: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "source_names": list(self.source_names),
            "source_types": list(self.source_types),
            "sample_counts": list(self.sample_counts),
            "total_count": self.total_count,
            "bounds_min": list(self.bounds_min),
            "bounds_max": list(self.bounds_max),
        }


@dataclass(slots=True, frozen=True)
class SDFGridSummary:
    """Structured summary for one SDF grid."""

    source_name: str
    boundary_type: str
    resolution: int
    resolution_source: str
    padding_ratio: float
    cell_size: float
    target_cell_size: float
    cells_per_particle_spacing: float
    startup_warn_gap: float
    startup_correction_enabled: bool
    startup_correction_target_gap: float
    startup_correction_max_shift: float
    startup_velocity_projection_enabled: bool
    startup_velocity_projection_gap: float
    startup_validity_warn_inside_fraction: float
    startup_validity_fail_inside_fraction: float
    startup_validity_warn_normal_fraction: float
    startup_validity_fail_normal_fraction: float
    startup_validity_warn_too_close_fraction: float
    startup_validity_fail_too_close_fraction: float
    startup_runtime_fail_action: str
    grid_min: tuple[float, ...]
    grid_max: tuple[float, ...]
    mesh_bounds_min: tuple[float, ...]
    mesh_bounds_max: tuple[float, ...]
    sdf_min: float
    sdf_max: float
    is_watertight: bool
    is_usable: bool
    warnings: tuple[str, ...]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "boundary_type": self.boundary_type,
            "resolution": self.resolution,
            "resolution_source": self.resolution_source,
            "padding_ratio": self.padding_ratio,
            "cell_size": self.cell_size,
            "target_cell_size": self.target_cell_size,
            "cells_per_particle_spacing": self.cells_per_particle_spacing,
            "startup_warn_gap": self.startup_warn_gap,
            "startup_correction_enabled": self.startup_correction_enabled,
            "startup_correction_target_gap": self.startup_correction_target_gap,
            "startup_correction_max_shift": self.startup_correction_max_shift,
            "startup_velocity_projection_enabled": self.startup_velocity_projection_enabled,
            "startup_velocity_projection_gap": self.startup_velocity_projection_gap,
            "startup_validity_warn_inside_fraction": self.startup_validity_warn_inside_fraction,
            "startup_validity_fail_inside_fraction": self.startup_validity_fail_inside_fraction,
            "startup_validity_warn_normal_fraction": self.startup_validity_warn_normal_fraction,
            "startup_validity_fail_normal_fraction": self.startup_validity_fail_normal_fraction,
            "startup_validity_warn_too_close_fraction": self.startup_validity_warn_too_close_fraction,
            "startup_validity_fail_too_close_fraction": self.startup_validity_fail_too_close_fraction,
            "startup_runtime_fail_action": self.startup_runtime_fail_action,
            "grid_min": list(self.grid_min),
            "grid_max": list(self.grid_max),
            "mesh_bounds_min": list(self.mesh_bounds_min),
            "mesh_bounds_max": list(self.mesh_bounds_max),
            "sdf_min": self.sdf_min,
            "sdf_max": self.sdf_max,
            "is_watertight": self.is_watertight,
            "is_usable": self.is_usable,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


@dataclass(slots=True, frozen=True)
class SDFRepresentationSummary:
    """Structured summary for the SDF boundary representation."""

    kind: str
    grid_summaries: tuple[SDFGridSummary, ...]
    all_grids_usable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "all_grids_usable": self.all_grids_usable,
            "grid_summaries": [grid.to_dict() for grid in self.grid_summaries],
        }


@dataclass(slots=True, frozen=True)
class VolumeMapRepresentationSummary:
    """Structured summary for a volume-map boundary representation."""

    kind: str
    grid_count: int
    support_radius: float
    volume_min: float
    volume_max: float
    volume_mean_nonzero: float
    cells_with_nonzero_volume: int
    total_cells: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "grid_count": self.grid_count,
            "support_radius": self.support_radius,
            "volume_min": self.volume_min,
            "volume_max": self.volume_max,
            "volume_mean_nonzero": self.volume_mean_nonzero,
            "cells_with_nonzero_volume": self.cells_with_nonzero_volume,
            "total_cells": self.total_cells,
        }


@dataclass(slots=True, frozen=True)
class SDFStartupGeometryCheck:
    """Startup fluid-to-wall geometry summary derived from mesh_sdf."""

    fluid_count: int
    inside_fraction: float
    signed_distance_min: float
    signed_distance_max: float
    wall_distance_min: float
    wall_distance_mean: float
    wall_distance_max: float
    too_close_count: int
    warn_gap: float
    normal_finite_fraction: float
    normal_norm_mean: float
    mean_abs_normal: tuple[float, float, float]
    status: Classification
    warnings: tuple[str, ...]
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fluid_count": self.fluid_count,
            "inside_fraction": self.inside_fraction,
            "signed_distance_min": self.signed_distance_min,
            "signed_distance_max": self.signed_distance_max,
            "wall_distance_min": self.wall_distance_min,
            "wall_distance_mean": self.wall_distance_mean,
            "wall_distance_max": self.wall_distance_max,
            "too_close_count": self.too_close_count,
            "warn_gap": self.warn_gap,
            "normal_finite_fraction": self.normal_finite_fraction,
            "normal_norm_mean": self.normal_norm_mean,
            "mean_abs_normal": list(self.mean_abs_normal),
            "status": self.status,
            "warnings": list(self.warnings),
            "failures": list(self.failures),
        }


@dataclass(slots=True, frozen=True)
class BoundaryValidationResult:
    """Structured result for representation-level boundary validation."""

    scene: str
    spacing: float
    requested_representations: tuple[str, ...]
    built_representation_kinds: tuple[str, ...]
    particle: ParticleRepresentationSummary | None
    sdf: SDFRepresentationSummary | None
    volume_map: VolumeMapRepresentationSummary | None
    sdf_startup_geometry: SDFStartupGeometryCheck | None
    source_checks: tuple[BoundarySourceCheck, ...]
    status: Classification
    passed: bool
    summary_lines: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene": self.scene,
            "spacing": self.spacing,
            "requested_representations": list(self.requested_representations),
            "built_representation_kinds": list(self.built_representation_kinds),
            "particle": None if self.particle is None else self.particle.to_dict(),
            "sdf": None if self.sdf is None else self.sdf.to_dict(),
            "volume_map": None if self.volume_map is None else self.volume_map.to_dict(),
            "sdf_startup_geometry": None if self.sdf_startup_geometry is None else self.sdf_startup_geometry.to_dict(),
            "source_checks": [check.to_dict() for check in self.source_checks],
            "status": self.status,
            "passed": self.passed,
            "summary_lines": list(self.summary_lines),
        }


def _merge_status(current: Classification, new: Classification) -> Classification:
    order = {"pass": 0, "warn": 1, "fail": 2}
    return new if order[new] > order[current] else current


def _load_scene(scene_path: Path) -> dict:
    return json.loads(scene_path.read_text())


def _apply_requested_representations(scene: dict, requested: tuple[str, ...]) -> dict:
    if not requested:
        return scene
    scene_copy = json.loads(json.dumps(scene))
    requested_set = {item.lower() for item in requested}
    for boundary in scene_copy.get("boundaries", []):
        boundary["representations"] = sorted(requested_set)
        if "sdf" in requested_set:
            sdf_cfg = dict(boundary.get("sdf", {}))
            sdf_cfg["enable"] = True
            boundary["sdf"] = sdf_cfg
    return scene_copy


def _scene_spacing(scene: dict) -> float:
    return float(scene["fluid"]["spacing"])


def _fluid_block_positions(scene: dict) -> np.ndarray | None:
    fluid_cfg = scene.get("fluid", {})
    if str(fluid_cfg.get("type", "")).lower() != "block":
        return None

    spacing = float(fluid_cfg["spacing"])
    fmin = np.asarray(fluid_cfg["min"], dtype=np.float64)
    fmax = np.asarray(fluid_cfg["max"], dtype=np.float64)
    dim = int(scene.get("meta", {}).get("dimensions", fmin.shape[0]))

    if dim == 2:
        xs = np.arange(fmin[0], fmax[0] + 0.5 * spacing, spacing)
        ys = np.arange(fmin[1], fmax[1] + 0.5 * spacing, spacing)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
        padded = np.zeros((pts.shape[0], 3), dtype=np.float64)
        padded[:, :2] = pts
        return padded

    xs = np.arange(fmin[0], fmax[0] + 0.5 * spacing, spacing)
    ys = np.arange(fmin[1], fmax[1] + 0.5 * spacing, spacing)
    zs = np.arange(fmin[2], fmax[2] + 0.5 * spacing, spacing)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float64)


def _summarize_particle(rep: MeshParticleBoundaryRepresentation, spacing: float) -> ParticleRepresentationSummary:
    positions = rep.project_particle_positions(dim=3, spacing=spacing, deduplicate=False)
    bounds_min = tuple(np.min(positions, axis=0).astype(np.float64).tolist())
    bounds_max = tuple(np.max(positions, axis=0).astype(np.float64).tolist())
    return ParticleRepresentationSummary(
        kind=rep.kind,
        source_names=tuple(source.source_name for source in rep.sources),
        source_types=tuple(source.boundary_type for source in rep.sources),
        sample_counts=tuple(sample.count for sample in rep.samples),
        total_count=rep.count,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
    )


def _summarize_sdf(rep: MeshSDFBoundaryRepresentation) -> SDFRepresentationSummary:
    grids: list[SDFGridSummary] = []
    for grid in rep.grids:
        policy = grid.source.policy
        quality = grid.quality
        grids.append(
            SDFGridSummary(
                source_name=grid.source.source_name,
                boundary_type=grid.source.source.boundary_type,
                resolution=grid.resolution,
                resolution_source=policy.resolution_source,
                padding_ratio=policy.padding_ratio,
                cell_size=grid.cell_size,
                target_cell_size=policy.target_cell_size,
                cells_per_particle_spacing=policy.cells_per_particle_spacing,
                startup_warn_gap=policy.startup.warn_gap,
                startup_correction_enabled=policy.startup.correction_enabled,
                startup_correction_target_gap=policy.startup.correction_target_gap,
                startup_correction_max_shift=policy.startup.correction_max_shift,
                startup_velocity_projection_enabled=policy.startup.velocity_projection_enabled,
                startup_velocity_projection_gap=policy.startup.velocity_projection_gap,
                startup_validity_warn_inside_fraction=policy.startup.validity_warn_inside_fraction,
                startup_validity_fail_inside_fraction=policy.startup.validity_fail_inside_fraction,
                startup_validity_warn_normal_fraction=policy.startup.validity_warn_normal_fraction,
                startup_validity_fail_normal_fraction=policy.startup.validity_fail_normal_fraction,
                startup_validity_warn_too_close_fraction=policy.startup.validity_warn_too_close_fraction,
                startup_validity_fail_too_close_fraction=policy.startup.validity_fail_too_close_fraction,
                startup_runtime_fail_action=policy.startup.runtime_fail_action,
                grid_min=tuple(grid.grid_min.astype(np.float64).tolist()),
                grid_max=tuple(grid.grid_max.astype(np.float64).tolist()),
                mesh_bounds_min=tuple(quality.mesh_bounds_min.astype(np.float64).tolist()),
                mesh_bounds_max=tuple(quality.mesh_bounds_max.astype(np.float64).tolist()),
                sdf_min=quality.sdf_min,
                sdf_max=quality.sdf_max,
                is_watertight=quality.is_watertight,
                is_usable=quality.is_usable,
                warnings=quality.warnings,
                errors=quality.errors,
            )
        )
    return SDFRepresentationSummary(
        kind=rep.kind,
        grid_summaries=tuple(grids),
        all_grids_usable=rep.all_grids_usable,
    )


def _summarize_sdf_startup_geometry(
    scene: dict,
    rep: MeshSDFBoundaryRepresentation,
    spacing: float,
) -> SDFStartupGeometryCheck | None:
    fluid_positions = _fluid_block_positions(scene)
    if fluid_positions is None or fluid_positions.size == 0:
        return None

    signed = rep.sample_signed_distance(fluid_positions)
    finite = np.isfinite(signed)
    if not np.any(finite):
        return SDFStartupGeometryCheck(
            fluid_count=int(fluid_positions.shape[0]),
            inside_fraction=0.0,
            signed_distance_min=float("nan"),
            signed_distance_max=float("nan"),
            wall_distance_min=float("nan"),
            wall_distance_mean=float("nan"),
            wall_distance_max=float("nan"),
            too_close_count=0,
            warn_gap=float("nan"),
            normal_finite_fraction=0.0,
            normal_norm_mean=float("nan"),
            mean_abs_normal=(float("nan"), float("nan"), float("nan")),
            status="fail",
            warnings=("no finite SDF startup queries for fluid positions",),
            failures=("no finite SDF startup queries for fluid positions",),
        )

    signed = signed[finite]
    wall_distance = np.abs(signed)
    warn_gap = float(rep.grids[0].source.policy.startup.warn_gap)
    gap_tol = max(1.0e-4, 1.0e-3 * warn_gap)
    too_close_count = int(np.count_nonzero(wall_distance < (warn_gap - gap_tol)))
    normals = rep.sample_wall_normal(fluid_positions[finite])
    normal_norm = np.linalg.norm(normals, axis=1)
    finite_normals = np.isfinite(normal_norm)
    normal_finite_fraction = float(np.mean(finite_normals)) if normal_norm.size else 0.0
    normal_norm_mean = float(np.mean(normal_norm[finite_normals])) if np.any(finite_normals) else float("nan")
    if np.any(finite_normals):
        mean_abs_normal = tuple(np.mean(np.abs(normals[finite_normals]), axis=0).astype(np.float64).tolist())
    else:
        mean_abs_normal = (float("nan"), float("nan"), float("nan"))
    inside_fraction = float(np.mean(signed < 0.0))
    policy = rep.grids[0].source.policy.startup
    warnings: list[str] = []
    failures: list[str] = []
    status: Classification = "pass"
    too_close_fraction = too_close_count / max(int(fluid_positions[finite].shape[0]), 1)

    if inside_fraction < policy.validity_fail_inside_fraction:
        failures.append(
            f"only {inside_fraction * 100.0:.1f}% of startup fluid samples are inside the mesh volume"
        )
        status = _merge_status(status, "fail")
    elif inside_fraction < policy.validity_warn_inside_fraction:
        warnings.append(
            f"only {inside_fraction * 100.0:.1f}% of startup fluid samples are inside the mesh volume"
        )
        status = _merge_status(status, "warn")

    if too_close_fraction > policy.validity_fail_too_close_fraction:
        failures.append(
            f"{too_close_count} startup fluid samples ({too_close_fraction * 100.0:.1f}%) are closer to the wall than warn_gap={warn_gap:.5f}"
        )
        status = _merge_status(status, "fail")
    elif too_close_fraction > policy.validity_warn_too_close_fraction:
        warnings.append(
            f"{too_close_count} startup fluid samples ({too_close_fraction * 100.0:.1f}%) are closer to the wall than warn_gap={warn_gap:.5f}"
        )
        status = _merge_status(status, "warn")

    if normal_finite_fraction < policy.validity_fail_normal_fraction:
        failures.append(
            f"only {normal_finite_fraction * 100.0:.1f}% of startup wall normals are finite"
        )
        status = _merge_status(status, "fail")
    elif normal_finite_fraction < policy.validity_warn_normal_fraction:
        warnings.append(
            f"only {normal_finite_fraction * 100.0:.1f}% of startup wall normals are finite"
        )
        status = _merge_status(status, "warn")

    if np.isfinite(normal_norm_mean) and abs(normal_norm_mean - 1.0) > 0.05:
        warnings.append(f"startup wall normals are not near unit length (mean norm={normal_norm_mean:.3f})")
        status = _merge_status(status, "warn")

    return SDFStartupGeometryCheck(
        fluid_count=int(fluid_positions.shape[0]),
        inside_fraction=inside_fraction,
        signed_distance_min=float(np.min(signed)),
        signed_distance_max=float(np.max(signed)),
        wall_distance_min=float(np.min(wall_distance)),
        wall_distance_mean=float(np.mean(wall_distance)),
        wall_distance_max=float(np.max(wall_distance)),
        too_close_count=too_close_count,
        warn_gap=warn_gap,
        normal_finite_fraction=normal_finite_fraction,
        normal_norm_mean=normal_norm_mean,
        mean_abs_normal=mean_abs_normal,
        status=status,
        warnings=tuple(warnings),
        failures=tuple(failures),
    )


def _summarize_volume_map(rep: VolumeMapBoundaryRepresentation) -> VolumeMapRepresentationSummary:
    total_cells = 0
    vol_all = []
    support_radius = 0.0
    for grid in rep.grids:
        v = grid.volume_values.ravel()
        vol_all.append(v)
        total_cells += len(v)
        support_radius = grid.support_radius
    if vol_all:
        combined = np.concatenate(vol_all)
        nonzero = combined[combined > 1e-6]
        return VolumeMapRepresentationSummary(
            kind=rep.kind,
            grid_count=len(rep.grids),
            support_radius=support_radius,
            volume_min=float(np.min(combined)),
            volume_max=float(np.max(combined)),
            volume_mean_nonzero=float(np.mean(nonzero)) if len(nonzero) > 0 else 0.0,
            cells_with_nonzero_volume=int(len(nonzero)),
            total_cells=total_cells,
        )
    return VolumeMapRepresentationSummary(
        kind=rep.kind,
        grid_count=0,
        support_radius=0.0,
        volume_min=0.0,
        volume_max=0.0,
        volume_mean_nonzero=0.0,
        cells_with_nonzero_volume=0,
        total_cells=0,
    )


def _build_source_checks(
    particle: ParticleRepresentationSummary | None,
    sdf: SDFRepresentationSummary | None,
) -> tuple[BoundarySourceCheck, ...]:
    source_map: dict[tuple[str, str], set[str]] = {}

    if particle is not None:
        for boundary_type, source_name in zip(particle.source_types, particle.source_names, strict=True):
            source_map.setdefault((boundary_type, source_name), set()).add("mesh_particles")

    if sdf is not None:
        for grid in sdf.grid_summaries:
            source_map.setdefault((grid.boundary_type, grid.source_name), set()).add("mesh_sdf")

    checks: list[BoundarySourceCheck] = []
    for (boundary_type, source_name), kinds in sorted(source_map.items()):
        has_particle = "mesh_particles" in kinds
        has_sdf = "mesh_sdf" in kinds
        passed = True
        message = "source built successfully"
        if has_sdf and not has_particle:
            message = "source has mesh_sdf without mesh_particles; runtime bridge remains particle-only"
        checks.append(
            BoundarySourceCheck(
                source_name=source_name,
                boundary_type=boundary_type,
                representation_kinds=tuple(sorted(kinds)),
                has_particle_representation=has_particle,
                has_sdf_representation=has_sdf,
                passed=passed,
                message=message,
            )
        )
    return tuple(checks)


def _build_summary_lines(
    particle: ParticleRepresentationSummary | None,
    sdf: SDFRepresentationSummary | None,
    volume_map: VolumeMapRepresentationSummary | None,
    sdf_startup_geometry: SDFStartupGeometryCheck | None,
    checks: tuple[BoundarySourceCheck, ...],
    status: Classification,
) -> tuple[str, ...]:
    lines: list[str] = [f"Boundary representation validation: {status.upper()}"]
    if particle is not None:
        lines.append(
            "mesh_particles: "
            f"sources={len(particle.source_names)} total_count={particle.total_count} "
            f"bounds_min={tuple(round(v, 4) for v in particle.bounds_min)} "
            f"bounds_max={tuple(round(v, 4) for v in particle.bounds_max)}"
        )
    if sdf is not None:
        for grid in sdf.grid_summaries:
            lines.append(
                "mesh_sdf: "
                f"source={grid.source_name} resolution={grid.resolution} "
                f"resolution_source={grid.resolution_source} cell_size={grid.cell_size:.5f} "
                f"usable={grid.is_usable} warnings={len(grid.warnings)} errors={len(grid.errors)}"
            )
    if volume_map is not None:
        lines.append(
            "volume_map: "
            f"grids={volume_map.grid_count} support_radius={volume_map.support_radius:.4f} "
            f"nonzero_cells={volume_map.cells_with_nonzero_volume}/{volume_map.total_cells} "
            f"volume_range=[{volume_map.volume_min:.4f}, {volume_map.volume_max:.4f}] "
            f"volume_mean_nonzero={volume_map.volume_mean_nonzero:.4f}"
        )
    if sdf_startup_geometry is not None:
        lines.append(
            "mesh_sdf startup: "
            f"inside_fraction={sdf_startup_geometry.inside_fraction:.3f} "
            f"wall_distance_min={sdf_startup_geometry.wall_distance_min:.5f} "
            f"wall_distance_mean={sdf_startup_geometry.wall_distance_mean:.5f} "
            f"too_close_count={sdf_startup_geometry.too_close_count} "
            f"normal_finite_fraction={sdf_startup_geometry.normal_finite_fraction:.3f} "
            f"normal_norm_mean={sdf_startup_geometry.normal_norm_mean:.3f} "
            f"status={sdf_startup_geometry.status.upper()} "
            f"warnings={len(sdf_startup_geometry.warnings)} "
            f"failures={len(sdf_startup_geometry.failures)}"
        )
        for warning in sdf_startup_geometry.warnings:
            lines.append(f"startup warning: {warning}")
        for failure in sdf_startup_geometry.failures:
            lines.append(f"startup failure: {failure}")
    for check in checks:
        lines.append(
            f"source={check.source_name} kinds={','.join(check.representation_kinds)} "
            f"status={'PASS' if check.passed else 'FAIL'}"
        )
    return tuple(lines)


def validate_boundary_representations(
    scene_path: Path,
    *,
    requested_representations: tuple[str, ...] = (),
) -> BoundaryValidationResult:
    scene_path = scene_path.resolve()
    scene = _load_scene(scene_path)
    scene = _apply_requested_representations(scene, requested_representations)
    spacing = _scene_spacing(scene)
    representations = load_boundary_representations_from_scene(scene, scene_path, spacing=spacing)
    built_kinds = tuple(rep.kind for rep in representations)

    particle_rep = next((rep for rep in representations if isinstance(rep, MeshParticleBoundaryRepresentation)), None)
    sdf_rep = next((rep for rep in representations if isinstance(rep, MeshSDFBoundaryRepresentation)), None)
    vm_rep = next((rep for rep in representations if isinstance(rep, VolumeMapBoundaryRepresentation)), None)

    particle_summary = None if particle_rep is None else _summarize_particle(particle_rep, spacing=spacing)
    sdf_summary = None if sdf_rep is None else _summarize_sdf(sdf_rep)
    vm_summary = None if vm_rep is None else _summarize_volume_map(vm_rep)
    sdf_startup_geometry = None if sdf_rep is None else _summarize_sdf_startup_geometry(scene, sdf_rep, spacing=spacing)

    checks = _build_source_checks(particle_summary, sdf_summary)
    status: Classification = "pass"
    if not built_kinds:
        status = _merge_status(status, "fail")
    if sdf_summary is not None and not sdf_summary.all_grids_usable:
        status = _merge_status(status, "fail")
    if sdf_startup_geometry is not None:
        status = _merge_status(status, sdf_startup_geometry.status)
    if not checks:
        status = _merge_status(status, "fail")

    passed = status != "fail"
    summary_lines = _build_summary_lines(particle_summary, sdf_summary, vm_summary, sdf_startup_geometry, checks, status)
    return BoundaryValidationResult(
        scene=str(scene_path),
        spacing=spacing,
        requested_representations=tuple(requested_representations),
        built_representation_kinds=built_kinds,
        particle=particle_summary,
        sdf=sdf_summary,
        volume_map=vm_summary,
        sdf_startup_geometry=sdf_startup_geometry,
        source_checks=checks,
        status=status,
        passed=passed,
        summary_lines=summary_lines,
    )
