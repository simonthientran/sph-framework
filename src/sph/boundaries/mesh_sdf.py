"""
SDF boundary representation for mesh geometry.

This is preprocessing groundwork only. It carries mesh-derived SDF data behind
the generic boundary-representation seam but does not change runtime boundary
physics, which still use the particle bridge by default.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sph.boundaries.contracts import BoundaryRepresentation, BoundaryRepresentationSource
from sph.boundaries.mesh_particles import MeshBoundarySourceSpec
from sph.io.geometry_loader import compute_sdf_grid, load_mesh, transform_mesh


_DEFAULT_SDF_PADDING_RATIO = 0.05
_DEFAULT_SDF_CELLS_PER_PARTICLE_SPACING = 2.0
_DEFAULT_SDF_MIN_RESOLUTION = 16
_DEFAULT_SDF_MAX_RESOLUTION = 128
_MIN_ACCEPTABLE_SDF_RESOLUTION = 8
_MIN_RECOMMENDED_CELLS_PER_PARTICLE_SPACING = 1.0
_MAX_RECOMMENDED_CELLS_PER_PARTICLE_SPACING = 4.0


@dataclass(slots=True, frozen=True)
class MeshSDFStartupPolicy:
    """Startup wall-gap safeguard policy resolved for one mesh source."""

    warn_gap: float
    correction_enabled: bool
    correction_target_gap: float
    correction_max_shift: float
    velocity_projection_enabled: bool
    velocity_projection_gap: float
    validity_warn_inside_fraction: float
    validity_fail_inside_fraction: float
    validity_warn_normal_fraction: float
    validity_fail_normal_fraction: float
    validity_warn_too_close_fraction: float
    validity_fail_too_close_fraction: float
    runtime_fail_action: str


@dataclass(slots=True, frozen=True)
class MeshSDFRuntimePolicy:
    """Optional runtime wall-contact guard policy resolved for one mesh source."""

    wall_guard_enabled: bool
    wall_guard_min_gap: float
    wall_guard_target_gap: float
    wall_guard_max_shift: float
    wall_guard_project_outward_velocity: bool


@dataclass(slots=True, frozen=True)
class MeshSDFPreprocessingPolicy:
    """Scene-resolved SDF preprocessing policy for one mesh source."""

    resolution: int
    padding_ratio: float
    target_cell_size: float
    cells_per_particle_spacing: float
    resolution_source: str
    startup: MeshSDFStartupPolicy
    runtime: MeshSDFRuntimePolicy


@dataclass(slots=True, frozen=True)
class MeshSDFSourceSpec:
    """SDF preprocessing request for one mesh boundary source."""

    source: MeshBoundarySourceSpec
    policy: MeshSDFPreprocessingPolicy

    @property
    def source_name(self) -> str:
        return self.source.source_name

    def as_representation_source(self) -> BoundaryRepresentationSource:
        return self.source.as_representation_source()


@dataclass(slots=True, frozen=True)
class MeshSDFQualityReport:
    """Quality summary for one SDF grid."""

    is_watertight: bool
    mesh_bounds_min: np.ndarray
    mesh_bounds_max: np.ndarray
    cells_per_particle_spacing: float
    sdf_min: float
    sdf_max: float
    warnings: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def is_usable(self) -> bool:
        return len(self.errors) == 0


@dataclass(slots=True)
class MeshSDFGrid:
    """One mesh-derived SDF grid."""

    source: MeshSDFSourceSpec
    sdf_values: np.ndarray
    grid_min: np.ndarray
    grid_max: np.ndarray
    cell_size: float
    quality: MeshSDFQualityReport

    @property
    def resolution(self) -> int:
        return int(self.sdf_values.shape[0])

    def sample_signed_distance(self, points: np.ndarray) -> np.ndarray:
        """Trilinearly interpolate signed distance values at query points."""
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"mesh_sdf queries require shape (N, 3), got {points.shape}.")

        result = np.full(points.shape[0], np.nan, dtype=np.float64)
        if points.size == 0:
            return result

        resolution = self.resolution
        if resolution <= 1:
            result[:] = float(self.sdf_values.reshape(-1)[0])
            return result

        grid_span = self.grid_max - self.grid_min
        step = grid_span / float(resolution - 1)
        inside = np.all((points >= self.grid_min[None, :]) & (points <= self.grid_max[None, :]), axis=1)
        if not np.any(inside):
            return result

        local = (points[inside] - self.grid_min[None, :]) / step[None, :]
        i0 = np.floor(local).astype(np.int64)
        i0 = np.clip(i0, 0, resolution - 2)
        frac = np.clip(local - i0.astype(np.float64), 0.0, 1.0)
        i1 = i0 + 1

        grid = self.sdf_values
        c000 = grid[i0[:, 0], i0[:, 1], i0[:, 2]]
        c001 = grid[i0[:, 0], i0[:, 1], i1[:, 2]]
        c010 = grid[i0[:, 0], i1[:, 1], i0[:, 2]]
        c011 = grid[i0[:, 0], i1[:, 1], i1[:, 2]]
        c100 = grid[i1[:, 0], i0[:, 1], i0[:, 2]]
        c101 = grid[i1[:, 0], i0[:, 1], i1[:, 2]]
        c110 = grid[i1[:, 0], i1[:, 1], i0[:, 2]]
        c111 = grid[i1[:, 0], i1[:, 1], i1[:, 2]]

        tx = frac[:, 0]
        ty = frac[:, 1]
        tz = frac[:, 2]

        c00 = c000 * (1.0 - tx) + c100 * tx
        c01 = c001 * (1.0 - tx) + c101 * tx
        c10 = c010 * (1.0 - tx) + c110 * tx
        c11 = c011 * (1.0 - tx) + c111 * tx
        c0 = c00 * (1.0 - ty) + c10 * ty
        c1 = c01 * (1.0 - ty) + c11 * ty
        interpolated = c0 * (1.0 - tz) + c1 * tz

        result[inside] = interpolated.astype(np.float64)
        return result

    def sample_wall_normal(self, points: np.ndarray) -> np.ndarray:
        """
        Estimate wall normals from the SDF gradient.

        The normal follows the signed-distance gradient, i.e. from interior
        (negative SDF) toward exterior (positive SDF).
        """
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"mesh_sdf queries require shape (N, 3), got {points.shape}.")

        normals = np.full(points.shape, np.nan, dtype=np.float64)
        if points.size == 0:
            return normals

        eps = max(0.5 * float(self.cell_size), 1.0e-6)
        basis = np.eye(3, dtype=np.float64) * eps
        grad = np.zeros_like(points, dtype=np.float64)
        for axis in range(3):
            plus = self.sample_signed_distance(points + basis[axis][None, :])
            minus = self.sample_signed_distance(points - basis[axis][None, :])
            valid = np.isfinite(plus) & np.isfinite(minus)
            grad[valid, axis] = (plus[valid] - minus[valid]) / (2.0 * eps)
            grad[~valid, axis] = np.nan

        norm = np.linalg.norm(grad, axis=1)
        valid_norm = np.isfinite(norm) & (norm > 1.0e-12)
        normals[valid_norm] = grad[valid_norm] / norm[valid_norm, None]
        return normals


@dataclass(slots=True)
class MeshSDFBoundaryRepresentation(BoundaryRepresentation):
    """Mesh-derived signed-distance-field representation."""

    grids: tuple[MeshSDFGrid, ...]

    @property
    def kind(self) -> str:
        return "mesh_sdf"

    @property
    def sources(self) -> tuple[BoundaryRepresentationSource, ...]:
        return tuple(grid.source.as_representation_source() for grid in self.grids)

    @property
    def all_grids_usable(self) -> bool:
        return all(grid.quality.is_usable for grid in self.grids)

    def sample_signed_distance(self, points: np.ndarray) -> np.ndarray:
        """
        Query signed distance against the closest available SDF grid.

        For multiple mesh sources, this chooses the value with the smallest
        absolute distance among all valid grid queries.
        """
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"mesh_sdf queries require shape (N, 3), got {points.shape}.")

        if not self.grids:
            return np.full(points.shape[0], np.nan, dtype=np.float64)

        sampled = np.vstack([grid.sample_signed_distance(points) for grid in self.grids])
        abs_sampled = np.abs(sampled)
        abs_sampled[~np.isfinite(abs_sampled)] = np.inf
        best_idx = np.argmin(abs_sampled, axis=0)
        best = sampled[best_idx, np.arange(sampled.shape[1])]
        best[~np.isfinite(best)] = np.nan
        return best

    def sample_wall_normal(self, points: np.ndarray) -> np.ndarray:
        """Query wall normals from the closest usable SDF grid."""
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"mesh_sdf queries require shape (N, 3), got {points.shape}.")

        if not self.grids:
            return np.full(points.shape, np.nan, dtype=np.float64)

        sampled = np.vstack([grid.sample_signed_distance(points) for grid in self.grids])
        normal_stack = np.stack([grid.sample_wall_normal(points) for grid in self.grids], axis=0)
        abs_sampled = np.abs(sampled)
        abs_sampled[~np.isfinite(abs_sampled)] = np.inf
        best_idx = np.argmin(abs_sampled, axis=0)
        normals = normal_stack[best_idx, np.arange(points.shape[0])]
        invalid = ~np.isfinite(np.linalg.norm(normals, axis=1))
        normals[invalid] = np.nan
        return normals

    def sample_wall_distance(self, points: np.ndarray) -> np.ndarray:
        """Absolute wall distance derived from the current SDF query."""
        return np.abs(self.sample_signed_distance(points))

    def project_particle_positions(
        self,
        dim: int,
        spacing: float,
        deduplicate: bool = False,
    ) -> np.ndarray:
        raise RuntimeError(
            "mesh_sdf is preprocessing-only groundwork and cannot yet be projected "
            "into runtime boundary particles."
        )


def _resolve_sdf_policy(raw_cfg: dict, spacing: float, mesh_extent: float) -> MeshSDFPreprocessingPolicy:
    sdf_cfg = raw_cfg.get("sdf", {})
    startup_cfg = sdf_cfg.get("startup_correction", {})
    runtime_cfg = sdf_cfg.get("runtime_wall_guard", {})

    padding_ratio = float(sdf_cfg.get("padding_ratio", raw_cfg.get("sdf_padding_ratio", _DEFAULT_SDF_PADDING_RATIO)))
    if padding_ratio < 0.0:
        raise ValueError(f"SDF padding_ratio must be non-negative (got {padding_ratio}).")

    explicit_resolution = sdf_cfg.get("resolution", raw_cfg.get("sdf_resolution"))
    if explicit_resolution is not None:
        resolution = int(explicit_resolution)
        resolution_source = "explicit"
    else:
        target_cell_size = float(spacing) / _DEFAULT_SDF_CELLS_PER_PARTICLE_SPACING
        padded_extent = mesh_extent * (1.0 + 2.0 * padding_ratio)
        resolution = int(np.ceil(padded_extent / max(target_cell_size, 1.0e-12)))
        resolution = max(_DEFAULT_SDF_MIN_RESOLUTION, min(_DEFAULT_SDF_MAX_RESOLUTION, resolution))
        resolution_source = "spacing_default"

    if resolution <= 0:
        raise ValueError(f"SDF resolution must be positive (got {resolution}).")

    padded_extent = mesh_extent * (1.0 + 2.0 * padding_ratio)
    target_cell_size = padded_extent / float(resolution)
    cells_per_particle_spacing = float(spacing) / max(target_cell_size, 1.0e-12)
    startup_warn_gap = float(sdf_cfg.get("startup_warn_gap", 0.25 * spacing))
    correction_enabled = bool(startup_cfg.get("enable", False))
    correction_target_gap = float(startup_cfg.get("target_gap", spacing))
    correction_max_shift = float(startup_cfg.get("max_shift", 0.5 * spacing))
    velocity_projection_enabled = bool(startup_cfg.get("project_outward_velocity", False))
    velocity_projection_gap = float(startup_cfg.get("velocity_projection_gap", startup_warn_gap))
    validity_cfg = sdf_cfg.get("startup_validity", {})
    validity_warn_inside_fraction = float(validity_cfg.get("warn_inside_fraction", 0.99))
    validity_fail_inside_fraction = float(validity_cfg.get("fail_inside_fraction", 0.95))
    validity_warn_normal_fraction = float(validity_cfg.get("warn_normal_fraction", 0.999))
    validity_fail_normal_fraction = float(validity_cfg.get("fail_normal_fraction", 0.95))
    validity_warn_too_close_fraction = float(validity_cfg.get("warn_too_close_fraction", 0.0))
    validity_fail_too_close_fraction = float(validity_cfg.get("fail_too_close_fraction", 0.10))
    runtime_fail_action = str(validity_cfg.get("runtime_fail_action", "warn")).lower()
    wall_guard_enabled = bool(runtime_cfg.get("enable", False))
    wall_guard_min_gap = float(runtime_cfg.get("min_gap", startup_warn_gap))
    wall_guard_target_gap = float(runtime_cfg.get("target_gap", max(wall_guard_min_gap, spacing)))
    wall_guard_max_shift = float(runtime_cfg.get("max_shift", 0.5 * spacing))
    wall_guard_project_outward_velocity = bool(runtime_cfg.get("project_outward_velocity", True))

    if startup_warn_gap < 0.0:
        raise ValueError(f"SDF startup_warn_gap must be non-negative (got {startup_warn_gap}).")
    if correction_target_gap < 0.0:
        raise ValueError(
            f"SDF startup correction target_gap must be non-negative (got {correction_target_gap})."
        )
    if correction_max_shift < 0.0:
        raise ValueError(
            f"SDF startup correction max_shift must be non-negative (got {correction_max_shift})."
        )
    if velocity_projection_gap < 0.0:
        raise ValueError(
            f"SDF startup velocity_projection_gap must be non-negative (got {velocity_projection_gap})."
        )
    if runtime_fail_action not in {"warn", "raise"}:
        raise ValueError(
            f"SDF startup runtime_fail_action must be 'warn' or 'raise' (got {runtime_fail_action})."
        )
    if wall_guard_min_gap < 0.0:
        raise ValueError(
            f"SDF runtime wall_guard min_gap must be non-negative (got {wall_guard_min_gap})."
        )
    if wall_guard_target_gap < wall_guard_min_gap:
        raise ValueError(
            "SDF runtime wall_guard target_gap must be greater than or equal to min_gap "
            f"(got target_gap={wall_guard_target_gap}, min_gap={wall_guard_min_gap})."
        )
    if wall_guard_max_shift < 0.0:
        raise ValueError(
            f"SDF runtime wall_guard max_shift must be non-negative (got {wall_guard_max_shift})."
        )

    return MeshSDFPreprocessingPolicy(
        resolution=resolution,
        padding_ratio=padding_ratio,
        target_cell_size=float(target_cell_size),
        cells_per_particle_spacing=float(cells_per_particle_spacing),
        resolution_source=resolution_source,
        startup=MeshSDFStartupPolicy(
            warn_gap=startup_warn_gap,
            correction_enabled=correction_enabled,
            correction_target_gap=correction_target_gap,
            correction_max_shift=correction_max_shift,
            velocity_projection_enabled=velocity_projection_enabled,
            velocity_projection_gap=velocity_projection_gap,
            validity_warn_inside_fraction=validity_warn_inside_fraction,
            validity_fail_inside_fraction=validity_fail_inside_fraction,
            validity_warn_normal_fraction=validity_warn_normal_fraction,
            validity_fail_normal_fraction=validity_fail_normal_fraction,
            validity_warn_too_close_fraction=validity_warn_too_close_fraction,
            validity_fail_too_close_fraction=validity_fail_too_close_fraction,
            runtime_fail_action=runtime_fail_action,
        ),
        runtime=MeshSDFRuntimePolicy(
            wall_guard_enabled=wall_guard_enabled,
            wall_guard_min_gap=wall_guard_min_gap,
            wall_guard_target_gap=wall_guard_target_gap,
            wall_guard_max_shift=wall_guard_max_shift,
            wall_guard_project_outward_velocity=wall_guard_project_outward_velocity,
        ),
    )


def parse_mesh_sdf_specs(
    scene: dict,
    scene_path: Path | None,
    spacing: float,
) -> list[MeshSDFSourceSpec]:
    """
    Parse optional SDF preprocessing requests for mesh boundaries.

    Supported boundary config forms:
    - ``"sdf": {"enable": true, "resolution": 48, "padding_ratio": 0.05}``
    - ``"representations": ["particles", "sdf"]`` plus optional
      ``"sdf_resolution": 48`` / ``"sdf_padding_ratio": 0.05``

    If resolution is omitted, it is derived from the current particle spacing so
    the default SDF grid cell size tracks the runtime discretization.
    """
    scene_dir = scene_path.parent if scene_path is not None else Path.cwd()
    specs: list[MeshSDFSourceSpec] = []
    for raw_cfg in scene.get("boundaries", []):
        boundary_type = str(raw_cfg.get("type", "")).lower()
        if boundary_type not in {"stl", "obj", "mesh"}:
            continue

        sdf_cfg = raw_cfg.get("sdf", {})
        rep_list = {str(item).lower() for item in raw_cfg.get("representations", [])}
        enable_sdf = bool(sdf_cfg.get("enable", False) or ("sdf" in rep_list))
        if not enable_sdf:
            continue

        mesh_path = Path(str(raw_cfg["file"]))
        if not mesh_path.is_absolute():
            mesh_path = (scene_dir / mesh_path).resolve()

        mesh = transform_mesh(
            load_mesh(str(mesh_path)),
            translation=np.asarray(raw_cfg.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64),
            rotation_axis=np.asarray(raw_cfg.get("rotation_axis", [0.0, 1.0, 0.0]), dtype=np.float64),
            rotation_angle_degrees=float(raw_cfg.get("rotation_angle", 0.0)),
            scale=np.asarray(raw_cfg.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64),
        )
        mesh_extent = float((mesh.bounds[1] - mesh.bounds[0]).max())
        policy = _resolve_sdf_policy(raw_cfg, spacing=spacing, mesh_extent=mesh_extent)
        specs.append(
            MeshSDFSourceSpec(
                source=MeshBoundarySourceSpec(
                    boundary_type=boundary_type,
                    mesh_path=mesh_path,
                    sampling=str(raw_cfg.get("sampling", "poisson")).lower(),
                    n_layers=int(raw_cfg.get("n_layers", 3)),
                    translation=np.asarray(raw_cfg.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64),
                    rotation_axis=np.asarray(raw_cfg.get("rotation_axis", [0.0, 1.0, 0.0]), dtype=np.float64),
                    rotation_angle=float(raw_cfg.get("rotation_angle", 0.0)),
                    scale=np.asarray(raw_cfg.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64),
                ),
                policy=policy,
            )
        )
    return specs


def _build_mesh_sdf_quality_report(
    mesh,
    spec: MeshSDFSourceSpec,
    sdf_values: np.ndarray,
    grid_min: np.ndarray,
    grid_max: np.ndarray,
    cell_size: float,
) -> MeshSDFQualityReport:
    del grid_min
    del grid_max

    warnings: list[str] = []
    errors: list[str] = []

    if spec.policy.resolution < _MIN_ACCEPTABLE_SDF_RESOLUTION:
        errors.append(
            f"resolution={spec.policy.resolution} is below the minimum acceptable "
            f"SDF resolution {_MIN_ACCEPTABLE_SDF_RESOLUTION}"
        )

    cells_per_spacing = float(spec.policy.cells_per_particle_spacing)
    if cells_per_spacing < _MIN_RECOMMENDED_CELLS_PER_PARTICLE_SPACING:
        warnings.append(
            f"cell_size={cell_size:.5f} is coarser than particle spacing={cells_per_spacing:.2f} "
            "cells per particle spacing"
        )
    if cells_per_spacing > _MAX_RECOMMENDED_CELLS_PER_PARTICLE_SPACING:
        warnings.append(
            f"cell_size={cell_size:.5f} is much finer than particle spacing={cells_per_spacing:.2f} "
            "cells per particle spacing"
        )

    sdf_min = float(np.min(sdf_values))
    sdf_max = float(np.max(sdf_values))
    if not np.isfinite(sdf_min) or not np.isfinite(sdf_max):
        errors.append("SDF grid contains non-finite values")
    if sdf_min >= 0.0:
        errors.append("SDF grid does not contain any interior samples (min >= 0)")
    if sdf_max <= 0.0:
        errors.append("SDF grid does not contain any exterior samples (max <= 0)")
    if not bool(mesh.is_watertight):
        warnings.append("mesh is not watertight; signed distances may be unreliable")

    return MeshSDFQualityReport(
        is_watertight=bool(mesh.is_watertight),
        mesh_bounds_min=np.asarray(mesh.bounds[0], dtype=np.float64),
        mesh_bounds_max=np.asarray(mesh.bounds[1], dtype=np.float64),
        cells_per_particle_spacing=cells_per_spacing,
        sdf_min=sdf_min,
        sdf_max=sdf_max,
        warnings=tuple(warnings),
        errors=tuple(errors),
    )


def _build_mesh_sdf_grid(spec: MeshSDFSourceSpec) -> MeshSDFGrid:
    mesh = transform_mesh(
        load_mesh(str(spec.source.mesh_path)),
        translation=spec.source.translation,
        rotation_axis=spec.source.rotation_axis,
        rotation_angle_degrees=spec.source.rotation_angle,
        scale=spec.source.scale,
    )
    sdf_values, grid_min, grid_max, cell_size = compute_sdf_grid(
        mesh,
        resolution=spec.policy.resolution,
        padding_ratio=spec.policy.padding_ratio,
    )
    quality = _build_mesh_sdf_quality_report(
        mesh=mesh,
        spec=spec,
        sdf_values=sdf_values,
        grid_min=grid_min,
        grid_max=grid_max,
        cell_size=cell_size,
    )
    return MeshSDFGrid(
        source=spec,
        sdf_values=np.asarray(sdf_values, dtype=np.float32),
        grid_min=np.asarray(grid_min, dtype=np.float64),
        grid_max=np.asarray(grid_max, dtype=np.float64),
        cell_size=float(cell_size),
        quality=quality,
    )


def load_mesh_sdf_representation_from_scene(
    scene: dict,
    scene_path: Path | None,
    spacing: float,
) -> MeshSDFBoundaryRepresentation | None:
    """Build the mesh-derived SDF boundary representation when explicitly requested."""
    specs = parse_mesh_sdf_specs(scene, scene_path, spacing=spacing)
    if not specs:
        return None

    grids = tuple(_build_mesh_sdf_grid(spec) for spec in specs)
    for grid in grids:
        policy = grid.source.policy
        print(
            "Boundary representation source: "
            f"{grid.source.source.boundary_type} '{grid.source.source_name}' -> "
            f"mesh_sdf(resolution={grid.resolution}, "
            f"resolution_source={policy.resolution_source}, "
            f"padding_ratio={policy.padding_ratio:.3f}, "
            f"cell_size={grid.cell_size:.5f})"
        )
        for warning in grid.quality.warnings:
            print(f"  mesh_sdf warning: {warning}")
        if not grid.quality.is_usable:
            errors = "; ".join(grid.quality.errors)
            raise ValueError(f"mesh_sdf for '{grid.source.source_name}' is not usable: {errors}")

    return MeshSDFBoundaryRepresentation(grids=grids)
