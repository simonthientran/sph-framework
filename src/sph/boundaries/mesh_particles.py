"""
Particle-boundary representation for mesh geometry.

This module is intentionally narrower than ``sph.io.geometry_loader``:
- ``sph.io.geometry_loader`` handles mesh import and preprocessing primitives
- this module turns scene boundary entries into the current runtime boundary
  representation: sampled boundary particles

That separation keeps the current particle-based path working while making it
explicit that mesh import and boundary physics representation are different
concerns.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sph.boundaries.contracts import BoundaryRepresentation, BoundaryRepresentationSource
from sph.io.geometry_loader import load_mesh, sample_surface_particles, sample_surface_poisson, transform_mesh


@dataclass(slots=True)
class MeshBoundarySourceSpec:
    """Scene-level mesh boundary source resolved to a concrete mesh file."""

    boundary_type: str
    mesh_path: Path
    sampling: str
    n_layers: int
    translation: np.ndarray
    rotation_axis: np.ndarray
    rotation_angle: float
    scale: np.ndarray

    @property
    def source_name(self) -> str:
        return self.mesh_path.name

    def as_representation_source(self) -> BoundaryRepresentationSource:
        return BoundaryRepresentationSource(
            boundary_type=self.boundary_type,
            source_name=self.source_name,
        )


@dataclass(slots=True)
class ParticleBoundarySamples:
    """Current mesh-boundary runtime representation: sampled surface particles."""

    source: MeshBoundarySourceSpec
    positions: np.ndarray
    normals: np.ndarray

    @property
    def count(self) -> int:
        return int(self.positions.shape[0])

    def positions_for_dim(self, dim: int) -> np.ndarray:
        if dim == self.positions.shape[1]:
            return self.positions
        if dim == 2 and self.positions.shape[1] == 3:
            return self.positions[:, :2]
        raise ValueError(
            f"Cannot project boundary samples of shape {self.positions.shape} into dim={dim}."
        )


@dataclass(slots=True)
class MeshParticleBoundaryRepresentation(BoundaryRepresentation):
    """Particle-sampled mesh boundary as one boundary-representation implementation."""

    samples: tuple[ParticleBoundarySamples, ...]

    @property
    def kind(self) -> str:
        return "mesh_particles"

    @property
    def sources(self) -> tuple[BoundaryRepresentationSource, ...]:
        return tuple(sample.source.as_representation_source() for sample in self.samples)

    @property
    def count(self) -> int:
        return sum(sample.count for sample in self.samples)

    def project_particle_positions(
        self,
        dim: int,
        spacing: float,
        deduplicate: bool = False,
    ) -> np.ndarray:
        projected = [np.asarray(sample.positions_for_dim(dim), dtype=np.float64) for sample in self.samples]
        combined = np.vstack(projected)
        if deduplicate:
            combined = _deduplicate_positions(combined, spacing)
        return combined


def parse_mesh_boundary_specs(
    scene: dict,
    scene_path: Path | None,
) -> list[MeshBoundarySourceSpec]:
    """Parse scene ``boundaries`` entries that resolve to mesh geometry."""

    scene_dir = scene_path.parent if scene_path is not None else Path.cwd()
    specs: list[MeshBoundarySourceSpec] = []
    for cfg in scene.get("boundaries", []):
        boundary_type = str(cfg.get("type", "")).lower()
        if boundary_type not in {"stl", "obj", "mesh"}:
            continue
        mesh_path = Path(str(cfg["file"]))
        if not mesh_path.is_absolute():
            mesh_path = (scene_dir / mesh_path).resolve()
        specs.append(
            MeshBoundarySourceSpec(
                boundary_type=boundary_type,
                mesh_path=mesh_path,
                sampling=str(cfg.get("sampling", "poisson")).lower(),
                n_layers=int(cfg.get("n_layers", 3)),
                translation=np.asarray(cfg.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64),
                rotation_axis=np.asarray(cfg.get("rotation_axis", [0.0, 1.0, 0.0]), dtype=np.float64),
                rotation_angle=float(cfg.get("rotation_angle", 0.0)),
                scale=np.asarray(cfg.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64),
            )
        )
    return specs


def _sample_particle_boundary(spec: MeshBoundarySourceSpec, spacing: float) -> ParticleBoundarySamples:
    mesh = transform_mesh(
        load_mesh(str(spec.mesh_path)),
        translation=spec.translation,
        rotation_axis=spec.rotation_axis,
        rotation_angle_degrees=spec.rotation_angle,
        scale=spec.scale,
    )
    if spec.sampling == "poisson":
        positions, normals = sample_surface_poisson(mesh, spacing, n_layers=spec.n_layers)
    else:
        positions, normals = sample_surface_particles(mesh, spacing, n_layers=spec.n_layers)
    return ParticleBoundarySamples(
        source=spec,
        positions=np.asarray(positions, dtype=np.float64),
        normals=np.asarray(normals, dtype=np.float64),
    )


def _deduplicate_positions(positions: np.ndarray, spacing: float) -> np.ndarray:
    if positions.size == 0:
        return positions
    rounded = np.round(positions / spacing).astype(np.int64)
    _, uniq_idx = np.unique(rounded, axis=0, return_index=True)
    return positions[np.sort(uniq_idx)]


def load_mesh_particle_representation_from_scene(
    scene: dict,
    scene_path: Path | None,
    spacing: float,
    dim: int | None = None,
) -> MeshParticleBoundaryRepresentation | None:
    """
    Build the particle-based mesh boundary representation from mesh scene entries.

    This is intentionally the current bridge representation only. Future SDF or
    volume-map boundaries should enter beside this path, not through it.
    """
    specs = parse_mesh_boundary_specs(scene, scene_path)
    if not specs:
        return None

    samples = tuple(_sample_particle_boundary(spec, spacing) for spec in specs)
    if dim is not None:
        for sample in samples:
            positions = sample.positions_for_dim(dim)
            print(
                f"Mesh boundary '{sample.source.source_name}': {len(positions)} particles "
                f"(sampling={sample.source.sampling}, layers={sample.source.n_layers})"
            )
        combined = MeshParticleBoundaryRepresentation(samples=samples).project_particle_positions(
            dim=dim,
            spacing=spacing,
            deduplicate=False,
        )
        print(f"Boundary particles total: {len(combined)}")
    return MeshParticleBoundaryRepresentation(samples=samples)


def load_particle_boundaries_from_scene(
    scene: dict,
    scene_path: Path | None,
    spacing: float,
    dim: int,
    deduplicate: bool = False,
) -> MeshParticleBoundaryRepresentation | None:
    """
    Backward-compatible helper for callers that still want a concrete particle
    representation from scene data.
    """
    representation = load_mesh_particle_representation_from_scene(
        scene=scene,
        scene_path=scene_path,
        spacing=spacing,
        dim=dim,
    )
    if representation is None:
        return None
    if deduplicate:
        # Trigger projection once so callers preserve previous side effects and
        # semantics when they explicitly ask for deduplication.
        representation.project_particle_positions(dim=dim, spacing=spacing, deduplicate=True)
    return representation
