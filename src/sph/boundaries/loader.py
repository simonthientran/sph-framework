"""Scene-to-boundary-representation loading helpers."""

from __future__ import annotations

from pathlib import Path

from sph.boundaries.contracts import BoundaryRepresentation
from sph.boundaries.mesh_particles import load_mesh_particle_representation_from_scene
from sph.boundaries.mesh_sdf import load_mesh_sdf_representation_from_scene
from sph.boundaries.volume_map import load_volume_map_representation_from_scene


def load_boundary_representations_from_scene(
    scene: dict,
    scene_path: Path | None,
    spacing: float,
    support_radius: float | None = None,
) -> tuple[BoundaryRepresentation, ...]:
    """
    Load all supported boundary representations requested by the current scene.

    The returned tuple can contain multiple parallel representations for the
    same mesh sources, e.g. particle samples, SDF groundwork, and/or volume
    maps.

    Args:
        scene: Normalized scene dictionary.
        scene_path: Path to the scene JSON file (for resolving relative paths).
        spacing: Particle spacing.
        support_radius: SPH kernel support radius (needed for volume maps).
    """
    representations: list[BoundaryRepresentation] = []

    particle_representation = load_mesh_particle_representation_from_scene(
        scene=scene,
        scene_path=scene_path,
        spacing=spacing,
    )
    if particle_representation is not None:
        representations.append(particle_representation)

    sdf_representation = load_mesh_sdf_representation_from_scene(
        scene=scene,
        scene_path=scene_path,
        spacing=spacing,
    )
    if sdf_representation is not None:
        representations.append(sdf_representation)

    # Volume map: requires SDF-enabled boundaries + support_radius
    if support_radius is not None:
        volume_map_requested = _scene_requests_volume_map(scene)
        if volume_map_requested:
            vm_rep = load_volume_map_representation_from_scene(
                scene=scene,
                scene_path=scene_path,
                spacing=spacing,
                support_radius=support_radius,
            )
            if vm_rep is not None:
                representations.append(vm_rep)

    return tuple(representations)


def _scene_requests_volume_map(scene: dict) -> bool:
    """Check whether any boundary in the scene requests volume-map handling."""
    domain_method = scene.get("domain", {}).get("boundary_handling_method", "particle")
    if domain_method == "volume_map":
        return True
    for bnd in scene.get("boundaries", []):
        reps = {str(r).lower() for r in bnd.get("representations", [])}
        if "volume_map" in reps:
            return True
        if bnd.get("boundary_handling_method", "").lower() == "volume_map":
            return True
    return False


def load_boundary_representation_from_scene(
    scene: dict,
    scene_path: Path | None,
    spacing: float,
    support_radius: float | None = None,
) -> BoundaryRepresentation | None:
    """
    Load the best supported boundary representation for the current scene.

    Prefers particle-based mesh representation (active runtime path).
    Volume-map representations are loaded alongside when requested but do
    not replace the particle path unless the consumer explicitly selects them.
    """
    for representation in load_boundary_representations_from_scene(
        scene=scene,
        scene_path=scene_path,
        spacing=spacing,
        support_radius=support_radius,
    ):
        if representation.kind == "mesh_particles":
            return representation
    return None
