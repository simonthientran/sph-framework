"""Scene-level fluid source loading for runtime particle layouts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sph.io.geometry_loader import compute_sdf_grid, fill_volume_from_sdf, load_mesh

_AXIS_NAMES = ("x", "y", "z")


@dataclass(slots=True, frozen=True)
class FluidParticleSource:
    """One scene-authored fluid source resolved to particles."""

    source_type: str
    source_name: str
    positions: np.ndarray
    initial_velocity: np.ndarray

    @property
    def count(self) -> int:
        return int(self.positions.shape[0])


@dataclass(slots=True, frozen=True)
class FluidParticleLayout:
    """Combined fluid particle layout used by the active runtime."""

    spacing: float
    dim: int
    sources: tuple[FluidParticleSource, ...]
    positions: np.ndarray
    velocities: np.ndarray

    @property
    def bounds_min(self) -> np.ndarray:
        if self.positions.size == 0:
            return np.zeros(self.dim, dtype=np.float64)
        return np.min(self.positions, axis=0)

    @property
    def bounds_max(self) -> np.ndarray:
        if self.positions.size == 0:
            return np.zeros(self.dim, dtype=np.float64)
        return np.max(self.positions, axis=0)


def _array_from_min_max(block: dict, dim: int) -> tuple[np.ndarray, np.ndarray] | None:
    if "min" in block and "max" in block:
        vmin = np.array(block["min"], dtype=np.float64)
        vmax = np.array(block["max"], dtype=np.float64)
        return vmin[:dim], vmax[:dim]

    mins = []
    maxs = []
    for axis in _AXIS_NAMES[:dim]:
        key_min = f"{axis}_min"
        key_max = f"{axis}_max"
        if key_min not in block or key_max not in block:
            return None
        mins.append(float(block[key_min]))
        maxs.append(float(block[key_max]))
    return np.array(mins, dtype=np.float64), np.array(maxs, dtype=np.float64)


def _vector_from_cfg(cfg: dict, dim: int) -> np.ndarray:
    vec = cfg.get("initial_velocity", cfg.get("velocity", [0.0] * dim))
    arr = np.array(vec, dtype=np.float64)
    if arr.shape[0] < dim:
        raise ValueError(f"initial_velocity must provide at least {dim} components (got {arr.shape[0]}).")
    return arr[:dim]


def resolve_scene_particle_spacing(scene: dict) -> float:
    """Resolve the single runtime particle spacing used by all fluid sources."""
    fluid_cfg = scene.get("fluid")
    if isinstance(fluid_cfg, dict) and "spacing" in fluid_cfg:
        return float(fluid_cfg["spacing"])

    fluids_cfg = scene.get("fluids", [])
    if isinstance(fluids_cfg, list) and fluids_cfg:
        spacings = [float(cfg.get("spacing")) for cfg in fluids_cfg if "spacing" in cfg]
        if not spacings:
            raise KeyError("Scene must define particle spacing in fluid.spacing or every fluids[] entry.")
        first = spacings[0]
        for spacing in spacings[1:]:
            if abs(spacing - first) > 1.0e-12:
                raise ValueError("All fluids[] entries must use the same spacing in the current runtime.")
        return first

    raise KeyError("Scene must define fluid particle spacing.")


def _grid_points_block(pmin: np.ndarray, pmax: np.ndarray, spacing: float, dim: int) -> np.ndarray:
    if dim == 2:
        x = np.arange(pmin[0], pmax[0], spacing, dtype=np.float64)
        y = np.arange(pmin[1], pmax[1], spacing, dtype=np.float64)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        return np.column_stack([xx.ravel(), yy.ravel()])

    x = np.arange(pmin[0], pmax[0], spacing, dtype=np.float64)
    y = np.arange(pmin[1], pmax[1], spacing, dtype=np.float64)
    z = np.arange(pmin[2], pmax[2], spacing, dtype=np.float64)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


def _resolve_mesh_path(raw_path: str, scene_path: Path | None) -> Path:
    mesh_path = Path(str(raw_path))
    if mesh_path.is_absolute():
        return mesh_path
    scene_dir = scene_path.parent if scene_path is not None else Path.cwd()
    return (scene_dir / mesh_path).resolve()


def _build_block_source(cfg: dict, spacing: float, dim: int, default_name: str) -> FluidParticleSource:
    bounds = _array_from_min_max(cfg, dim)
    if bounds is None:
        raise KeyError(f"Block fluid source '{default_name}' must define min/max bounds.")
    pmin, pmax = bounds
    positions = _grid_points_block(pmin, pmax, spacing=spacing, dim=dim)
    velocity = _vector_from_cfg(cfg, dim)
    return FluidParticleSource(
        source_type="block",
        source_name=str(cfg.get("name", default_name)),
        positions=np.asarray(positions, dtype=np.float64),
        initial_velocity=velocity,
    )


def _build_cylinder_source(cfg: dict, spacing: float, dim: int, default_name: str) -> FluidParticleSource:
    """Generate fluid particles in a cylinder (disk cross-section clipped from a box grid)."""
    if dim != 3:
        raise ValueError("Cylinder fluid sources require dim=3.")
    bounds = _array_from_min_max(cfg, dim)
    if bounds is None:
        raise KeyError(f"Cylinder fluid source '{default_name}' must define min/max bounds.")
    pmin, pmax = bounds
    positions = _grid_points_block(pmin, pmax, spacing=spacing, dim=dim)

    axis_center = np.array(cfg.get("axis_center", [0.0, 0.0]), dtype=np.float64)
    radius = float(cfg["radius"])

    r = np.sqrt((positions[:, 1] - axis_center[0]) ** 2 + (positions[:, 2] - axis_center[1]) ** 2)
    positions = positions[r < radius]

    velocity = _vector_from_cfg(cfg, dim)
    return FluidParticleSource(
        source_type="cylinder",
        source_name=str(cfg.get("name", default_name)),
        positions=np.asarray(positions, dtype=np.float64),
        initial_velocity=velocity,
    )


def _build_mesh_fill_source(
    cfg: dict,
    spacing: float,
    dim: int,
    scene_path: Path | None,
    default_name: str,
) -> FluidParticleSource:
    if dim != 3:
        raise ValueError("mesh_fill fluid sources currently require dim=3.")
    mesh_path = _resolve_mesh_path(str(cfg["file"]), scene_path)
    mesh = load_mesh(str(mesh_path))
    resolution = int(cfg.get("sdf_resolution", cfg.get("resolution", 32)))
    padding_ratio = float(cfg.get("sdf_padding_ratio", cfg.get("padding_ratio", 0.05)))
    inside_threshold = float(cfg.get("inside_threshold", 0.0))
    sdf_values, grid_min, grid_max, _ = compute_sdf_grid(
        mesh,
        resolution=resolution,
        padding_ratio=padding_ratio,
    )
    positions = fill_volume_from_sdf(
        sdf_values,
        grid_min,
        grid_max,
        dx=spacing,
        inside_threshold=inside_threshold,
    )
    velocity = _vector_from_cfg(cfg, dim)
    return FluidParticleSource(
        source_type="mesh_fill",
        source_name=str(cfg.get("name", mesh_path.name if mesh_path.name else default_name)),
        positions=np.asarray(positions, dtype=np.float64),
        initial_velocity=velocity,
    )


def _deduplicate_particles(
    positions: np.ndarray,
    velocities: np.ndarray,
    spacing: float,
) -> tuple[np.ndarray, np.ndarray]:
    if positions.size == 0:
        return positions, velocities
    rounded = np.round(positions / spacing).astype(np.int64)
    _, uniq_idx = np.unique(rounded, axis=0, return_index=True)
    uniq_idx = np.sort(uniq_idx)
    return positions[uniq_idx], velocities[uniq_idx]


def load_fluid_particle_layout_from_scene(
    scene: dict,
    scene_path: Path | None,
    dim: int,
) -> FluidParticleLayout:
    """
    Load runtime fluid particles from scene-authored sources.

    Supported forms:
    - legacy single source: ``scene["fluid"]``
    - multi-source scene: ``scene["fluids"]``
      - ``type: "block"`` or ``"box"``
      - ``type: "mesh_fill"`` or ``"mesh_volume"``
    """
    spacing = resolve_scene_particle_spacing(scene)
    raw_sources = scene.get("fluids")
    sources: list[FluidParticleSource] = []

    if isinstance(raw_sources, list) and raw_sources:
        for index, cfg in enumerate(raw_sources):
            source_type = str(cfg.get("type", "block")).lower()
            if "spacing" in cfg and abs(float(cfg["spacing"]) - spacing) > 1.0e-12:
                raise ValueError("All fluid sources must use the same spacing in the current runtime.")
            default_name = f"fluid_{index}"
            if source_type in {"block", "box"}:
                sources.append(_build_block_source(cfg, spacing=spacing, dim=dim, default_name=default_name))
            elif source_type in {"cylinder", "disk"}:
                sources.append(_build_cylinder_source(cfg, spacing=spacing, dim=dim, default_name=default_name))
            elif source_type in {"mesh_fill", "mesh_volume"}:
                sources.append(
                    _build_mesh_fill_source(
                        cfg,
                        spacing=spacing,
                        dim=dim,
                        scene_path=scene_path,
                        default_name=default_name,
                    )
                )
            else:
                raise ValueError(f"Unsupported fluid source type '{source_type}'.")
    else:
        fluid_cfg = scene.get("fluid")
        if not isinstance(fluid_cfg, dict):
            raise KeyError("Scene must define either fluid or fluids[].")
        sources.append(_build_block_source(fluid_cfg, spacing=spacing, dim=dim, default_name="fluid"))

    position_blocks: list[np.ndarray] = []
    velocity_blocks: list[np.ndarray] = []
    for source in sources:
        if source.positions.size == 0:
            continue
        position_blocks.append(source.positions)
        velocity_blocks.append(np.repeat(source.initial_velocity[None, :], source.count, axis=0))

    if position_blocks:
        positions = np.vstack(position_blocks)
        velocities = np.vstack(velocity_blocks)
        if len(position_blocks) > 1:
            positions, velocities = _deduplicate_particles(positions, velocities, spacing=spacing)
    else:
        positions = np.zeros((0, dim), dtype=np.float64)
        velocities = np.zeros((0, dim), dtype=np.float64)

    return FluidParticleLayout(
        spacing=spacing,
        dim=dim,
        sources=tuple(sources),
        positions=positions,
        velocities=velocities,
    )
