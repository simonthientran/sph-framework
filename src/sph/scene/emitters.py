"""Scene-level emitter and inflow helpers for the active runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_AXIS_NAMES = ("x", "y", "z")


@dataclass(slots=True, frozen=True)
class BoxEmitterSpec:
    """One scene-authored box emitter."""

    name: str
    pmin: np.ndarray
    pmax: np.ndarray
    velocity: np.ndarray
    interval_steps: int
    start_step: int
    end_step: int | None
    max_particles: int | None
    min_clearance: float
    max_active_fluid_particles: int | None

    @property
    def dim(self) -> int:
        return int(self.pmin.shape[0])

    @property
    def dominant_axis(self) -> int:
        return int(np.argmax(np.abs(self.velocity)))


@dataclass(slots=True)
class BoxEmitterState:
    """Runtime state for one box emitter."""

    spec: BoxEmitterSpec
    emitted_particles: int = 0

    def is_active_on_step(self, step: int) -> bool:
        if step < self.spec.start_step:
            return False
        if self.spec.end_step is not None and step > self.spec.end_step:
            return False
        if self.spec.max_particles is not None and self.emitted_particles >= self.spec.max_particles:
            return False
        return (step - self.spec.start_step) % max(self.spec.interval_steps, 1) == 0


def _array_from_min_max(block: dict, dim: int) -> tuple[np.ndarray, np.ndarray] | None:
    if "min" in block and "max" in block:
        return (
            np.array(block["min"], dtype=np.float64)[:dim],
            np.array(block["max"], dtype=np.float64)[:dim],
        )
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


def load_box_emitters_from_scene(scene: dict, scene_path: Path | None, dim: int) -> tuple[BoxEmitterState, ...]:
    """Parse optional box emitters from the scene."""
    del scene_path
    emitters: list[BoxEmitterState] = []
    for index, cfg in enumerate(scene.get("emitters", [])):
        emitter_type = str(cfg.get("type", "box")).lower()
        if emitter_type not in {"box", "inlet"}:
            raise ValueError(f"Unsupported emitter type '{emitter_type}'.")
        bounds = _array_from_min_max(cfg, dim)
        if bounds is None:
            raise KeyError(f"Emitter {index} must define min/max bounds.")
        pmin, pmax = bounds
        velocity = np.array(cfg.get("velocity", [0.0] * dim), dtype=np.float64)[:dim]
        if velocity.shape[0] != dim:
            raise ValueError(f"Emitter {index} velocity must have {dim} components.")
        interval_steps = int(cfg.get("interval_steps", 1))
        if interval_steps <= 0:
            raise ValueError(f"Emitter {index} interval_steps must be positive.")
        max_particles_cfg = cfg.get("max_particles")
        max_particles = int(max_particles_cfg) if max_particles_cfg is not None else None
        min_clearance = float(cfg.get("min_clearance", 0.8))
        if min_clearance < 0.0:
            raise ValueError(f"Emitter {index} min_clearance must be non-negative.")
        emitters.append(
            BoxEmitterState(
                spec=BoxEmitterSpec(
                    name=str(cfg.get("name", f"emitter_{index}")),
                    pmin=pmin,
                    pmax=pmax,
                    velocity=velocity,
                    interval_steps=interval_steps,
                    start_step=int(cfg.get("start_step", 0)),
                    end_step=int(cfg["end_step"]) if cfg.get("end_step") is not None else None,
                    max_particles=max_particles,
                    min_clearance=min_clearance,
                    max_active_fluid_particles=(
                        int(cfg["max_active_fluid_particles"])
                        if cfg.get("max_active_fluid_particles") is not None
                        else None
                    ),
                )
            )
        )
    return tuple(emitters)


def build_emitter_slab_positions(spec: BoxEmitterSpec, spacing: float) -> np.ndarray:
    """Build one spacing-thick slab of particles for a box emitter."""
    axis = spec.dominant_axis
    if abs(spec.velocity[axis]) < 1.0e-12:
        raise ValueError(f"Emitter '{spec.name}' requires non-zero velocity along at least one axis.")

    coords: list[np.ndarray] = []
    for dim_idx in range(spec.dim):
        if dim_idx == axis:
            face = spec.pmin[dim_idx] if spec.velocity[dim_idx] >= 0.0 else spec.pmax[dim_idx]
            coords.append(np.array([face], dtype=np.float64))
        else:
            coords.append(np.arange(spec.pmin[dim_idx], spec.pmax[dim_idx], spacing, dtype=np.float64))

    if any(axis_coords.size == 0 for axis_coords in coords):
        return np.zeros((0, spec.dim), dtype=np.float64)

    mesh = np.meshgrid(*coords, indexing="ij")
    return np.stack([axis_values.ravel() for axis_values in mesh], axis=1).astype(np.float64)
