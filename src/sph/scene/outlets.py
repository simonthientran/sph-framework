"""Scene-level outlet and outflow helpers for the active runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_AXIS_NAMES = ("x", "y", "z")


@dataclass(slots=True, frozen=True)
class BoxOutletSpec:
    """One scene-authored box outlet."""

    name: str
    pmin: np.ndarray
    pmax: np.ndarray
    direction: np.ndarray | None

    @property
    def dim(self) -> int:
        return int(self.pmin.shape[0])


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


def load_box_outlets_from_scene(scene: dict, scene_path: Path | None, dim: int) -> tuple[BoxOutletSpec, ...]:
    """Parse optional box outlets from the scene."""
    del scene_path
    outlets: list[BoxOutletSpec] = []
    for index, cfg in enumerate(scene.get("outlets", [])):
        outlet_type = str(cfg.get("type", "box")).lower()
        if outlet_type not in {"box", "outlet", "outflow"}:
            raise ValueError(f"Unsupported outlet type '{outlet_type}'.")
        bounds = _array_from_min_max(cfg, dim)
        if bounds is None:
            raise KeyError(f"Outlet {index} must define min/max bounds.")
        pmin, pmax = bounds
        if np.any(pmax < pmin):
            raise ValueError(f"Outlet {index} max bounds must be >= min bounds.")

        direction = None
        if cfg.get("direction") is not None:
            direction = np.array(cfg["direction"], dtype=np.float64)[:dim]
            if direction.shape[0] != dim:
                raise ValueError(f"Outlet {index} direction must have {dim} components.")
            norm = float(np.linalg.norm(direction))
            if norm <= 1.0e-12:
                raise ValueError(f"Outlet {index} direction must be non-zero when provided.")
            direction = direction / norm

        outlets.append(
            BoxOutletSpec(
                name=str(cfg.get("name", f"outlet_{index}")),
                pmin=pmin,
                pmax=pmax,
                direction=direction,
            )
        )
    return tuple(outlets)
