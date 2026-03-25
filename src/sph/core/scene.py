"""Scene metadata and export configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SceneMetadata:
    """Describes a scene for UI/pipeline presentation."""

    path: Path
    name: str
    dimensions: int
    intended_use: str
    preferred_overlay: str | None
    notes: str | None


@dataclass(slots=True)
class ExportTarget:
    enabled: bool
    every: int
    directory: Path


@dataclass(slots=True)
class ExportSettings:
    csv: ExportTarget
    vtk: ExportTarget

    @property
    def has_exports(self) -> bool:
        return self.csv.enabled or self.vtk.enabled


def parse_scene_metadata(scene: dict[str, Any], path: Path) -> SceneMetadata:
    meta = scene.get("meta", {})
    name = str(meta.get("name") or path.stem)
    dims = int(meta.get("dimensions", scene.get("dimensions", 2)))
    intended_use = str(meta.get("use") or meta.get("category") or "general")
    preferred_overlay = meta.get("preferred_overlay")
    notes = meta.get("notes")
    return SceneMetadata(
        path=path,
        name=name,
        dimensions=dims,
        intended_use=intended_use,
        preferred_overlay=str(preferred_overlay) if preferred_overlay else None,
        notes=str(notes) if notes else None,
    )


def parse_export_settings(scene: dict[str, Any], scene_path: Path | None = None) -> ExportSettings:
    export_cfg = scene.get("export", {})
    base_dir = scene_path.parent if scene_path is not None else Path(".")

    def _target(cfg: dict, default_dir: str) -> ExportTarget:
        enabled = bool(cfg.get("enable", False))
        every = max(1, int(cfg.get("every", 10)))
        directory = cfg.get("dir", default_dir)
        directory_path = Path(directory)
        if not directory_path.is_absolute():
            directory_path = base_dir / directory_path
        return ExportTarget(enabled=enabled, every=every, directory=directory_path)

    csv = _target(export_cfg.get("csv", {}), "out/csv")
    vtk = _target(export_cfg.get("vtk", {}), "out/vtk")
    return ExportSettings(csv=csv, vtk=vtk)


def load_scene_metadata(path: Path) -> SceneMetadata:
    import json

    with Path(path).open("r", encoding="utf-8") as handle:
        scene = json.load(handle)
    return parse_scene_metadata(scene, Path(path))
