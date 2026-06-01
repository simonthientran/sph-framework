"""
SPlisHSPlasH-compatible scene schema with validation and runtime normalization.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _vec3(values: list[float] | tuple[float, ...] | None, default: list[float]) -> list[float]:
    raw = list(values if values is not None else default)
    if len(raw) < 3:
        raw.extend(default[len(raw) : 3])
    return [float(raw[0]), float(raw[1]), float(raw[2])]


def _resolve_scene_path(raw_path: str, scene_path: Path | None) -> str:
    path = Path(str(raw_path))
    if path.is_absolute() or scene_path is None:
        return str(path)
    candidates = [
        (scene_path.parent / path).resolve(),
        (scene_path.parent.parent / path).resolve() if scene_path.parent.parent != scene_path.parent else None,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return str(candidate)
    return str((scene_path.parent / path).resolve())


def _box_container_bounds(rb: "RigidBody", dim: int) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Return world-space bounds for the canonical ``box.stl`` container asset.

    ``box.stl`` is normalized to unit size and transformed around its center,
    so bounds are translation ± 0.5 * scale.
    """
    if Path(rb.geometry_file).name.lower() != "box.stl":
        return None
    center = np.asarray(rb.translation[:dim], dtype=np.float64)
    scale = np.asarray(rb.scale[:dim], dtype=np.float64)
    half = 0.5 * scale
    return center - half, center + half


@dataclass(slots=True)
class DFSPHConfig:
    min_iterations: int = 2
    max_iterations: int = 100
    max_error: float = 0.01
    max_iterations_v: int = 100
    max_error_v: float = 0.1
    enable_divergence_solver: bool = True


@dataclass(slots=True)
class Configuration:
    time_step_size: float = 0.001
    stop_at: float = -1.0
    pause_at: float = -1.0
    simulation_method: int = 4
    particle_radius: float = 0.025
    sim_2d: bool = False
    enable_z_sort: bool = True
    gravitation: list[float] = field(default_factory=lambda: [0.0, -9.81, 0.0])
    cfl_method: int = 1
    cfl_factor: float = 0.4
    cfl_max_time_step_size: float = 0.005
    cfl_min_time_step_size: float = 1.0e-6
    DFSPH: DFSPHConfig = field(default_factory=DFSPHConfig)
    particle_attributes: str = "velocity;density;pressure"
    density_diffusion_delta: float = 0.1
    enable_density_diffusion: bool = True
    density_floor_ratio: float = 0.6


@dataclass(slots=True)
class RigidBody:
    geometry_file: str
    translation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation_axis: list[float] = field(default_factory=lambda: [0.0, 1.0, 0.0])
    rotation_angle: float = 0.0
    scale: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    is_wall: bool = True
    color: list[float] = field(default_factory=lambda: [0.1, 0.4, 0.6, 1.0])
    dense_mode: int = 0
    resolution_sdf: list[int] = field(default_factory=lambda: [20, 20, 20])
    invert: bool = False
    n_layers: int = 3


@dataclass(slots=True)
class FluidBlock:
    start: list[float]
    end: list[float]
    initial_velocity: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    id: str = "Fluid"
    translation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass(slots=True)
class Emitter:
    width: int = 5
    height: int = 5
    translation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation_axis: list[float] = field(default_factory=lambda: [0.0, 1.0, 0.0])
    rotation_angle: float = 0.0
    velocity: float = 1.0
    emit_start_time: float = 0.0
    emit_end_time: float = 1.0e10
    type: int = 0
    id: str = "Fluid"
    max_emitter_particles: int = 10000
    emitter_reuse_particles: bool = False
    emitter_box_min: list[float] = field(default_factory=lambda: [-10.0, -10.0, -10.0])
    emitter_box_max: list[float] = field(default_factory=lambda: [10.0, 10.0, 10.0])


@dataclass(slots=True)
class Material:
    """Per-fluid-phase material definition.

    Mirrors SPlisHSPlasH Materials (docs §4.6): physical properties,
    numerical method selection, and visualization settings are coupled
    to a fluid phase via ``id``.
    """
    id: str = "Fluid"
    density0: float = 1000.0
    viscosity: float = 0.01
    viscosity_method: int = 1
    surface_tension: float = 0.0
    surface_tension_method: int = 0
    xsph_factor: float = 0.05
    color_field: str = "velocity"
    color_map_type: int = 1
    render_min_value: float = 0.0
    render_max_value: float = 5.0


@dataclass(slots=True)
class Exporter:
    enabled: bool = True
    vtk: bool = True
    vtk_dir: str = "output"
    vtk_interval: int = 10
    fps: float = 25.0


# SPlisHSPlasH colorMapType → color map name string
# Ref: SPlisHSPlasH docs §4.6.2 (colorMapType enum)
COLORMAP_TYPE_MAP: dict[int, str] = {
    0: "none",
    1: "jet",
    2: "plasma",
    3: "coolwarm",
    4: "blue_white_red",
    5: "seismic",
}

# Reverse lookup for scene authoring (name → int)
COLORMAP_NAME_TO_INT: dict[str, int] = {v: k for k, v in COLORMAP_TYPE_MAP.items()}


def _extract_splishsplash_viscosity(mat_data: dict[str, Any]) -> float:
    """Extract viscosity coefficient from a SPlisHSPlasH material dict.

    SPlisHSPlasH nests the viscosity value inside a method-specific sub-block
    (e.g. ``"Standard viscosity": {"viscosity": 0.01}``). This helper checks
    the top-level first, then scans nested dicts for a ``viscosity`` key.
    """
    top = mat_data.get("viscosity")
    if top is not None and not isinstance(top, dict):
        return float(top)
    for v in mat_data.values():
        if isinstance(v, dict) and "viscosity" in v:
            val = v["viscosity"]
            if not isinstance(val, dict):
                return float(val)
    return 0.0


# SPlisHSPlasH simulationMethod → solver type string
# Ref: SPlisHSPlasH docs §4.1.4 (simulationMethod enum)
SIMULATION_METHOD_MAP: dict[int, str] = {
    0: "wcsph",
    1: "pcisph",
    2: "pbf",
    3: "iisph",
    4: "dfsph",
    5: "pf",
    6: "icsph",
}

# SPlisHSPlasH boundaryHandlingMethod → boundary method string
# Ref: SPlisHSPlasH docs §4.1.4 (boundaryHandlingMethod enum)
BOUNDARY_METHOD_MAP: dict[int, str] = {
    0: "particle",
    1: "density_map",
    2: "volume_map",
}


@dataclass(slots=True)
class Scene:
    configuration: Configuration
    rigid_bodies: list[RigidBody] = field(default_factory=list)
    fluid_blocks: list[FluidBlock] = field(default_factory=list)
    emitters: list[Emitter] = field(default_factory=list)
    materials: list[Material] = field(default_factory=list)
    exporter: Exporter = field(default_factory=Exporter)

    @classmethod
    def from_json(cls, path: str | Path) -> "Scene":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls._parse(data)

    @classmethod
    def _parse(cls, data: dict[str, Any]) -> "Scene":
        if not is_splishsplash_scene_dict(data):
            return _parse_legacy_scene(data)

        cfg_data = dict(data.get("Configuration", {}))
        dfsph_data = dict(cfg_data.get("DFSPH", {}))
        dfsph = DFSPHConfig(
            min_iterations=int(dfsph_data.get("minIterations", 2)),
            max_iterations=int(dfsph_data.get("maxIterations", 100)),
            max_error=float(dfsph_data.get("maxError", 0.01)),
            max_iterations_v=int(dfsph_data.get("maxIterationsV", 100)),
            max_error_v=float(dfsph_data.get("maxErrorV", 0.1)),
            enable_divergence_solver=bool(dfsph_data.get("enableDivergenceSolver", True)),
        )
        cfg = Configuration(
            time_step_size=float(cfg_data.get("timeStepSize", 0.001)),
            stop_at=float(cfg_data.get("stopAt", -1.0)),
            pause_at=float(cfg_data.get("pauseAt", -1.0)),
            simulation_method=int(cfg_data.get("simulationMethod", 4)),
            particle_radius=float(cfg_data.get("particleRadius", 0.025)),
            sim_2d=bool(cfg_data.get("sim2D", False)),
            enable_z_sort=bool(cfg_data.get("enableZSort", True)),
            gravitation=_vec3(cfg_data.get("gravitation"), [0.0, -9.81, 0.0]),
            cfl_method=int(cfg_data.get("cflMethod", 1)),
            cfl_factor=float(cfg_data.get("cflFactor", 0.4)),
            cfl_max_time_step_size=float(cfg_data.get("cflMaxTimeStepSize", 0.005)),
            cfl_min_time_step_size=float(cfg_data.get("cflMinTimeStepSize", 1.0e-6)),
            DFSPH=dfsph,
            particle_attributes=str(cfg_data.get("particleAttributes", "velocity;density;pressure")),
            density_diffusion_delta=float(cfg_data.get("densityDiffusionDelta", 0.1)),
            enable_density_diffusion=bool(cfg_data.get("enableDensityDiffusion", True)),
            density_floor_ratio=float(
                cfg_data.get("densityFloorRatio", cfg_data.get("density_floor_ratio", 0.6))
            ),
        )

        rigid_bodies = [
            RigidBody(
                geometry_file=str(rb.get("geometryFile", "")),
                translation=_vec3(rb.get("translation"), [0.0, 0.0, 0.0]),
                rotation_axis=_vec3(rb.get("rotationAxis"), [0.0, 1.0, 0.0]),
                rotation_angle=float(rb.get("rotationAngle", 0.0)),
                scale=_vec3(rb.get("scale"), [1.0, 1.0, 1.0]),
                is_wall=bool(rb.get("isWall", True)),
                color=list(rb.get("color", [0.1, 0.4, 0.6, 1.0])),
                dense_mode=int(rb.get("denseMode", 0)),
                resolution_sdf=list(rb.get("resolutionSDF", rb.get("resolution_sdf", [20, 20, 20]))),
                invert=bool(rb.get("invert", False)),
                n_layers=int(rb.get("n_layers", rb.get("nLayers", 3))),
            )
            for rb in data.get("RigidBodies", [])
        ]

        fluid_blocks = [
            FluidBlock(
                start=_vec3(fb.get("start"), [0.0, 0.0, 0.0]),
                end=_vec3(fb.get("end"), [0.0, 0.0, 0.0]),
                initial_velocity=_vec3(fb.get("initialVelocity"), [0.0, 0.0, 0.0]),
                id=str(fb.get("id", "Fluid")),
                translation=_vec3(fb.get("translation"), [0.0, 0.0, 0.0]),
            )
            for fb in data.get("FluidBlocks", [])
        ]

        emitters = [
            Emitter(
                width=int(em.get("width", 5)),
                height=int(em.get("height", 5)),
                translation=_vec3(em.get("translation"), [0.0, 0.0, 0.0]),
                rotation_axis=_vec3(em.get("rotationAxis"), [0.0, 1.0, 0.0]),
                rotation_angle=float(em.get("rotationAngle", 0.0)),
                velocity=float(em.get("velocity", 1.0)),
                emit_start_time=float(em.get("emitStartTime", 0.0)),
                emit_end_time=float(em.get("emitEndTime", 1.0e10)),
                type=int(em.get("type", 0)),
                id=str(em.get("id", "Fluid")),
                max_emitter_particles=int(em.get("maxEmitterParticles", 10000)),
                emitter_reuse_particles=bool(em.get("emitterReuseParticles", False)),
                emitter_box_min=_vec3(em.get("emitterBoxMin"), [-10.0, -10.0, -10.0]),
                emitter_box_max=_vec3(em.get("emitterBoxMax"), [10.0, 10.0, 10.0]),
            )
            for em in data.get("Emitters", [])
        ]

        materials = [
            Material(
                id=str(mat.get("id", "Fluid")),
                density0=float(mat.get("density0", 1000.0)),
                viscosity=_extract_splishsplash_viscosity(mat),
                viscosity_method=int(mat.get("viscosityMethod", 1)),
                surface_tension=float(mat.get("surfaceTension", 0.0)),
                surface_tension_method=int(mat.get("surfaceTensionMethod", 0)),
                xsph_factor=float(mat.get("xsph", mat.get("xsphFactor", 0.05))),
                color_field=str(mat.get("colorField", "velocity")),
                color_map_type=int(mat.get("colorMapType", 1)),
                render_min_value=float(mat.get("renderMinValue", 0.0)),
                render_max_value=float(mat.get("renderMaxValue", 5.0)),
            )
            for mat in data.get("Materials", [])
        ]
        if not materials:
            materials = [Material()]

        exp_data = data.get("Exporter", data.get("export", {}))
        exporter = Exporter(
            enabled=bool(exp_data.get("enabled", True)),
            vtk=bool(exp_data.get("vtk", True)),
            vtk_dir=str(exp_data.get("vtk_dir", exp_data.get("output_dir", "output"))),
            vtk_interval=int(exp_data.get("vtk_interval", 10)),
            fps=float(exp_data.get("fps", 25.0)),
        )

        return cls(
            configuration=cfg,
            rigid_bodies=rigid_bodies,
            fluid_blocks=fluid_blocks,
            emitters=emitters,
            materials=materials,
            exporter=exporter,
        )

    def to_runtime_scene_dict(
        self,
        scene_path: Path | None = None,
        raw_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cfg = self.configuration
        cfg_data = (raw_data or {}).get("Configuration", {})
        dim = 2 if cfg.sim_2d else 3
        primary_material = self.materials[0] if self.materials else Material()
        scene_name = Path(scene_path).stem if scene_path is not None else "scene"

        boundary_method_int = int(cfg_data.get("boundaryHandlingMethod", 0))
        boundary_method = BOUNDARY_METHOD_MAP.get(boundary_method_int, "particle")

        boundaries: list[dict[str, Any]] = []
        for rb in self.rigid_bodies:
            suffix = Path(rb.geometry_file).suffix.lower()
            boundary_type = "mesh"
            if suffix == ".stl":
                boundary_type = "stl"
            elif suffix == ".obj":
                boundary_type = "obj"
            boundaries.append(
                {
                    "type": boundary_type,
                    "file": _resolve_scene_path(rb.geometry_file, scene_path),
                    "sampling": "poisson",
                    "n_layers": max(1, int(rb.n_layers)),
                    "translation": rb.translation,
                    "rotation_axis": rb.rotation_axis,
                    "rotation_angle": rb.rotation_angle,
                    "scale": rb.scale,
                    "invert": rb.invert,
                    "is_wall": rb.is_wall,
                    "boundary_handling_method": boundary_method,
                }
            )

        fluids: list[dict[str, Any]] = []
        spacing = 2.0 * float(cfg.particle_radius)  # SplishSplash: spacing = diameter
        container_bounds = None
        if dim == 3 and self.rigid_bodies:
            container_bounds = _box_container_bounds(self.rigid_bodies[0], dim)
        for index, block in enumerate(self.fluid_blocks):
            translation = block.translation
            start = [block.start[i] + translation[i] for i in range(dim)]
            end = [block.end[i] + translation[i] for i in range(dim)]

            # Imported 3D dam-break scenes can initialize as a near full-depth
            # slab in box containers, which makes a large fraction of particles
            # wall-adjacent from step 0. Keep one-spacing clearance on the far
            # depth side while preserving the left-wall dam contact.
            if dim == 3 and container_bounds is not None:
                cmin, cmax = container_bounds
                eps = 1.0e-9
                z_axis = 2
                spans_depth = (
                    abs(start[z_axis] - float(cmin[z_axis])) <= eps
                    and abs(end[z_axis] - float(cmax[z_axis])) <= eps
                )
                if spans_depth:
                    end[z_axis] = max(start[z_axis] + spacing, end[z_axis] - spacing)

            fluids.append(
                {
                    "type": "block",
                    "name": block.id or f"fluid_{index}",
                    "spacing": spacing,
                    "min": start,
                    "max": end,
                    "initial_velocity": block.initial_velocity[:dim],
                }
            )

        emitters: list[dict[str, Any]] = []
        for index, emitter in enumerate(self.emitters):
            velocity_dir = _axis_angle_direction(emitter.rotation_axis, emitter.rotation_angle)[:dim]
            velocity = (np.asarray(velocity_dir, dtype=np.float64) * float(emitter.velocity)).tolist()
            spacing_thickness = spacing
            center = np.asarray(emitter.translation[:dim], dtype=np.float64)
            if dim == 2:
                extents = np.array([spacing_thickness, max(emitter.width, 1) * spacing], dtype=np.float64)
            else:
                extents = np.array(
                    [spacing_thickness, max(emitter.width, 1) * spacing, max(emitter.height, 1) * spacing],
                    dtype=np.float64,
                )
            axis = int(np.argmax(np.abs(velocity_dir[:dim])))
            ordered = np.full(dim, spacing_thickness, dtype=np.float64)
            transverse = [idx for idx in range(dim) if idx != axis]
            for local_idx, world_idx in enumerate(transverse):
                ordered[world_idx] = extents[min(local_idx + 1, extents.shape[0] - 1)]
            pmin = (center - 0.5 * ordered).tolist()
            pmax = (center + 0.5 * ordered).tolist()
            emitters.append(
                {
                    "type": "inlet",
                    "name": f"emitter_{index}",
                    "min": pmin,
                    "max": pmax,
                    "velocity": velocity,
                    "start_step": max(int(round(emitter.emit_start_time / max(cfg.time_step_size, 1.0e-8))), 0),
                    "end_step": (
                        int(round(emitter.emit_end_time / max(cfg.time_step_size, 1.0e-8)))
                        if math.isfinite(emitter.emit_end_time) and emitter.emit_end_time < 1.0e9
                        else None
                    ),
                    "interval_steps": 1,
                    "max_particles": emitter.max_emitter_particles,
                }
            )

        export = {
            "particle_attributes": cfg.particle_attributes,
            "csv": {"enable": False, "dir": "out/csv"},
            "vtk": {
                "enable": bool(self.exporter.enabled and self.exporter.vtk),
                "every": max(1, int(self.exporter.vtk_interval)),
                "dir": self.exporter.vtk_dir,
            },
        }

        neighbors: dict[str, Any] = {}
        if dim == 3:
            # Imported SPlisHSPlasH-style 3D scenes do not carry the runtime
            # neighbor support config this codebase expects. If we omit it, the
            # simulator falls back to support_radius = 6 * spacing, which makes
            # wall coupling dominate too much of the initial dam-break column.
            neighbors = {
                "type": "kdtree",
                "support_radius": 3.0 * spacing,
            }

        solver_type = SIMULATION_METHOD_MAP.get(cfg.simulation_method, "dfsph")
        solver_dict: dict[str, Any] = {"type": solver_type}

        if solver_type == "dfsph":
            solver_dict["dfsph"] = {
                "eta_cd": float(cfg.DFSPH.max_error),
                "eta_df": float(cfg.DFSPH.max_error_v),
                "max_iter_cd": int(cfg.DFSPH.max_iterations),
                "max_iter_df": int(cfg.DFSPH.max_iterations_v),
                "enable_divergence_solver": bool(cfg.DFSPH.enable_divergence_solver),
                "density_diffusion": {
                    "enable": bool(cfg.enable_density_diffusion),
                    "delta": float(cfg.density_diffusion_delta),
                    "floor_ratio": float(cfg.density_floor_ratio),
                },
            }
        elif solver_type == "wcsph":
            wcsph_data = cfg_data.get("WCSPH", {})
            solver_dict["wcsph"] = {
                "stiffness": float(wcsph_data.get("stiffness", 50000)),
                "exponent": float(wcsph_data.get("exponent", 7)),
            }

        materials_list: list[dict[str, Any]] = []
        for mat in self.materials:
            materials_list.append({
                "id": mat.id,
                "density0": float(mat.density0),
                "viscosity": float(mat.viscosity),
                "viscosityMethod": mat.viscosity_method,
                "xsph": float(mat.xsph_factor),
                "surfaceTension": float(mat.surface_tension),
                "surfaceTensionMethod": mat.surface_tension_method,
                "colorField": mat.color_field,
                "colorMapType": COLORMAP_TYPE_MAP.get(mat.color_map_type, "jet"),
                "renderMinValue": float(mat.render_min_value),
                "renderMaxValue": float(mat.render_max_value),
            })
        if not materials_list:
            materials_list = [{
                "id": "Fluid",
                "density0": 1000.0,
                "viscosity": 0.0,
                "viscosityMethod": 1,
                "xsph": 0.0,
                "colorField": "velocity",
                "colorMapType": "jet",
                "renderMinValue": 0.0,
                "renderMaxValue": 5.0,
            }]

        periodic_x = bool(cfg_data.get("periodicX", cfg_data.get("periodic_x", False)))
        domain: dict[str, Any] = {"periodic_x": periodic_x}
        raw_domain_min = cfg_data.get("domainMin", cfg_data.get("domain_min"))
        raw_domain_max = cfg_data.get("domainMax", cfg_data.get("domain_max"))
        if raw_domain_min is not None:
            domain["min"] = list(raw_domain_min[:dim])
        if raw_domain_max is not None:
            domain["max"] = list(raw_domain_max[:dim])

        normalized: dict[str, Any] = {
            "meta": {"name": scene_name, "dimensions": dim, "use": "general"},
            "domain": domain,
            "solver": solver_dict,
            "materials": materials_list,
            "material": {"rho0": float(primary_material.density0)},
            "time": {
                "mode": "fixed" if int(cfg.cfl_method) == 0 else "cfl",
                "dt_fixed": float(cfg.time_step_size),
                "dt_max": float(cfg.cfl_max_time_step_size),
                "dt_min": float(cfg.cfl_min_time_step_size),
                "cfl": float(cfg.cfl_factor),
                "stop_at": float(cfg.stop_at),
                "pause_at": float(cfg.pause_at),
            },
            "forces": {
                "gravity": cfg.gravitation[:dim],
                "viscosity": {
                    "enable": float(primary_material.viscosity) > 0.0,
                    "nu": float(primary_material.viscosity),
                },
                "xsph": {
                    "enable": float(primary_material.xsph_factor) > 0.0,
                    "eps": float(primary_material.xsph_factor),
                },
            },
            "performance": {
                "enable_z_sort": bool(cfg.enable_z_sort),
                "sort_every": 10,
            },
            "neighbors": neighbors,
            "fluids": fluids,
            "boundaries": boundaries,
            "emitters": emitters,
            "export": export,
        }
        return normalized


def _axis_angle_direction(axis: list[float], angle_degrees: float) -> list[float]:
    vec = [1.0, 0.0, 0.0]
    axis_arr = _normalize_vector(axis)
    angle = math.radians(float(angle_degrees))
    c = math.cos(angle)
    s = math.sin(angle)
    x, y, z = axis_arr
    rot = [
        [(c + x * x * (1 - c)), (x * y * (1 - c) - z * s), (x * z * (1 - c) + y * s)],
        [(y * x * (1 - c) + z * s), (c + y * y * (1 - c)), (y * z * (1 - c) - x * s)],
        [(z * x * (1 - c) - y * s), (z * y * (1 - c) + x * s), (c + z * z * (1 - c))],
    ]
    out = [
        rot[0][0] * vec[0] + rot[0][1] * vec[1] + rot[0][2] * vec[2],
        rot[1][0] * vec[0] + rot[1][1] * vec[1] + rot[1][2] * vec[2],
        rot[2][0] * vec[0] + rot[2][1] * vec[1] + rot[2][2] * vec[2],
    ]
    return _normalize_vector(out)


def _normalize_vector(values: list[float] | tuple[float, ...]) -> list[float]:
    arr = [float(values[0]), float(values[1]), float(values[2])]
    norm = math.sqrt(arr[0] * arr[0] + arr[1] * arr[1] + arr[2] * arr[2])
    if norm <= 1.0e-12:
        return [1.0, 0.0, 0.0]
    return [arr[0] / norm, arr[1] / norm, arr[2] / norm]


def is_splishsplash_scene_dict(data: dict[str, Any]) -> bool:
    return any(key in data for key in ("Configuration", "RigidBodies", "FluidBlocks", "Materials", "Exporter"))


def normalize_own_scene_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize our own scene format into a canonical internal form.

    SPlisHSPlasH-inspired normalization that ensures every consumer of the
    scene dict sees a consistent shape regardless of authoring shortcuts.

    Structural principles (ref: SPlisHSPlasH docs §4, §9):

    1. ``fluid`` (singular dict) is promoted to ``fluids`` (list)
    2. ``solver.type`` is always present (scene-type-aware default)
    3. Legacy ``prefer_iterative`` is stripped
    4. Solver-specific sub-blocks (``solver.dfsph``, ``solver.wcsph``)
       are flattened into the solver dict for uniform Simulator access
    5. WCSPH ``stiffness`` is mapped to ``eos_k``
    6. ``material.eos.k`` is promoted to ``solver.eos_k`` when missing
    7. Duplicate viscosity declarations are consolidated
    8. Boundary handling method defaults are set
    9. Required top-level blocks get safe defaults
    """
    out: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, dict):
            out[k] = dict(v)
        elif isinstance(v, list):
            out[k] = list(v)
        else:
            out[k] = v

    meta = out.setdefault("meta", {})
    if isinstance(meta, dict):
        meta.setdefault("dimensions", 2)
        meta.setdefault("version", 1)
    dim = int(meta.get("dimensions", 2))
    scene_type = str(meta.get("scene_type", ""))

    # --- fluids: always a list ------------------------------------------------
    if "fluid" in out and "fluids" not in out:
        fluid_singular = out.pop("fluid")
        if isinstance(fluid_singular, dict):
            out["fluids"] = [fluid_singular]
    if "fluids" not in out:
        out["fluids"] = []

    # --- solver: always present with explicit type ----------------------------
    solver = out.get("solver")
    if solver is None or not isinstance(solver, dict):
        solver = {}
        out["solver"] = solver
    else:
        solver = dict(solver)
        out["solver"] = solver

    solver.pop("prefer_iterative", None)

    if "type" not in solver:
        if scene_type == "free_surface":
            solver["type"] = "wcsph"
        elif scene_type == "internal_flow":
            solver["type"] = "dfsph"
        else:
            solver["type"] = "dfsph"

    # --- solver sub-block flattening ------------------------------------------
    # SPlisHSPlasH-inspired: solver-specific params can be authored in a named
    # sub-block (e.g. ``solver.dfsph: {eta_cd: 0.01}``) which is flattened
    # into the top-level solver dict for uniform downstream access.
    # Ref: SPlisHSPlasH Configuration.DFSPH / Configuration.WCSPH pattern.
    solver_type = solver["type"]
    solver_sub = solver.pop(solver_type, None)
    if isinstance(solver_sub, dict):
        for k, v in solver_sub.items():
            if k not in solver:
                solver[k] = v

    # Remove sub-blocks for non-active solvers (keep the config clean)
    for other_solver in ("dfsph", "wcsph", "pcisph", "pbf", "iisph", "pf", "icsph"):
        if other_solver != solver_type:
            solver.pop(other_solver, None)

    # --- WCSPH stiffness → eos_k mapping (SPlisHSPlasH uses "stiffness") -----
    if "stiffness" in solver and "eos_k" not in solver:
        solver["eos_k"] = float(solver.pop("stiffness"))
    elif "stiffness" in solver:
        solver.pop("stiffness")

    # --- eos_k consolidation: material.eos.k → solver.eos_k ------------------
    mat = out.get("material", {})
    if isinstance(mat, dict):
        eos = mat.get("eos", {})
        if isinstance(eos, dict) and "k" in eos and "eos_k" not in solver:
            solver["eos_k"] = float(eos["k"])

    # --- material defaults ----------------------------------------------------
    out.setdefault("material", {"rho0": 1000.0})

    # --- forces: ensure block exists ------------------------------------------
    forces = out.get("forces")
    if forces is None or not isinstance(forces, dict):
        forces = {}
        out["forces"] = forces
    else:
        forces = dict(forces)
        out["forces"] = forces
    if "gravity" not in forces:
        forces["gravity"] = [0.0, -9.81] if dim == 2 else [0.0, -9.81, 0.0]

    # --- viscosity consolidation: resolve once --------------------------------
    mat_visc = mat.get("viscosity", {}) if isinstance(mat, dict) else {}
    force_visc = forces.get("viscosity", {})
    if force_visc and not mat_visc:
        if isinstance(out.get("material"), dict):
            out["material"]["viscosity"] = dict(force_visc)
    elif mat_visc and not force_visc:
        forces["viscosity"] = dict(mat_visc)

    # --- time defaults --------------------------------------------------------
    out.setdefault("time", {
        "mode": "cfl", "cfl": 0.4,
        "dt_min": 1e-5, "dt_max": 0.001, "steps": 500,
    })

    # --- neighbors defaults ---------------------------------------------------
    out.setdefault("neighbors", {})

    # --- boundary handling method defaults ------------------------------------
    # Ref: SPlisHSPlasH boundaryHandlingMethod (§4.1.4)
    # Our current runtime only supports "particle" (Akinci et al. 2012).
    domain = out.get("domain")
    if isinstance(domain, dict):
        domain.setdefault("boundary_handling_method", "particle")

    boundaries = out.get("boundaries")
    if isinstance(boundaries, list):
        for b in boundaries:
            if isinstance(b, dict):
                b.setdefault("boundary_handling_method", "particle")
                b.setdefault("is_wall", True)

    # --- materials: SPlisHSPlasH-style per-phase behavior + visualization -----
    # Ref: SPlisHSPlasH docs §4.6 (Materials block)
    #
    # The materials array is the canonical source for per-fluid-phase physical
    # properties (density, viscosity, XSPH, surface tension) AND visualization
    # settings (colorField, colorMapType, renderMinValue, renderMaxValue).
    #
    # When scenes are authored without an explicit materials block, we
    # synthesize one from the legacy material/forces/meta paths.
    # Backward-compat: material.rho0 and forces.viscosity/xsph are kept in
    # sync so older consumers still work.
    existing_materials = out.get("materials")
    if isinstance(existing_materials, list) and existing_materials:
        materials = existing_materials
        for m in materials:
            m.setdefault("id", "Fluid")
            m.setdefault("density0", 1000.0)
            m.setdefault("colorField", "velocity")
            m.setdefault("colorMapType", "jet")
            m.setdefault("renderMinValue", 0.0)
            m.setdefault("renderMaxValue", 5.0)
            m.setdefault("viscosity", 0.0)
            m.setdefault("xsph", 0.0)
    else:
        mat_rho0 = float(out.get("material", {}).get("rho0", 1000.0))

        f_visc = forces.get("viscosity", {})
        visc_enabled = bool(f_visc.get("enable", False))
        visc_nu = float(f_visc.get("nu", 0.0)) if visc_enabled else 0.0

        f_xsph = forces.get("xsph", {})
        xsph_enabled = bool(f_xsph.get("enable", False))
        xsph_eps = float(f_xsph.get("eps", f_xsph.get("epsilon", 0.0))) if xsph_enabled else 0.0

        color_field = str(meta.get("preferred_overlay", "velocity"))
        color_map = str(meta.get("color_map", "jet"))

        materials = [{
            "id": "Fluid",
            "density0": mat_rho0,
            "viscosity": visc_nu,
            "xsph": xsph_eps,
            "colorField": color_field,
            "colorMapType": color_map,
            "renderMinValue": 0.0,
            "renderMaxValue": 5.0,
        }]

    out["materials"] = materials

    # Keep backward-compat material/forces in sync with materials[0]
    primary = materials[0]
    if isinstance(out.get("material"), dict):
        out["material"]["rho0"] = float(primary.get("density0", 1000.0))
    else:
        out["material"] = {"rho0": float(primary.get("density0", 1000.0))}

    mat_visc = float(primary.get("viscosity", 0.0))
    mat_xsph = float(primary.get("xsph", 0.0))
    if mat_visc > 0.0:
        forces.setdefault("viscosity", {})
        forces["viscosity"]["enable"] = True
        forces["viscosity"]["nu"] = mat_visc
    if mat_xsph > 0.0:
        forces.setdefault("xsph", {})
        forces["xsph"]["enable"] = True
        forces["xsph"]["eps"] = mat_xsph

    return out


def normalize_scene_dict(data: dict[str, Any], scene_path: Path | None = None) -> dict[str, Any]:
    if is_splishsplash_scene_dict(data):
        converted = Scene._parse(data).to_runtime_scene_dict(
            scene_path=scene_path, raw_data=data,
        )
        return normalize_own_scene_dict(converted)
    return normalize_own_scene_dict(data)


def load_scene_dict(path: str | Path) -> dict[str, Any]:
    scene_path = Path(path)
    with scene_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return normalize_scene_dict(raw, scene_path=scene_path)


def _parse_legacy_scene(data: dict[str, Any]) -> Scene:
    dim = int(data.get("meta", {}).get("dimensions", data.get("dim", data.get("dimensions", 2))))
    material = data.get("material", {})
    forces = data.get("forces", {})
    time_cfg = data.get("time", {})
    solver_cfg = data.get("solver", {})
    visc_cfg = forces.get("viscosity", {})
    xsph_cfg = forces.get("xsph", {})
    performance_cfg = data.get("performance", {})
    export_cfg = data.get("export", {})

    fluid_sources = list(data.get("fluids", []))
    if not fluid_sources and isinstance(data.get("fluid"), dict):
        fluid_sources = [dict(data["fluid"])]
    blocks: list[FluidBlock] = []
    for source in fluid_sources:
        if str(source.get("type", "block")).lower() not in {"block", "box"}:
            continue
        pmin = list(source.get("min", [source.get("x_min", 0.0), source.get("y_min", 0.0), source.get("z_min", 0.0)]))
        pmax = list(source.get("max", [source.get("x_max", 0.0), source.get("y_max", 0.0), source.get("z_max", 0.0)]))
        blocks.append(
            FluidBlock(
                start=_vec3(pmin, [0.0, 0.0, 0.0]),
                end=_vec3(pmax, [0.0, 0.0, 0.0]),
                initial_velocity=_vec3(source.get("initial_velocity", source.get("velocity")), [0.0, 0.0, 0.0]),
                id=str(source.get("name", source.get("id", "Fluid"))),
            )
        )

    rigid_bodies: list[RigidBody] = []
    for boundary in data.get("boundaries", []):
        boundary_type = str(boundary.get("type", "")).lower()
        if boundary_type not in {"stl", "obj", "mesh"}:
            continue
        rigid_bodies.append(
            RigidBody(
                geometry_file=str(boundary.get("file", "")),
                translation=_vec3(boundary.get("translation"), [0.0, 0.0, 0.0]),
                rotation_axis=_vec3(boundary.get("rotation_axis"), [0.0, 1.0, 0.0]),
                rotation_angle=float(boundary.get("rotation_angle", 0.0)),
                scale=_vec3(boundary.get("scale"), [1.0, 1.0, 1.0]),
                n_layers=int(boundary.get("n_layers", 3)),
            )
        )

    configuration = Configuration(
        time_step_size=float(time_cfg.get("dt_fixed", time_cfg.get("dt_max", 0.001))),
        stop_at=float(time_cfg.get("stop_at", -1.0)),
        pause_at=float(time_cfg.get("pause_at", -1.0)),
        simulation_method=4 if str(solver_cfg.get("type", "dfsph")).lower() == "dfsph" else 0,
        particle_radius=float(data.get("particle_spacing", fluid_sources[0].get("spacing", 0.025) if fluid_sources else 0.025)),
        sim_2d=dim == 2,
        enable_z_sort=bool(performance_cfg.get("enable_z_sort", False)),
        gravitation=list(forces.get("gravity", [0.0, -9.81, 0.0])),
        cfl_method=1 if str(time_cfg.get("mode", "cfl")).lower() != "fixed" else 0,
        cfl_factor=float(time_cfg.get("cfl", 0.4)),
        cfl_max_time_step_size=float(time_cfg.get("dt_max", 0.005)),
        cfl_min_time_step_size=float(time_cfg.get("dt_min", 1.0e-6)),
        DFSPH=DFSPHConfig(
            max_iterations=int(solver_cfg.get("max_iter_cd", 100)),
            max_error=float(solver_cfg.get("eta_cd", 0.01)),
            max_iterations_v=int(solver_cfg.get("max_iter_df", 100)),
            max_error_v=float(solver_cfg.get("eta_df", 0.1)),
        ),
        particle_attributes=str(export_cfg.get("particle_attributes", "velocity;density;pressure")),
        density_diffusion_delta=float(solver_cfg.get("density_diffusion", {}).get("delta", 0.1)),
        enable_density_diffusion=bool(solver_cfg.get("density_diffusion", {}).get("enable", True)),
        density_floor_ratio=float(solver_cfg.get("density_diffusion", {}).get("floor_ratio", 0.6)),
    )
    material_out = Material(
        id="Fluid",
        density0=float(material.get("rho0", 1000.0)),
        viscosity=float(visc_cfg.get("nu", 0.0)) if bool(visc_cfg.get("enable", False)) else 0.0,
        xsph_factor=float(xsph_cfg.get("eps", xsph_cfg.get("epsilon", 0.035))) if bool(xsph_cfg.get("enable", True)) else 0.0,
    )
    exporter = Exporter(
        enabled=bool(export_cfg.get("vtk", {}).get("enable", False)) if isinstance(export_cfg, dict) else True,
        vtk=bool(export_cfg.get("vtk", {}).get("enable", False)) if isinstance(export_cfg, dict) else False,
        vtk_dir=str(export_cfg.get("vtk", {}).get("dir", "out/vtk")) if isinstance(export_cfg, dict) else "out/vtk",
        vtk_interval=int(export_cfg.get("vtk", {}).get("every", 10)) if isinstance(export_cfg, dict) else 10,
    )
    return Scene(
        configuration=configuration,
        rigid_bodies=rigid_bodies,
        fluid_blocks=blocks,
        materials=[material_out],
        exporter=exporter,
    )
