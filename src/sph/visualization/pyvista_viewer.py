from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

from sph.core.backend import RuntimeStats, SimulationStateView
from sph.core.simulation import SimulationRunner
from sph.io.geometry_loader import load_mesh


def _pad_points_to_3d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if values.shape[1] == 3:
        return values.copy()
    padded = np.zeros((values.shape[0], 3), dtype=np.float64)
    padded[:, : values.shape[1]] = values
    return padded


def _trimesh_to_polydata(mesh) -> pv.PolyData:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.size == 0:
        return pv.PolyData(vertices)
    vtk_faces = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).ravel()
    return pv.PolyData(vertices, vtk_faces)


@dataclass(slots=True)
class ViewerFrameSummary:
    step: int
    fluid_count: int
    boundary_count: int
    scalar_name: str
    scalar_min: float
    scalar_max: float


class PyVistaSceneViewer:
    """Interactive 3D viewer built around the current runtime state."""

    AVAILABLE_SCALARS = (
        "speed",
        "density",
        "density_deviation",
        "pressure",
        "neighbor_count",
        "free_surface_score",
        "low_density_flag",
        "low_neighbor_flag",
    )

    def __init__(
        self,
        scene_path: Path | str,
        *,
        backend_name: str = "numba_cpu",
        scalar: str = "speed",
        off_screen: bool = False,
        window_size: tuple[int, int] = (1400, 900),
        show_boundary_mesh: bool = True,
        fluid_point_size: float = 14.0,
        boundary_point_size: float = 9.0,
        render_points_as_spheres: bool = True,
    ):
        self.scene_path = Path(scene_path)
        self.backend_name = backend_name
        self.scalar_name = self._normalize_scalar_name(scalar)
        self.show_boundary_mesh = bool(show_boundary_mesh)
        self.fluid_point_size = float(fluid_point_size)
        self.boundary_point_size = float(boundary_point_size)
        self.render_points_as_spheres = bool(render_points_as_spheres)

        self.runner = SimulationRunner(self.scene_path, backend_name=self.backend_name)
        self.plotter = pv.Plotter(off_screen=off_screen, window_size=window_size)
        self._fluid_actor = None
        self._boundary_actor = None
        self._domain_actor = None
        self._mesh_actors: list = []
        self._status_actor = None
        self._help_actor = None
        self._last_runtime: RuntimeStats | None = None
        self._last_summary: ViewerFrameSummary | None = None
        self._boundary_meshes = self._load_boundary_meshes() if self.show_boundary_mesh else []

        self._configure_plotter()
        self.refresh()

    @property
    def last_summary(self) -> ViewerFrameSummary | None:
        return self._last_summary

    def available_scalars(self) -> tuple[str, ...]:
        return self.AVAILABLE_SCALARS

    def refresh(self) -> ViewerFrameSummary:
        state = self.runner.state
        self._rebuild_scene(state)
        summary = self._build_summary(state)
        self._last_summary = summary
        self._update_text(summary)
        return summary

    def advance(self, steps: int = 1) -> ViewerFrameSummary:
        summary = self.refresh()
        for _ in range(max(steps, 0)):
            result = self.runner.step()
            self._last_runtime = result.runtime
            summary = self.refresh()
        return summary

    def cycle_scalar(self) -> ViewerFrameSummary:
        scalars = self.AVAILABLE_SCALARS
        index = scalars.index(self.scalar_name)
        self.scalar_name = scalars[(index + 1) % len(scalars)]
        return self.refresh()

    def save_screenshot(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.plotter.render()
        self.plotter.screenshot(str(path))
        return path

    def record_frames(
        self,
        directory: str | Path,
        *,
        steps: int,
        prefix: str = "frame",
    ) -> list[Path]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        outputs: list[Path] = []
        outputs.append(self.save_screenshot(directory / f"{prefix}_0000.png"))
        for index in range(1, max(int(steps), 0) + 1):
            self.advance(1)
            outputs.append(self.save_screenshot(directory / f"{prefix}_{index:04d}.png"))
        return outputs

    def show(self, auto_close: bool = False) -> None:
        self.plotter.show(auto_close=auto_close)

    def close(self) -> None:
        self.plotter.close()

    def _configure_plotter(self) -> None:
        self.plotter.set_background("#f4f7fb", top="#dfeaf7")
        self.plotter.enable_anti_aliasing("ssaa")
        self.plotter.add_axes()
        self.plotter.show_grid(color="#9fb3c8")
        self.plotter.add_key_event("space", lambda: self.advance(1))
        self.plotter.add_key_event("s", lambda: self.cycle_scalar())
        self.plotter.add_key_event("r", lambda: self.refresh())
        self._help_actor = self.plotter.add_text(
            "space: step   s: cycle scalar   r: refresh",
            position="lower_left",
            font_size=10,
            color="black",
        )

    def _build_summary(self, state: SimulationStateView) -> ViewerFrameSummary:
        scalar_values = self._fluid_scalar_values(state, self.scalar_name)
        if scalar_values.size:
            scalar_min = float(np.nanmin(scalar_values))
            scalar_max = float(np.nanmax(scalar_values))
        else:
            scalar_min = 0.0
            scalar_max = 0.0
        return ViewerFrameSummary(
            step=int(self.runner.backend.frame_index),
            fluid_count=int(state.fluid_positions.shape[0]),
            boundary_count=int(state.boundary_positions.shape[0]),
            scalar_name=self.scalar_name,
            scalar_min=scalar_min,
            scalar_max=scalar_max,
        )

    def _update_text(self, summary: ViewerFrameSummary) -> None:
        if self._status_actor is not None:
            self.plotter.remove_actor(self._status_actor)
        runtime_bits: list[str] = []
        if self._last_runtime is not None:
            runtime_bits.append(f"dt={self._last_runtime.dt:.3e}")
            runtime_bits.append(f"iter_cd={self._last_runtime.solver_metrics.get('iter_cd', 0):.0f}")
            runtime_bits.append(f"iter_df={self._last_runtime.solver_metrics.get('iter_df', 0):.0f}")
        text = (
            f"{self.runner.scene_name}  step={summary.step}\n"
            f"fluid={summary.fluid_count} boundary={summary.boundary_count}\n"
            f"scalar={summary.scalar_name} range=[{summary.scalar_min:.4g}, {summary.scalar_max:.4g}]"
        )
        if runtime_bits:
            text += "\n" + "  ".join(runtime_bits)
        self._status_actor = self.plotter.add_text(
            text,
            position="upper_left",
            font_size=11,
            color="black",
            name="viewer_status",
        )

    def _rebuild_scene(self, state: SimulationStateView) -> None:
        fluid_poly = self._build_fluid_polydata(state)
        boundary_poly = self._build_boundary_polydata(state)

        if self._fluid_actor is not None:
            self.plotter.remove_actor(self._fluid_actor)
        if self._boundary_actor is not None:
            self.plotter.remove_actor(self._boundary_actor)
        if self._domain_actor is not None:
            self.plotter.remove_actor(self._domain_actor)
        for actor in self._mesh_actors:
            self.plotter.remove_actor(actor)
        self._mesh_actors.clear()

        if boundary_poly.n_points:
            self._boundary_actor = self.plotter.add_mesh(
                boundary_poly,
                color="#535c68",
                point_size=self.boundary_point_size,
                render_points_as_spheres=self.render_points_as_spheres,
                opacity=0.75,
                name="boundary_particles",
            )
        else:
            self._boundary_actor = None

        if fluid_poly.n_points:
            self._fluid_actor = self.plotter.add_mesh(
                fluid_poly,
                scalars=self.scalar_name,
                cmap="viridis",
                clim=self._scalar_limits(fluid_poly[self.scalar_name]),
                point_size=self.fluid_point_size,
                render_points_as_spheres=self.render_points_as_spheres,
                scalar_bar_args={"title": self.scalar_name.replace("_", " ")},
                name="fluid_particles",
            )
        else:
            self._fluid_actor = None

        if self.show_boundary_mesh:
            for index, mesh_poly in enumerate(self._boundary_meshes):
                actor = self.plotter.add_mesh(
                    mesh_poly,
                    color="#7f8c8d",
                    style="wireframe",
                    line_width=1.5,
                    opacity=0.35,
                    name=f"boundary_mesh_{index}",
                )
                self._mesh_actors.append(actor)

        bounds = tuple(np.concatenate([state.domain_min, state.domain_max]).tolist())
        self._domain_actor = self.plotter.add_mesh(
            pv.Box(bounds=bounds),
            style="wireframe",
            color="#b03a2e",
            line_width=1.0,
            opacity=0.2,
            name="domain_box",
        )
        self.plotter.reset_camera_clipping_range()
        self.plotter.render()

    def _build_fluid_polydata(self, state: SimulationStateView) -> pv.PolyData:
        points = _pad_points_to_3d(state.fluid_positions)
        poly = pv.PolyData(points)
        poly["speed"] = np.asarray(state.fluid_speed, dtype=np.float64)
        poly["density"] = np.asarray(state.fluid_density, dtype=np.float64)
        poly["density_deviation"] = np.asarray(state.fluid_density_deviation, dtype=np.float64)
        poly["pressure"] = np.asarray(state.fluid_pressure, dtype=np.float64)
        poly["neighbor_count"] = np.asarray(state.fluid_neighbor_counts, dtype=np.float64)
        poly["free_surface_score"] = np.asarray(state.fluid_free_surface_score, dtype=np.float64)
        poly["low_density_flag"] = np.asarray(state.fluid_low_density_mask, dtype=np.float64)
        poly["low_neighbor_flag"] = np.asarray(state.fluid_low_neighbor_mask, dtype=np.float64)
        poly["velocity"] = _pad_points_to_3d(state.fluid_velocities)
        return poly

    def _build_boundary_polydata(self, state: SimulationStateView) -> pv.PolyData:
        points = _pad_points_to_3d(state.boundary_positions)
        poly = pv.PolyData(points)
        if points.shape[0]:
            poly["speed"] = np.linalg.norm(state.boundary_velocities, axis=1).astype(np.float64)
        else:
            poly["speed"] = np.zeros(0, dtype=np.float64)
        return poly

    def _load_boundary_meshes(self) -> list[pv.PolyData]:
        with self.scene_path.open("r", encoding="utf-8") as handle:
            scene = json.load(handle)
        meshes: list[pv.PolyData] = []
        for entry in scene.get("boundaries", []):
            mesh_file = entry.get("file")
            if not mesh_file:
                continue
            mesh_path = (self.scene_path.parent / mesh_file).resolve()
            if not mesh_path.exists():
                continue
            mesh = load_mesh(str(mesh_path))
            meshes.append(_trimesh_to_polydata(mesh))
        return meshes

    def _scalar_limits(self, values: np.ndarray) -> tuple[float, float]:
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return (0.0, 1.0)
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0
        return (vmin, vmax)

    def _fluid_scalar_values(self, state: SimulationStateView, scalar_name: str) -> np.ndarray:
        mapping = {
            "speed": state.fluid_speed,
            "density": state.fluid_density,
            "density_deviation": state.fluid_density_deviation,
            "pressure": state.fluid_pressure,
            "neighbor_count": state.fluid_neighbor_counts,
            "free_surface_score": state.fluid_free_surface_score,
            "low_density_flag": state.fluid_low_density_mask.astype(np.float64),
            "low_neighbor_flag": state.fluid_low_neighbor_mask.astype(np.float64),
        }
        return np.asarray(mapping[scalar_name], dtype=np.float64)

    def _normalize_scalar_name(self, name: str) -> str:
        scalar_name = str(name).strip().lower()
        if scalar_name not in self.AVAILABLE_SCALARS:
            available = ", ".join(self.AVAILABLE_SCALARS)
            raise ValueError(f"Unsupported scalar '{name}'. Available scalars: {available}")
        return scalar_name


def list_viewer_scalars() -> tuple[str, ...]:
    return PyVistaSceneViewer.AVAILABLE_SCALARS
