from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

from sph.io.geometry_loader import load_mesh
from sph.io.playback_diagnostics import load_playback_diagnostics


def _trimesh_to_polydata(mesh) -> pv.PolyData:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.size == 0:
        return pv.PolyData(vertices)
    vtk_faces = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).ravel()
    return pv.PolyData(vertices, vtk_faces)


def _extract_point_scalars(poly: pv.PolyData) -> tuple[str, ...]:
    scalar_names: list[str] = []
    for name in poly.point_data.keys():
        array = np.asarray(poly.point_data[name])
        if array.ndim == 1 and array.shape[0] == poly.n_points:
            scalar_names.append(name)
    return tuple(scalar_names)


def _subset_polydata(poly: pv.PolyData, mask: np.ndarray) -> pv.PolyData:
    mask = np.asarray(mask, dtype=bool)
    points = np.asarray(poly.points, dtype=np.float64)
    subset = pv.PolyData(points[mask])
    for name in poly.point_data.keys():
        array = np.asarray(poly.point_data[name])
        subset.point_data[name] = array[mask]
    return subset


def collect_vtk_frame_paths(path: str | Path) -> list[Path]:
    source = Path(path)
    if source.is_file():
        return [source]
    return sorted(p for p in source.glob("*.vtk") if p.is_file())


@dataclass(slots=True)
class PlaybackFrameSummary:
    frame_index: int
    frame_count: int
    fluid_count: int
    boundary_count: int
    scalar_name: str
    scalar_min: float
    scalar_max: float
    frame_path: Path
    diagnostics: dict | None


class PyVistaPlaybackViewer:
    """Playback/inspection viewer for exported VTK particle frames."""

    def __init__(
        self,
        frame_source: str | Path,
        *,
        scene_path: str | Path | None = None,
        scalar: str = "particle_speed",
        off_screen: bool = False,
        window_size: tuple[int, int] = (1400, 900),
        show_boundary_mesh: bool = True,
        fluid_point_size: float = 14.0,
        boundary_point_size: float = 9.0,
        render_points_as_spheres: bool = True,
        playback_interval_ms: int = 180,
    ):
        self.frame_paths = collect_vtk_frame_paths(frame_source)
        if not self.frame_paths:
            raise ValueError(f"No VTK frames found in '{frame_source}'.")

        self.scene_path = Path(scene_path) if scene_path is not None else None
        self.show_boundary_mesh = bool(show_boundary_mesh)
        self.fluid_point_size = float(fluid_point_size)
        self.boundary_point_size = float(boundary_point_size)
        self.render_points_as_spheres = bool(render_points_as_spheres)
        self.playback_interval_ms = int(playback_interval_ms)

        self.plotter = pv.Plotter(off_screen=off_screen, window_size=window_size)
        self._fluid_actor = None
        self._boundary_actor = None
        self._mesh_actors: list = []
        self._status_actor = None
        self._help_actor = None
        self._frame_index = 0
        self._playing = False
        self._boundary_meshes = self._load_boundary_meshes() if self.show_boundary_mesh else []
        self._frame_diagnostics: dict | None = None

        first_frame = pv.read(self.frame_paths[0])
        if not isinstance(first_frame, pv.PolyData):
            first_frame = first_frame.extract_surface()
        self.available_scalars = _extract_point_scalars(first_frame)
        self.scalar_name = self._normalize_scalar_name(scalar)
        self._configure_plotter()
        self._apply_frame_polydata(first_frame)

    @property
    def frame_index(self) -> int:
        return self._frame_index

    @property
    def current_frame_path(self) -> Path:
        return self.frame_paths[self._frame_index]

    def current_summary(self) -> PlaybackFrameSummary:
        fluid_poly, boundary_poly = self._current_split_polydata()
        fluid_scalars = np.asarray(fluid_poly.point_data[self.scalar_name], dtype=np.float64)
        if fluid_scalars.size:
            scalar_min = float(np.nanmin(fluid_scalars))
            scalar_max = float(np.nanmax(fluid_scalars))
        else:
            scalar_min = 0.0
            scalar_max = 0.0
        return PlaybackFrameSummary(
            frame_index=self._frame_index,
            frame_count=len(self.frame_paths),
            fluid_count=int(fluid_poly.n_points),
            boundary_count=int(boundary_poly.n_points),
            scalar_name=self.scalar_name,
            scalar_min=scalar_min,
            scalar_max=scalar_max,
            frame_path=self.current_frame_path,
            diagnostics=self._frame_diagnostics,
        )

    def next_frame(self, loop: bool = True) -> PlaybackFrameSummary:
        if self._frame_index + 1 >= len(self.frame_paths):
            if not loop:
                return self.current_summary()
            self._frame_index = 0
        else:
            self._frame_index += 1
        self._load_current_frame()
        return self.current_summary()

    def previous_frame(self, loop: bool = True) -> PlaybackFrameSummary:
        if self._frame_index == 0:
            if not loop:
                return self.current_summary()
            self._frame_index = len(self.frame_paths) - 1
        else:
            self._frame_index -= 1
        self._load_current_frame()
        return self.current_summary()

    def set_frame(self, frame_index: int) -> PlaybackFrameSummary:
        self._frame_index = max(0, min(int(frame_index), len(self.frame_paths) - 1))
        self._load_current_frame()
        return self.current_summary()

    def toggle_play(self) -> None:
        self._playing = not self._playing
        self._update_text()

    def cycle_scalar(self) -> PlaybackFrameSummary:
        index = self.available_scalars.index(self.scalar_name)
        self.scalar_name = self.available_scalars[(index + 1) % len(self.available_scalars)]
        self._load_current_frame()
        return self.current_summary()

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
        frame_limit: int | None = None,
        prefix: str = "playback",
    ) -> list[Path]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        outputs: list[Path] = []
        total = len(self.frame_paths) if frame_limit is None else min(int(frame_limit), len(self.frame_paths))
        start = self._frame_index
        outputs.append(self.save_screenshot(directory / f"{prefix}_{0:04d}.png"))
        for offset in range(1, total):
            self.set_frame((start + offset) % len(self.frame_paths))
            outputs.append(self.save_screenshot(directory / f"{prefix}_{offset:04d}.png"))
        self.set_frame(start)
        return outputs

    def show(self, auto_close: bool = False) -> None:
        self.plotter.show(auto_close=auto_close)

    def close(self) -> None:
        self.plotter.close()

    def _configure_plotter(self) -> None:
        self.plotter.set_background("#f5f6fa", top="#e0ecff")
        self.plotter.enable_anti_aliasing("ssaa")
        self.plotter.add_axes()
        self.plotter.show_grid(color="#93a7bc")
        self.plotter.add_key_event("space", self.toggle_play)
        self.plotter.add_key_event("Right", lambda: self.next_frame(loop=True))
        self.plotter.add_key_event("Left", lambda: self.previous_frame(loop=True))
        self.plotter.add_key_event("s", lambda: self.cycle_scalar())
        self.plotter.add_timer_event(max_steps=1_000_000, duration=self.playback_interval_ms, callback=self._on_timer)
        self._help_actor = self.plotter.add_text(
            "space: play/pause   left/right: step   s: cycle scalar",
            position="lower_left",
            font_size=10,
            color="black",
        )

    def _on_timer(self, _step: int) -> None:
        if not self._playing:
            return
        self.next_frame(loop=True)

    def _load_current_frame(self) -> None:
        poly = pv.read(self.current_frame_path)
        if not isinstance(poly, pv.PolyData):
            poly = poly.extract_surface()
        self._frame_diagnostics = load_playback_diagnostics(self.current_frame_path.with_suffix(".json"))
        self._apply_frame_polydata(poly)

    def _apply_frame_polydata(self, poly: pv.PolyData) -> None:
        self._frame_polydata = poly
        if self._frame_diagnostics is None:
            self._frame_diagnostics = load_playback_diagnostics(self.current_frame_path.with_suffix(".json"))
        fluid_poly, boundary_poly = self._split_frame(poly)

        if self._fluid_actor is not None:
            self.plotter.remove_actor(self._fluid_actor)
        if self._boundary_actor is not None:
            self.plotter.remove_actor(self._boundary_actor)
        for actor in self._mesh_actors:
            self.plotter.remove_actor(actor)
        self._mesh_actors.clear()

        if boundary_poly.n_points:
            self._boundary_actor = self.plotter.add_mesh(
                boundary_poly,
                color="#4b5563",
                point_size=self.boundary_point_size,
                render_points_as_spheres=self.render_points_as_spheres,
                opacity=0.75,
                name="playback_boundary",
            )
        else:
            self._boundary_actor = None

        if fluid_poly.n_points:
            self._fluid_actor = self.plotter.add_mesh(
                fluid_poly,
                scalars=self.scalar_name,
                cmap="viridis",
                clim=self._scalar_limits(np.asarray(fluid_poly.point_data[self.scalar_name], dtype=np.float64)),
                point_size=self.fluid_point_size,
                render_points_as_spheres=self.render_points_as_spheres,
                scalar_bar_args={"title": self.scalar_name.replace("_", " ")},
                name="playback_fluid",
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
                    name=f"playback_boundary_mesh_{index}",
                )
                self._mesh_actors.append(actor)

        self._update_text()
        self.plotter.reset_camera_clipping_range()
        self.plotter.render()

    def _current_split_polydata(self) -> tuple[pv.PolyData, pv.PolyData]:
        return self._split_frame(self._frame_polydata)

    def _split_frame(self, poly: pv.PolyData) -> tuple[pv.PolyData, pv.PolyData]:
        if "is_boundary" not in poly.point_data:
            empty = pv.PolyData(np.zeros((0, 3), dtype=np.float64))
            return poly, empty
        is_boundary = np.asarray(poly.point_data["is_boundary"], dtype=np.float64) > 0.5
        return _subset_polydata(poly, ~is_boundary), _subset_polydata(poly, is_boundary)

    def _update_text(self) -> None:
        if self._status_actor is not None:
            self.plotter.remove_actor(self._status_actor)
        summary = self.current_summary()
        status = "PLAY" if self._playing else "PAUSE"
        text = (
            f"frame={summary.frame_index + 1}/{summary.frame_count}  {status}\n"
            f"fluid={summary.fluid_count} boundary={summary.boundary_count}\n"
            f"scalar={summary.scalar_name} range=[{summary.scalar_min:.4g}, {summary.scalar_max:.4g}]\n"
            f"{summary.frame_path.name}"
        )
        diagnostics = summary.diagnostics
        if diagnostics is not None:
            solver_metrics = diagnostics.get("solver_metrics", {})
            text += (
                f"\nstep={diagnostics.get('step', '-')}"
                f" dt={float(diagnostics.get('dt', 0.0)):.3e}"
                f" wall={float(diagnostics.get('wall_time_ms', 0.0)):.2f}ms"
            )
            text += (
                f"\nrho_mean={float(diagnostics.get('rho_mean', 0.0)):.2f}"
                f" rho_max={float(diagnostics.get('rho_max', 0.0)):.2f}"
                f" vmax={float(diagnostics.get('velocity_max', 0.0)):.3e}"
            )
            text += (
                f"\nneighbor_mean={float(diagnostics.get('neighbor_mean', 0.0)):.2f}"
                f" stability={diagnostics.get('stability', '-')}"
                f" iter_cd={float(solver_metrics.get('iter_cd', 0.0)):.0f}"
                f" iter_df={float(solver_metrics.get('iter_df', 0.0)):.0f}"
            )
            solver_health = str(diagnostics.get("solver_health_summary", "")).strip()
            if solver_health:
                text += f"\nhealth={solver_health}"
        else:
            text += "\ndiagnostics=missing"
        self._status_actor = self.plotter.add_text(
            text,
            position="upper_left",
            font_size=11,
            color="black",
            name="playback_status",
        )

    def _scalar_limits(self, values: np.ndarray) -> tuple[float, float]:
        if values.size == 0:
            return (0.0, 1.0)
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0
        return (vmin, vmax)

    def _normalize_scalar_name(self, scalar_name: str) -> str:
        if scalar_name not in self.available_scalars:
            available = ", ".join(self.available_scalars)
            raise ValueError(f"Unsupported playback scalar '{scalar_name}'. Available scalars: {available}")
        return scalar_name

    def _load_boundary_meshes(self) -> list[pv.PolyData]:
        if self.scene_path is None:
            return []
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
            meshes.append(_trimesh_to_polydata(load_mesh(str(mesh_path))))
        return meshes
