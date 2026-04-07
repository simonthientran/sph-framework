"""
Precomputed SPH playback viewer.

This viewer computes a simulation into an in-memory frame cache first, then
plays the cached frames smoothly in VTK. It also supports off-screen capture
for headless smoke checks.

Usage:
    PYTHONPATH=src python visualizer/fluid_viewer_pro.py \
        --scene scenes/examples/dam_break_3d.json \
        --steps 800 \
        --fps 30
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import threading
import time

import numpy as np
import vtk
import vtk.util.numpy_support as vnp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _make_polydata(positions: np.ndarray, scalars: np.ndarray | None = None) -> vtk.vtkPolyData:
    positions = np.asarray(positions, dtype=np.float32)
    n = int(len(positions))
    pts = vtk.vtkPoints()
    pts.SetData(vnp.numpy_to_vtk(positions, deep=True))

    ids = np.arange(n, dtype=np.int64)
    cells = np.column_stack([np.ones(n, dtype=np.int64), ids]).ravel()
    verts = vtk.vtkCellArray()
    verts.ImportLegacyFormat(vnp.numpy_to_vtkIdTypeArray(cells, deep=True))

    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetVerts(verts)
    if scalars is not None:
        arr = vnp.numpy_to_vtk(np.asarray(scalars, dtype=np.float32), deep=True)
        arr.SetName("speed")
        pd.GetPointData().SetScalars(arr)
        pd.GetPointData().SetActiveScalars("speed")
    return pd


def _velocity_lut() -> vtk.vtkLookupTable:
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfColors(256)
    lut.SetHueRange(0.66, 0.0)
    lut.SetSaturationRange(0.95, 1.0)
    lut.SetValueRange(0.80, 1.0)
    lut.SetAlphaRange(1.0, 1.0)
    lut.Build()
    return lut


def _build_primary_mapper(radius: float, lut: vtk.vtkLookupTable) -> vtk.vtkPointGaussianMapper:
    mapper = vtk.vtkPointGaussianMapper()
    mapper.SetScaleFactor(radius * 1.9)
    mapper.EmissiveOff()
    mapper.SetLookupTable(lut)
    mapper.SetColorModeToMapScalars()
    mapper.ScalarVisibilityOn()
    return mapper


def _build_ssfr_mapper(radius: float) -> vtk.vtkOpenGLFluidMapper:
    mapper = vtk.vtkOpenGLFluidMapper()
    mapper.SetSurfaceFilterMethod(vtk.vtkOpenGLFluidMapper.NarrowRange)
    mapper.SetSurfaceFilterIterations(5)
    mapper.SetSurfaceFilterRadius(8)
    mapper.SetParticleRadius(radius * 3.0)
    mapper.SetRefractiveIndex(1.33)
    mapper.SetAttenuationColor(0.08, 0.38, 0.82)
    mapper.SetAttenuationScale(0.8)
    mapper.SetAdditionalReflection(0.30)
    mapper.SetRefractionScale(0.07)
    mapper.SetOpaqueColor(0.04, 0.18, 0.55)
    mapper.SetDisplayMode(vtk.vtkOpenGLFluidMapper.FilteredOpaqueSurface)
    return mapper


@dataclass(slots=True)
class CachedFrame:
    positions: np.ndarray
    speeds: np.ndarray
    meta: dict


class SimulationCache:
    """Precompute simulation frames in the background."""

    def __init__(self, scene_path: str | Path, n_steps: int, record_every: int = 2, backend: str = "auto"):
        from sph.core.simulation import SimulationRunner

        self.scene_path = Path(scene_path)
        self.n_steps = int(n_steps)
        self.record_every = max(1, int(record_every))

        if backend == "auto" or backend == "numba_cuda":
            try:
                self.runner = SimulationRunner(self.scene_path, backend_name="numba_cuda")
                print("Backend: CUDA")
            except Exception as e:
                print(f"CUDA unavailable ({e}), using CPU")
                self.runner = SimulationRunner(self.scene_path, backend_name="numba_cpu")
        else:
            self.runner = SimulationRunner(self.scene_path, backend_name=backend)
            print(f"Backend: {backend}")

        # Disable scene exports while caching frames in memory.
        self.runner.backend.export_settings.csv.enabled = False
        self.runner.backend.export_settings.vtk.enabled = False
        self.runner._export_manager.settings.csv.enabled = False
        self.runner._export_manager.settings.vtk.enabled = False

        sim = self.runner.backend.sim
        self.initial_positions = sim.fluid.positions.copy()
        self._boundary_positions = (
            sim.boundary.positions.copy() if sim.boundary is not None and sim.boundary.n > 0 else np.zeros((0, sim.dim))
        )
        self._spacing = float(sim.spacing)

        self.frames: list[CachedFrame] = []
        self.ready = False
        self.failed = False
        self.error_message = ""
        self.progress = 0.0
        self.completed_steps = 0
        self.sim_time = 0.0

    def spacing(self) -> float:
        return self._spacing

    def boundary_positions(self) -> np.ndarray:
        return self._boundary_positions

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    def get_frame(self, index: int) -> CachedFrame:
        if not self.frames:
            raise RuntimeError("No cached frames are available.")
        index = max(0, min(int(index), len(self.frames) - 1))
        return self.frames[index]

    def run(self) -> None:
        try:
            self.frames.clear()
            self._record_frame(step=0, dt=0.0, rho_err=0.0, iter_cd="-", iter_df="-")
            for step_idx in range(1, self.n_steps + 1):
                result = self.runner.step()
                self.sim_time += float(result.runtime.dt)
                self.completed_steps = step_idx
                self.progress = step_idx / max(self.n_steps, 1)

                if step_idx % self.record_every == 0 or step_idx == self.n_steps:
                    metrics = result.runtime.solver_metrics if hasattr(result.runtime, "solver_metrics") else {}
                    self._record_frame(
                        step=step_idx,
                        dt=float(result.runtime.dt),
                        rho_err=float(result.runtime.rho_error_mean * 100.0),
                        iter_cd=metrics.get("iter_cd", "-"),
                        iter_df=metrics.get("iter_df", "-"),
                    )
            self.ready = True
        except Exception as exc:  # pragma: no cover - interactive failure path
            self.failed = True
            self.error_message = str(exc)

    def _record_frame(
        self,
        *,
        step: int,
        dt: float,
        rho_err: float,
        iter_cd: object,
        iter_df: object,
    ) -> None:
        fluid = self.runner.backend.sim.fluid
        positions = fluid.positions.copy()
        speeds = np.linalg.norm(fluid.velocities, axis=1).astype(np.float32, copy=False)
        vmax = float(speeds.max()) if speeds.size else 0.0
        meta = {
            "step": int(step),
            "time": float(self.sim_time),
            "dt": float(dt),
            "rho_err": float(rho_err),
            "iter_cd": iter_cd,
            "iter_df": iter_df,
            "vmax": vmax,
            "fluid_count": int(positions.shape[0]),
        }
        self.frames.append(CachedFrame(positions=positions, speeds=speeds.copy(), meta=meta))


class SPHFluidViewerPro:
    WINDOW_W = 1400
    WINDOW_H = 900

    def __init__(
        self,
        scene_path: str,
        *,
        n_steps: int = 600,
        fps: int = 30,
        record_every: int = 2,
        use_ssfr: bool = False,
        off_screen: bool = False,
        auto_screenshot_frame: int | None = None,
        quit_after_screenshot: bool = False,
        backend: str = "auto",
    ):
        self.cache = SimulationCache(scene_path, n_steps=n_steps, record_every=record_every, backend=backend)
        self.scene_path = str(scene_path)
        self.dx = self.cache.spacing()
        self.playback_fps = max(1, int(fps))
        self.use_ssfr = bool(use_ssfr)
        self.off_screen = bool(off_screen)
        self.auto_screenshot_frame = auto_screenshot_frame
        self.quit_after_screenshot = bool(quit_after_screenshot)

        self.frame = 0
        self.playing = False
        self.speed = 1.0
        self._cache_thread: threading.Thread | None = None
        self._playback_started = False
        self._last_frame_time = time.perf_counter()
        self._auto_capture_done = False

        self.lut = _velocity_lut()
        self._build_scene()

    def _build_scene(self) -> None:
        self.renderer = vtk.vtkRenderer()
        self.renderer.GradientBackgroundOn()
        self.renderer.SetBackground(0.02, 0.02, 0.04)
        self.renderer.SetBackground2(0.06, 0.07, 0.12)

        self.window = vtk.vtkRenderWindow()
        self.window.SetSize(self.WINDOW_W, self.WINDOW_H)
        self.window.SetWindowName("SPH Framework - Professional Playback Viewer")
        if self.off_screen:
            self.window.SetOffScreenRendering(1)
        self.window.AddRenderer(self.renderer)

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.window)
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

        initial_speeds = np.zeros(len(self.cache.initial_positions), dtype=np.float32)
        self.pd = _make_polydata(self.cache.initial_positions, initial_speeds)

        if self.use_ssfr:
            mapper = _build_ssfr_mapper(self.dx)
            mapper.SetInputData(self.pd)
            vol_prop = vtk.vtkVolumeProperty()
            vol_prop.SetDiffuse(0.7)
            vol_prop.SetSpecular(0.8)
            vol_prop.SetSpecularPower(120.0)
            self.fluid_prop: vtk.vtkProp3D = vtk.vtkVolume()
            self.fluid_prop.SetMapper(mapper)
            self.fluid_prop.SetProperty(vol_prop)
            self.renderer.AddVolume(self.fluid_prop)
        else:
            mapper = _build_primary_mapper(self.dx, self.lut)
            mapper.SetInputData(self.pd)
            self.fluid_prop = vtk.vtkActor()
            self.fluid_prop.SetMapper(mapper)
            self.fluid_prop.GetProperty().SetAmbient(0.10)
            self.renderer.AddActor(self.fluid_prop)
        self.fluid_mapper = mapper

        self._add_container_wireframe()
        self._add_floor()
        self._add_lighting()
        self._build_ui()

        self.interactor.AddObserver("KeyPressEvent", self._on_key)
        self.interactor.AddObserver("TimerEvent", self._on_timer)

    def _panel(
        self,
        text: str,
        x: int,
        y: int,
        *,
        size: int = 14,
        color: tuple[float, float, float] = (0.85, 0.90, 0.95),
        bg: tuple[float, float, float] = (0.04, 0.05, 0.08),
        bg_opacity: float = 0.78,
        bold: bool = False,
    ) -> vtk.vtkTextActor:
        actor = vtk.vtkTextActor()
        actor.SetInput(text)
        prop = actor.GetTextProperty()
        prop.SetFontSize(size)
        prop.SetColor(*color)
        prop.SetFontFamilyToArial()
        prop.SetBackgroundColor(*bg)
        prop.SetBackgroundOpacity(bg_opacity)
        if bold:
            prop.BoldOn()
        actor.SetPosition(x, y)
        self.renderer.AddViewProp(actor)
        return actor

    def _build_ui(self) -> None:
        self._panel("SPH FRAMEWORK", 20, 862, size=28, color=(0.30, 0.75, 1.00), bg_opacity=0.0, bold=True)
        self._panel("Precomputed Playback  -  DFSPH  -  3D", 260, 870, size=13, color=(0.45, 0.55, 0.65), bg_opacity=0.0)
        self.diag_actor = self._panel("  Computing...", 20, 20)
        self.play_actor = self._panel("  Loading cache...", 430, 22, size=14, color=(0.65, 0.80, 0.95))
        self._panel(
            "  CONTROLS\n"
            "  Space  -  Play / Pause\n"
            "  Left/Right - Step frames\n"
            "  [ ]    -  Speed x0.5 / x2\n"
            "  R      -  Rewind\n"
            "  S      -  Screenshot\n"
            "  E      -  Export VTK\n"
            "  Q      -  Quit",
            1215,
            20,
            size=13,
            color=(0.50, 0.60, 0.70),
        )
        self.status_actor = self._panel("  COMPUTING 0%", 1160, 862, size=14, color=(1.0, 0.75, 0.20))
        self.loading_actor = self._panel("  COMPUTING SIMULATION", 420, 430, size=28, color=(0.40, 0.80, 1.0), bold=True)

    def _add_container_wireframe(self) -> None:
        boundary = self.cache.boundary_positions()
        if boundary.size == 0:
            return
        mn = boundary.min(axis=0)
        mx = boundary.max(axis=0)
        cube = vtk.vtkCubeSource()
        cube.SetBounds(mn[0], mx[0], mn[1], mx[1], mn[2], mx[2])
        cube.Update()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(cube.GetOutput())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetRepresentationToWireframe()
        actor.GetProperty().SetColor(0.25, 0.30, 0.42)
        actor.GetProperty().SetLineWidth(1.2)
        actor.GetProperty().SetOpacity(0.35)
        self.renderer.AddActor(actor)

    def _add_floor(self) -> None:
        boundary = self.cache.boundary_positions()
        mn = boundary.min(axis=0) if boundary.size else np.zeros(3)
        mx = boundary.max(axis=0) if boundary.size else np.ones(3)
        y0 = float(mn[1]) - 0.005
        pad = 0.3

        plane = vtk.vtkPlaneSource()
        plane.SetOrigin(mn[0] - pad, y0, mn[2] - pad)
        plane.SetPoint1(mx[0] + pad, y0, mn[2] - pad)
        plane.SetPoint2(mn[0] - pad, y0, mx[2] + pad)
        plane.SetResolution(20, 20)
        plane.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(plane.GetOutput())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.08, 0.09, 0.12)
        actor.GetProperty().SetSpecular(0.4)
        actor.GetProperty().SetSpecularPower(60.0)
        actor.GetProperty().SetAmbient(0.3)
        actor.GetProperty().SetDiffuse(0.5)
        self.renderer.AddActor(actor)

    def _add_lighting(self) -> None:
        self.renderer.RemoveAllLights()
        for pos, focal, intensity, color in [
            ((2.0, 4.0, 3.0), (0.5, 0.25, 0.25), 1.0, (1.00, 0.98, 0.95)),
            ((-2.0, 1.0, 1.0), (0.5, 0.25, 0.25), 0.4, (0.80, 0.90, 1.00)),
            ((0.0, -1.0, -3.0), (0.5, 0.25, 0.25), 0.3, (0.60, 0.80, 1.00)),
        ]:
            light = vtk.vtkLight()
            light.SetPosition(*pos)
            light.SetFocalPoint(*focal)
            light.SetIntensity(intensity)
            light.SetColor(*color)
            self.renderer.AddLight(light)

    def _start_cache_thread(self) -> None:
        if self._cache_thread is not None:
            return
        self._cache_thread = threading.Thread(target=self.cache.run, daemon=True)
        self._cache_thread.start()

    def _update_loading(self) -> None:
        pct = int(self.cache.progress * 100.0)
        filled = int(self.cache.progress * 30)
        bar = "#" * filled + "." * (30 - filled)
        self.loading_actor.SetInput(
            "  COMPUTING SIMULATION\n\n"
            f"  [{bar}] {pct:3d}%\n\n"
            f"  {self.cache.completed_steps} / {self.cache.n_steps} steps"
        )
        self.diag_actor.SetInput(
            "  CACHE BUILD\n"
            f"  Scene     {Path(self.scene_path).name}\n"
            f"  Steps     {self.cache.completed_steps:6d} / {self.cache.n_steps}\n"
            f"  Frames    {self.cache.n_frames:6d}\n"
            f"  Mode      {'SSFR' if self.use_ssfr else 'points'}"
        )
        self.play_actor.SetInput("  Waiting for cached frames...")
        self.status_actor.SetInput(f"  COMPUTING {pct}%")
        self.status_actor.GetTextProperty().SetColor(1.0, 0.75, 0.20)

    def _apply_frame(self, frame_index: int) -> None:
        frame = self.cache.get_frame(frame_index)
        self.frame = max(0, min(frame_index, self.cache.n_frames - 1))
        self.pd = _make_polydata(frame.positions, frame.speeds)
        self.fluid_mapper.SetInputData(self.pd)

        vmax = max(float(frame.meta["vmax"]), 0.01)
        if not self.use_ssfr:
            self.fluid_mapper.SetScalarRange(0.0, max(vmax * 0.8, 0.01))

        pct = 100.0 * self.frame / max(self.cache.n_frames - 1, 1)
        filled = int(pct / 5.0)
        bar = "#" * filled + "." * (20 - filled)
        symbol = ">" if self.playing else "||"
        self.diag_actor.SetInput(
            "  PLAYBACK\n"
            f"  Frame     {self.frame:6d} / {self.cache.n_frames - 1}\n"
            f"  Step      {frame.meta['step']:6d}\n"
            f"  Time      {frame.meta['time']:.4f} s\n"
            f"  dt        {frame.meta['dt']:.2e} s\n"
            "  -----------------------\n"
            f"  Particles {frame.meta['fluid_count']:6d}\n"
            f"  v_max     {frame.meta['vmax']:.4f} m/s\n"
            f"  rho_err   {frame.meta['rho_err']:.3f} %\n"
            f"  iter_cd   {frame.meta['iter_cd']}\n"
            f"  iter_df   {frame.meta['iter_df']}"
        )
        self.play_actor.SetInput(
            f"  {symbol}  {bar}  {pct:5.1f}%  x{self.speed:.1f}"
        )
        self.loading_actor.SetInput("")
        if self.playing:
            self.status_actor.SetInput("  PLAYING")
            self.status_actor.GetTextProperty().SetColor(0.20, 1.00, 0.40)
        else:
            self.status_actor.SetInput("  PAUSED")
            self.status_actor.GetTextProperty().SetColor(0.70, 0.70, 0.70)

    def _on_timer(self, _obj: object, _event: str) -> None:
        if self.cache.failed:
            self.playing = False
            self.loading_actor.SetInput(f"  CACHE FAILED\n\n  {self.cache.error_message}")
            self.status_actor.SetInput("  FAILED")
            self.status_actor.GetTextProperty().SetColor(1.0, 0.30, 0.30)
            self.window.Render()
            return

        if not self.cache.ready:
            self._update_loading()
            self.window.Render()
            return

        if not self._playback_started:
            self._playback_started = True
            self.playing = True
            self._last_frame_time = time.perf_counter()
            self._apply_frame(0)
            self.window.Render()
            return

        if self.playing and self.frame < self.cache.n_frames - 1:
            now = time.perf_counter()
            target = 1.0 / max(self.playback_fps * self.speed, 1.0)
            if now - self._last_frame_time >= target:
                self._apply_frame(self.frame + 1)
                self._last_frame_time = now
        elif self.frame >= self.cache.n_frames - 1:
            self.playing = False
            self._apply_frame(self.frame)

        if self.auto_screenshot_frame is not None and not self._auto_capture_done:
            target_frame = max(0, min(int(self.auto_screenshot_frame), self.cache.n_frames - 1))
            if self.frame >= target_frame:
                self._auto_capture_done = True
                self._save_screenshot()
                if self.quit_after_screenshot:
                    self.interactor.TerminateApp()
                    return

        self.window.Render()

    def _on_key(self, _obj: object, _event: str) -> None:
        key = self.interactor.GetKeySym()
        if key == "space" and self.cache.ready:
            self.playing = not self.playing
            self._last_frame_time = time.perf_counter()
            self._apply_frame(self.frame)
            self.window.Render()
        elif key == "Right" and self.cache.ready:
            self.playing = False
            self._apply_frame(min(self.frame + 1, self.cache.n_frames - 1))
            self.window.Render()
        elif key == "Left" and self.cache.ready:
            self.playing = False
            self._apply_frame(max(self.frame - 1, 0))
            self.window.Render()
        elif key.lower() == "r" and self.cache.ready:
            self.playing = False
            self._apply_frame(0)
            self.window.Render()
        elif key == "bracketright":
            self.speed = min(self.speed * 2.0, 16.0)
            if self.cache.ready:
                self._apply_frame(self.frame)
                self.window.Render()
        elif key == "bracketleft":
            self.speed = max(self.speed * 0.5, 0.125)
            if self.cache.ready:
                self._apply_frame(self.frame)
                self.window.Render()
        elif key.lower() == "s":
            self._save_screenshot()
        elif key.lower() == "e" and self.cache.ready:
            self._export_current_vtk()
        elif key.lower() == "q":
            self.interactor.TerminateApp()

    def _save_screenshot(self, path: str | None = None) -> str:
        if path is None:
            Path("out/screenshots").mkdir(parents=True, exist_ok=True)
            path = f"out/screenshots/sph_frame_{self.frame:05d}.png"
        self.window.Render()
        window_to_image = vtk.vtkWindowToImageFilter()
        window_to_image.SetInput(self.window)
        window_to_image.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(path)
        writer.SetInputData(window_to_image.GetOutput())
        writer.Write()
        print(f"Screenshot: {path}")
        return path

    def _export_current_vtk(self) -> str:
        Path("out/export").mkdir(parents=True, exist_ok=True)
        path = f"out/export/sph_frame_{self.frame:05d}.vtk"
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(path)
        writer.SetInputData(self.pd)
        writer.Write()
        print(f"Exported VTK: {path}")
        return path

    def _run_offscreen_capture(self) -> None:
        self.cache.run()
        if self.cache.failed:
            raise RuntimeError(self.cache.error_message)
        self._playback_started = True
        target = 0 if self.auto_screenshot_frame is None else max(
            0, min(int(self.auto_screenshot_frame), self.cache.n_frames - 1)
        )
        self.playing = False
        self._apply_frame(target)
        self.window.Render()
        self._save_screenshot()

    def run(self) -> None:
        cam = self.renderer.GetActiveCamera()
        cam.SetPosition(1.8, 0.8, 1.2)
        cam.SetFocalPoint(0.5, 0.25, 0.25)
        cam.SetViewUp(0.0, 1.0, 0.0)
        self.renderer.ResetCameraClippingRange()

        if self.off_screen:
            self.window.Render()
            self._run_offscreen_capture()
            return

        self.interactor.Initialize()
        self.window.Render()
        self._start_cache_thread()
        self.interactor.CreateRepeatingTimer(32)
        self.interactor.Start()


def main() -> None:
    parser = argparse.ArgumentParser(description="Precomputed SPH playback viewer")
    parser.add_argument("--scene", default="scenes/examples/dam_break_3d.json")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--record-every", type=int, default=2)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--ssfr", action="store_true", help="Use VTK screen-space fluid rendering")
    parser.add_argument("--off-screen", action="store_true", help="Run without opening a GUI window")
    parser.add_argument("--auto-screenshot-frame", type=int, default=None)
    parser.add_argument("--quit-after-screenshot", action="store_true")
    parser.add_argument(
        "--backend", default="auto",
        choices=["auto", "numba_cuda", "numba_cpu"],
        help="Simulation backend: auto tries CUDA first, falls back to CPU")
    args = parser.parse_args()

    print(f"VTK {vtk.vtkVersion.GetVTKVersion()}")
    print("Viewer: precomputed playback")
    print(f"Scene:   {args.scene}")
    print(f"Steps:   {args.steps}")
    print(f"FPS:     {args.fps}")
    print(f"Backend: {args.backend}")

    viewer = SPHFluidViewerPro(
        args.scene,
        n_steps=args.steps,
        fps=args.fps,
        record_every=args.record_every,
        use_ssfr=args.ssfr,
        off_screen=args.off_screen,
        auto_screenshot_frame=args.auto_screenshot_frame,
        quit_after_screenshot=args.quit_after_screenshot,
        backend=args.backend,
    )
    viewer.run()


if __name__ == "__main__":
    main()
