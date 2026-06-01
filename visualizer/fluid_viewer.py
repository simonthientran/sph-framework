"""
Professional SPH fluid viewer using VTK screen-space fluid rendering (SSFR).
Produces a smooth refractive fluid surface from raw particle positions.

Requires VTK 9.x:
    pip install vtk

Usage:
    PYTHONPATH=src python visualizer/fluid_viewer.py \\
        --scene scenes/examples/dam_break_3d.json \\
        --steps 2000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import vtk
import vtk.util.numpy_support as vnp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ---------------------------------------------------------------------------
# VTK helpers
# ---------------------------------------------------------------------------

def _make_polydata(positions: np.ndarray) -> vtk.vtkPolyData:
    pts = vtk.vtkPoints()
    pts.SetData(vnp.numpy_to_vtk(positions.astype(np.float32), deep=True))

    verts = vtk.vtkCellArray()
    ids = np.arange(len(positions), dtype=np.int64)
    cells = np.column_stack([np.ones(len(positions), dtype=np.int64), ids]).ravel()
    verts.ImportLegacyFormat(vnp.numpy_to_vtkIdTypeArray(cells, deep=True))

    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetVerts(verts)
    return pd


def _build_fluid_mapper(radius: float) -> vtk.vtkOpenGLFluidMapper:
    mapper = vtk.vtkOpenGLFluidMapper()

    # Surface smoothing — narrow-range filter, 3 smoothing passes
    mapper.SetSurfaceFilterMethod(vtk.vtkOpenGLFluidMapper.NarrowRange)
    mapper.SetSurfaceFilterIterations(3)
    mapper.SetSurfaceFilterRadius(5)

    # Depth reconstruction particle radius
    mapper.SetParticleRadius(radius * 2.5)

    # Water optical properties
    mapper.SetRefractiveIndex(1.33)
    mapper.SetAttenuationColor(0.1, 0.5, 0.9)
    mapper.SetAttenuationScale(0.5)
    mapper.SetAdditionalReflection(0.3)
    mapper.SetRefractionScale(0.07)
    mapper.SetParticleColorPower(0.1)
    mapper.SetParticleColorScale(0.57)
    mapper.SetOpaqueColor(0.05, 0.25, 0.65)      # deep water colour

    mapper.SetDisplayMode(vtk.vtkOpenGLFluidMapper.UnfilteredOpaqueSurface)

    return mapper


def _build_fallback_mapper(radius: float) -> vtk.vtkPointGaussianMapper:
    """Sphere-impostor fallback when SSFR is unavailable."""
    mapper = vtk.vtkPointGaussianMapper()
    mapper.SetScaleFactor(radius * 1.2)
    mapper.EmissiveOff()
    mapper.SetSplatShaderCode(
        "//VTK::Color::Impl\n"
        "float dist = dot(offsetVCVSOutput.xy, offsetVCVSOutput.xy);\n"
        "if (dist > 1.0) { discard; }\n"
        "float nz = sqrt(1.0 - dist);\n"
        "vec3 n = normalize(vec3(offsetVCVSOutput.xy, nz));\n"
        "vec3 L = normalize(vec3(1.0, 2.0, 1.5));\n"
        "float diff = max(dot(n, L), 0.0);\n"
        "float spec = pow(max(dot(reflect(-L,n), vec3(0,0,1)), 0.0), 40.0);\n"
        "ambientColor  = vec3(0.05, 0.15, 0.40);\n"
        "diffuseColor  = vec3(0.20, 0.55, 0.90) * diff;\n"
        "gl_FragColor  = vec4(ambientColor + diffuseColor + spec * 0.5, 1.0);\n"
    )
    return mapper


# ---------------------------------------------------------------------------
# Main viewer class
# ---------------------------------------------------------------------------

class SPHFluidViewer:
    def __init__(self, scene_path: str, n_steps: int = 5000,
                 screenshot_step: int | None = None,
                 use_fallback: bool = False):
        from sph.core.simulation import SimulationRunner

        self.runner = SimulationRunner(Path(scene_path), backend_name="numba_cpu")
        self.n_steps = n_steps
        self.screenshot_step = screenshot_step
        self.step = 0
        self.sim_time = 0.0
        self._sim = self.runner.backend.sim  # direct handle for positions/spacing
        self.dx = float(self._sim.spacing)
        self.use_fallback = use_fallback

        self._build_scene()

    # ------------------------------------------------------------------
    def _build_scene(self) -> None:
        # Renderer — dark gradient background
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.05, 0.05, 0.08)
        self.renderer.GradientBackgroundOn()
        self.renderer.SetBackground2(0.02, 0.02, 0.05)

        # Window
        self.window = vtk.vtkRenderWindow()
        self.window.SetSize(1400, 900)
        self.window.SetWindowName("SPH Framework  —  Professional Fluid Viewer")
        self.window.AddRenderer(self.renderer)

        # Interactor
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.window)
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

        # --- Initial fluid geometry ---
        pos = self._sim.fluid.positions
        self.pd = _make_polydata(pos)

        if self.use_fallback:
            # vtkPointGaussianMapper IS a vtkMapper → use vtkActor
            self.mapper = _build_fallback_mapper(self.dx)
            self.mapper.SetInputData(self.pd)
            self.fluid_actor: vtk.vtkProp3D = vtk.vtkActor()
            self.fluid_actor.SetMapper(self.mapper)
            prop = self.fluid_actor.GetProperty()
            prop.SetDiffuse(0.8)
            prop.SetSpecular(0.4)
            prop.SetSpecularPower(80.0)
        else:
            # vtkOpenGLFluidMapper IS a vtkAbstractVolumeMapper → use vtkVolume
            self.mapper = _build_fluid_mapper(self.dx)
            self.mapper.SetInputData(self.pd)
            vol_prop = vtk.vtkVolumeProperty()
            vol_prop.SetAmbient(0.2)
            vol_prop.SetDiffuse(0.8)
            vol_prop.SetSpecular(0.4)
            vol_prop.SetSpecularPower(80.0)
            self.fluid_actor = vtk.vtkVolume()
            self.fluid_actor.SetMapper(self.mapper)
            self.fluid_actor.SetProperty(vol_prop)
        self.renderer.AddVolume(self.fluid_actor) if isinstance(
            self.fluid_actor, vtk.vtkVolume
        ) else self.renderer.AddActor(self.fluid_actor)

        self._add_boundary_wireframe()
        self._add_lighting()
        self._add_text_overlay()

        # Timer at ~30 FPS; 5 sim steps per frame
        self.interactor.AddObserver("TimerEvent", self._on_timer)
        self._timer_interval_ms = 32

    # ------------------------------------------------------------------
    def _add_boundary_wireframe(self) -> None:
        boundary = self._sim.boundary
        if boundary is None or boundary.n == 0:
            return
        p = boundary.positions
        mn, mx = p.min(axis=0), p.max(axis=0)

        cube = vtk.vtkCubeSource()
        cube.SetBounds(mn[0], mx[0], mn[1], mx[1], mn[2], mx[2])
        cube.Update()

        m = vtk.vtkPolyDataMapper()
        m.SetInputData(cube.GetOutput())

        a = vtk.vtkActor()
        a.SetMapper(m)
        a.GetProperty().SetRepresentationToWireframe()
        a.GetProperty().SetColor(0.3, 0.3, 0.4)
        a.GetProperty().SetLineWidth(1.5)
        a.GetProperty().SetOpacity(0.4)
        self.renderer.AddActor(a)

    # ------------------------------------------------------------------
    def _add_lighting(self) -> None:
        self.renderer.RemoveAllLights()

        specs = [
            # (position,        focal_point,         intensity, color)
            ((2.0, 4.0, 3.0),  (0.5, 0.25, 0.25), 1.0, (1.00, 0.98, 0.95)),  # key
            ((-2.0, 1.0, 1.0), (0.5, 0.25, 0.25), 0.4, (0.80, 0.90, 1.00)),  # fill
            ((0.0, -1.0, -3.0),(0.5, 0.25, 0.25), 0.3, (0.60, 0.80, 1.00)),  # rim
        ]
        for pos, fp, intensity, color in specs:
            light = vtk.vtkLight()
            light.SetPosition(*pos)
            light.SetFocalPoint(*fp)
            light.SetIntensity(intensity)
            light.SetColor(*color)
            self.renderer.AddLight(light)

    # ------------------------------------------------------------------
    def _add_text_overlay(self) -> None:
        # Title bar — top-left
        self.title_actor = vtk.vtkTextActor()
        self.title_actor.SetInput("SPH Framework  |  DFSPH Solver")
        prop = self.title_actor.GetTextProperty()
        prop.SetFontSize(22)
        prop.SetColor(0.4, 0.8, 1.0)
        prop.SetFontFamilyToArial()
        prop.BoldOn()
        self.title_actor.GetPositionCoordinate().SetCoordinateSystemToDisplay()
        self.title_actor.SetPosition(20, 860)
        self.renderer.AddViewProp(self.title_actor)

        # Diagnostics panel — bottom-left
        self.diag_actor = vtk.vtkTextActor()
        prop2 = self.diag_actor.GetTextProperty()
        prop2.SetFontSize(17)
        prop2.SetColor(0.9, 0.9, 0.9)
        prop2.SetFontFamilyToArial()
        prop2.SetBackgroundColor(0.0, 0.0, 0.0)
        prop2.SetBackgroundOpacity(0.6)
        self.diag_actor.GetPositionCoordinate().SetCoordinateSystemToDisplay()
        self.diag_actor.SetPosition(20, 20)
        self.renderer.AddViewProp(self.diag_actor)
        self._update_diag(0.0, 0.0, 0.0, "?", "?")

    def _update_diag(self, vmax: float, rho_err_pct: float, dt: float,
                     iter_cd: object, iter_df: object) -> None:
        self.diag_actor.SetInput(
            f"  Step:      {self.step:6d}\n"
            f"  Time:      {self.sim_time:.4f} s\n"
            f"  dt:        {dt:.2e}\n"
            f"  Particles: {self._sim.fluid.n:6d}\n"
            f"  v_max:     {vmax:.4f} m/s\n"
            f"  rho_err:   {rho_err_pct:.2f}%\n"
            f"  iter_cd:   {iter_cd}    iter_df: {iter_df}"
        )

    # ------------------------------------------------------------------
    def _on_timer(self, obj: object, event: str) -> None:
        if event != "TimerEvent" or self.runner is None:
            return
        if self.step >= self.n_steps:
            return

        # 5 sim steps per render frame
        last_rt = None
        for _ in range(5):
            if self.step >= self.n_steps:
                break
            rt = self.runner.step()
            self.sim_time += rt.runtime.dt
            self.step += 1
            last_rt = rt

        if last_rt is None:
            return

        stats = last_rt.runtime
        fluid = self._sim.fluid
        pos = fluid.positions
        vel = fluid.velocities
        speed = np.linalg.norm(vel, axis=1)
        vmax = float(speed.max()) if speed.size else 0.0

        # Update particle positions in-place
        self.pd.GetPoints().SetData(
            vnp.numpy_to_vtk(pos.astype(np.float32), deep=True))
        self.pd.Modified()

        # Diagnostics
        metrics = getattr(stats, "solver_metrics", {}) if hasattr(stats, "solver_metrics") else {}
        iter_cd = metrics.get("iter_cd", getattr(stats, "iter_cd", "?"))
        iter_df = metrics.get("iter_df", getattr(stats, "iter_df", "?"))
        self._update_diag(
            vmax=vmax,
            rho_err_pct=stats.rho_error_mean * 100.0,
            dt=stats.dt,
            iter_cd=iter_cd,
            iter_df=iter_df,
        )

        # Screenshot at requested step
        if self.screenshot_step and self.step >= self.screenshot_step:
            self._save_screenshot(f"out/fluid_viewer_step{self.step:04d}.png")
            self.screenshot_step = None  # fire once

        self.window.Render()

    # ------------------------------------------------------------------
    def _save_screenshot(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.window.Render()
        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(self.window)
        w2i.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(path)
        writer.SetInputData(w2i.GetOutput())
        writer.Write()
        print(f"Screenshot saved: {path}")

    # ------------------------------------------------------------------
    def run(self) -> None:
        cam = self.renderer.GetActiveCamera()
        cam.SetPosition(1.8, 0.8, 1.2)
        cam.SetFocalPoint(0.5, 0.25, 0.25)
        cam.SetViewUp(0.0, 1.0, 0.0)
        self.renderer.ResetCameraClippingRange()

        self.interactor.Initialize()
        self.window.Render()
        self.interactor.CreateTimer(self._timer_interval_ms)
        if self.screenshot_step is not None:
            while self.screenshot_step is not None and self.step < self.n_steps:
                self._on_timer(self.interactor, "TimerEvent")
            return
        self.interactor.Start()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Professional SPH fluid viewer — VTK screen-space fluid rendering")
    parser.add_argument("--scene", default="scenes/examples/dam_break_3d.json")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--screenshot-step", type=int, default=200,
                        help="Save a PNG at this simulation step (0 = disabled)")
    parser.add_argument("--fallback", action="store_true",
                        help="Use sphere-impostor fallback instead of SSFR")
    args = parser.parse_args()

    print(f"VTK {vtk.vtkVersion.GetVTKVersion()}")
    try:
        vtk.vtkOpenGLFluidMapper()
        print("vtkOpenGLFluidMapper: YES")
    except AttributeError:
        print("vtkOpenGLFluidMapper: NO — switching to fallback")
        args.fallback = True

    print(f"Loading scene: {args.scene}")
    viewer = SPHFluidViewer(
        scene_path=args.scene,
        n_steps=args.steps,
        screenshot_step=args.screenshot_step or None,
        use_fallback=args.fallback,
    )
    print("Starting viewer... (close window to exit)")
    viewer.run()


if __name__ == "__main__":
    main()
