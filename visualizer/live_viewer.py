"""
SPH Live Viewer — real-time visualization of running SPH simulations.

Zero coupling to simulation internals: only reads fluid.positions,
fluid.velocities, fluid.densities, fluid.rho0, and calls sim.step().

Layout (5 panels):
  LEFT col (full height): particle scatter colored by speed
  TOP CENTER:             vx(y) SPH vs analytical
  TOP RIGHT:              vmax / analytical (converges to 1.0)
  BOTTOM CENTER:          density error %
  BOTTOM RIGHT:           L2 error %

Usage:
  python visualizer/live_viewer.py [--solver wcsph|dfsph] [scene.json]
"""
from __future__ import annotations

import sys
import os
import time
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Allow running directly from the repo root
_repo = Path(__file__).parent.parent
sys.path.insert(0, str(_repo / "src"))


# ---------------------------------------------------------------------------

class SPHLiveViewer:
    """
    Real-time viewer for an SPH simulation.

    Parameters
    ----------
    simulator : Simulator
        Any object that exposes .step(), .fluid, and optionally .boundary.
    scene : dict
        Raw scene JSON dict used to extract display parameters.
    """

    def __init__(self, simulator, scene: dict):
        self.sim   = simulator
        self.scene = scene

        # --- domain bounds for the particle scatter plot ---
        domain = scene.get("domain", {})
        d_min  = domain.get("min", [0.0, 0.0])
        d_max  = domain.get("max", [1.0, 1.0])
        self.x_min = float(d_min[0])
        self.x_max = float(d_max[0])
        self.y_min = float(d_min[1])
        self.y_max = float(d_max[1])

        # --- fluid region bounds (for analytical profile) ---
        fluid_cfg   = scene.get("fluid", {})
        fy_min      = float(fluid_cfg.get("min", d_min)[1])
        fy_max      = float(fluid_cfg.get("max", d_max)[1])
        self.y_center = (fy_min + fy_max) / 2.0
        self.L        = (fy_max - fy_min) / 2.0

        # --- physics params for analytical overlay ---
        forces  = scene.get("forces", {})
        gravity = forces.get("gravity", [0.0, 0.0])
        self.f  = float(abs(gravity[0]))   # body force magnitude in x

        visc    = forces.get("viscosity", {})
        self.nu = float(visc.get("nu", 1.0))

        self.vmax_ana = self.f * self.L**2 / (2.0 * self.nu) if self.nu > 0 else 1.0

        # --- history buffers (capped at 300 frames) ---
        self._step_hist  = []
        self._vmax_hist  = []
        self._rho_hist   = []
        self._l2_hist    = []
        self._HIST       = 300

        self._frame      = 0
        self._t_last     = time.perf_counter()
        self._fps        = 0.0

        self._build_figure()

    # ------------------------------------------------------------------

    def _build_figure(self):
        dark  = "#1a1a2e"
        panel = "#16213e"
        tick  = "#e0e0e0"

        self.fig = plt.figure(figsize=(14, 8), facecolor=dark)
        gs = GridSpec(2, 3, figure=self.fig, hspace=0.42, wspace=0.38)

        self.ax_main = self.fig.add_subplot(gs[:, 0])   # particle scatter
        self.ax_prof = self.fig.add_subplot(gs[0, 1])   # velocity profile
        self.ax_vmax = self.fig.add_subplot(gs[0, 2])   # vmax convergence
        self.ax_rho  = self.fig.add_subplot(gs[1, 1])   # density error
        self.ax_l2   = self.fig.add_subplot(gs[1, 2])   # L2 error

        for ax in (self.ax_main, self.ax_prof,
                   self.ax_vmax, self.ax_rho, self.ax_l2):
            ax.set_facecolor(panel)
            ax.tick_params(colors=tick, labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor("#444")

        self.fig.patch.set_facecolor(dark)
        self._title = self.fig.suptitle(
            "SPH Live Viewer", color=tick, fontsize=12, fontweight="bold"
        )

    # ------------------------------------------------------------------

    def _extract_profile(self, fluid, n_bins: int = 20):
        """Bin-averaged vx(y) profile. Returns (y_centered, vx_mean, populated)."""
        y_raw = fluid.positions[:, 1]
        y_min, y_max = y_raw.min(), y_raw.max()
        bins    = np.linspace(y_min, y_max + 1e-10, n_bins + 1)
        centers = 0.5 * (bins[:-1] + bins[1:])
        vx_mean = np.zeros(n_bins)
        pop     = np.zeros(n_bins, dtype=bool)

        for k in range(n_bins):
            mask = (y_raw >= bins[k]) & (y_raw < bins[k + 1])
            if mask.sum() > 0:
                vx_mean[k] = fluid.velocities[mask, 0].mean()
                pop[k]     = True

        return centers - self.y_center, vx_mean, pop

    def _compute_L2(self, y_c: np.ndarray, vx: np.ndarray,
                    pop: np.ndarray) -> float:
        """Relative L2 error vs analytical Poiseuille profile."""
        if self.nu <= 0 or self.f <= 0:
            return 0.0
        interior = (np.abs(y_c) < 0.9 * self.L) & pop
        if interior.sum() == 0:
            return 0.0
        vx_ana = self.f / (2.0 * self.nu) * (self.L**2 - y_c**2)
        err  = np.sqrt(np.mean((vx[interior] - vx_ana[interior])**2))
        norm = np.sqrt(np.mean(vx_ana[interior]**2)) + 1e-12
        return err / norm * 100.0

    # ------------------------------------------------------------------

    def _update(self, _frame):
        steps_per_frame = 10
        for _ in range(steps_per_frame):
            self.sim.step()
        fluid = self.sim.fluid

        # ── FPS ───────────────────────────────────────────────────────
        now          = time.perf_counter()
        self._fps    = 1.0 / max(now - self._t_last, 1e-6)
        self._t_last = now
        self._frame += steps_per_frame

        # ── Particle scatter ─────────────────────────────────────────
        ax = self.ax_main
        ax.cla()
        ax.set_facecolor("#16213e")
        speed = np.linalg.norm(fluid.velocities, axis=1)
        ax.scatter(
            fluid.positions[:, 0], fluid.positions[:, 1],
            c=speed, cmap="plasma", s=4,
            vmin=0, vmax=max(speed.max(), 1e-10),
        )
        if hasattr(self.sim, "boundary") and self.sim.boundary is not None:
            bp = self.sim.boundary.positions
            ax.scatter(bp[:, 0], bp[:, 1], c="#555", s=2, alpha=0.5)

        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min - 0.01, self.y_max + 0.01)
        ax.set_aspect("equal")
        ax.set_title(
            f"Step {self.sim.current_step}  |  {self._fps:.1f} fps",
            color="#e0e0e0", fontsize=9,
        )
        ax.set_xlabel("x", color="#aaa", fontsize=8)
        ax.set_ylabel("y", color="#aaa", fontsize=8)
        ax.tick_params(colors="#e0e0e0", labelsize=7)

        # ── Velocity profile ─────────────────────────────────────────
        y_c, vx_bins, pop = self._extract_profile(fluid)

        ax = self.ax_prof
        ax.cla()
        ax.set_facecolor("#16213e")
        ax.plot(vx_bins[pop], y_c[pop], "o-",
                color="#e94560", ms=3, lw=1.5, label="SPH")
        if self.f > 0 and self.nu > 0:
            y_fine  = np.linspace(-self.L, self.L, 200)
            vx_fine = np.maximum(
                self.f / (2.0 * self.nu) * (self.L**2 - y_fine**2), 0.0
            )
            ax.plot(vx_fine, y_fine, "--", color="#4ecca3", lw=1.5,
                    label="Analytical")
        ax.set_title("vx(y)", color="#e0e0e0", fontsize=9)
        ax.set_xlabel("vx", color="#aaa", fontsize=8)
        ax.set_ylabel("y − y_c", color="#aaa", fontsize=8)
        ax.tick_params(colors="#e0e0e0", labelsize=7)
        ax.legend(fontsize=7, labelcolor="#e0e0e0", facecolor="#1a1a2e",
                  edgecolor="#444")

        # ── History metrics ──────────────────────────────────────────
        vmax_sim = fluid.velocities[:, 0].max()
        rho_err  = (np.abs(fluid.densities - fluid.rho0).mean()
                    / fluid.rho0 * 100.0)
        l2       = self._compute_L2(y_c, vx_bins, pop)

        self._step_hist.append(self.sim.current_step)
        self._vmax_hist.append(vmax_sim / self.vmax_ana if self.vmax_ana > 0 else 0)
        self._rho_hist.append(rho_err)
        self._l2_hist.append(l2)

        # cap history
        H = self._HIST
        sh = self._step_hist[-H:]

        # (ax, data, title, color, ref_line)
        panels = [
            (self.ax_vmax, self._vmax_hist[-H:],
             "vmax / analytical", "#4ecca3", 1.0),
            (self.ax_rho,  self._rho_hist[-H:],
             "density error %",   "#f5a623", 2.0),
            (self.ax_l2,   self._l2_hist[-H:],
             "L2 error %",        "#7b68ee", 10.0),
        ]
        for ax, data, title, color, ref in panels:
            ax.cla()
            ax.set_facecolor("#16213e")
            ax.plot(sh, data, color=color, lw=1.5)
            ax.axhline(ref, color="#666", ls="--", lw=0.8)
            ax.set_title(title, color="#e0e0e0", fontsize=9)
            ax.tick_params(colors="#e0e0e0", labelsize=7)
            ax.set_xlabel("step", color="#aaa", fontsize=7)

        self._title.set_text(
            f"SPH Live Viewer  |  step {self.sim.current_step}"
            f"  |  {self._fps:.1f} fps"
        )
        return []

    # ------------------------------------------------------------------

    def run(self, interval_ms: int = 50):
        """Start the live animation loop (blocks until window closed)."""
        self.anim = animation.FuncAnimation(
            self.fig,
            self._update,
            interval=interval_ms,
            blit=False,
            cache_frame_data=False,
        )
        plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="SPH Live Viewer")
    p.add_argument(
        "scene",
        nargs="?",
        default=str(_repo / "scenes" / "examples" / "pipe_flow_2d.json"),
        help="Path to scene JSON (default: pipe_flow_2d.json)",
    )
    p.add_argument(
        "--solver",
        choices=["wcsph", "dfsph"],
        default="wcsph",
        help="Solver to use (default: wcsph)",
    )
    p.add_argument(
        "--interval",
        type=int,
        default=50,
        help="Animation update interval in ms (default: 50)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    from sph.simulator_new import Simulator

    scene_path = Path(args.scene)
    print(f"Scene:  {scene_path}")
    print(f"Solver: {args.solver}")
    print("Building simulation...")

    sim = Simulator(scene_path, solver=args.solver)

    with open(scene_path) as fh:
        scene_dict = json.load(fh)

    print(f"Particles: {sim.fluid.n} fluid"
          + (f", {sim.boundary.n} boundary" if sim.boundary else ""))
    print("Launching viewer... (close window to exit)")

    viewer = SPHLiveViewer(sim, scene_dict)
    viewer.run(interval_ms=args.interval)
