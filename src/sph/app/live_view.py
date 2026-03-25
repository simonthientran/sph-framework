"""Real-time particle visualization using PyQtGraph."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from sph.core.backend import SimulationStateView


class ParticleView(pg.GraphicsLayoutWidget):
    """Scatter-based visualization with switchable scalar overlays."""

    COLOR_MODES = {
        "type": "Particle Type",
        "speed": "Velocity Magnitude",
        "density": "Density",
        "pressure": "Pressure",
    }

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent=parent)
        self.setBackground("k")
        self.view = self.addPlot()
        self.view.showGrid(x=True, y=True, alpha=0.2)
        self.view.setLabel("left", "y [m]")
        self.view.setLabel("bottom", "x [m]")
        self.view.setAspectLocked(True, ratio=1.0)

        self.fluid_item = pg.ScatterPlotItem(size=6.0, pen=pg.mkPen(None))
        self.boundary_item = pg.ScatterPlotItem(
            size=4.0,
            brush=pg.mkBrush(200, 200, 200, 160),
            pen=pg.mkPen(120, 120, 120, 220),
        )
        self.view.addItem(self.boundary_item)
        self.view.addItem(self.fluid_item)

        self._color_mode = "type"
        self._state_cache: Optional[SimulationStateView] = None
        self._color_maps = {
            "speed": pg.ColorMap(
                [0.0, 0.4, 0.8, 1.0],
                [(5, 35, 120), (0, 180, 255), (255, 230, 90), (255, 80, 0)],
            ),
            "density": pg.ColorMap(
                [0.0, 0.5, 1.0],
                [(30, 30, 110), (80, 200, 120), (255, 235, 80)],
            ),
            "pressure": pg.ColorMap(
                [0.0, 0.5, 1.0],
                [(20, 20, 80), (60, 60, 200), (255, 120, 120)],
            ),
        }
        self._type_brush = pg.mkBrush(50, 200, 255, 200)
        self._range_cache: Dict[str, Tuple[float, float]] = {
            "speed": (0.0, 1.0),
            "density": (950.0, 1050.0),
            "pressure": (-1000.0, 1000.0),
        }
        self._flash_pen = pg.mkPen(None)

    # ------------------------------------------------------------------ configuration
    def set_domain(self, domain_min: np.ndarray, domain_max: np.ndarray) -> None:
        x0, y0 = float(domain_min[0]), float(domain_min[1])
        x1, y1 = float(domain_max[0]), float(domain_max[1])
        self.view.setXRange(x0, x1, padding=0.02)
        self.view.setYRange(y0, y1, padding=0.02)

    def set_color_mode(self, mode: str) -> None:
        if mode == self._color_mode:
            return
        if mode not in self.COLOR_MODES:
            raise ValueError(f"Unsupported color mode '{mode}'")
        self._color_mode = mode
        if self._state_cache is not None:
            self._render_state(self._state_cache)

    # ------------------------------------------------------------------ rendering
    def render_state(self, state: SimulationStateView) -> None:
        self._state_cache = state
        self.set_domain(state.domain_min, state.domain_max)
        self._render_state(state)

    def _render_state(self, state: SimulationStateView) -> None:
        fluid_positions = state.fluid_positions
        if fluid_positions.size == 0:
            self.fluid_item.setData([], [])
        else:
            brushes = self._compute_fluid_brushes(state)
            self.fluid_item.setData(fluid_positions[:, 0], fluid_positions[:, 1], brush=brushes)

        boundary_positions = state.boundary_positions
        if boundary_positions.size == 0:
            self.boundary_item.setData([], [])
        else:
            self.boundary_item.setData(boundary_positions[:, 0], boundary_positions[:, 1])

    def _compute_fluid_brushes(self, state: SimulationStateView) -> list:
        if self._color_mode == "type" or state.fluid_positions.size == 0:
            return [self._type_brush] * state.fluid_positions.shape[0]

        scalars = self._extract_scalar_field(state)
        if scalars.size == 0:
            return [self._type_brush] * state.fluid_positions.shape[0]

        lo, hi = self._get_scalar_range(self._color_mode, scalars)
        norm = np.clip((scalars - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
        cmap = self._color_maps[self._color_mode]
        colors = cmap.map(norm, mode="byte")
        return [pg.mkBrush(int(c[0]), int(c[1]), int(c[2]), 230) for c in colors]

    def _extract_scalar_field(self, state: SimulationStateView) -> np.ndarray:
        if self._color_mode == "speed":
            return state.fluid_speed
        if self._color_mode == "density":
            return state.fluid_density
        if self._color_mode == "pressure":
            return state.fluid_pressure
        return np.zeros(state.fluid_positions.shape[0], dtype=np.float64)

    def _get_scalar_range(self, mode: str, scalars: np.ndarray) -> Tuple[float, float]:
        if scalars.size == 0:
            return self._range_cache.get(mode, (0.0, 1.0))
        lo = float(np.percentile(scalars, 5))
        hi = float(np.percentile(scalars, 95))
        if hi - lo < 1e-6:
            center = float(np.mean(scalars))
            delta = max(abs(center) * 0.01, 1.0)
            lo = center - delta
            hi = center + delta
        prev_lo, prev_hi = self._range_cache.get(mode, (lo, hi))
        blend = 0.35
        lo = prev_lo * (1.0 - blend) + lo * blend
        hi = prev_hi * (1.0 - blend) + hi * blend
        if hi - lo < 1e-9:
            hi = lo + 1e-9
        span = hi - lo
        lo -= 0.05 * span
        hi += 0.05 * span
        self._range_cache[mode] = (lo, hi)
        return lo, hi

    def flash_stability(self, stability: str) -> None:
        color = {
            "pass": (0, 180, 0),
            "warn": (255, 165, 0),
            "fail": (255, 60, 60),
        }.get(stability, (200, 200, 200))
        pen = pg.mkPen(color, width=2)
        self.view.getViewBox().setBorder(pen)
        QtCore.QTimer.singleShot(400, lambda: self.view.getViewBox().setBorder(self._flash_pen))
