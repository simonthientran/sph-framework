"""
Live diagnostic chart strip — CFD-style convergence and field monitors.

Layout (bottom of the main window):
┌── DENSITY ERROR ─────┬── |v| MAX ────────┬── TIME STEP Δt ───┬── VELOCITY PROFILE ┐
│  [rolling line plot] │ [rolling line plot]│ [rolling line plot]│ [spatial snapshot] │
│  step axis ──────────│ step axis ─────────│ step axis ─────────│ x [m] ─────────── │
└──────────────────────┴───────────────────┴────────────────────┴────────────────────┘
"""

from __future__ import annotations

from collections import deque
from typing import Optional, Sequence

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from sph.core.backend import SimulationStateView, StepDiagnostics

# ── Theme ────────────────────────────────────────────────────────────────────
_BG       = "#0c1018"
_BG_STRIP = "#0f1420"
_TICK_PEN = pg.mkPen((65, 85, 110))
_TICK_TXT = pg.mkPen((95, 118, 148))
_GRID_A   = 0.15
_TEXT_DIM = "#3d5068"
_TEXT_MED = "#6a7d96"
_TEXT_HI  = "#c8d4e4"
_BORDER   = "#1a2535"

_ACCENT_GREEN  = "#4ade80"
_ACCENT_CYAN   = "#00b4d8"
_ACCENT_PURPLE = "#a78bfa"
_ACCENT_AMBER  = "#f59e0b"

_HISTORY      = 500   # rolling window length (steps)
_PROFILE_BINS = 48

# Compact 8-pt font for axis tick labels
_TICK_FONT = QtGui.QFont()
_TICK_FONT.setPointSize(8)


# ── PlotWidget factory ───────────────────────────────────────────────────────

def _make_plot(x_label: str = "Step") -> pg.PlotWidget:
    """
    Styled dark PlotWidget with both axes visible and compact tick labels.
    x_label: text shown below the bottom axis tick marks (short, e.g. "Step").
    """
    plot = pg.PlotWidget(background=_BG)
    plot.setContentsMargins(0, 0, 0, 0)

    vb = plot.getViewBox()
    vb.setBorder(pg.mkPen(None))
    vb.setMouseEnabled(x=False, y=False)

    # ── Left (y) axis ────────────────────────────────────────────────────────
    left = plot.getAxis("left")
    left.setPen(_TICK_PEN)
    left.setTextPen(_TICK_TXT)
    left.setTickFont(_TICK_FONT)
    left.setStyle(tickLength=-4, tickTextOffset=2, stopAxisAtTick=(True, True))
    left.setWidth(44)

    # ── Bottom (x) axis — always visible ────────────────────────────────────
    bottom = plot.getAxis("bottom")
    bottom.setPen(_TICK_PEN)
    bottom.setTextPen(_TICK_TXT)
    bottom.setTickFont(_TICK_FONT)
    bottom.setStyle(tickLength=-3, tickTextOffset=2)
    bottom.setHeight(28)          # enough for ticks + 1 row of labels
    # Unit label sits right of the axis ticks, not as a rotated label
    if x_label:
        bottom.setLabel(
            f'<span style="color:{_TEXT_DIM}; font-size:8px">{x_label}</span>'
        )

    for ax in ("right", "top"):
        plot.getAxis(ax).hide()

    plot.showGrid(x=False, y=True, alpha=_GRID_A)
    plot.setMenuEnabled(False)
    return plot


# ── MiniTimeSeries ────────────────────────────────────────────────────────────

class MiniTimeSeries(QtWidgets.QWidget):
    """Rolling time-series with compact title, y-axis, and step-number x-axis."""

    def __init__(
        self,
        title: str,
        unit: str,
        color: str,
        ref_lines: Sequence[tuple[float, str]] = (),
        fmt: str = ".3g",
        logy: bool = False,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._title_str = title
        self._unit = unit
        self._fmt = fmt

        vbox = QtWidgets.QVBoxLayout(self)
        vbox.setContentsMargins(2, 2, 2, 0)
        vbox.setSpacing(1)

        # ── Title row ────────────────────────────────────────────────────────
        self._title_lbl = QtWidgets.QLabel(title)
        self._title_lbl.setFixedHeight(14)
        self._title_lbl.setStyleSheet(
            f"background:transparent; color:{_TEXT_MED}; "
            f"font-size:9px; font-weight:700; letter-spacing:0.8px;"
        )
        vbox.addWidget(self._title_lbl)

        # ── Plot ─────────────────────────────────────────────────────────────
        self._plot = _make_plot(x_label="Step")
        if logy:
            self._plot.setLogMode(x=False, y=True)
        vbox.addWidget(self._plot, stretch=1)

        self._curve = pg.PlotCurveItem(
            pen=pg.mkPen(color, width=1.5), antialias=True,
        )
        self._plot.addItem(self._curve)

        for val, ref_color in ref_lines:
            self._plot.addItem(pg.InfiniteLine(
                pos=val, angle=0,
                pen=pg.mkPen(ref_color, width=1.0,
                             style=QtCore.Qt.PenStyle.DashLine),
            ))

        # ── Data buffers ──────────────────────────────────────────────────────
        self._steps: deque[int]   = deque(maxlen=_HISTORY)
        self._vals:  deque[float] = deque(maxlen=_HISTORY)

    def push(self, step: int, value: float) -> None:
        self._steps.append(step)
        self._vals.append(value)

        n = len(self._steps)
        x = np.fromiter(self._steps, dtype=np.float64, count=n)
        y = np.fromiter(self._vals,  dtype=np.float64, count=n)

        self._curve.setData(x, y)

        # Scrolling x-window: always show last _HISTORY steps
        self._plot.setXRange(x[-1] - _HISTORY, x[-1] + 4, padding=0)

        # Y auto-range with small head-room
        if n > 1:
            lo, hi = float(y.min()), float(y.max())
            span = (hi - lo) if hi > lo else max(abs(hi) * 0.05, 1e-9)
            self._plot.setYRange(lo - span * 0.06, hi + span * 0.14, padding=0)

        # Title: quantity  current-value  unit
        self._title_lbl.setText(
            f"<span style='color:{_TEXT_MED}'>{self._title_str}</span>"
            f"&nbsp;&nbsp;"
            f"<span style='color:{_TEXT_HI}; font-family:monospace'>"
            f"{value:{self._fmt}}&thinsp;{self._unit}</span>"
        )
        self._title_lbl.setTextFormat(QtCore.Qt.TextFormat.RichText)

    def reset(self) -> None:
        self._steps.clear()
        self._vals.clear()
        self._curve.setData([], [])
        self._title_lbl.setText(self._title_str)


# ── MiniSpatialChart ──────────────────────────────────────────────────────────

class MiniSpatialChart(QtWidgets.QWidget):
    """
    Velocity profile: binned mean |v| vs x-position (current-step snapshot).
    Shows spatial velocity distribution across the domain — equivalent to a
    line probe in STAR-CCM+ / CFD-Post.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)

        vbox = QtWidgets.QVBoxLayout(self)
        vbox.setContentsMargins(2, 2, 2, 0)
        vbox.setSpacing(1)

        self._title_lbl = QtWidgets.QLabel("VELOCITY PROFILE")
        self._title_lbl.setFixedHeight(14)
        self._title_lbl.setStyleSheet(
            f"background:transparent; color:{_TEXT_MED}; "
            f"font-size:9px; font-weight:700; letter-spacing:0.8px;"
        )
        vbox.addWidget(self._title_lbl)

        # x-axis: physical coordinates in metres
        self._plot = _make_plot(x_label="x [m]")
        vbox.addWidget(self._plot, stretch=1)

        self._curve = pg.PlotCurveItem(
            pen=pg.mkPen(_ACCENT_CYAN, width=1.5), antialias=True,
        )
        self._baseline = pg.PlotCurveItem(pen=pg.mkPen(None))
        self._fill = pg.FillBetweenItem(
            self._baseline, self._curve,
            brush=pg.mkBrush(0, 180, 216, 28),
        )
        self._plot.addItem(self._fill)
        self._plot.addItem(self._curve)

        self._x0: float = 0.0
        self._x1: float = 1.0

    def update_profile(self, state: SimulationStateView) -> None:
        pos = state.fluid_positions
        if pos.shape[0] < 4:
            return
        x0 = float(state.domain_min[0])
        x1 = float(state.domain_max[0])
        if x1 <= x0:
            return

        xp = pos[:, 0]
        sp = state.fluid_speed

        edges = np.linspace(x0, x1, _PROFILE_BINS + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        idx = np.clip(
            ((xp - x0) / (x1 - x0) * _PROFILE_BINS).astype(np.int32),
            0, _PROFILE_BINS - 1,
        )
        s = np.zeros(_PROFILE_BINS)
        c = np.zeros(_PROFILE_BINS, dtype=np.int32)
        np.add.at(s, idx, sp)
        np.add.at(c, idx, 1)
        v = np.where(c > 0, s / np.maximum(c, 1), 0.0)

        self._curve.setData(centers, v)
        self._baseline.setData(centers, np.zeros_like(v))

        if x0 != self._x0 or x1 != self._x1:
            self._plot.setXRange(x0, x1, padding=0.02)
            self._x0, self._x1 = x0, x1

        v_max = float(v.max())
        self._plot.setYRange(0, max(v_max * 1.15, 0.01), padding=0)

        self._title_lbl.setText(
            f"<span style='color:{_TEXT_MED}'>VELOCITY PROFILE</span>"
            f"&nbsp;&nbsp;"
            f"<span style='color:{_TEXT_HI}; font-family:monospace'>"
            f"peak {v_max:.3f}&thinsp;m/s</span>"
        )
        self._title_lbl.setTextFormat(QtCore.Qt.TextFormat.RichText)

    def reset(self) -> None:
        self._curve.setData([], [])
        self._baseline.setData([], [])
        self._title_lbl.setText("VELOCITY PROFILE")


# ── ChartStrip ────────────────────────────────────────────────────────────────

class ChartStrip(QtWidgets.QWidget):
    """
    Bottom chart strip — four mini charts in a horizontal row.

    1. Density Error [%]  — convergence quality (warn/fail threshold lines)
    2. |v| max [m/s]      — flow dynamics
    3. Time step Δt [s]   — adaptive CFL stepping (log scale)
    4. Velocity profile   — spatial |v| distribution (current frame)
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("chart_strip")
        self.setFixedHeight(172)          # tall enough for x-axis labels
        self.setStyleSheet(f"QWidget#chart_strip {{ background:{_BG_STRIP}; }}")

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        sep = QtWidgets.QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background:{_BORDER};")
        outer.addWidget(sep)

        row = QtWidgets.QWidget()
        row.setStyleSheet(f"background:{_BG_STRIP};")
        hbox = QtWidgets.QHBoxLayout(row)
        hbox.setContentsMargins(6, 4, 6, 4)
        hbox.setSpacing(1)
        outer.addWidget(row, stretch=1)

        self._rho_err = MiniTimeSeries(
            title="DENSITY ERROR", unit="%", color=_ACCENT_GREEN,
            ref_lines=[(4.0, _ACCENT_AMBER), (8.0, "#f43f5e")],
            fmt=".2f",
        )
        hbox.addWidget(self._rho_err, stretch=1)
        hbox.addWidget(_vline())

        self._vmax = MiniTimeSeries(
            title="|v| MAX", unit="m/s", color=_ACCENT_CYAN, fmt=".3f",
        )
        hbox.addWidget(self._vmax, stretch=1)
        hbox.addWidget(_vline())

        self._dt_chart = MiniTimeSeries(
            title="TIME STEP  Δt", unit="s", color=_ACCENT_PURPLE,
            fmt=".2e", logy=True,
        )
        hbox.addWidget(self._dt_chart, stretch=1)
        hbox.addWidget(_vline())

        self._vprofile = MiniSpatialChart()
        hbox.addWidget(self._vprofile, stretch=1)

    def update(  # type: ignore[override]
        self,
        step: int,
        diag: StepDiagnostics,
        state: SimulationStateView,
    ) -> None:
        self._rho_err.push(step, diag.rho_error_mean * 100.0)
        self._vmax.push(step, diag.velocity_max)
        self._dt_chart.push(step, diag.dt)
        self._vprofile.update_profile(state)

    def reset(self) -> None:
        self._rho_err.reset()
        self._vmax.reset()
        self._dt_chart.reset()
        self._vprofile.reset()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _vline() -> QtWidgets.QFrame:
    f = QtWidgets.QFrame()
    f.setFrameShape(QtWidgets.QFrame.Shape.VLine)
    f.setFixedWidth(1)
    f.setStyleSheet(f"background:{_BORDER}; border:none;")
    return f
