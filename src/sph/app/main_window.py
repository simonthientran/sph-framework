"""
SPH Framework — Professional CFD GUI
Siemens Xcelerator + NVIDIA Omniverse aesthetics.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QLabel, QPushButton, QSlider,
    QFileDialog, QGroupBox, QGridLayout,
    QProgressBar, QComboBox, QDoubleSpinBox,
    QSpinBox, QCheckBox, QTextEdit,
    QStatusBar, QToolBar, QSizePolicy, QMessageBox,
    QFrame,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QPainter, QColor
import pyqtgraph as pg
import pyqtgraph.opengl as gl


# ══════════════════════════════════════════════════════
# TASK 1 — PALETTE + build_stylesheet
# ══════════════════════════════════════════════════════

PALETTE = {
    'dark': {
        'bg_base':     '#080a0f',
        'bg_panel':    '#0e1118',
        'bg_widget':   '#151820',
        'bg_hover':    '#1c2130',
        'border':      '#1e2535',
        'border_acc':  '#00d4ff',
        'accent':      '#00d4ff',
        'accent_dim':  '#005f80',
        'accent_glow': '#003d52',
        'text_hi':     '#f0f4ff',
        'text_mid':    '#8892a4',
        'text_dim':    '#3d4a5c',
        'success':     '#00e676',
        'warning':     '#ffd740',
        'danger':      '#ff5252',
        'separator':   '#0e1118',
    },
    'light': {
        'bg_base':     '#f4f6fb',
        'bg_panel':    '#ffffff',
        'bg_widget':   '#f8fafd',
        'bg_hover':    '#eef2f9',
        'border':      '#dde3ef',
        'border_acc':  '#0057a8',
        'accent':      '#0057a8',
        'accent_dim':  '#cce0ff',
        'accent_glow': '#e8f1ff',
        'text_hi':     '#0a0f1e',
        'text_mid':    '#4a5568',
        'text_dim':    '#94a3b8',
        'success':     '#00875a',
        'warning':     '#b76e00',
        'danger':      '#cc0000',
        'separator':   '#f4f6fb',
    },
}

# Backwards-compat alias used in method bodies
COLORS = {
    'bg_dark':     PALETTE['dark']['bg_base'],
    'bg_panel':    PALETTE['dark']['bg_panel'],
    'bg_widget':   PALETTE['dark']['bg_widget'],
    'bg_hover':    PALETTE['dark']['bg_hover'],
    'accent':      PALETTE['dark']['accent'],
    'accent2':     PALETTE['dark']['accent_dim'],
    'accent_dim':  PALETTE['dark']['accent_glow'],
    'text_bright': PALETTE['dark']['text_hi'],
    'text_mid':    PALETTE['dark']['text_mid'],
    'text_dim':    PALETTE['dark']['text_dim'],
    'border':      PALETTE['dark']['border'],
    'success':     PALETTE['dark']['success'],
    'warning':     PALETTE['dark']['warning'],
    'error':       PALETTE['dark']['danger'],
}


def build_stylesheet(mode: str = 'dark') -> str:
    c = PALETTE[mode]
    return f"""
/* ── Base ─────────────────────────────────────────── */
QMainWindow, QDialog {{
    background: {c['bg_base']};
    color: {c['text_hi']};
}}
QWidget {{
    background: transparent;
    color: {c['text_hi']};
    font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif;
    font-size: 12px;
}}

/* ── Menu ─────────────────────────────────────────── */
QMenuBar {{
    background: {c['bg_panel']};
    color: {c['text_mid']};
    border-bottom: 1px solid {c['border']};
    padding: 2px 4px;
    spacing: 2px;
}}
QMenuBar::item {{
    padding: 4px 10px;
    border-radius: 3px;
}}
QMenuBar::item:selected {{
    background: {c['bg_hover']};
    color: {c['accent']};
}}
QMenu {{
    background: {c['bg_panel']};
    color: {c['text_hi']};
    border: 1px solid {c['border']};
    padding: 4px 0;
}}
QMenu::item {{
    padding: 6px 24px 6px 12px;
}}
QMenu::item:selected {{
    background: {c['accent_glow']};
    color: {c['accent']};
}}
QMenu::separator {{
    height: 1px;
    background: {c['border']};
    margin: 4px 8px;
}}

/* ── Toolbar ──────────────────────────────────────── */
QToolBar {{
    background: {c['bg_panel']};
    border-bottom: 1px solid {c['border']};
    padding: 4px 8px;
    spacing: 4px;
}}

/* ── GroupBox ─────────────────────────────────────── */
QGroupBox {{
    background: {c['bg_panel']};
    border: 1px solid {c['border']};
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 10px;
    padding-left: 8px;
    padding-right: 8px;
    padding-bottom: 8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    left: 10px;
    color: {c['accent']};
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 2px;
    background: {c['bg_panel']};
}}

/* ── Buttons ──────────────────────────────────────── */
QPushButton {{
    background: {c['bg_widget']};
    color: {c['text_mid']};
    border: 1px solid {c['border']};
    border-radius: 4px;
    padding: 5px 14px;
    font-size: 11px;
    font-weight: 500;
    min-height: 26px;
}}
QPushButton:hover {{
    background: {c['bg_hover']};
    border-color: {c['accent']};
    color: {c['accent']};
}}
QPushButton:pressed {{
    background: {c['accent_glow']};
}}
QPushButton:disabled {{
    color: {c['text_dim']};
    border-color: {c['border']};
}}
QPushButton#run_btn {{
    background: {c['accent_glow']};
    color: {c['accent']};
    border: 1px solid {c['accent']};
    font-weight: 700;
    letter-spacing: 1px;
}}
QPushButton#run_btn:hover {{
    background: {c['accent']};
    color: {c['bg_base']};
}}
QPushButton#danger_btn {{
    color: {c['danger']};
    border-color: {c['danger']};
}}
QPushButton#danger_btn:hover {{
    background: {c['danger']};
    color: {c['bg_base']};
}}

/* ── Inputs ───────────────────────────────────────── */
QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {{
    background: {c['bg_widget']};
    color: {c['text_hi']};
    border: 1px solid {c['border']};
    border-radius: 4px;
    padding: 3px 8px;
    min-height: 24px;
    font-family: 'IBM Plex Mono', 'Consolas', monospace;
    font-size: 11px;
}}
QSpinBox:focus, QDoubleSpinBox:focus,
QComboBox:focus, QLineEdit:focus {{
    border-color: {c['accent']};
    background: {c['accent_glow']};
}}
QComboBox::drop-down {{
    border: none;
    width: 24px;
}}
QComboBox QAbstractItemView {{
    background: {c['bg_panel']};
    color: {c['text_hi']};
    border: 1px solid {c['border']};
    selection-background-color: {c['accent_glow']};
    selection-color: {c['accent']};
    padding: 2px;
}}

/* ── Sliders ──────────────────────────────────────── */
QSlider::groove:horizontal {{
    background: {c['bg_widget']};
    height: 3px;
    border-radius: 2px;
    border: 1px solid {c['border']};
}}
QSlider::sub-page:horizontal {{
    background: {c['accent']};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {c['accent']};
    border: 2px solid {c['bg_panel']};
    width: 14px;
    height: 14px;
    border-radius: 7px;
    margin: -6px 0;
}}
QSlider::handle:horizontal:hover {{
    background: {c['text_hi']};
}}

/* ── CheckBox ─────────────────────────────────────── */
QCheckBox {{
    color: {c['text_mid']};
    spacing: 8px;
    font-size: 11px;
}}
QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {c['border']};
    border-radius: 3px;
    background: {c['bg_widget']};
}}
QCheckBox::indicator:checked {{
    background: {c['accent']};
    border-color: {c['accent']};
}}
QCheckBox:hover {{
    color: {c['text_hi']};
}}

/* ── Tabs ─────────────────────────────────────────── */
QTabWidget::pane {{
    background: {c['bg_panel']};
    border: 1px solid {c['border']};
    border-top: none;
    border-radius: 0 0 6px 6px;
}}
QTabBar::tab {{
    background: transparent;
    color: {c['text_dim']};
    border-bottom: 2px solid transparent;
    padding: 7px 16px;
    font-size: 11px;
    letter-spacing: 0.5px;
}}
QTabBar::tab:selected {{
    color: {c['accent']};
    border-bottom-color: {c['accent']};
}}
QTabBar::tab:hover:!selected {{
    color: {c['text_mid']};
}}

/* ── Console ──────────────────────────────────────── */
QTextEdit {{
    background: {c['bg_base']};
    color: #5af78e;
    border: 1px solid {c['border']};
    border-radius: 4px;
    font-family: 'IBM Plex Mono', 'Consolas', monospace;
    font-size: 10px;
    padding: 4px;
}}

/* ── Splitter ─────────────────────────────────────── */
QSplitter::handle {{
    background: {c['border']};
    width: 1px;
}}

/* ── Scrollbar ────────────────────────────────────── */
QScrollBar:vertical {{
    background: {c['bg_base']};
    width: 5px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {c['border']};
    border-radius: 3px;
    min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{
    background: {c['accent_dim']};
}}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QScrollBar:horizontal {{ height: 5px; }}
QScrollBar::handle:horizontal {{
    background: {c['border']};
    border-radius: 3px;
}}

/* ── Status bar ───────────────────────────────────── */
QStatusBar {{
    background: {c['bg_panel']};
    border-top: 1px solid {c['border']};
    color: {c['text_dim']};
    font-size: 10px;
    font-family: 'IBM Plex Mono', 'Consolas', monospace;
}}
QStatusBar::item {{ border: none; }}

/* ── Progress bar ─────────────────────────────────── */
QProgressBar {{
    background: {c['bg_widget']};
    border: 1px solid {c['border']};
    border-radius: 3px;
    color: {c['text_hi']};
    max-height: 14px;
}}
QProgressBar::chunk {{
    background: {c['accent']};
    border-radius: 2px;
}}

/* ── Labels ───────────────────────────────────────── */
QLabel#app_title {{
    color: {c['accent']};
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 3px;
}}
QLabel#app_sub {{
    color: {c['text_dim']};
    font-size: 10px;
    letter-spacing: 1.5px;
}}
QLabel#stat_val {{
    color: {c['text_hi']};
    font-family: 'IBM Plex Mono', 'Consolas', monospace;
    font-size: 17px;
    font-weight: 600;
}}
QLabel#stat_unit {{
    color: {c['text_dim']};
    font-size: 9px;
    letter-spacing: 1px;
}}
"""


# ══════════════════════════════════════════════════════
# TASK 3 — SCIENTIFIC COLORMAPS
# ══════════════════════════════════════════════════════

def _turbo(t: np.ndarray):
    r = np.clip(0.1357 + t * (4.5974 - t * (42.3277 - t * (130.5887 - t * (150.5846 - t * 58.1854)))), 0, 1)
    g = np.clip(0.0914 + t * (2.1856 + t * (4.8052 - t * (14.0776 - t * (4.2066 + t * 2.3758)))), 0, 1)
    b = np.clip(0.1067 + t * (12.5925 - t * (60.1097 - t * (109.0745 - t * (88.5066 - t * 26.8183)))), 0, 1)
    return r, g, b


def _plasma(t: np.ndarray):
    r = np.clip(0.050 + t * (2.694 - t * 1.744), 0, 1)
    g = np.clip(0.030 - t * 0.030 + t ** 2 * 0.010 + t ** 3 * 0.90, 0, 1)
    b = np.clip(0.527 + t * (1.000 - t * 2.527), 0, 1)
    return r, g, b


def _viridis(t: np.ndarray):
    r = np.clip(0.267 + t * (-0.003 + t * (1.398 - t * 0.662)), 0, 1)
    g = np.clip(0.005 + t * (1.085 - t * 0.090), 0, 1)
    b = np.clip(0.329 + t * (1.748 - t * 2.400 + t ** 2 * 0.853), 0, 1)
    return r, g, b


def _inferno(t: np.ndarray):
    r = np.clip(t * (3.0 - t * 1.5), 0, 1)
    g = np.clip(t * t * (3.0 - t * 2.5), 0, 1)
    b = np.clip(0.5 * (1 - np.cos(t * np.pi)), 0, 1) * (1 - t)
    return r, g, b


def _hot(t: np.ndarray):
    return np.clip(t * 3.0, 0, 1), np.clip(t * 3.0 - 1, 0, 1), np.clip(t * 3.0 - 2, 0, 1)


def _cool(t: np.ndarray):
    return t, 1 - t, np.ones_like(t)


def _coolwarm(t: np.ndarray):
    """Diverging blue→white→red map (CFD pressure look): t=0 blue (low),
    t=0.5 near-white (~0), t=1 red (high)."""
    t = np.clip(t, 0.0, 1.0)
    lo = np.array([0.23, 0.30, 0.75])   # blue
    mid = np.array([0.95, 0.95, 0.95])  # near-white
    hi = np.array([0.71, 0.02, 0.15])   # red
    s = np.clip(t * 2.0, 0, 1)[:, None]            # 0..0.5 → lo→mid
    s2 = np.clip((t - 0.5) * 2.0, 0, 1)[:, None]   # 0.5..1 → mid→hi
    lower = lo[None, :] * (1 - s) + mid[None, :] * s
    upper = mid[None, :] * (1 - s2) + hi[None, :] * s2
    out = np.where((t[:, None] < 0.5), lower, upper)
    return out[:, 0], out[:, 1], out[:, 2]


_CMAP_FNS = {
    'Turbo':   _turbo,
    'Plasma':  _plasma,
    'Viridis': _viridis,
    'Inferno': _inferno,
    'Hot':     _hot,
    'Cool':    _cool,
    'Coolwarm': _coolwarm,
}
COLORMAP_OPTIONS = tuple(_CMAP_FNS.keys())


def apply_colormap(values: np.ndarray, name: str = 'Turbo', alpha: float = 0.92) -> np.ndarray:
    n = len(values)
    vmax = float(np.percentile(values, 95)) if n else 1.0
    vmin = float(values.min()) if n else 0.0
    t = np.clip((values - vmin) / (vmax - vmin + 1e-10), 0, 1)
    fn = _CMAP_FNS.get(name, _turbo)
    r, g, b = fn(t)
    return np.stack([r, g, b, np.full(n, alpha, dtype=np.float32)], axis=1).astype(np.float32)


# ══════════════════════════════════════════════════════
# SIMULATION WORKER THREAD
# ══════════════════════════════════════════════════════

class SimWorker(QThread):
    step_done = pyqtSignal(dict)
    sim_error = pyqtSignal(str)

    def __init__(self, runner):
        super().__init__()
        self.runner = runner
        self._running = False
        self._paused = False

    def run(self):
        self._running = True
        while self._running:
            if self._paused:
                time.sleep(0.05)
                continue
            try:
                rt = self.runner.step()
                fl = self.runner.backend.sim.fluid
                pos = fl.positions.copy()
                spd = np.linalg.norm(fl.velocities, axis=1).copy()
                sim = self.runner.backend.sim
                metrics = {
                    'step':    int(sim.current_step),
                    'time':    float(getattr(sim, 't', 0.0)),
                    'dt':      float(rt.runtime.dt),
                    'vmax':    float(spd.max()) if len(spd) else 0.0,
                    'rho_err': float(rt.runtime.rho_error_mean) * 100.0,
                    'iter_cd': rt.runtime.solver_metrics.get('iter_cd', 0),
                    'iter_df': rt.runtime.solver_metrics.get('iter_df', 0),
                    'n_fluid': int(fl.n),
                    'reynolds_number': float(rt.runtime.reynolds_number),
                    'regime':  str(rt.runtime.regime),
                    'positions': pos,
                    'speeds':    spd,
                    'velocities': fl.velocities.copy(),
                    'p_display': (fl.p_display.copy()
                                  if hasattr(fl, 'p_display') else None),
                }
                self.step_done.emit(metrics)
            except Exception as e:
                self.sim_error.emit(str(e))
                break

    def pause(self):  self._paused = True
    def resume(self): self._paused = False
    def stop(self):   self._running = False; self._paused = False


# ══════════════════════════════════════════════════════
# TASK 2 — STAT WIDGET (Siemens Xcelerator KPI card)
# ══════════════════════════════════════════════════════

class StatWidget(QWidget):
    """
    KPI card with left accent bar that changes colour by threshold.

    ┌─────────────────────┐
    │ 2.88             %  │  ← large monospace value + unit
    │ L2 ERROR            │  ← dimmed uppercase label
    └─────────────────────┘
    """

    def __init__(
        self,
        label: str,
        unit: str = '',
        warn_threshold=None,
        danger_threshold=None,
    ):
        super().__init__()
        self._warn = warn_threshold
        self._danger = danger_threshold
        self._unit = unit
        self._dark = True
        self._bar_color = PALETTE['dark']['accent']

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 8, 8)
        layout.setSpacing(2)

        row = QHBoxLayout()
        self.val_lbl = QLabel('—')
        self.val_lbl.setObjectName('stat_val')
        self.unit_lbl = QLabel(unit)
        self.unit_lbl.setObjectName('stat_unit')
        self.unit_lbl.setAlignment(
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)
        row.addWidget(self.val_lbl)
        row.addStretch()
        row.addWidget(self.unit_lbl)
        layout.addLayout(row)

        self.lbl = QLabel(label.upper())
        self.lbl.setObjectName('stat_unit')
        layout.addWidget(self.lbl)

        self.setMinimumHeight(58)
        self._update_card_bg()

    def _update_card_bg(self):
        c = PALETTE['dark' if self._dark else 'light']
        self.setStyleSheet(
            f'StatWidget {{ background: {c["bg_widget"]}; '
            f'border: 1px solid {c["border"]}; border-radius: 4px; }}'
        )

    def set_dark(self, dark: bool):
        self._dark = dark
        self._update_card_bg()

    def set_value(self, v, fmt: str | None = None):
        if fmt:
            text = fmt.format(v)
        elif isinstance(v, float):
            text = f'{v:.4f}' if abs(v) < 10 else f'{v:.1f}'
        elif isinstance(v, int):
            text = f'{v:,}'
        else:
            text = str(v)
        self.val_lbl.setText(text)

        c = PALETTE['dark' if self._dark else 'light']
        color = c['text_hi']
        bar = c['accent']
        if (self._danger is not None
                and isinstance(v, (int, float)) and v > self._danger):
            color = c['danger']
            bar = c['danger']
        elif (self._warn is not None
              and isinstance(v, (int, float)) and v > self._warn):
            color = c['warning']
            bar = c['warning']

        self.val_lbl.setStyleSheet(
            f'color: {color}; '
            f'font-family: IBM Plex Mono, Consolas, monospace; '
            f'font-size: 17px; font-weight: 600;'
        )
        self._bar_color = bar
        self.update()

    def paintEvent(self, e):
        super().paintEvent(e)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(0, 4, 3, self.height() - 8, QColor(self._bar_color))
        p.end()


# ══════════════════════════════════════════════════════
# MAIN WINDOW
# ══════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(
        self,
        scene_paths: Optional[list] = None,
        initial_scene: Optional[Path] = None,
    ):
        super().__init__()
        self.runner = None
        self.worker: Optional[SimWorker] = None
        self._history: dict[str, list] = {
            'time': [], 'vmax': [], 'rho_err': [], 'iter_cd': []
        }
        self._show_boundary = True
        self._scene_R: float = 0.07
        self._last_frame_time: float = time.time()
        self._pause_at_step: int = 0
        self._dark_mode = True
        self._fps = 0.0
        self._colorbar_vmin = 0.0
        self._colorbar_vmax = 1.0
        # Part A: clipping plane (display-only section through the fluid).
        self._clip_axis: int | None = None      # None / 0=X / 1=Y / 2=Z
        self._clip_frac: float = 1.0             # slider fraction 0..1
        self._fluid_bbox: tuple[np.ndarray, np.ndarray] | None = None
        # Part B: inlet/outlet markers.
        self._show_io_markers = True

        self.setWindowTitle('SPH Framework  —  Professional Edition')
        self.resize(1600, 950)
        self.setMinimumSize(1200, 700)
        self.setStyleSheet(build_stylesheet('dark'))

        self._build_menu()
        self._build_toolbar()
        self._build_central()
        self._build_statusbar()

        if initial_scene:
            self._load_scene(str(initial_scene))
        elif scene_paths:
            self._load_scene(str(scene_paths[0]))

    # ── Menu ──────────────────────────────────────────

    def _build_menu(self):
        from PyQt6.QtGui import QKeySequence, QAction

        def act(menu, text, slot, shortcut=None, checkable=False):
            a = QAction(text, self)
            a.triggered.connect(slot)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            if checkable:
                a.setCheckable(True)
            menu.addAction(a)
            return a

        mb = self.menuBar()

        fm = mb.addMenu('File')
        act(fm, 'Open Scene...', self._open_scene, 'Ctrl+O')
        fm.addSeparator()
        act(fm, 'Export VTK...', self._export_vtk)
        act(fm, 'Screenshot...', self._screenshot)
        fm.addSeparator()
        act(fm, 'Quit', self.close, 'Ctrl+Q')

        sm = mb.addMenu('Simulation')
        act(sm, 'Run',       self._run,       'Space')
        act(sm, 'Pause',     self._pause,     'P')
        act(sm, 'Reset',     self._reset,     'Ctrl+R')
        sm.addSeparator()
        act(sm, 'Step Once', self._step_once, 'N')

        vm = mb.addMenu('View')
        act(vm, 'Reset Camera', self._reset_camera)
        act(vm, 'Focus bend', self._focus_bend)
        vm.addSeparator()
        self._act_boundary = act(
            vm, 'Show Boundary Particles',
            self._toggle_boundary, checkable=True)
        self._act_boundary.setChecked(True)
        self._act_axes = act(
            vm, 'Show axes', self._toggle_axes, checkable=True)
        self._act_axes.setChecked(True)
        vm.addSeparator()
        self._act_theme = vm.addAction('Light Mode', self._toggle_theme)

        hm = mb.addMenu('Help')
        act(hm, 'About', self._about)

    # ── TASK 4 — Toolbar ──────────────────────────────

    def _build_toolbar(self):
        tb = self.addToolBar('main')
        tb.setMovable(False)
        tb.setIconSize(QSize(16, 16))

        def spacer(w: int = 8):
            s = QWidget()
            s.setFixedWidth(w)
            tb.addWidget(s)

        def sep():
            f = QFrame()
            f.setFrameShape(QFrame.Shape.VLine)
            c = PALETTE['dark' if self._dark_mode else 'light']
            f.setStyleSheet(f'color: {c["border"]}; max-width: 1px;')
            f.setFixedHeight(22)
            tb.addWidget(f)

        def btn(text: str, slot, obj: str = '', shortcut: str = ''):
            b = QPushButton(text)
            if obj:
                b.setObjectName(obj)
            b.clicked.connect(slot)
            b.setFixedHeight(28)
            b.setMinimumWidth(70)
            if shortcut:
                b.setShortcut(shortcut)
            tb.addWidget(b)
            return b

        spacer(8)
        logo = QLabel('SPH FRAMEWORK')
        logo.setObjectName('app_title')
        tb.addWidget(logo)
        sub = QLabel('  DFSPH · 3D · GPU')
        sub.setObjectName('app_sub')
        tb.addWidget(sub)
        spacer(20)
        sep()
        spacer(8)

        btn('⬡  Open', self._open_scene, shortcut='Ctrl+O')
        spacer(4)
        self._btn_run   = btn('▶  Run',   self._run,   'run_btn', 'Space')
        self._btn_pause = btn('⏸  Pause', self._pause)
        self._btn_reset = btn('⏹  Reset', self._reset, shortcut='Ctrl+R')
        self._btn_pause.setEnabled(False)
        spacer(4)
        sep()
        spacer(4)
        btn('Step', self._step_once, shortcut='N')
        spacer(4)
        sep()
        spacer(4)
        btn('Screenshot', self._screenshot, shortcut='S')
        btn('Export VTK', self._export_vtk, shortcut='E')

        stretch = QWidget()
        stretch.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(stretch)

        self._theme_btn = QPushButton('☀')
        self._theme_btn.setFixedSize(28, 28)
        self._theme_btn.setToolTip('Toggle light / dark mode')
        self._theme_btn.clicked.connect(self._toggle_theme)
        tb.addWidget(self._theme_btn)
        spacer(8)

    # ── Central ────────────────────────────────────────

    def _build_central(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_viewport())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([260, 900, 310])
        layout.addWidget(splitter)

    def _build_left_panel(self) -> QWidget:
        c = PALETTE['dark']
        panel = QWidget()
        panel.setFixedWidth(260)
        panel.setStyleSheet(
            f'background: {c["bg_panel"]}; border-right: 1px solid {c["border"]};')
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Scene group
        scene_grp = QGroupBox('SCENE')
        sg = QGridLayout(scene_grp)
        sg.setSpacing(6)
        self._lbl_scene = QLabel('No scene loaded')
        self._lbl_scene.setWordWrap(True)
        self._lbl_scene.setStyleSheet(f'color: {c["text_mid"]}; font-size: 11px;')
        sg.addWidget(self._lbl_scene, 0, 0, 1, 2)
        sg.addWidget(QLabel('Backend:'), 1, 0)
        self._combo_backend = QComboBox()
        self._combo_backend.addItems(['CPU', 'CUDA (auto)'])
        sg.addWidget(self._combo_backend, 1, 1)
        load_btn = QPushButton('Load Scene...')
        load_btn.setObjectName('run_btn')
        load_btn.clicked.connect(self._open_scene)
        sg.addWidget(load_btn, 2, 0, 1, 2)
        layout.addWidget(scene_grp)

        # Solver params
        params_grp = QGroupBox('SOLVER')
        pg_l = QGridLayout(params_grp)
        pg_l.setSpacing(6)

        pg_l.addWidget(QLabel('Max iter CD:'), 0, 0)
        self._spin_max_cd = QSpinBox()
        self._spin_max_cd.setRange(1, 200)
        self._spin_max_cd.setValue(100)
        pg_l.addWidget(self._spin_max_cd, 0, 1)

        pg_l.addWidget(QLabel('Max iter DF:'), 1, 0)
        self._spin_max_df = QSpinBox()
        self._spin_max_df.setRange(1, 200)
        self._spin_max_df.setValue(100)
        pg_l.addWidget(self._spin_max_df, 1, 1)

        pg_l.addWidget(QLabel('CFL:'), 2, 0)
        self._spin_cfl = QDoubleSpinBox()
        self._spin_cfl.setRange(0.05, 1.0)
        self._spin_cfl.setSingleStep(0.05)
        self._spin_cfl.setValue(0.4)
        pg_l.addWidget(self._spin_cfl, 2, 1)

        pg_l.addWidget(QLabel('dt max:'), 3, 0)
        self._spin_dtmax = QDoubleSpinBox()
        self._spin_dtmax.setRange(1e-5, 0.1)
        self._spin_dtmax.setDecimals(4)
        self._spin_dtmax.setSingleStep(0.001)
        self._spin_dtmax.setValue(0.005)
        pg_l.addWidget(self._spin_dtmax, 3, 1)

        self._chk_diffusion = QCheckBox('Density diffusion')
        self._chk_diffusion.setChecked(True)
        pg_l.addWidget(self._chk_diffusion, 4, 0, 1, 2)

        pg_l.addWidget(QLabel('Pause at step:'), 5, 0)
        self._spin_pause_at = QSpinBox()
        self._spin_pause_at.setRange(0, 999999)
        self._spin_pause_at.setValue(0)
        self._spin_pause_at.valueChanged.connect(
            lambda v: setattr(self, '_pause_at_step', v))
        pg_l.addWidget(self._spin_pause_at, 5, 1)
        layout.addWidget(params_grp)

        # Visualization
        vis_grp = QGroupBox('VISUALIZATION')
        vg = QGridLayout(vis_grp)
        vg.setSpacing(6)

        vg.addWidget(QLabel('Color by:'), 0, 0)
        self._combo_color = QComboBox()
        self._combo_color.addItems(
            ['Speed', 'Secondary', 'Density error', 'Pressure', 'Solid color'])
        self._combo_color.setToolTip(
            'Pressure is RELATIVE and qualitative (WCSPH). '
            'Velocity is the quantitatively validated field.')
        self._combo_color.currentTextChanged.connect(self._on_color_mode_changed)
        vg.addWidget(self._combo_color, 0, 1)

        # Honest note shown only in Pressure mode.
        self._lbl_pressure_note = QLabel(
            'Relative pressure — qualitative (WCSPH);\nvelocity is quantitatively validated.')
        self._lbl_pressure_note.setWordWrap(True)
        self._lbl_pressure_note.setStyleSheet(
            f'color: {PALETTE["dark"]["warning"]}; font-size: 9px;')
        self._lbl_pressure_note.setVisible(False)
        vg.addWidget(self._lbl_pressure_note, 5, 0, 1, 2)

        vg.addWidget(QLabel('Colormap:'), 1, 0)
        self._combo_colormap = QComboBox()
        self._combo_colormap.addItems(list(COLORMAP_OPTIONS))
        self._combo_colormap.currentTextChanged.connect(self._on_colormap_changed)
        vg.addWidget(self._combo_colormap, 1, 1)

        # Particle-size multiplier 0.5x–2.0x (slider int 5–20 → /10), default 1.0x
        vg.addWidget(QLabel('Particle size:'), 2, 0)
        self._slider_size = QSlider(Qt.Orientation.Horizontal)
        self._slider_size.setRange(5, 20)
        self._slider_size.setValue(10)
        self._slider_size.setToolTip('Particle size multiplier (0.5x – 2.0x)')
        self._slider_size.valueChanged.connect(self._on_size_changed)
        vg.addWidget(self._slider_size, 2, 1)

        # Part C: diagnostic velocity-arrow overlay at the bend (default off).
        self._chk_vectors = QCheckBox('Show flow vectors')
        self._chk_vectors.setChecked(False)
        self._chk_vectors.setToolTip('Overlay short velocity arrows at the bend')
        self._chk_vectors.toggled.connect(self._on_vectors_toggled)
        vg.addWidget(self._chk_vectors, 3, 0, 1, 2)

        # Part B: inlet/outlet markers toggle.
        self._chk_io = QCheckBox('Show inlet/outlet')
        self._chk_io.setChecked(True)
        self._chk_io.setToolTip('Green INLET ring + red OUTLET ring')
        self._chk_io.toggled.connect(self._on_io_toggled)
        vg.addWidget(self._chk_io, 4, 0, 1, 2)
        layout.addWidget(vis_grp)

        # Part A: clipping plane (StarCCM-style section through the fluid).
        clip_grp = QGroupBox('CLIP')
        cg = QGridLayout(clip_grp)
        cg.setSpacing(6)
        cg.addWidget(QLabel('Axis:'), 0, 0)
        self._combo_clip = QComboBox()
        self._combo_clip.addItems(['Off', 'X', 'Y', 'Z'])
        self._combo_clip.currentTextChanged.connect(self._on_clip_axis_changed)
        cg.addWidget(self._combo_clip, 0, 1)
        cg.addWidget(QLabel('Position:'), 1, 0)
        self._slider_clip = QSlider(Qt.Orientation.Horizontal)
        self._slider_clip.setRange(0, 100)
        self._slider_clip.setValue(100)
        self._slider_clip.setToolTip('Plane position across the fluid bbox (keeps the lower side)')
        self._slider_clip.valueChanged.connect(self._on_clip_pos_changed)
        cg.addWidget(self._slider_clip, 1, 1)
        layout.addWidget(clip_grp)

        # Part B: live flow-drive controls.
        flow_grp = QGroupBox('FLOW')
        fg = QGridLayout(flow_grp)
        fg.setSpacing(6)
        self._lbl_drive = QLabel('Drive magnitude:')
        fg.addWidget(self._lbl_drive, 0, 0)
        self._spin_drive = QDoubleSpinBox()
        self._spin_drive.setRange(0.0, 5.0)
        self._spin_drive.setSingleStep(0.1)
        self._spin_drive.setDecimals(2)
        self._spin_drive.setValue(2.0)
        self._spin_drive.setToolTip('Tangent body-force magnitude (drives the flow, live)')
        self._spin_drive.valueChanged.connect(self._on_drive_changed)
        fg.addWidget(self._spin_drive, 0, 1)

        # Inlet velocity (inlet/outlet scenes) — clamped to the validated
        # laminar range so the flow stays a developable profile, not a slug.
        self._lbl_inletv = QLabel('Inlet velocity:')
        fg.addWidget(self._lbl_inletv, 1, 0)
        self._spin_inletv = QDoubleSpinBox()
        self._spin_inletv.setRange(0.02, 0.50)
        self._spin_inletv.setSingleStep(0.01)
        self._spin_inletv.setDecimals(2)
        self._spin_inletv.setValue(0.10)
        self._spin_inletv.setToolTip('Inlet velocity [m/s] — validated laminar range')
        self._spin_inletv.valueChanged.connect(self._on_inletv_changed)
        fg.addWidget(self._spin_inletv, 1, 1)

        # Outlet is a fixed atmosphere (p = 0) — read-only, not a control.
        fg.addWidget(QLabel('Outlet:'), 2, 0)
        self._lbl_outlet_bc = QLabel('Atmosphere (p = 0)')
        self._lbl_outlet_bc.setStyleSheet(
            f'color: {PALETTE["dark"]["text_mid"]}; font-size: 10px;')
        fg.addWidget(self._lbl_outlet_bc, 2, 1)

        # Read-only note for tangent-driven (open outlet) scenes.
        self._lbl_outlet_note = QLabel('open (tangent-driven)')
        self._lbl_outlet_note.setStyleSheet(
            f'color: {PALETTE["dark"]["text_dim"]}; font-size: 10px;')
        fg.addWidget(self._lbl_outlet_note, 3, 0, 1, 2)
        layout.addWidget(flow_grp)
        self._flow_grp = flow_grp

        layout.addStretch()

        # Console
        log_grp = QGroupBox('CONSOLE')
        lg = QVBoxLayout(log_grp)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFixedHeight(140)
        lg.addWidget(self._log)
        layout.addWidget(log_grp)

        return panel

    def _build_viewport(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        pg.setConfigOptions(antialias=True)
        self.gl_widget = gl.GLViewWidget()
        # Part D: keep viewport background dark (#0a0d14) in dark mode
        self.gl_widget.setBackgroundColor(pg.mkColor(10, 13, 20))
        self.gl_widget.setCameraPosition(distance=3.5, elevation=25, azimuth=45)

        grid = gl.GLGridItem()
        grid.setColor(pg.mkColor(30, 37, 53, 80))
        grid.setSize(3, 3)
        grid.setSpacing(0.1, 0.1)
        self.gl_widget.addItem(grid)

        # Part C: industrial fluid look — real-world particle size (pxMode=False
        # gives depth scaling), translucent edges, depth test so near particles
        # occlude far ones. Size is set per-frame from the scene spacing.
        self._scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            size=0.016,
            color=(0.0, 0.83, 1.0, 0.95),
            pxMode=False)
        self._scatter.setGLOptions('translucent')
        try:
            self._scatter.setData(antialias=True)
        except Exception:
            pass
        self.gl_widget.addItem(self._scatter)

        # Boundary wall: faint, smaller, gray — visible but not dominant.
        self._boundary_scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            size=0.008,
            color=(0.45, 0.50, 0.60, 0.10),
            pxMode=False)
        self._boundary_scatter.setGLOptions('translucent')
        self.gl_widget.addItem(self._boundary_scatter)

        # Default real-world particle radius (updated on scene load).
        self._particle_world_size = 0.016

        # Part C: thin cyan velocity arrows at the bend (diagnostic overlay).
        self._flow_vectors = gl.GLLinePlotItem(
            pos=np.zeros((2, 3), dtype=np.float32),
            color=(0.0, 0.9, 1.0, 0.85),
            width=1.4,
            mode='lines',
            antialias=True)
        self._flow_vectors.setGLOptions('translucent')
        self._flow_vectors.setVisible(False)
        self.gl_widget.addItem(self._flow_vectors)

        # Part B: inlet (green) and outlet (red) ring markers + text labels.
        self._inlet_ring = gl.GLLinePlotItem(
            pos=np.zeros((2, 3), dtype=np.float32),
            color=(0.0, 0.9, 0.35, 0.95), width=2.5, mode='line_strip', antialias=True)
        self._inlet_ring.setGLOptions('translucent')
        self._inlet_ring.setVisible(False)
        self.gl_widget.addItem(self._inlet_ring)
        self._outlet_ring = gl.GLLinePlotItem(
            pos=np.zeros((2, 3), dtype=np.float32),
            color=(1.0, 0.25, 0.25, 0.95), width=2.5, mode='line_strip', antialias=True)
        self._outlet_ring.setGLOptions('translucent')
        self._outlet_ring.setVisible(False)
        self.gl_widget.addItem(self._outlet_ring)
        self._inlet_label = None
        self._outlet_label = None
        try:
            self._inlet_label = gl.GLTextItem(pos=np.zeros(3), text='INLET', color=(0, 230, 90))
            self._outlet_label = gl.GLTextItem(pos=np.zeros(3), text='OUTLET', color=(255, 70, 70))
            self.gl_widget.addItem(self._inlet_label)
            self.gl_widget.addItem(self._outlet_label)
            self._inlet_label.setVisible(False)
            self._outlet_label.setVisible(False)
        except Exception:
            self._inlet_label = self._outlet_label = None

        # Part A: small XYZ orientation gizmo (X=red, Y=green, Z=blue) at the
        # fluid bbox origin, length ~0.5*pipe radius. Positioned on scene load.
        self._axes_items: list = []
        self._axes_labels: list = []
        for col in ((1.0, 0.25, 0.25, 1.0), (0.25, 1.0, 0.35, 1.0), (0.3, 0.5, 1.0, 1.0)):
            ln = gl.GLLinePlotItem(
                pos=np.zeros((2, 3), dtype=np.float32), color=col,
                width=2.5, mode='lines', antialias=True)
            ln.setGLOptions('translucent')
            self.gl_widget.addItem(ln)
            self._axes_items.append(ln)
        try:
            for txt, col in (('X', (255, 60, 60)), ('Y', (60, 255, 90)), ('Z', (80, 130, 255))):
                t = gl.GLTextItem(pos=np.zeros(3), text=txt, color=col)
                self.gl_widget.addItem(t)
                self._axes_labels.append(t)
        except Exception:
            self._axes_labels = []
        self._show_axes = True

        # Part C: gl viewport + large colorbar docked at its right edge.
        view_row = QWidget()
        vr = QHBoxLayout(view_row)
        vr.setContentsMargins(0, 0, 0, 0)
        vr.setSpacing(0)
        vr.addWidget(self.gl_widget, stretch=1)
        vr.addWidget(self._build_big_colorbar())

        layout.addWidget(view_row, stretch=1)
        layout.addWidget(self._build_playback_bar())
        return container

    def _build_big_colorbar(self) -> QWidget:
        """Tall vertical colorbar docked beside the 3D viewport (StarCCM look)."""
        c = PALETTE['dark']
        panel = QWidget()
        panel.setFixedWidth(86)
        panel.setStyleSheet(f'background: {c["bg_base"]};')
        col = QVBoxLayout(panel)
        col.setContentsMargins(6, 8, 8, 8)
        col.setSpacing(4)

        self._big_cbar_title = QLabel('Speed [m/s]')
        self._big_cbar_title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._big_cbar_title.setWordWrap(True)
        self._big_cbar_title.setStyleSheet(
            f'color: {c["accent"]}; font-size: 9px; font-weight: 700; letter-spacing: 1px;')
        col.addWidget(self._big_cbar_title)

        lbl_style = (f'color: {c["text_mid"]}; '
                     f'font-family: IBM Plex Mono, Consolas, monospace; font-size: 9px;')
        bar_row = QWidget()
        br = QHBoxLayout(bar_row)
        br.setContentsMargins(0, 0, 0, 0)
        br.setSpacing(4)

        # Stacked min/mid/max labels at the right of the bar.
        labels = QVBoxLayout()
        labels.setContentsMargins(0, 0, 0, 0)
        labels.setSpacing(0)
        self._big_cbar_max = QLabel('1.000')
        self._big_cbar_mid = QLabel('0.500')
        self._big_cbar_min = QLabel('0.000')
        for lb in (self._big_cbar_max, self._big_cbar_mid, self._big_cbar_min):
            lb.setStyleSheet(lbl_style)
            lb.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        labels.addWidget(self._big_cbar_max)
        labels.addStretch()
        labels.addWidget(self._big_cbar_mid)
        labels.addStretch()
        labels.addWidget(self._big_cbar_min)

        self._big_cbar_widget = pg.PlotWidget()
        self._big_cbar_widget.setFixedWidth(28)
        self._big_cbar_widget.hideAxis('bottom')
        self._big_cbar_widget.hideAxis('left')
        self._big_cbar_widget.setBackground(pg.mkColor(8, 10, 15))
        self._big_cbar_widget.setMouseEnabled(x=False, y=False)
        self._big_cbar_widget.setMenuEnabled(False)
        self._big_cbar_img = pg.ImageItem()
        self._big_cbar_widget.addItem(self._big_cbar_img)

        br.addWidget(self._big_cbar_widget)
        br.addLayout(labels)
        # The bar fills ~all available height; spacers keep it ~60% if tall.
        col.addStretch(1)
        col.addWidget(bar_row, stretch=6)
        col.addStretch(1)
        self._refresh_colorbar_image()
        return panel

    def _build_playback_bar(self) -> QWidget:
        c = PALETTE['dark']
        bar = QWidget()
        bar.setFixedHeight(34)
        bar.setStyleSheet(
            f'background: {c["bg_panel"]}; border-top: 1px solid {c["border"]};')
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(16)

        self._lbl_time = QLabel('t = 0.0000 s')
        self._lbl_time.setStyleSheet(
            f'color: {c["accent"]}; '
            f'font-family: IBM Plex Mono, Consolas, monospace; font-size: 12px;')
        layout.addWidget(self._lbl_time)

        layout.addStretch()

        self._lbl_step = QLabel('Step: 0')
        self._lbl_step.setStyleSheet(
            f'color: {c["text_mid"]}; font-size: 11px;')
        layout.addWidget(self._lbl_step)

        self._lbl_fps = QLabel('FPS: —')
        self._lbl_fps.setStyleSheet(
            f'color: {c["text_dim"]}; font-size: 11px;')
        layout.addWidget(self._lbl_fps)

        self._lbl_status = QLabel('● IDLE')
        self._lbl_status.setStyleSheet(
            f'color: {c["text_dim"]}; font-size: 11px;')
        layout.addWidget(self._lbl_status)

        return bar

    # ── TASK 5 — Right panel ───────────────────────────

    def _build_right_panel(self) -> QWidget:
        c = PALETTE['dark']
        panel = QWidget()
        panel.setFixedWidth(310)
        panel.setStyleSheet(
            f'background: {c["bg_panel"]}; border-left: 1px solid {c["border"]};')
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # 2×4 KPI grid
        stats_grp = QGroupBox('LIVE DIAGNOSTICS')
        dg = QGridLayout(stats_grp)
        dg.setSpacing(5)

        self._stat_vmax = StatWidget('V MAX',      'm/s')
        self._stat_rho  = StatWidget('ρ ERROR',    '%',
                                     warn_threshold=2.0,
                                     danger_threshold=5.0)
        self._stat_icd  = StatWidget('ITER CD',    '',
                                     warn_threshold=5,
                                     danger_threshold=20)
        self._stat_idf  = StatWidget('ITER DF',    '')
        self._stat_dt   = StatWidget('Δt',         's')
        self._stat_n    = StatWidget('PARTICLES',  '')
        self._stat_re   = StatWidget('Re',         '')
        self._stat_fps  = StatWidget('FPS',        '')

        # backwards compat aliases used in existing code paths
        self._stat_rho_err  = self._stat_rho
        self._stat_iter_cd  = self._stat_icd
        self._stat_iter_df  = self._stat_idf

        dg.addWidget(self._stat_vmax, 0, 0)
        dg.addWidget(self._stat_rho,  0, 1)
        dg.addWidget(self._stat_icd,  1, 0)
        dg.addWidget(self._stat_idf,  1, 1)
        dg.addWidget(self._stat_dt,   2, 0)
        dg.addWidget(self._stat_n,    2, 1)
        dg.addWidget(self._stat_re,   3, 0)
        dg.addWidget(self._stat_fps,  3, 1)
        layout.addWidget(stats_grp)

        # Time series charts
        charts_grp = QGroupBox('TIME SERIES')
        cg = QVBoxLayout(charts_grp)
        cg.setSpacing(4)

        plot_bg = pg.mkColor(8, 10, 15)
        pg.setConfigOption('background', c['bg_widget'])
        pg.setConfigOption('foreground', c['text_mid'])

        self._plot_vmax = pg.PlotWidget(title='v_max  [m/s]')
        self._plot_vmax.setBackground(plot_bg)
        self._plot_vmax.setFixedHeight(110)
        self._plot_vmax.showGrid(x=True, y=True, alpha=0.12)
        self._curve_vmax = self._plot_vmax.plot(
            pen=pg.mkPen(c['accent'], width=2))
        cg.addWidget(self._plot_vmax)

        self._plot_rho = pg.PlotWidget(title='ρ error  [%]')
        self._plot_rho.setBackground(plot_bg)
        self._plot_rho.setFixedHeight(110)
        self._plot_rho.showGrid(x=True, y=True, alpha=0.12)
        self._curve_rho = self._plot_rho.plot(
            pen=pg.mkPen(c['warning'], width=2))
        cg.addWidget(self._plot_rho)

        self._plot_profile = pg.PlotWidget(title='v(r) profile')
        self._plot_profile.setBackground(plot_bg)
        self._plot_profile.setFixedHeight(130)
        self._plot_profile.showGrid(x=True, y=True, alpha=0.12)
        self._curve_profile_sim = self._plot_profile.plot(
            pen=pg.mkPen(c['accent'], width=2), name='SPH')
        self._curve_profile_ana = self._plot_profile.plot(
            pen=pg.mkPen(c['warning'], width=1, style=Qt.PenStyle.DashLine),
            name='Poiseuille')
        cg.addWidget(self._plot_profile)
        layout.addWidget(charts_grp)

        # Colorbar
        self._cbar_grp = QGroupBox('COLORMAP SCALE · SPEED')
        cbar_layout = QVBoxLayout(self._cbar_grp)
        cbar_layout.addWidget(self._build_colorbar())
        layout.addWidget(self._cbar_grp)

        layout.addStretch()

        # Export
        exp_grp = QGroupBox('EXPORT')
        eg = QVBoxLayout(exp_grp)
        eg.setSpacing(6)
        btn_vtk = QPushButton('Export VTK Frame')
        btn_vtk.clicked.connect(self._export_vtk)
        eg.addWidget(btn_vtk)
        btn_ss = QPushButton('Screenshot (PNG)')
        btn_ss.clicked.connect(self._screenshot)
        eg.addWidget(btn_ss)
        layout.addWidget(exp_grp)

        return panel

    # ── TASK 6 — Status bar ────────────────────────────

    def _build_statusbar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)

        mono = 'font-family: IBM Plex Mono, Consolas, monospace; font-size: 10px;'
        self._sb_scene = QLabel('  No scene loaded')
        self._sb_scene.setStyleSheet(f'color: #3d4a5c; {mono}')

        self._sb_metrics = QLabel()
        self._sb_metrics.setStyleSheet(f'color: #5a6a7e; {mono}')

        self._sb_right = QLabel('SPH Framework  ·  DFSPH Solver  ·  v1.0  ')
        self._sb_right.setStyleSheet(f'color: #1e2535; {mono}')

        sb.addWidget(self._sb_scene)
        sb.addPermanentWidget(self._sb_metrics)
        sb.addPermanentWidget(self._sb_right)

        # legacy alias used by some code paths
        self._sb_left = self._sb_scene

    def _update_status(self, m: dict) -> None:
        self._sb_metrics.setText(
            f"  v_max: {m.get('vmax', 0):.4f} m/s"
            f"  ·  Re: {m.get('reynolds_number', 0):.0f}"
            f"  ·  t: {m.get('time', 0):.4f} s"
            f"  ·  {m.get('n_fluid', 0):,} particles  "
        )

    def _update_status_bar(self, m: dict) -> None:
        self._update_status(m)

    # ── Colorbar ───────────────────────────────────────

    def _build_colorbar(self) -> QWidget:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(4, 4, 4, 4)
        row.setSpacing(8)

        c = PALETTE['dark']
        lbl_style = (f'color: {c["text_mid"]}; '
                     f'font-family: IBM Plex Mono, Consolas, monospace; '
                     f'font-size: 10px;')
        self._lbl_cbar_max = QLabel('1.000')
        self._lbl_cbar_max.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._lbl_cbar_max.setStyleSheet(lbl_style)

        self._colorbar_widget = pg.PlotWidget()
        self._colorbar_widget.setFixedWidth(60)
        self._colorbar_widget.setFixedHeight(200)
        self._colorbar_widget.hideAxis('bottom')
        self._colorbar_widget.hideAxis('left')
        self._colorbar_widget.setBackground(pg.mkColor(8, 10, 15))
        self._colorbar_widget.setMouseEnabled(x=False, y=False)
        self._colorbar_widget.setMenuEnabled(False)
        self._colorbar_img = pg.ImageItem()
        self._colorbar_widget.addItem(self._colorbar_img)
        # backwards-compat alias
        self._cbar_item = self._colorbar_img
        self._refresh_colorbar_image()

        self._lbl_cbar_min = QLabel('0.000')
        self._lbl_cbar_min.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._lbl_cbar_min.setStyleSheet(lbl_style)

        labels = QVBoxLayout()
        labels.setContentsMargins(0, 0, 0, 0)
        labels.setSpacing(2)
        labels.addWidget(self._lbl_cbar_max)
        labels.addStretch()
        labels.addWidget(self._lbl_cbar_min)
        row.addLayout(labels)
        row.addWidget(self._colorbar_widget)
        return container

    def _refresh_colorbar_image(self) -> None:
        """Build a vertical gradient bar (1 px wide × 256 px tall, low→high)."""
        # Pressure mode forces the diverging Coolwarm map (standard CFD look).
        if self._combo_color.currentText() == 'Pressure':
            name = 'Coolwarm'
        else:
            name = self._combo_colormap.currentText()
        fn = _CMAP_FNS.get(name, _turbo)
        n = 256
        t = np.linspace(0.0, 1.0, n)  # bottom = low, top = high
        r, g, b = fn(t)
        img = np.zeros((1, n, 4), dtype=np.uint8)
        img[0, :, 0] = (r * 255).astype(np.uint8)
        img[0, :, 1] = (g * 255).astype(np.uint8)
        img[0, :, 2] = (b * 255).astype(np.uint8)
        img[0, :, 3] = 255
        for imgitem, widget in (
            (getattr(self, '_colorbar_img', None), getattr(self, '_colorbar_widget', None)),
            (getattr(self, '_big_cbar_img', None), getattr(self, '_big_cbar_widget', None)),
        ):
            if imgitem is None or widget is None:
                continue
            imgitem.setImage(img)
            imgitem.setRect(0.0, 0.0, 1.0, float(n))
            widget.setXRange(0.0, 1.0, padding=0)
            widget.setYRange(0.0, float(n), padding=0)

    def _color_by_label(self) -> str:
        """Title (field + unit) for the active Color-by mode."""
        return {
            'Speed': 'Velocity magnitude [m/s]',
            'Secondary': 'Secondary [m/s]',
            'Density error': 'ρ error',
            'Pressure': 'Pressure (relative, qualitative) [Pa]',
            'Solid color': 'Solid',
        }.get(self._combo_color.currentText(), self._combo_color.currentText())

    def _update_colorbar_labels(self) -> None:
        """Sync both colorbars' min/mid/max labels and the big-bar title."""
        vmin, vmax = self._colorbar_vmin, self._colorbar_vmax
        vmid = 0.5 * (vmin + vmax)
        # Pressure values can be large (Pa); use compact formatting.
        pressure = self._combo_color.currentText() == 'Pressure'
        fmt = (lambda v: f'{v:+.3g}') if pressure else (lambda v: f'{v:.4f}')
        fmt_big = (lambda v: f'{v:+.2g}') if pressure else (lambda v: f'{v:.3f}')
        if hasattr(self, '_lbl_cbar_min'):
            self._lbl_cbar_min.setText(fmt(vmin))
            self._lbl_cbar_max.setText(fmt(vmax))
        if hasattr(self, '_big_cbar_min'):
            self._big_cbar_min.setText(fmt_big(vmin))
            self._big_cbar_mid.setText(fmt_big(vmid))
            self._big_cbar_max.setText(fmt_big(vmax))
            self._big_cbar_title.setText(self._color_by_label())

    def _rebuild_colorbar(self, *_args) -> None:
        """Rebuild gradient when colormap selection changes."""
        self._refresh_colorbar_image()

    def _on_colormap_changed(self, _text: str) -> None:
        self._refresh_colorbar_image()
        self._rerender_current()

    def _on_size_changed(self, _v: int) -> None:
        """Live-update particle size when the multiplier slider moves."""
        if self.runner is not None:
            self._scatter.setData(size=self._particle_size_world())

    def _on_color_mode_changed(self, mode: str) -> None:
        """Relabel the colorbar and re-render the current frame in the new mode."""
        if hasattr(self, '_cbar_grp'):
            self._cbar_grp.setTitle(f'COLORMAP SCALE · {mode.upper()}')
        if hasattr(self, '_big_cbar_title'):
            self._big_cbar_title.setText(self._color_by_label())
        # Pressure mode → diverging Coolwarm bar + honest qualitative note.
        self._refresh_colorbar_image()
        if hasattr(self, '_lbl_pressure_note'):
            self._lbl_pressure_note.setVisible(mode == 'Pressure')
        self._rerender_current()

    def _on_vectors_toggled(self, on: bool) -> None:
        """Show/hide the bend velocity-arrow overlay."""
        if not on:
            self._flow_vectors.setVisible(False)
            return
        self._flow_vectors.setVisible(True)
        if self.runner is not None and self.runner.backend.sim.fluid.n > 0:
            fl = self.runner.backend.sim.fluid
            self._update_flow_vectors(
                fl.positions.astype(np.float64),
                fl.velocities.astype(np.float64))

    def _update_flow_vectors(self, positions: np.ndarray, velocities: np.ndarray) -> None:
        """Draw short cyan arrows along velocity for ~40 particles at the bend."""
        if positions.shape[1] < 3:
            self._flow_vectors.setData(pos=np.zeros((2, 3), dtype=np.float32))
            return
        x, y = positions[:, 0], positions[:, 1]
        bend = (x > 0.25) & (x < 0.45) & (y > 0.02) & (y < 0.18)
        idx = np.where(bend)[0]
        if idx.size == 0:
            self._flow_vectors.setData(pos=np.zeros((2, 3), dtype=np.float32))
            return
        if idx.size > 40:
            idx = idx[np.linspace(0, idx.size - 1, 40).astype(int)]
        p = positions[idx]
        v = velocities[idx]
        speed = np.linalg.norm(v, axis=1, keepdims=True)
        direction = v / (speed + 1e-12)
        # Arrow length scaled to speed, clamped so they stay subtle.
        scale = self._particle_world_size * 6.0
        length = np.clip(speed, 0.0, 0.05) / 0.05 * scale
        ends = p + direction * length
        # Interleave start/end points for 'lines' mode (pairs of vertices).
        segs = np.empty((idx.size * 2, 3), dtype=np.float32)
        segs[0::2] = p.astype(np.float32)
        segs[1::2] = ends.astype(np.float32)
        self._flow_vectors.setData(pos=segs, color=(0.0, 0.9, 1.0, 0.85))

    def _apply_colormap(self, values: np.ndarray, mode: str) -> np.ndarray:
        n = len(values)
        # Part B: clamp the colour scale to the 99th percentile so a lone
        # outlier particle never washes out the field. nan-safe.
        if n:
            finite = values[np.isfinite(values)]
            vmax = float(np.percentile(finite, 99)) if finite.size else 1.0
            vmin = float(finite.min()) if finite.size else 0.0
        else:
            vmax, vmin = 1.0, 0.0
        self._colorbar_vmin = vmin
        self._colorbar_vmax = vmax
        t = np.clip((np.nan_to_num(values) - vmin) / (vmax - vmin + 1e-10), 0, 1)
        fn = _CMAP_FNS.get(mode, _turbo)
        r, g, b = fn(t)
        colors = np.zeros((n, 4), dtype=np.float32)
        colors[:, 0] = r; colors[:, 1] = g; colors[:, 2] = b; colors[:, 3] = 0.95
        return colors

    def _secondary_magnitude(
        self, positions: np.ndarray, velocities: np.ndarray,
    ) -> np.ndarray:
        """Per-particle velocity magnitude perpendicular to the local main-flow
        axis, highlighting the Dean / secondary swirl at the bend."""
        n = positions.shape[0]
        dim = positions.shape[1]
        vx = velocities[:, 0]
        vy = velocities[:, 1] if dim > 1 else np.zeros(n)
        vz = velocities[:, 2] if dim > 2 else np.zeros(n)
        x = positions[:, 0]
        y = positions[:, 1] if dim > 1 else np.zeros(n)
        perp = np.zeros(n, dtype=np.float64)
        horiz = x < 0.25            # horizontal arm: main flow along x
        vert = y > 0.25             # vertical arm:   main flow along y
        bend = ~(horiz | vert)      # bend region:    treat main axis as x
        perp[horiz] = np.sqrt(vy[horiz] ** 2 + vz[horiz] ** 2)
        perp[vert] = np.sqrt(vx[vert] ** 2 + vz[vert] ** 2)
        perp[bend] = np.sqrt(vy[bend] ** 2 + vz[bend] ** 2)
        return perp

    def _particle_colors(
        self, speeds: np.ndarray, values: np.ndarray | None = None,
    ) -> np.ndarray:
        mode = self._combo_color.currentText()
        n = len(speeds)
        if mode == 'Solid color':
            colors = np.zeros((n, 4), dtype=np.float32)
            colors[:, 0] = 0.0; colors[:, 1] = 0.83; colors[:, 2] = 1.0; colors[:, 3] = 0.92
            return colors
        if mode == 'Speed':
            return self._apply_colormap(speeds, self._combo_colormap.currentText())
        if mode == 'Secondary':
            # ``values`` carries the precomputed perpendicular magnitude.
            field = values if values is not None else speeds
            return self._apply_colormap(field, self._combo_colormap.currentText())
        if mode == 'Pressure' and values is not None:
            # Signed display pressure k*(rho-rho0): map directly so negatives
            # (low/inner) → cool, positives (high/outer/stagnation) → warm.
            return self._apply_colormap(values, self._combo_colormap.currentText())
        if values is None:
            values = speeds
        vmax = float(values.max()) if n else 1.0
        t = np.clip(values / (vmax * 0.8 + 1e-10), 0.0, 1.0)
        return self._apply_colormap(t, self._combo_colormap.currentText())

    # ── Theme toggle ───────────────────────────────────

    def _toggle_theme(self) -> None:
        if self._dark_mode:
            self._dark_mode = False
            self._act_theme.setText('Dark Mode')
            self._theme_btn.setText('🌙')
            self.setStyleSheet(build_stylesheet('light'))
            self.gl_widget.setBackgroundColor(pg.mkColor(244, 246, 251))
            plot_bg = pg.mkColor(248, 250, 253)
            cbar_bg = pg.mkColor(248, 250, 253)
        else:
            self._dark_mode = True
            self._act_theme.setText('Light Mode')
            self._theme_btn.setText('☀')
            self.setStyleSheet(build_stylesheet('dark'))
            self.gl_widget.setBackgroundColor(pg.mkColor(10, 13, 20))
            plot_bg = pg.mkColor(8, 10, 15)
            cbar_bg = pg.mkColor(8, 10, 15)

        for plot in (self._plot_vmax, self._plot_rho, self._plot_profile):
            plot.setBackground(plot_bg)
        self._colorbar_widget.setBackground(cbar_bg)
        if hasattr(self, '_big_cbar_widget'):
            self._big_cbar_widget.setBackground(cbar_bg)

        for w in (self._stat_vmax, self._stat_rho, self._stat_icd, self._stat_idf,
                  self._stat_dt, self._stat_n, self._stat_re, self._stat_fps):
            w.set_dark(self._dark_mode)

    # ── Actions ────────────────────────────────────────

    def _open_scene(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open Scene', 'scenes',
            'Scene Files (*.json);;All Files (*)')
        if path:
            self._load_scene(path)

    def _update_boundary_display(self):
        if not hasattr(self, 'runner') or self.runner is None:
            return
        fl = self.runner.backend.sim.fluid
        if getattr(fl, 'dim', 2) != 3:
            return
        bd = getattr(self.runner.backend.sim, 'boundary', None)
        if bd is None or not hasattr(bd, 'positions') or len(bd.positions) == 0:
            return
        # Faint, smaller wall particles (~0.5x the fluid world size).
        bsize = 0.5 * getattr(self, '_particle_world_size', 0.016)
        self._boundary_scatter.setData(
            pos=bd.positions.astype(np.float32), size=bsize)
        self._boundary_scatter.setVisible(self._show_boundary)

    def _load_scene(self, path: str):
        try:
            from sph.core.simulation import SimulationRunner

            self.runner = SimulationRunner(Path(path))
            self._log_msg('Backend: CPU')
            fl = self.runner.backend.sim.fluid
            bd = self.runner.backend.sim.boundary

            # Part C: real-world particle size = 1.6 * particle spacing so
            # particles nearly touch like a filled fluid (fallback 0.01).
            spacing = self._scene_spacing(fl)
            self._particle_world_size = 1.6 * spacing

            self._update_boundary_display()
            if fl.n > 0:
                self._scatter.setData(
                    pos=fl.positions.astype(np.float32),
                    size=self._particle_size_world())

            n_bd = len(bd.positions) if bd is not None and hasattr(bd, 'positions') else 0
            if (bd is not None and hasattr(bd, 'positions')
                    and len(bd.positions) > 0 and bd.positions.shape[1] == 3):
                cy = bd.positions[:, 1].mean()
                cz = bd.positions[:, 2].mean()
                r_bd = np.sqrt((bd.positions[:, 1] - cy) ** 2 + (bd.positions[:, 2] - cz) ** 2)
                self._scene_R = float(np.percentile(r_bd, 95))
            else:
                self._scene_R = 0.07

            # Cache boundary positions + support radius for the Pressure-mode
            # wall-deficiency neighbour count.
            self._boundary_pos_cache = (
                bd.positions.copy() if (bd is not None and hasattr(bd, 'positions')
                                        and len(bd.positions) > 0) else None)
            self._support_radius = float(getattr(self.runner.backend.sim, 'support_radius', 0.04))

            self._lbl_scene.setText(f'{Path(path).name}\n{fl.n} fluid · {n_bd} bnd')
            self._log_msg(f'Scene loaded: {Path(path).name}')
            self._sb_scene.setText(f'  {Path(path).name}')
            self._history = {k: [] for k in self._history}

            # Clear any stale flow-vector overlay from the previous scene.
            if hasattr(self, '_flow_vectors'):
                self._flow_vectors.setData(pos=np.zeros((2, 3), dtype=np.float32))
                self._flow_vectors.setVisible(self._chk_vectors.isChecked())

            # Part A: recompute clip bounds from the fluid bbox; reset to Off.
            if fl.n > 0:
                self._fluid_bbox = (fl.positions.min(0).copy(), fl.positions.max(0).copy())
            else:
                self._fluid_bbox = None
            self._clip_axis = None
            self._clip_frac = 1.0
            if hasattr(self, '_combo_clip'):
                self._combo_clip.blockSignals(True)
                self._combo_clip.setCurrentText('Off')
                self._combo_clip.blockSignals(False)
                self._slider_clip.blockSignals(True)
                self._slider_clip.setValue(100)
                self._slider_clip.blockSignals(False)

            # Part B: place inlet/outlet markers for this scene.
            self._update_io_markers()
            self._refresh_colorbar_image()
            self._update_colorbar_labels()

            # Part A/B: orientation gizmo + live flow-drive controls.
            self._update_axes_gizmo()
            self._sync_flow_controls()

            # Part D: auto-frame camera to fit all particles.
            self._auto_frame_camera(fl, bd)

        except Exception as e:
            self._log_msg(f'ERROR: {e}', error=True)

    def _scene_spacing(self, fluid) -> float:
        """Particle spacing from scene config, falling back to fluid.h or 0.01."""
        try:
            scene = self.runner.backend.sim.scene
            fcfg = scene.get('fluid') or (scene.get('fluids') or [{}])[0]
            sp = fcfg.get('spacing')
            if sp is not None and float(sp) > 0.0:
                return float(sp)
        except Exception:
            pass
        h = getattr(fluid, 'h', 0.0)
        return float(h) if h and h > 0.0 else 0.01

    def _particle_size_world(self) -> float:
        """World particle size scaled by the size-multiplier slider (0.5x–2.0x)."""
        mult = self._slider_size.value() / 10.0
        return self._particle_world_size * mult

    def _auto_frame_camera(self, fluid, boundary) -> None:
        """Set camera distance/centre from the particle bounding-box diagonal."""
        pts = []
        if fluid is not None and getattr(fluid, 'n', 0) > 0:
            pts.append(fluid.positions)
        if (boundary is not None and hasattr(boundary, 'positions')
                and len(boundary.positions) > 0):
            pts.append(boundary.positions)
        if not pts:
            return
        allp = np.vstack(pts)
        finite = allp[np.isfinite(allp).all(axis=1)]
        if finite.shape[0] == 0:
            return
        pmin = finite.min(axis=0)
        pmax = finite.max(axis=0)
        center = 0.5 * (pmin + pmax)
        diag = float(np.linalg.norm(pmax - pmin))
        if finite.shape[1] == 2:
            center = np.array([center[0], center[1], 0.0])
        from pyqtgraph import Vector
        self.gl_widget.opts['center'] = Vector(float(center[0]), float(center[1]), float(center[2]))
        self.gl_widget.setCameraPosition(
            distance=max(diag * 1.4, 0.1), elevation=25, azimuth=45)

    def _run(self):
        if not self.runner:
            self._log_msg('Load a scene first.', error=True)
            return
        if self.worker and self.worker.isRunning():
            self.worker.resume()
        else:
            self.worker = SimWorker(self.runner)
            self.worker.step_done.connect(self._on_step)
            self.worker.sim_error.connect(self._on_error)
            self.worker.start()
        self._btn_run.setEnabled(False)
        self._btn_pause.setEnabled(True)
        self._set_status('● RUNNING', PALETTE['dark']['success'])

    def _pause(self):
        if self.worker:
            self.worker.pause()
        self._btn_run.setEnabled(True)
        self._btn_pause.setEnabled(False)
        self._set_status('⏸ PAUSED', PALETTE['dark']['warning'])

    def _reset(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self.worker = None
        self._btn_run.setEnabled(True)
        self._btn_pause.setEnabled(False)
        self._set_status('● IDLE', PALETTE['dark']['text_dim'])
        self._history = {k: [] for k in self._history}
        if self.runner:
            self.runner.reset()
            fl = self.runner.backend.sim.fluid
            self._scatter.setData(pos=fl.positions.astype(np.float32))
        self._log_msg('Simulation reset.')

    def _step_once(self):
        if not self.runner:
            return
        try:
            rt = self.runner.step()
            fl = self.runner.backend.sim.fluid
            sim = self.runner.backend.sim
            spd = np.linalg.norm(fl.velocities, axis=1)
            self._on_step({
                'step':      int(sim.current_step),
                'time':      float(getattr(sim, 't', 0.0)),
                'dt':        float(rt.runtime.dt),
                'vmax':      float(spd.max()) if len(spd) else 0.0,
                'rho_err':   float(rt.runtime.rho_error_mean) * 100.0,
                'iter_cd':   rt.runtime.solver_metrics.get('iter_cd', 0),
                'iter_df':   rt.runtime.solver_metrics.get('iter_df', 0),
                'n_fluid':   int(fl.n),
                'reynolds_number': float(rt.runtime.reynolds_number),
                'regime':    str(rt.runtime.regime),
                'positions': fl.positions.copy(),
                'speeds':    spd.copy(),
                'velocities': fl.velocities.copy(),
            })
        except Exception as e:
            self._on_error(str(e))

    def _profile_geometry(self) -> tuple[float, float, float]:
        cy, cz, R = 0.0, 0.0, self._scene_R
        try:
            scene = self.runner.backend.sim.scene
            wall = scene.get('domain', {}).get('cylinder_wall')
            if wall is None:
                fluids = scene.get('fluids')
                wall = fluids[0] if fluids else scene.get('fluid')
            if wall is not None:
                center = wall.get('center')
                if center is not None and len(center) >= 2:
                    cy, cz = float(center[0]), float(center[1])
                if wall.get('radius') is not None:
                    R = float(wall['radius'])
        except Exception:
            pass
        return cy, cz, R

    def _update_profile(self, m: dict):
        pos = m['positions']
        vel = m.get('velocities')
        if vel is None or pos.ndim != 2 or pos.shape[1] != 3:
            return
        cy, cz, R = self._profile_geometry()
        r = np.sqrt((pos[:, 1] - cy) ** 2 + (pos[:, 2] - cz) ** 2)
        vx = vel[:, 0]
        r_bins = np.linspace(0.0, R, 15)
        r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
        vx_mean = np.array([
            vx[(r >= r_bins[k]) & (r < r_bins[k + 1])].mean()
            if ((r >= r_bins[k]) & (r < r_bins[k + 1])).sum() > 0 else 0.0
            for k in range(len(r_bins) - 1)
        ])
        vmax = vx_mean.max() if vx_mean.max() > 0 else 1.0
        vx_ana = vmax * (1.0 - r_centers ** 2 / R ** 2)
        self._curve_profile_sim.setData(r_centers, vx_mean)
        self._curve_profile_ana.setData(r_centers, vx_ana)

    # ── Part A: clip plane + unified fluid draw ───────────────────────────
    def _clip_mask(self, positions: np.ndarray) -> np.ndarray:
        """Keep-mask for the clipping plane (True = render). Keeps the lower
        side: position[axis] <= plane_value. axis Off → keep all."""
        n = positions.shape[0]
        if self._clip_axis is None or self._fluid_bbox is None or n == 0:
            return np.ones(n, dtype=bool)
        ax = self._clip_axis
        if ax >= positions.shape[1]:
            return np.ones(n, dtype=bool)
        lo = float(self._fluid_bbox[0][ax])
        hi = float(self._fluid_bbox[1][ax])
        plane = lo + self._clip_frac * (hi - lo)
        return positions[:, ax] <= plane + 1e-9

    def _draw_fluid(self, positions: np.ndarray, speeds: np.ndarray,
                    velocities: np.ndarray, p_display: np.ndarray | None = None) -> None:
        """Apply finite-guard, clip plane, colour map and size to the scatter."""
        positions = np.asarray(positions, dtype=np.float64)
        speeds = np.asarray(speeds, dtype=np.float64)
        velocities = np.asarray(velocities, dtype=np.float64)
        if p_display is not None:
            p_display = np.asarray(p_display, dtype=np.float64)

        finite = np.isfinite(positions).all(axis=1)
        if speeds.shape[0] == positions.shape[0]:
            finite &= np.isfinite(speeds)
        keep = finite & self._clip_mask(positions)
        positions = positions[keep]
        speeds = speeds[keep]
        if velocities.shape[0] == keep.shape[0]:
            velocities = velocities[keep]
        if p_display is not None and p_display.shape[0] == keep.shape[0]:
            p_display = p_display[keep]

        if positions.shape[0] == 0:
            self._scatter.setData(pos=np.zeros((0, 3), dtype=np.float32))
        else:
            mode = self._combo_color.currentText()
            if mode == 'Pressure' and p_display is not None:
                colors = self._pressure_colors(positions, p_display)
            elif mode == 'Speed':
                colors = self._velocity_colors(positions, speeds)
            else:
                values = None
                if mode == 'Secondary':
                    values = self._secondary_magnitude(positions, velocities)
                colors = self._particle_colors(speeds, values)
            self._scatter.setData(
                pos=positions.astype(np.float32), color=colors,
                size=self._particle_size_world())
        self._update_colorbar_labels()

    def _velocity_colors(self, positions: np.ndarray, speeds: np.ndarray) -> np.ndarray:
        """Part A: reference-CFD velocity look. Scale = [0, 99th-pct of INTERIOR
        speeds] (wall-deficient particles excluded from the max so transient
        outliers don't wash out the field), full blue→red colormap (Turbo
        default). All particles coloured on that scale."""
        n = len(speeds)
        counts = self._neighbor_counts(positions)
        med = float(np.median(counts)) if n else 1.0
        interior = counts >= 0.6 * max(med, 1.0)
        s_int = speeds[interior] if interior.any() else speeds
        s_int = s_int[np.isfinite(s_int)]
        vmax = float(np.percentile(s_int, 99)) if s_int.size else 1.0
        vmax = max(vmax, 1e-9)
        self._colorbar_vmin = 0.0
        self._colorbar_vmax = vmax
        t = np.clip(np.nan_to_num(speeds) / vmax, 0.0, 1.0)
        fn = _CMAP_FNS.get(self._combo_colormap.currentText(), _turbo)
        r, g, b = fn(t)
        colors = np.zeros((n, 4), dtype=np.float32)
        colors[:, 0] = r; colors[:, 1] = g; colors[:, 2] = b; colors[:, 3] = 0.95
        return colors

    def _neighbor_counts(self, positions: np.ndarray) -> np.ndarray:
        """Per-fluid-particle FLUID-neighbour count within the support radius.
        Fluid-only (boundary excluded) so wall-adjacent particles — which have
        fluid on one side only — show a genuinely low count and are detected as
        kernel-deficient. Computed in Pressure/Speed modes (cost paid on demand)."""
        try:
            from scipy.spatial import cKDTree
        except Exception:
            return np.full(positions.shape[0], 1.0)
        h = float(getattr(self, '_support_radius', 0.0)) or 0.04
        tree = cKDTree(positions)
        counts = tree.query_ball_point(positions, r=h, return_length=True) - 1
        return counts.astype(np.float64)

    def _pressure_colors(self, positions: np.ndarray, p_display: np.ndarray) -> np.ndarray:
        """FIX1+2: gray out wall-deficient particles, spatially SMOOTH the
        pressure field (neighbour-average removes WCSPH per-particle density
        noise) and colour with a diverging Coolwarm map over a symmetric,
        outlier-proof (5–95th percentile) range of the interior, smoothed field."""
        n = positions.shape[0]
        gray = np.array([0.45, 0.45, 0.45, 0.35], dtype=np.float32)
        if n == 0:
            return np.zeros((0, 4), dtype=np.float32)
        try:
            from scipy.spatial import cKDTree
        except Exception:
            cKDTree = None

        h = float(getattr(self, '_support_radius', 0.0)) or 0.04
        if cKDTree is not None:
            tree = cKDTree(positions)
            nbrs = tree.query_ball_point(positions, r=h)
            counts = np.array([len(lst) - 1 for lst in nbrs], dtype=np.float64)
            # neighbour-average p_display → smooth field (display-only)
            p_smooth = np.array([float(p_display[lst].mean()) if lst else float(p_display[i])
                                 for i, lst in enumerate(nbrs)], dtype=np.float64)
        else:
            counts = np.full(n, 1.0)
            p_smooth = p_display.copy()

        med = float(np.median(counts))
        deficient = counts < 0.6 * max(med, 1.0)
        interior = ~deficient
        self._pressure_deficient_count = int(deficient.sum())

        # RELATIVE pressure: centre the diverging map on the interior median
        # (absolute 0 is meaningless in WCSPH — the whole field is offset by the
        # density floor). A near-uniform field then reads neutral/white, and
        # deviations show blue (low) / red (high). Symmetric, 5–95th pct, robust.
        p_int = p_smooth[interior] if interior.any() else p_smooth
        if p_int.size:
            centre = float(np.median(p_int))
            lo = float(np.percentile(p_int, 5))
            hi = float(np.percentile(p_int, 95))
        else:
            centre, lo, hi = 0.0, -1.0, 1.0
        dev = max(abs(lo - centre), abs(hi - centre), 1e-9)
        self._colorbar_vmin = centre - dev
        self._colorbar_vmax = centre + dev

        # Diverging Coolwarm on the SMOOTHED field; clamp so outliers saturate.
        t = np.clip((p_smooth - centre) / (2.0 * dev) + 0.5, 0.0, 1.0)
        r, g, b = _coolwarm(t)
        colors = np.zeros((n, 4), dtype=np.float32)
        colors[:, 0] = r; colors[:, 1] = g; colors[:, 2] = b; colors[:, 3] = 0.95
        colors[deficient] = gray   # wall-deficient → neutral gray
        return colors

    def _rerender_current(self) -> None:
        """Redraw the current (paused) frame — used by clip/colormode toggles."""
        if self.runner is None or self.runner.backend.sim.fluid.n == 0:
            return
        fl = self.runner.backend.sim.fluid
        pos = fl.positions.astype(np.float64)
        vel = fl.velocities.astype(np.float64)
        pdisp = fl.p_display.copy() if hasattr(fl, 'p_display') else None
        self._draw_fluid(pos, np.linalg.norm(vel, axis=1), vel, pdisp)

    def _on_clip_axis_changed(self, text: str) -> None:
        self._clip_axis = {'Off': None, 'X': 0, 'Y': 1, 'Z': 2}.get(text, None)
        self._rerender_current()

    def _on_clip_pos_changed(self, v: int) -> None:
        self._clip_frac = float(v) / 100.0
        self._rerender_current()

    # ── Part B: inlet / outlet markers ────────────────────────────────────
    def _on_io_toggled(self, on: bool) -> None:
        self._show_io_markers = bool(on)
        self._apply_io_visibility()
        if on:
            self._update_io_markers()

    def _apply_io_visibility(self) -> None:
        vis = self._show_io_markers
        for item in (self._inlet_ring, self._outlet_ring,
                     self._inlet_label, self._outlet_label):
            if item is not None:
                item.setVisible(vis)

    @staticmethod
    def _ring_points(center: np.ndarray, normal: np.ndarray,
                     radius: float, segs: int = 48) -> np.ndarray:
        """Closed circle of `radius` centred at `center`, normal to `normal`."""
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        ref = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(normal, ref); u /= (np.linalg.norm(u) + 1e-12)
        w = np.cross(normal, u)
        th = np.linspace(0.0, 2 * np.pi, segs + 1)
        pts = (center[None, :] + radius * (np.cos(th)[:, None] * u[None, :]
                                           + np.sin(th)[:, None] * w[None, :]))
        return pts.astype(np.float32)

    def _io_geometry(self):
        """Return (inlet_center, inlet_normal, outlet_center, outlet_normal, r).
        Curved-elbow aware; falls back to fluid-bbox flow extremes."""
        r = float(getattr(self, '_scene_R', 0.045)) or 0.045
        fl = self.runner.backend.sim.fluid if self.runner is not None else None
        if fl is None or getattr(fl, 'n', 0) == 0 or fl.positions.shape[1] < 3:
            return None
        p = fl.positions
        scene = getattr(self.runner.backend.sim, 'scene', {})
        forces = scene.get('forces', {}) if isinstance(scene, dict) else {}
        if str(forces.get('body_force_mode', '')).lower() == 'tangent':
            # Elbow: inlet at min-x end (centerline y≈R_bend), outlet at max-y end.
            r_bend = 0.135
            x_in = float(p[:, 0].min())
            inlet_c = np.array([x_in, r_bend, 0.0])
            inlet_n = np.array([1.0, 0.0, 0.0])
            y_out = float(p[:, 1].max())
            x_out = float(np.median(p[p[:, 1] > p[:, 1].max() - 0.03, 0])) if (p[:, 1] > p[:, 1].max() - 0.03).any() else float(p[:, 0].max())
            outlet_c = np.array([x_out, y_out, 0.0])
            outlet_n = np.array([0.0, 1.0, 0.0])
            return inlet_c, inlet_n, outlet_c, outlet_n, r
        # Generic fallback: inlet = min-x face, outlet = max-x face.
        pmin = p.min(0); pmax = p.max(0)
        c = 0.5 * (pmin + pmax)
        inlet_c = np.array([pmin[0], c[1], c[2]])
        outlet_c = np.array([pmax[0], c[1], c[2]])
        n = np.array([1.0, 0.0, 0.0])
        return inlet_c, n, outlet_c, n, r

    def _update_io_markers(self) -> None:
        geom = self._io_geometry()
        if geom is None:
            return
        inlet_c, inlet_n, outlet_c, outlet_n, r = geom
        self._inlet_ring.setData(pos=self._ring_points(inlet_c, inlet_n, r))
        self._outlet_ring.setData(pos=self._ring_points(outlet_c, outlet_n, r))
        if self._inlet_label is not None:
            self._inlet_label.setData(pos=inlet_c + np.array([0, -r * 1.4, 0]), text='INLET')
        if self._outlet_label is not None:
            self._outlet_label.setData(pos=outlet_c + np.array([r * 1.4, 0, 0]), text='OUTLET')
        self._apply_io_visibility()

    # ── Part A: XYZ orientation gizmo ─────────────────────────────────────
    def _toggle_axes(self, checked: bool) -> None:
        self._show_axes = bool(checked)
        self._apply_axes_visibility()

    def _apply_axes_visibility(self) -> None:
        for it in self._axes_items + self._axes_labels:
            if it is not None:
                it.setVisible(self._show_axes)

    def _update_axes_gizmo(self) -> None:
        """Small XYZ gizmo in the lower-left corner, pushed clear of the
        inlet/outlet rings (which span centreline ± r)."""
        r = float(getattr(self, '_scene_R', 0.045)) or 0.045
        L = 0.4 * r                         # subtle, short axes
        if self._fluid_bbox is not None:
            origin = self._fluid_bbox[0].astype(np.float64).copy()
            origin[:] -= 2.2 * r            # > r clear of any inlet/outlet ring
        else:
            origin = np.zeros(3)
        dirs = (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]))
        for ln, d in zip(self._axes_items, dirs):
            seg = np.vstack([origin, origin + L * d]).astype(np.float32)
            ln.setData(pos=seg)
        for lb, d in zip(self._axes_labels, dirs):
            if lb is not None:
                lb.setData(pos=origin + L * 1.25 * d)
        self._apply_axes_visibility()

    # ── Part B: live flow-drive controls ──────────────────────────────────
    def _sync_flow_controls(self) -> None:
        """Detect the loaded scene's drive mode; populate + show/hide controls."""
        sim = self.runner.backend.sim if self.runner is not None else None
        mode = str(getattr(sim, '_body_force_mode', 'constant')).lower() if sim else 'constant'
        has_io = getattr(sim, '_inlet_yz', None) is not None if sim else False

        tangent = (mode == 'tangent')
        # Drive magnitude (tangent scenes)
        self._lbl_drive.setVisible(tangent)
        self._spin_drive.setVisible(tangent)
        if tangent:
            self._spin_drive.blockSignals(True)
            self._spin_drive.setValue(float(getattr(sim, '_body_force_magnitude', 2.0)))
            self._spin_drive.blockSignals(False)

        # Inlet velocity (inlet/outlet scenes); outlet = read-only atmosphere
        self._lbl_inletv.setVisible(has_io)
        self._spin_inletv.setVisible(has_io)
        self._lbl_outlet_bc.setVisible(has_io)
        if has_io:
            self._spin_inletv.blockSignals(True)
            iv = float(getattr(sim, '_inlet_velocity', 0.10))
            self._spin_inletv.setValue(min(max(iv, 0.02), 0.50))
            self._spin_inletv.blockSignals(False)

        # Outlet note: shown for tangent (open) scenes, hidden for io scenes.
        self._lbl_outlet_note.setVisible(tangent and not has_io)
        # If neither mode applies (constant gravity), show drive disabled note.
        if not tangent and not has_io:
            self._lbl_drive.setVisible(True)
            self._spin_drive.setVisible(True)
            self._spin_drive.setEnabled(False)
            self._lbl_drive.setText('Drive (constant g):')
        else:
            self._spin_drive.setEnabled(True)
            self._lbl_drive.setText('Drive magnitude:')

    def _on_drive_changed(self, v: float) -> None:
        """Push the drive magnitude into the live sim (read each step)."""
        if self.runner is not None:
            sim = self.runner.backend.sim
            if hasattr(sim, '_body_force_magnitude'):
                sim._body_force_magnitude = float(v)

    def _on_inletv_changed(self, v: float) -> None:
        if self.runner is not None:
            sim = self.runner.backend.sim
            if hasattr(sim, '_inlet_velocity'):
                sim._inlet_velocity = float(v)

    def _on_step(self, m: dict):
        positions = np.asarray(m['positions'], dtype=np.float64)
        speeds = np.asarray(m['speeds'], dtype=np.float64)
        velocities = np.asarray(
            m.get('velocities', np.zeros_like(positions)), dtype=np.float64)

        # Parts A/B/C: finite-guard + clip plane + colour map (display-only).
        self._draw_fluid(positions, speeds, velocities, m.get('p_display'))

        # Part C: refresh the bend velocity-arrow overlay when enabled.
        if self._chk_vectors.isChecked():
            self._update_flow_vectors(positions, velocities)

        self._stat_vmax.set_value(m['vmax'])
        self._stat_rho.set_value(m['rho_err'])
        iter_cd = m['iter_cd']
        if isinstance(iter_cd, float):
            iter_cd = int(iter_cd)
        self._stat_icd.set_value(iter_cd)
        self._stat_idf.set_value(m['iter_df'])
        self._stat_dt.set_value(m['dt'])
        self._stat_n.set_value(m['n_fluid'])
        re_val = m.get('reynolds_number', 0.0)
        regime_val = m.get('regime', 'LAMINAR')
        self._stat_re.set_value(f"{re_val:.1f} ({regime_val})")

        _now = time.time()
        _fps = 1.0 / max(_now - self._last_frame_time, 1e-6)
        self._last_frame_time = _now
        self._fps = _fps
        self._stat_fps.set_value(_fps)
        self._lbl_fps.setText(f'FPS: {_fps:.1f}')
        self._lbl_time.setText(f"t = {m['time']:.4f} s  |  step {m['step']}")
        self._lbl_step.setText(f"Step: {m['step']}")
        self._update_status(m)

        max_pts = 500
        self._history['time'].append(m['time'])
        self._history['vmax'].append(m['vmax'])
        self._history['rho_err'].append(m['rho_err'])
        self._history['iter_cd'].append(
            float(m['iter_cd']) if isinstance(m['iter_cd'], (int, float)) else 0.0)
        for k in self._history:
            if len(self._history[k]) > max_pts:
                self._history[k].pop(0)

        t = self._history['time']
        self._curve_vmax.setData(t, self._history['vmax'])
        self._curve_rho.setData(t, self._history['rho_err'])
        self._update_profile(m)

        if self._pause_at_step > 0 and m['step'] >= self._pause_at_step:
            self._pause()

    def _on_error(self, msg: str):
        self._log_msg(f'SIM ERROR: {msg}', error=True)
        self._pause()

    def _toggle_boundary(self, checked: bool):
        self._show_boundary = checked
        self._boundary_scatter.setVisible(checked)

    def _reset_camera(self):
        self.gl_widget.setCameraPosition(distance=3.5, elevation=25, azimuth=45)

    def _focus_bend(self):
        """Frame the camera on the bend at a 3/4 angle to show secondary flow."""
        from pyqtgraph import Vector
        center = np.array([0.30, 0.06, 0.0])
        diag = 0.18
        if self.runner is not None:
            fl = self.runner.backend.sim.fluid
            if getattr(fl, 'n', 0) > 0 and fl.positions.shape[1] >= 3:
                p = fl.positions
                bend = ((p[:, 0] > 0.25) & (p[:, 0] < 0.45) &
                        (p[:, 1] > 0.02) & (p[:, 1] < 0.18))
                pts = p[bend] if bend.sum() > 5 else p
                finite = pts[np.isfinite(pts).all(axis=1)]
                if finite.shape[0] > 0:
                    pmin, pmax = finite.min(axis=0), finite.max(axis=0)
                    center = 0.5 * (pmin + pmax)
                    diag = max(float(np.linalg.norm(pmax - pmin)), 0.1)
        self.gl_widget.opts['center'] = Vector(
            float(center[0]), float(center[1]), float(center[2]))
        self.gl_widget.setCameraPosition(
            distance=max(diag * 2.2, 0.15), elevation=28, azimuth=35)

    def _export_vtk(self):
        if not self.runner:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export VTK', 'out/export.vtk', 'VTK Files (*.vtk)')
        if path:
            try:
                paths = self.runner.export_snapshot(csv=False, vtk=True)
                self._log_msg(f'Exported: {paths.get("vtk", path)}')
            except Exception as e:
                self._log_msg(f'Export error: {e}', error=True)

    def _screenshot(self):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Screenshot', 'out/screenshot.png', 'PNG Images (*.png)')
        if path:
            try:
                img = self.gl_widget.renderToArray((1920, 1080))
                import imageio
                imageio.imwrite(path, img)
                self._log_msg(f'Screenshot: {path}')
            except Exception as e:
                self._log_msg(f'Screenshot error: {e}', error=True)

    def _about(self):
        QMessageBox.about(self, 'About',
            'SPH Framework\n'
            'DFSPH Solver · 3D · GPU-Ready\n\n'
            'Siemens Xcelerator + NVIDIA Omniverse aesthetics\n'
            'Bender & Koschier 2017')

    def _set_status(self, text: str, color: str):
        self._lbl_status.setText(text)
        self._lbl_status.setStyleSheet(f'color: {color}; font-size: 11px;')

    def _log_msg(self, msg: str, error: bool = False):
        color = PALETTE['dark']['danger'] if error else PALETTE['dark']['text_mid']
        self._log.append(f'<span style="color:{color}">{msg}</span>')

    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
            self.worker.wait()
        event.accept()


# ══════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════

def launch():
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    app.setApplicationName('SPH Framework')
    app.setApplicationVersion('1.0')
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    launch()
