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


_CMAP_FNS = {
    'Turbo':   _turbo,
    'Plasma':  _plasma,
    'Viridis': _viridis,
    'Inferno': _inferno,
    'Hot':     _hot,
    'Cool':    _cool,
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
        vm.addSeparator()
        self._act_boundary = act(
            vm, 'Show Boundary Particles',
            self._toggle_boundary, checkable=True)
        self._act_boundary.setChecked(True)
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
        self._combo_color.addItems(['Speed', 'Density error', 'Pressure', 'Solid color'])
        vg.addWidget(self._combo_color, 0, 1)

        vg.addWidget(QLabel('Colormap:'), 1, 0)
        self._combo_colormap = QComboBox()
        self._combo_colormap.addItems(list(COLORMAP_OPTIONS))
        self._combo_colormap.currentTextChanged.connect(self._on_colormap_changed)
        vg.addWidget(self._combo_colormap, 1, 1)

        vg.addWidget(QLabel('Particle size:'), 2, 0)
        self._slider_size = QSlider(Qt.Orientation.Horizontal)
        self._slider_size.setRange(1, 20)
        self._slider_size.setValue(6)
        vg.addWidget(self._slider_size, 2, 1)
        layout.addWidget(vis_grp)

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
        self.gl_widget.setBackgroundColor(pg.mkColor(8, 10, 15))
        self.gl_widget.setCameraPosition(distance=3.5, elevation=25, azimuth=45)

        grid = gl.GLGridItem()
        grid.setColor(pg.mkColor(30, 37, 53, 80))
        grid.setSize(3, 3)
        grid.setSpacing(0.1, 0.1)
        self.gl_widget.addItem(grid)

        self._scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            size=6.0,
            color=(0.0, 0.83, 1.0, 0.9),
            pxMode=True)
        self._scatter.setGLOptions('additive')
        self.gl_widget.addItem(self._scatter)

        self._boundary_scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            size=2.0,
            color=(0.12, 0.16, 0.25, 0.15),
            pxMode=True)
        self.gl_widget.addItem(self._boundary_scatter)

        layout.addWidget(self.gl_widget)
        layout.addWidget(self._build_playback_bar())
        return container

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
        cbar_grp = QGroupBox('COLORMAP SCALE')
        cbar_layout = QVBoxLayout(cbar_grp)
        cbar_layout.addWidget(self._build_colorbar())
        layout.addWidget(cbar_grp)

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
        self._colorbar_img.setImage(img)
        # Map the 1×256 image to fill the plot: x∈[0,1], y∈[0,256]
        self._colorbar_img.setRect(0.0, 0.0, 1.0, float(n))
        self._colorbar_widget.setXRange(0.0, 1.0, padding=0)
        self._colorbar_widget.setYRange(0.0, float(n), padding=0)

    def _rebuild_colorbar(self, *_args) -> None:
        """Rebuild gradient when colormap selection changes."""
        self._refresh_colorbar_image()

    def _on_colormap_changed(self, _text: str) -> None:
        self._refresh_colorbar_image()
        if self.runner and self.runner.backend.sim.fluid.n > 0:
            fl = self.runner.backend.sim.fluid
            speeds = np.linalg.norm(fl.velocities, axis=1)
            colors = self._particle_colors(speeds)
            self._scatter.setData(
                pos=fl.positions.astype(np.float32), color=colors)

    def _apply_colormap(self, values: np.ndarray, mode: str) -> np.ndarray:
        n = len(values)
        vmax = float(np.percentile(values, 95)) if n else 1.0
        vmin = float(values.min()) if n else 0.0
        self._colorbar_vmin = vmin
        self._colorbar_vmax = vmax
        t = np.clip((values - vmin) / (vmax - vmin + 1e-10), 0, 1)
        fn = _CMAP_FNS.get(mode, _turbo)
        r, g, b = fn(t)
        colors = np.zeros((n, 4), dtype=np.float32)
        colors[:, 0] = r; colors[:, 1] = g; colors[:, 2] = b; colors[:, 3] = 0.92
        return colors

    def _particle_colors(
        self, speeds: np.ndarray, values: np.ndarray | None = None,
    ) -> np.ndarray:
        mode = self._combo_color.currentText()
        n = len(speeds)
        if mode == 'Solid color':
            colors = np.zeros((n, 4), dtype=np.float32)
            colors[:, 0] = 0.0; colors[:, 1] = 0.83; colors[:, 2] = 1.0; colors[:, 3] = 0.92
            return colors
        if values is None:
            values = speeds
        if mode == 'Speed':
            return self._apply_colormap(speeds, self._combo_colormap.currentText())
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
            self.gl_widget.setBackgroundColor(pg.mkColor(8, 10, 15))
            plot_bg = pg.mkColor(8, 10, 15)
            cbar_bg = pg.mkColor(8, 10, 15)

        for plot in (self._plot_vmax, self._plot_rho, self._plot_profile):
            plot.setBackground(plot_bg)
        self._colorbar_widget.setBackground(cbar_bg)

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
        self._boundary_scatter.setData(pos=bd.positions.astype(np.float32))
        self._boundary_scatter.setVisible(self._show_boundary)

    def _load_scene(self, path: str):
        try:
            from sph.core.simulation import SimulationRunner

            self.runner = SimulationRunner(Path(path))
            self._log_msg('Backend: CPU')
            fl = self.runner.backend.sim.fluid
            bd = self.runner.backend.sim.boundary

            self._update_boundary_display()
            if fl.n > 0:
                self._scatter.setData(pos=fl.positions.astype(np.float32))

            n_bd = len(bd.positions) if bd is not None and hasattr(bd, 'positions') else 0
            if (bd is not None and hasattr(bd, 'positions')
                    and len(bd.positions) > 0 and bd.positions.shape[1] == 3):
                cy = bd.positions[:, 1].mean()
                cz = bd.positions[:, 2].mean()
                r_bd = np.sqrt((bd.positions[:, 1] - cy) ** 2 + (bd.positions[:, 2] - cz) ** 2)
                self._scene_R = float(np.percentile(r_bd, 95))
            else:
                self._scene_R = 0.07

            self._lbl_scene.setText(f'{Path(path).name}\n{fl.n} fluid · {n_bd} bnd')
            self._log_msg(f'Scene loaded: {Path(path).name}')
            self._sb_scene.setText(f'  {Path(path).name}')
            self._history = {k: [] for k in self._history}

        except Exception as e:
            self._log_msg(f'ERROR: {e}', error=True)

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

    def _on_step(self, m: dict):
        speeds = m['speeds']
        colors = self._particle_colors(speeds)
        base_size = float(self._slider_size.value())
        if len(speeds) > 0:
            slow = float(np.percentile(speeds, 10))
            sizes = np.where(speeds < slow, 3.0, 6.0) * (base_size / 6.0)
        else:
            sizes = base_size
        self._scatter.setData(
            pos=m['positions'].astype(np.float32), color=colors, size=sizes)
        self._lbl_cbar_min.setText(f'{self._colorbar_vmin:.4f}')
        self._lbl_cbar_max.setText(f'{self._colorbar_vmax:.4f}')

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
