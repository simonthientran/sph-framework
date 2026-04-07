"""
SPH Framework — Professional GUI
NVIDIA Omniverse-inspired dark theme.
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
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QFont
import pyqtgraph as pg
import pyqtgraph.opengl as gl


# ══════════════════════════════════════════════════════
# OMNIVERSE COLOR PALETTE
# ══════════════════════════════════════════════════════

COLORS = {
    'bg_dark':     '#0d0d0f',
    'bg_panel':    '#141416',
    'bg_widget':   '#1a1a1e',
    'bg_hover':    '#222228',
    'accent':      '#00b4d8',
    'accent2':     '#0077b6',
    'accent_dim':  '#004e7c',
    'text_bright': '#e8eaed',
    'text_mid':    '#9aa0a6',
    'text_dim':    '#5f6368',
    'border':      '#2d2d35',
    'success':     '#00c853',
    'warning':     '#ffab00',
    'error':       '#f44336',
}

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {COLORS['bg_dark']};
    color: {COLORS['text_bright']};
    font-family: 'Segoe UI', 'SF Pro Display', Arial, sans-serif;
    font-size: 12px;
}}
QMenuBar {{
    background-color: {COLORS['bg_panel']};
    color: {COLORS['text_bright']};
    border-bottom: 1px solid {COLORS['border']};
    padding: 2px 0px;
}}
QMenuBar::item:selected {{
    background-color: {COLORS['bg_hover']};
    color: {COLORS['accent']};
}}
QMenu {{
    background-color: {COLORS['bg_panel']};
    color: {COLORS['text_bright']};
    border: 1px solid {COLORS['border']};
    padding: 4px 0px;
}}
QMenu::item:selected {{
    background-color: {COLORS['accent_dim']};
}}
QToolBar {{
    background-color: {COLORS['bg_panel']};
    border-bottom: 1px solid {COLORS['border']};
    spacing: 4px;
    padding: 4px;
}}
QSplitter::handle {{
    background-color: {COLORS['border']};
    width: 1px;
}}
QGroupBox {{
    background-color: {COLORS['bg_panel']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 8px;
    font-size: 11px;
    font-weight: 600;
    color: {COLORS['text_mid']};
    letter-spacing: 1px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    color: {COLORS['accent']};
    background-color: {COLORS['bg_panel']};
}}
QPushButton {{
    background-color: {COLORS['bg_widget']};
    color: {COLORS['text_bright']};
    border: 1px solid {COLORS['border']};
    border-radius: 3px;
    padding: 5px 12px;
    font-size: 12px;
    min-height: 24px;
}}
QPushButton:hover {{
    background-color: {COLORS['bg_hover']};
    border-color: {COLORS['accent']};
    color: {COLORS['accent']};
}}
QPushButton:pressed {{
    background-color: {COLORS['accent_dim']};
}}
QPushButton#accent_btn {{
    background-color: {COLORS['accent_dim']};
    color: {COLORS['accent']};
    border-color: {COLORS['accent']};
    font-weight: 600;
}}
QPushButton#accent_btn:hover {{
    background-color: {COLORS['accent']};
    color: {COLORS['bg_dark']};
}}
QPushButton#danger_btn {{
    border-color: {COLORS['error']};
    color: {COLORS['error']};
}}
QPushButton#danger_btn:hover {{
    background-color: {COLORS['error']};
    color: white;
}}
QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {COLORS['bg_widget']};
    color: {COLORS['text_bright']};
    border: 1px solid {COLORS['border']};
    border-radius: 3px;
    padding: 3px 6px;
    min-height: 22px;
}}
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {COLORS['accent']};
}}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background-color: {COLORS['bg_panel']};
    color: {COLORS['text_bright']};
    border: 1px solid {COLORS['border']};
    selection-background-color: {COLORS['accent_dim']};
}}
QSlider::groove:horizontal {{
    background: {COLORS['bg_widget']};
    height: 4px;
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {COLORS['accent']};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {COLORS['accent']};
    width: 12px;
    height: 12px;
    border-radius: 6px;
    margin: -4px 0;
}}
QProgressBar {{
    background-color: {COLORS['bg_widget']};
    border: 1px solid {COLORS['border']};
    border-radius: 3px;
    text-align: center;
    color: {COLORS['text_bright']};
    font-size: 11px;
    max-height: 16px;
}}
QProgressBar::chunk {{
    background-color: {COLORS['accent']};
    border-radius: 2px;
}}
QTabWidget::pane {{
    background-color: {COLORS['bg_panel']};
    border: 1px solid {COLORS['border']};
    border-top: none;
}}
QTabBar::tab {{
    background-color: {COLORS['bg_dark']};
    color: {COLORS['text_dim']};
    border: 1px solid {COLORS['border']};
    border-bottom: none;
    padding: 6px 14px;
    font-size: 11px;
}}
QTabBar::tab:selected {{
    background-color: {COLORS['bg_panel']};
    color: {COLORS['accent']};
    border-bottom: 2px solid {COLORS['accent']};
}}
QTabBar::tab:hover:!selected {{
    color: {COLORS['text_bright']};
    background-color: {COLORS['bg_widget']};
}}
QTextEdit {{
    background-color: {COLORS['bg_widget']};
    color: #a8d8a8;
    border: 1px solid {COLORS['border']};
    border-radius: 3px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 11px;
    padding: 4px;
}}
QCheckBox {{ color: {COLORS['text_bright']}; spacing: 6px; }}
QCheckBox::indicator {{
    width: 14px; height: 14px;
    border: 1px solid {COLORS['border']};
    border-radius: 2px;
    background: {COLORS['bg_widget']};
}}
QCheckBox::indicator:checked {{
    background: {COLORS['accent']};
    border-color: {COLORS['accent']};
}}
QLabel#title_label {{
    color: {COLORS['accent']};
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 2px;
}}
QLabel#subtitle_label {{
    color: {COLORS['text_dim']};
    font-size: 11px;
    letter-spacing: 1px;
}}
QLabel#stat_value {{
    color: {COLORS['text_bright']};
    font-family: 'Consolas', monospace;
    font-size: 13px;
    font-weight: 600;
}}
QLabel#stat_label {{
    color: {COLORS['text_dim']};
    font-size: 10px;
    letter-spacing: 1px;
}}
QStatusBar {{
    background-color: {COLORS['bg_panel']};
    border-top: 1px solid {COLORS['border']};
    color: {COLORS['text_mid']};
    font-size: 11px;
}}
QScrollBar:vertical {{
    background: {COLORS['bg_widget']};
    width: 6px;
    border-radius: 3px;
}}
QScrollBar::handle:vertical {{
    background: {COLORS['border']};
    border-radius: 3px;
    min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{
    background: {COLORS['accent_dim']};
}}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0px; }}
"""


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
                    'time':    float(sim.t),
                    'dt':      float(rt.runtime.dt),
                    'vmax':    float(spd.max()) if len(spd) else 0.0,
                    'rho_err': float(rt.runtime.rho_error_mean) * 100.0,
                    'iter_cd': rt.runtime.solver_metrics.get('iter_cd', 0),
                    'iter_df': rt.runtime.solver_metrics.get('iter_df', 0),
                    'n_fluid': int(fl.n),
                    'positions': pos,
                    'speeds': spd,
                }
                self.step_done.emit(metrics)
            except Exception as e:
                self.sim_error.emit(str(e))
                break

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def stop(self):
        self._running = False
        self._paused = False


# ══════════════════════════════════════════════════════
# STAT WIDGET
# ══════════════════════════════════════════════════════

class StatWidget(QWidget):
    def __init__(self, label: str, unit: str = ''):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        self.value_label = QLabel('—')
        self.value_label.setObjectName('stat_value')
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        lbl_text = f'{label}  {unit}'.strip().upper()
        lbl = QLabel(lbl_text)
        lbl.setObjectName('stat_label')
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.value_label)
        layout.addWidget(lbl)

        self.setStyleSheet(f"""
            StatWidget {{
                background: {COLORS['bg_widget']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
            }}
        """)

    def set_value(self, v):
        if isinstance(v, float):
            self.value_label.setText(f'{v:.4f}')
        else:
            self.value_label.setText(str(v))


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

        self.setWindowTitle('SPH Framework  —  Professional Edition')
        self.resize(1600, 950)
        self.setMinimumSize(1200, 700)
        self.setStyleSheet(STYLESHEET)

        self._build_menu()
        self._build_toolbar()
        self._build_central()
        self._build_statusbar()

        # Auto-load first scene if provided
        if initial_scene:
            self._load_scene(str(initial_scene))
        elif scene_paths:
            self._load_scene(str(scene_paths[0]))

    # ── Menu ─────────────────────────────────────────

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

        hm = mb.addMenu('Help')
        act(hm, 'About', self._about)

    # ── Toolbar ──────────────────────────────────────

    def _build_toolbar(self):
        tb = QToolBar('Main')
        tb.setIconSize(QSize(18, 18))
        tb.setMovable(False)
        self.addToolBar(tb)

        def btn(text, slot, obj_name=''):
            b = QPushButton(text)
            if obj_name:
                b.setObjectName(obj_name)
            b.clicked.connect(slot)
            b.setFixedHeight(28)
            tb.addWidget(b)
            return b

        btn('Open Scene', self._open_scene)
        tb.addSeparator()
        self._btn_run   = btn('▶  Run',   self._run,   'accent_btn')
        self._btn_pause = btn('⏸  Pause', self._pause)
        self._btn_reset = btn('⏹  Reset', self._reset)
        self._btn_pause.setEnabled(False)
        tb.addSeparator()
        btn('Step', self._step_once)
        tb.addSeparator()
        btn('Screenshot', self._screenshot)
        btn('Export VTK', self._export_vtk)

        spacer = QWidget()
        spacer.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(spacer)

        title = QLabel(' SPH FRAMEWORK ')
        title.setObjectName('title_label')
        tb.addWidget(title)

        sub = QLabel(' DFSPH · 3D · GPU-Ready  ')
        sub.setObjectName('subtitle_label')
        tb.addWidget(sub)

    # ── Central ──────────────────────────────────────

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
        splitter.setSizes([260, 900, 300])
        layout.addWidget(splitter)

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(260)
        panel.setStyleSheet(
            f'background: {COLORS["bg_panel"]};'
            f'border-right: 1px solid {COLORS["border"]};')
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Scene info
        scene_grp = QGroupBox('Scene')
        sg = QGridLayout(scene_grp)
        sg.setSpacing(6)
        self._lbl_scene = QLabel('No scene loaded')
        self._lbl_scene.setWordWrap(True)
        self._lbl_scene.setStyleSheet(
            f'color: {COLORS["text_mid"]}; font-size: 11px;')
        sg.addWidget(self._lbl_scene, 0, 0, 1, 2)
        load_btn = QPushButton('Load Scene...')
        load_btn.setObjectName('accent_btn')
        load_btn.clicked.connect(self._open_scene)
        sg.addWidget(load_btn, 1, 0, 1, 2)
        layout.addWidget(scene_grp)

        # Solver params
        params_grp = QGroupBox('Solver Parameters')
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

        pg_l.addWidget(QLabel('CFL factor:'), 2, 0)
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

        layout.addWidget(params_grp)

        # Visualization
        vis_grp = QGroupBox('Visualization')
        vg = QGridLayout(vis_grp)
        vg.setSpacing(6)

        vg.addWidget(QLabel('Color by:'), 0, 0)
        self._combo_color = QComboBox()
        self._combo_color.addItems(
            ['Speed', 'Density error', 'Pressure', 'Solid color'])
        vg.addWidget(self._combo_color, 0, 1)

        vg.addWidget(QLabel('Particle size:'), 1, 0)
        self._slider_size = QSlider(Qt.Orientation.Horizontal)
        self._slider_size.setRange(1, 20)
        self._slider_size.setValue(6)
        vg.addWidget(self._slider_size, 1, 1)

        layout.addWidget(vis_grp)
        layout.addStretch()

        # Console
        log_grp = QGroupBox('Console')
        lg = QVBoxLayout(log_grp)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFixedHeight(150)
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
        self.gl_widget.setBackgroundColor(pg.mkColor(COLORS['bg_dark']))
        self.gl_widget.setCameraPosition(distance=3.5, elevation=25, azimuth=45)

        grid = gl.GLGridItem()
        grid.setColor(pg.mkColor(50, 50, 60, 80))
        grid.setSize(4, 4)
        grid.setSpacing(0.25, 0.25)
        self.gl_widget.addItem(grid)

        self._scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            size=6,
            color=(0.2, 0.6, 1.0, 0.9),
            pxMode=True)
        self.gl_widget.addItem(self._scatter)

        self._boundary_scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            size=3,
            color=(0.3, 0.3, 0.4, 0.3),
            pxMode=True)
        self.gl_widget.addItem(self._boundary_scatter)

        layout.addWidget(self.gl_widget)
        layout.addWidget(self._build_playback_bar())
        return container

    def _build_playback_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(36)
        bar.setStyleSheet(
            f'background: {COLORS["bg_panel"]};'
            f'border-top: 1px solid {COLORS["border"]};')
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(12)

        self._lbl_time = QLabel('t = 0.0000 s')
        self._lbl_time.setStyleSheet(
            f'color: {COLORS["accent"]}; font-family: Consolas, monospace; font-size: 12px;')
        layout.addWidget(self._lbl_time)

        layout.addStretch()

        self._lbl_step = QLabel('Step: 0')
        self._lbl_step.setStyleSheet(
            f'color: {COLORS["text_mid"]}; font-size: 11px;')
        layout.addWidget(self._lbl_step)

        self._lbl_status = QLabel('● IDLE')
        self._lbl_status.setStyleSheet(
            f'color: {COLORS["text_dim"]}; font-size: 11px;')
        layout.addWidget(self._lbl_status)

        return bar

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(300)
        panel.setStyleSheet(
            f'background: {COLORS["bg_panel"]};'
            f'border-left: 1px solid {COLORS["border"]};')
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Stats grid
        stats_grp = QGroupBox('Live Diagnostics')
        sg = QGridLayout(stats_grp)
        sg.setSpacing(6)

        self._stat_vmax    = StatWidget('v max',     'm/s')
        self._stat_rho_err = StatWidget('ρ error',   '%')
        self._stat_iter_cd = StatWidget('iter CD',   '')
        self._stat_iter_df = StatWidget('iter DF',   '')
        self._stat_dt      = StatWidget('dt',        's')
        self._stat_n       = StatWidget('particles', '')

        sg.addWidget(self._stat_vmax,    0, 0)
        sg.addWidget(self._stat_rho_err, 0, 1)
        sg.addWidget(self._stat_iter_cd, 1, 0)
        sg.addWidget(self._stat_iter_df, 1, 1)
        sg.addWidget(self._stat_dt,      2, 0)
        sg.addWidget(self._stat_n,       2, 1)
        layout.addWidget(stats_grp)

        # Charts
        charts_grp = QGroupBox('Time Series')
        cg = QVBoxLayout(charts_grp)
        cg.setSpacing(4)

        pg.setConfigOption('background', COLORS['bg_widget'])
        pg.setConfigOption('foreground', COLORS['text_mid'])

        self._plot_vmax = pg.PlotWidget(title='v_max')
        self._plot_vmax.setFixedHeight(130)
        self._plot_vmax.showGrid(x=True, y=True, alpha=0.15)
        self._curve_vmax = self._plot_vmax.plot(
            pen=pg.mkPen(COLORS['accent'], width=2))
        cg.addWidget(self._plot_vmax)

        self._plot_rho = pg.PlotWidget(title='ρ error %')
        self._plot_rho.setFixedHeight(130)
        self._plot_rho.showGrid(x=True, y=True, alpha=0.15)
        self._curve_rho = self._plot_rho.plot(
            pen=pg.mkPen(COLORS['warning'], width=2))
        cg.addWidget(self._plot_rho)

        layout.addWidget(charts_grp)
        layout.addStretch()

        # Export
        exp_grp = QGroupBox('Export')
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

    def _build_statusbar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)
        self._sb_left = QLabel('Ready')
        self._sb_right = QLabel('SPH Framework  |  DFSPH Solver  |  3D')
        self._sb_right.setStyleSheet(f'color: {COLORS["text_dim"]};')
        sb.addWidget(self._sb_left)
        sb.addPermanentWidget(self._sb_right)

    # ── Actions ──────────────────────────────────────

    def _open_scene(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open Scene', 'scenes',
            'Scene Files (*.json);;All Files (*)')
        if path:
            self._load_scene(path)

    def _load_scene(self, path: str):
        try:
            from sph.core.simulation import SimulationRunner
            self.runner = SimulationRunner(
                Path(path), backend_name='numba_cpu')

            fl = self.runner.backend.sim.fluid
            bd = self.runner.backend.sim.boundary

            # Boundary particles
            if bd is not None and hasattr(bd, 'positions') and len(bd.positions) > 0:
                self._boundary_scatter.setData(
                    pos=bd.positions.astype(np.float32))
                self._boundary_scatter.setVisible(self._show_boundary)

            # Fluid particles
            if fl.n > 0:
                self._scatter.setData(pos=fl.positions.astype(np.float32))

            n_bd = len(bd.positions) if bd is not None and hasattr(bd, 'positions') else 0
            self._lbl_scene.setText(f'{Path(path).name}\n{fl.n} fluid · {n_bd} boundary')
            self._log_msg(f'Scene loaded: {Path(path).name}')
            self._log_msg(f'Fluid: {fl.n} · Boundary: {n_bd}')
            self._sb_left.setText(f'Scene: {Path(path).name}')
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
        self._set_status('● RUNNING', COLORS['success'])

    def _pause(self):
        if self.worker:
            self.worker.pause()
        self._btn_run.setEnabled(True)
        self._btn_pause.setEnabled(False)
        self._set_status('⏸ PAUSED', COLORS['warning'])

    def _reset(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self.worker = None
        self._btn_run.setEnabled(True)
        self._btn_pause.setEnabled(False)
        self._set_status('● IDLE', COLORS['text_dim'])
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
                'time':      float(sim.t),
                'dt':        float(rt.runtime.dt),
                'vmax':      float(spd.max()) if len(spd) else 0.0,
                'rho_err':   float(rt.runtime.rho_error_mean) * 100.0,
                'iter_cd':   rt.runtime.solver_metrics.get('iter_cd', 0),
                'iter_df':   rt.runtime.solver_metrics.get('iter_df', 0),
                'n_fluid':   int(fl.n),
                'positions': fl.positions.copy(),
                'speeds':    spd.copy(),
            })
        except Exception as e:
            self._on_error(str(e))

    def _on_step(self, m: dict):
        colors = self._speed_to_color(m['speeds'])
        self._scatter.setData(
            pos=m['positions'].astype(np.float32),
            color=colors,
            size=self._slider_size.value())

        self._stat_vmax.set_value(m['vmax'])
        self._stat_rho_err.set_value(m['rho_err'])
        self._stat_iter_cd.set_value(m['iter_cd'])
        self._stat_iter_df.set_value(m['iter_df'])
        self._stat_dt.set_value(m['dt'])
        self._stat_n.set_value(m['n_fluid'])
        self._lbl_time.setText(f"t = {m['time']:.4f} s")
        self._lbl_step.setText(f"Step: {m['step']}")

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

    def _on_error(self, msg: str):
        self._log_msg(f'SIM ERROR: {msg}', error=True)
        self._pause()

    def _speed_to_color(self, speeds: np.ndarray) -> np.ndarray:
        mode = self._combo_color.currentText()
        n = len(speeds)
        colors = np.zeros((n, 4), dtype=np.float32)
        colors[:, 3] = 0.9

        if mode == 'Solid color':
            colors[:, 0] = 0.2
            colors[:, 1] = 0.6
            colors[:, 2] = 1.0
            return colors

        vmax = speeds.max() if n else 0.0
        t = np.clip(speeds / (vmax * 0.8 + 1e-10), 0.0, 1.0)

        # Plasma-like: blue → cyan → yellow → red
        colors[:, 0] = np.clip(1.5 * t - 0.2, 0.0, 1.0)
        colors[:, 1] = np.clip(np.sin(t * np.pi) * 0.9, 0.0, 1.0)
        colors[:, 2] = np.clip(1.0 - 1.5 * t, 0.0, 1.0)
        return colors

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
            'Inspired by SPlisHSPlasH\n'
            'Bender & Koschier 2017')

    def _set_status(self, text: str, color: str):
        self._lbl_status.setText(text)
        self._lbl_status.setStyleSheet(
            f'color: {color}; font-size: 11px;')

    def _log_msg(self, msg: str, error: bool = False):
        color = COLORS['error'] if error else COLORS['text_mid']
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
