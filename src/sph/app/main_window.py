"""Qt main window for the SPH live application."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PyQt6 import QtCore, QtWidgets

from sph.app.controller import SimulationController
from sph.app.live_view import ParticleView
from sph.core.backend import StepDiagnostics
from sph.core.scene import SceneMetadata, load_scene_metadata
from sph.core.simulation import StepResult


class MainWindow(QtWidgets.QMainWindow):
    """Top-level GUI wiring the controller, particle view, and diagnostics panel."""

    _STABILITY_STYLES: Dict[str, Tuple[str, str]] = {
        "pass": ("#0f5132", "#d1e7dd"),
        "warn": ("#8a4f00", "#fff3cd"),
        "fail": ("#842029", "#f8d7da"),
    }
    _WARNING_STYLES: Dict[str, Tuple[str, str]] = {
        "ok": ("#0f5132", "#e9f7ef"),
        "warn": ("#7a4100", "#fff4e5"),
        "fail": ("#842029", "#fde2e1"),
    }

    def __init__(
        self,
        scene_paths: List[Path],
        initial_scene: Optional[Path] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        if not scene_paths:
            raise ValueError("At least one scene must be provided.")

        self.scene_paths = [Path(p) for p in scene_paths]
        self.scene_metadata_map: Dict[Path, SceneMetadata] = {}
        for path in self.scene_paths:
            self._ensure_scene_metadata(path)

        self.current_scene = initial_scene or self.scene_paths[0]
        if self.current_scene not in self.scene_paths:
            self.scene_paths.insert(0, self.current_scene)
            self._ensure_scene_metadata(self.current_scene)

        self.controller = SimulationController(self.current_scene)
        self.scene_metadata_map[self.current_scene] = self.controller.scene_metadata
        preferred_overlay = self.controller.scene_metadata.preferred_overlay
        self._viz_mode = preferred_overlay if preferred_overlay in ParticleView.COLOR_MODES else "type"
        self.setWindowTitle("SPH Live Viewer")
        self.resize(1280, 780)

        central = QtWidgets.QWidget(self)
        layout = QtWidgets.QHBoxLayout(central)
        self.setCentralWidget(central)

        self.particle_view = ParticleView()
        layout.addWidget(self.particle_view, stretch=3)

        side_panel = QtWidgets.QVBoxLayout()
        layout.addLayout(side_panel, stretch=2)

        controls = self._build_controls_box()
        side_panel.addWidget(controls)
        self.diag_box = self._build_diagnostics_box()
        side_panel.addWidget(self.diag_box)
        side_panel.addStretch(1)
        self._reset_metrics()

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(0)
        self.timer.timeout.connect(self._advance_simulation)

        self._sync_scene_combo()
        self._refresh_view(initial=True)
        self.statusBar().showMessage("Ready.")

    # ------------------------------------------------------------------ UI construction
    def _build_controls_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Simulation Controls", self)
        layout = QtWidgets.QGridLayout(box)

        self.scene_combo = QtWidgets.QComboBox(box)
        self.scene_combo.currentIndexChanged.connect(self._on_scene_changed)
        layout.addWidget(QtWidgets.QLabel("Scene:"), 0, 0)
        layout.addWidget(self.scene_combo, 0, 1, 1, 3)

        self.start_btn = QtWidgets.QPushButton("Start", box)
        self.start_btn.clicked.connect(self.start)
        layout.addWidget(self.start_btn, 1, 0)

        self.pause_btn = QtWidgets.QPushButton("Pause", box)
        self.pause_btn.clicked.connect(self.pause)
        layout.addWidget(self.pause_btn, 1, 1)

        self.step_btn = QtWidgets.QPushButton("Single Step", box)
        self.step_btn.clicked.connect(self.single_step)
        layout.addWidget(self.step_btn, 1, 2)

        self.reset_btn = QtWidgets.QPushButton("Reset", box)
        self.reset_btn.clicked.connect(self.reset_simulation)
        layout.addWidget(self.reset_btn, 1, 3)

        self.solver_label = QtWidgets.QLabel(self.controller.solver_name)
        layout.addWidget(QtWidgets.QLabel("Solver:"), 2, 0)
        layout.addWidget(self.solver_label, 2, 1, 1, 3)

        self.viz_combo = QtWidgets.QComboBox(box)
        for key, label in ParticleView.COLOR_MODES.items():
            self.viz_combo.addItem(label, key)
        self.viz_combo.currentIndexChanged.connect(self._on_viz_mode_changed)
        layout.addWidget(QtWidgets.QLabel("Particle colors:"), 3, 0)
        layout.addWidget(self.viz_combo, 3, 1, 1, 3)

        return box

    def _build_diagnostics_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Diagnostics", self)
        layout = QtWidgets.QVBoxLayout(box)
        layout.setSpacing(8)
        self.metrics: Dict[str, QtWidgets.QLabel] = {}

        sections = [
            (
                "Run",
                [
                    ("scene", "Scene"),
                    ("solver", "Solver"),
                    ("viz_mode", "Particle colors"),
                    ("step", "Step"),
                    ("dt", "dt [s]"),
                ],
            ),
            (
                "Fluid",
                [
                    ("counts", "Fluid / Boundary count"),
                    ("velocity", "|v|max [m/s]"),
                    ("rho", "ρ min / avg / max"),
                    ("rho_err", "ρ err [%]"),
                ],
            ),
            (
                "Pressure & Neighbors",
                [
                    ("pressure", "p min / avg / max"),
                    ("neighbors", "Neighbors min / avg / max"),
                ],
            ),
            (
                "Performance",
                [
                    ("perf_step", "Step ms / FPS"),
                    ("perf_avg", "Avg ms / FPS"),
                ],
            ),
        ]

        for title, entries in sections:
            section = QtWidgets.QGroupBox(title, box)
            form = QtWidgets.QFormLayout(section)
            form.setContentsMargins(8, 4, 8, 8)
            for key, label in entries:
                value = QtWidgets.QLabel("—")
                value.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                form.addRow(label, value)
                self.metrics[key] = value
            layout.addWidget(section)

        health_box = QtWidgets.QGroupBox("Health", box)
        health_layout = QtWidgets.QVBoxLayout(health_box)
        stability = QtWidgets.QLabel("—")
        stability.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        stability.setFixedHeight(32)
        stability.setStyleSheet("border-radius:6px; padding:6px; background:#2b2b2b; color:#dddddd;")
        self.metrics["stability"] = stability
        health_layout.addWidget(stability)

        warnings = QtWidgets.QLabel("—")
        warnings.setWordWrap(True)
        warnings.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        warnings.setStyleSheet("border-radius:4px; padding:4px; background:#2b2b2b; color:#dddddd;")
        self.metrics["warnings"] = warnings
        health_layout.addWidget(warnings)
        layout.addWidget(health_box)

        summary = QtWidgets.QLabel("—")
        summary.setWordWrap(True)
        summary.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        summary.setStyleSheet("border:1px solid #3a3a3a; border-radius:4px; padding:6px;")
        self.metrics["summary"] = summary
        layout.addWidget(summary)
        layout.addStretch(1)

        return box

    def _ensure_scene_metadata(self, path: Path) -> SceneMetadata:
        meta = self.scene_metadata_map.get(path)
        if meta is None:
            try:
                meta = load_scene_metadata(path)
            except Exception:
                meta = SceneMetadata(
                    path=path,
                    name=path.stem,
                    dimensions=2,
                    intended_use="general",
                    preferred_overlay=None,
                    notes=None,
                )
            self.scene_metadata_map[path] = meta
        return meta

    def _current_scene_metadata(self) -> SceneMetadata:
        return self.scene_metadata_map.get(self.current_scene) or self._ensure_scene_metadata(self.current_scene)

    def _apply_scene_metadata(self) -> None:
        meta = self.controller.scene_metadata
        self.scene_metadata_map[self.current_scene] = meta
        if hasattr(self, "metrics") and "scene" in self.metrics:
            self.metrics["scene"].setText(meta.name)
        preferred = meta.preferred_overlay
        if preferred and preferred in ParticleView.COLOR_MODES:
            self._set_visualization_mode(preferred, update_combo=True, refresh_view=True)
        if meta.notes:
            self.statusBar().showMessage(f"{meta.name}: {meta.notes}")

    def _sync_scene_combo(self) -> None:
        self.scene_combo.blockSignals(True)
        self.scene_combo.clear()
        for idx, path in enumerate(self.scene_paths):
            meta = self._ensure_scene_metadata(path)
            self.scene_combo.addItem(meta.name, path)
            if path == self.current_scene:
                self.scene_combo.setCurrentIndex(idx)
        self.scene_combo.blockSignals(False)
        self.solver_label.setText(self.controller.solver_name)
        if hasattr(self, "metrics"):
            if "scene" in self.metrics:
                self.metrics["scene"].setText(self._current_scene_metadata().name)
            if "solver" in self.metrics:
                self.metrics["solver"].setText(self.controller.solver_name)
        self._apply_scene_metadata()

    # ------------------------------------------------------------------ controls
    def start(self) -> None:
        if not self.timer.isActive():
            self.timer.start()
            self.statusBar().showMessage("Running…")

    def pause(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.statusBar().showMessage("Paused.")

    def single_step(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
        self._advance_simulation()

    def reset_simulation(self) -> None:
        self.pause()
        self.controller.reset()
        self._refresh_view(initial=True)
        self._apply_scene_metadata()
        self.statusBar().showMessage("Simulation reset.")

    def _on_scene_changed(self, index: int) -> None:
        path = self.scene_combo.itemData(index)
        if not isinstance(path, Path):
            return
        if path == self.current_scene:
            return
        self.pause()
        self.current_scene = path
        self.controller = SimulationController(self.current_scene)
        self._sync_scene_combo()
        self._refresh_view(initial=True)
        self._apply_scene_metadata()
        self.statusBar().showMessage(f"Scene loaded: {self.current_scene.name}")

    # ------------------------------------------------------------------ simulation loop
    def _advance_simulation(self) -> None:
        try:
            result = self.controller.step()
        except Exception as exc:  # pragma: no cover - surfaced in UI
            self.pause()
            QtWidgets.QMessageBox.critical(self, "Simulation error", str(exc))
            return
        self._apply_step_result(result)

    def _refresh_view(self, initial: bool = False) -> None:
        state = self.controller.state
        if state is None:
            return
        self.particle_view.render_state(state)
        if initial:
            self._reset_metrics()

    def _apply_step_result(self, result: StepResult) -> None:
        state = result.state
        diag = result.diagnostics
        self.particle_view.render_state(state)

        viz_label = self._viz_mode_label(self._viz_mode)
        self.metrics["scene"].setText(diag.scene_name)
        self.metrics["solver"].setText(diag.solver)
        self.metrics["viz_mode"].setText(viz_label)

        self.metrics["step"].setText(str(diag.step))
        self.metrics["dt"].setText(f"{diag.dt:.3e}")
        self.metrics["velocity"].setText(f"{diag.velocity_max:.4f}")
        self.metrics["rho"].setText(f"{diag.rho_min:.1f} / {diag.rho_mean:.1f} / {diag.rho_max:.1f}")
        self.metrics["rho_err"].setText(f"{diag.rho_error_mean * 100.0:.2f}")
        self.metrics["pressure"].setText(
            f"{diag.pressure_min:.1f} / {diag.pressure_mean:.1f} / {diag.pressure_max:.1f}"
        )
        self.metrics["neighbors"].setText(
            f"{diag.neighbor_min} / {diag.neighbor_mean:.1f} / {diag.neighbor_max}"
        )
        self.metrics["counts"].setText(f"{diag.fluid_count} / {diag.boundary_count}")

        perf = diag.performance
        self.metrics["perf_step"].setText(f"{perf.wall_time_ms:.1f} / {perf.steps_per_second:.1f}")
        self.metrics["perf_avg"].setText(f"{perf.rolling_wall_time_ms:.1f} / {perf.rolling_steps_per_second:.1f}")

        self._apply_stability_style(diag.stability)
        warnings, severity = self._collect_warnings(diag)
        self._set_warning_text(warnings, severity)

        summary = (
            f"{diag.scene_name} | solver={diag.solver} | "
            f"colors={viz_label} | ρ err={diag.rho_error_mean * 100.0:.2f}% | "
            f"|v|max={diag.velocity_max:.3f} m/s | "
            f"{perf.wall_time_ms:.1f}ms ({perf.steps_per_second:.1f} fps, "
            f"avg {perf.rolling_wall_time_ms:.1f}ms)"
        )
        if warnings:
            summary += f" | alerts={severity.upper()}"

        self.metrics["summary"].setText(summary)
        self.particle_view.flash_stability(diag.stability)
        self.statusBar().showMessage(summary)

    # ------------------------------------------------------------------ visualization helpers
    def _on_viz_mode_changed(self, index: int) -> None:
        mode = self.viz_combo.itemData(index)
        if isinstance(mode, str):
            self._set_visualization_mode(mode, update_combo=False, refresh_view=True)

    def _set_visualization_mode(self, mode: str, update_combo: bool = True, refresh_view: bool = False) -> None:
        if mode not in ParticleView.COLOR_MODES:
            return
        self._viz_mode = mode
        if update_combo:
            idx = self.viz_combo.findData(mode)
            if idx >= 0 and idx != self.viz_combo.currentIndex():
                self.viz_combo.blockSignals(True)
                self.viz_combo.setCurrentIndex(idx)
                self.viz_combo.blockSignals(False)
        label = self._viz_mode_label(mode)
        if "viz_mode" in self.metrics:
            self.metrics["viz_mode"].setText(label)
        self.particle_view.set_color_mode(mode)
        if refresh_view and self.controller.state is not None:
            self.particle_view.render_state(self.controller.state)

    def _viz_mode_label(self, mode: str) -> str:
        return ParticleView.COLOR_MODES.get(mode, mode.title())

    # ------------------------------------------------------------------ diagnostics helpers
    def _reset_metrics(self) -> None:
        if not hasattr(self, "metrics"):
            return
        for key, label in self.metrics.items():
            if key in {"scene", "solver"}:
                continue
            label.setText("—")
        if "scene" in self.metrics:
            self.metrics["scene"].setText(self._current_scene_metadata().name)
        if "solver" in self.metrics:
            self.metrics["solver"].setText(self.controller.solver_name)
        self._apply_stability_style(None)
        self._set_warning_text([], "ok")
        self._set_visualization_mode(self._viz_mode, update_combo=True, refresh_view=False)

    def _apply_stability_style(self, status: Optional[str]) -> None:
        label = self.metrics.get("stability")
        if label is None:
            return
        if not status:
            label.setText("—")
            label.setStyleSheet("border-radius:6px; padding:6px; background:#2b2b2b; color:#dddddd;")
            return
        status_lower = status.lower()
        fg, bg = self._STABILITY_STYLES.get(status_lower, ("#d1d1d1", "#2b2b2b"))
        label.setText(status.upper())
        label.setStyleSheet(
            f"border-radius:6px; padding:6px; font-weight:600; "
            f"color:{fg}; background:{bg};"
        )

    def _set_warning_text(self, warnings: List[str], severity: str) -> None:
        label = self.metrics.get("warnings")
        if label is None:
            return
        severity = severity if severity in self._WARNING_STYLES else "warn"
        fg, bg = self._WARNING_STYLES[severity]
        if not warnings:
            label.setText("No alerts")
            fg, bg = self._WARNING_STYLES["ok"]
        else:
            label.setText(" • ".join(warnings))
        label.setStyleSheet(f"border-radius:4px; padding:4px; color:{fg}; background:{bg};")

    def _collect_warnings(self, diagnostics: StepDiagnostics) -> Tuple[List[str], str]:
        warnings: List[str] = []
        severity_level = 0

        if diagnostics.neighbor_mean > 0 and diagnostics.neighbor_min < 0.6 * diagnostics.neighbor_mean:
            warnings.append("Neighbor coverage low")
            severity_level = max(severity_level, 1)

        rho_err = diagnostics.rho_error_mean
        if rho_err > 0.04:
            warnings.append("Density drift")
            severity_level = max(severity_level, 2 if rho_err > 0.08 else 1)

        pressure_span = diagnostics.pressure_max - diagnostics.pressure_min
        pressure_peak = max(abs(diagnostics.pressure_min), abs(diagnostics.pressure_max), 1.0)
        if pressure_peak > 500.0 and pressure_span > pressure_peak * 1.5:
            warnings.append("Pressure swing")
            severity_level = max(severity_level, 1)

        severity = ["ok", "warn", "fail"][min(severity_level, 2)]
        return warnings, severity
