from __future__ import annotations

from sph.core.diagnostics.detector import InstabilityDetector
from sph.core.diagnostics.metrics import StepMetrics
from sph.core.diagnostics.sinks import ConsoleSink, CsvSink, DebugSnapshotSink
from sph.core.state import ParticleState
import numpy as np


class DiagnosticsManager:
    def __init__(
        self,
        *,
        console_sink: ConsoleSink,
        csv_sink: CsvSink,
        debug_snapshot_sink: DebugSnapshotSink,
        instability_detector: InstabilityDetector,
    ) -> None:
        self.console_sink = console_sink
        self.csv_sink = csv_sink
        self.debug_snapshot_sink = debug_snapshot_sink
        self.instability_detector = instability_detector

    def process(self, *, metrics: StepMetrics, state: ParticleState, neigh_counts: np.ndarray) -> None:
        detected = self.instability_detector.evaluate(metrics, state)
        if detected:
            metrics.flags.extend(detected)
        self.console_sink.publish(metrics, neigh_counts)
        self.csv_sink.publish(metrics)
        if detected:
            snap_path = self.debug_snapshot_sink.publish(step=metrics.step, state=state, neigh_counts=neigh_counts)
            if snap_path is not None:
                print(
                    "[RUNTIME][WARN] particle instability detected "
                    f"flags={','.join(detected)} snapshot={snap_path}"
                )

