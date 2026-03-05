from __future__ import annotations

from pathlib import Path

import numpy as np

from sph.core.diagnostics.metrics import StepMetrics
from sph.core.state import ParticleState


class ConsoleSink:
    def __init__(self, *, log_every: int, neigh_hist_every: int) -> None:
        self.log_every = int(max(1, log_every))
        self.neigh_hist_every = int(max(1, neigh_hist_every))

    def publish(self, metrics: StepMetrics, neigh_counts: np.ndarray) -> None:
        if metrics.step == 1 or (metrics.step % self.log_every) == 0:
            print(
                f"[STEP {metrics.step:04d}] dt={metrics.dt:.3e} "
                f"|v|max={metrics.v_max:.3e} "
                f"rho(min/avg/max)={metrics.rho_min:.2f}/{metrics.rho_avg:.2f}/{metrics.rho_max:.2f} "
                f"p(min/avg/max)={metrics.p_min:.2f}/{metrics.p_avg:.2f}/{metrics.p_max:.2f} "
                f"neigh(min/avg/max)={metrics.neigh_min}/{metrics.neigh_avg:.1f}/{metrics.neigh_max} "
                f"dt_reasons={','.join(metrics.dt_reason_codes) if metrics.dt_reason_codes else '-'} "
                f"flags={','.join(metrics.flags) if metrics.flags else '-'}"
            )
        if (metrics.step % self.neigh_hist_every) == 0:
            self._print_neighbor_histogram(neigh_counts)

    @staticmethod
    def _print_neighbor_histogram(neigh_counts: np.ndarray) -> None:
        if neigh_counts.size == 0:
            print("[NEIGH] no fluid particles")
            return
        bins = [(0, 10), (10, 20), (20, 40), (40, 80), (80, None)]
        print("[NEIGH] histogram")
        for lo, hi in bins:
            if hi is None:
                count = int(np.count_nonzero(neigh_counts >= lo))
                print(f"[NEIGH] {lo:>2}+ : {count}")
            else:
                count = int(np.count_nonzero((neigh_counts >= lo) & (neigh_counts < hi)))
                print(f"[NEIGH] {lo:>2}-{hi:<2}: {count}")
        low_pct = 100.0 * float(np.count_nonzero(neigh_counts < 5)) / float(neigh_counts.size)
        print(f"[NEIGH] low(<5)={low_pct:.2f}%")


class CsvSink:
    def __init__(self, output_path: Path, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.output_path = output_path
        self._initialized = False

    def publish(self, metrics: StepMetrics) -> None:
        if not self.enabled:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if self._initialized else "w"
        with self.output_path.open(mode, encoding="utf-8") as f:
            if not self._initialized:
                f.write(
                    "step,time,dt,v_max,rho_min,rho_avg,rho_max,p_min,p_avg,p_max,"
                    "neigh_min,neigh_avg,neigh_max,dt_reason_codes,flags\n"
                )
                self._initialized = True
            f.write(
                f"{metrics.step},{metrics.time:.9e},{metrics.dt:.9e},{metrics.v_max:.9e},"
                f"{metrics.rho_min:.9e},{metrics.rho_avg:.9e},{metrics.rho_max:.9e},"
                f"{metrics.p_min:.9e},{metrics.p_avg:.9e},{metrics.p_max:.9e},"
                f"{metrics.neigh_min},{metrics.neigh_avg:.9e},{metrics.neigh_max},"
                f"\"{'|'.join(metrics.dt_reason_codes)}\",\"{'|'.join(metrics.flags)}\"\n"
            )


class DebugSnapshotSink:
    def __init__(self, *, enabled: bool, output_dir: Path) -> None:
        self.enabled = bool(enabled)
        self.output_dir = output_dir

    def publish(self, *, step: int, state: ParticleState, neigh_counts: np.ndarray) -> Path | None:
        if not self.enabled:
            return None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / f"step_{int(step):04d}.csv"
        fluid_ids = state.fluid_indices
        with out_path.open("w", encoding="utf-8") as f:
            f.write("x,y,vx,vy,rho,p,neighbor_count\n")
            for k, i in enumerate(fluid_ids):
                f.write(
                    f"{state.pos[i,0]:.9e},{state.pos[i,1]:.9e},"
                    f"{state.vel[i,0]:.9e},{state.vel[i,1]:.9e},"
                    f"{state.rho[i]:.9e},{state.p[i]:.9e},{int(neigh_counts[k])}\n"
                )
        return out_path

