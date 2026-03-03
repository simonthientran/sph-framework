from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ACTIVE_X_LIMIT = 1e8


@dataclass(frozen=True)
class VxProfileConfig:
    enable: bool
    every: int
    bins: int
    axis: int
    component: int
    use_x_slice: bool
    x_mid: float | None
    x_slice_width: float
    y_min: float | None
    y_max: float | None
    y0: float | None
    y_extent_mode: str
    y_margin: float
    channel_height: float | None
    gx: float | None
    nu: float | None
    out_file: Path


@dataclass(frozen=True)
class VxProfileSample:
    step: int
    y_centers: list[float]
    counts: list[int]
    mean_vx: list[float]
    vmax_vx: list[float]
    std_vx: list[float]
    vx_analytic: list[float]
    empty_bins: int
    l2: float
    linf: float


def build_vx_profile_config(
    *,
    scene: dict,
    solver_cfg: dict,
    support_radius: float,
    domain_min: np.ndarray | None,
    domain_max: np.ndarray | None,
    scene_name: str,
) -> VxProfileConfig | None:
    """
    Build robust vx-profile diagnostics config from scene + solver settings.

    Backward compatibility:
    - Existing `solver.debug_vx_profile` keys still work.
    - Old `x_window` is mapped to `x_slice_width` and enables x-slice mode.
    """
    cfg = solver_cfg.get("debug_vx_profile", {})
    if not bool(cfg.get("enable", False)):
        return None

    bins = max(2, int(cfg.get("bins", 8)))
    every = max(1, int(cfg.get("every", 10)))
    axis = int(cfg.get("axis", 1))
    component = int(cfg.get("component", 0))

    legacy_x_window = cfg.get("x_window", None)
    has_legacy_window = legacy_x_window is not None
    use_x_slice = bool(cfg.get("use_x_slice", has_legacy_window))
    x_slice_width = float(cfg.get("x_slice_width", legacy_x_window if has_legacy_window else 2.0 * support_radius))
    x_mid_val = cfg.get("x_mid", None)
    x_mid = float(x_mid_val) if x_mid_val is not None else None

    y_min = float(cfg["y_min"]) if "y_min" in cfg else None
    y_max = float(cfg["y_max"]) if "y_max" in cfg else None
    y0 = float(cfg["y0"]) if "y0" in cfg else None
    y_extent_mode = str(cfg.get("y_extent_mode", "config")).lower()
    if y_extent_mode not in {"config", "slice_auto"}:
        y_extent_mode = "config"
    y_margin = float(cfg.get("y_margin", 0.0))
    channel_height = float(cfg["H"]) if "H" in cfg else None

    gx = float(cfg["gx"]) if "gx" in cfg else None
    nu = float(cfg["nu"]) if "nu" in cfg else None
    if gx is None:
        body_force = np.asarray(scene.get("forces", {}).get("body_force", [0.0, 0.0]), dtype=np.float64)
        gx = float(body_force[component]) if component < body_force.size else 0.0
    if nu is None:
        nu = float(scene.get("material", {}).get("viscosity", {}).get("nu", 0.0))

    out_file = Path(cfg.get("out_file", f"out/{scene_name}/vx_profile_bins.csv"))

    # If H is configured and y0 is not, use:
    # - explicit y_min when present
    # - otherwise domain_min on the sampling axis
    # - otherwise centered channel default y0=-H/2
    if y0 is None and channel_height is not None:
        if y_min is not None:
            y0 = float(y_min)
        else:
            # Requested default for centered channels: y0 = -H/2.
            y0 = float(-0.5 * channel_height)

    # Keep backward compatibility for explicit y-range config.
    if y_min is None and y0 is not None:
        y_min = float(y0)
    if y_max is None and channel_height is not None and y0 is not None:
        y_max = float(y0 + channel_height)

    # If x_mid is not explicit, use domain center.
    if x_mid is None and domain_min is not None and domain_max is not None and component < domain_min.size:
        x_mid = 0.5 * float(domain_min[component] + domain_max[component])

    return VxProfileConfig(
        enable=True,
        every=every,
        bins=bins,
        axis=axis,
        component=component,
        use_x_slice=use_x_slice,
        x_mid=x_mid,
        x_slice_width=float(max(x_slice_width, 0.0)),
        y_min=y_min,
        y_max=y_max,
        y0=y0,
        y_extent_mode=y_extent_mode,
        y_margin=float(max(y_margin, 0.0)),
        channel_height=channel_height,
        gx=gx,
        nu=nu,
        out_file=out_file,
    )


class VxProfileDiagnostics:
    """
    Robust vx(y) profile diagnostics for Poiseuille-like channel benchmarks.

    CSV output (one row per bin):
    step, bin_id, y_center, n, mean_vx, vmax_vx, std_vx, vx_analytic, empty
    """

    def __init__(self, cfg: VxProfileConfig):
        self.cfg = cfg
        self.cfg.out_file.parent.mkdir(parents=True, exist_ok=True)
        with self.cfg.out_file.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "step",
                    "bin_id",
                    "y_center",
                    "n",
                    "mean_vx",
                    "vmax_vx",
                    "std_vx",
                    "vx_analytic",
                    "empty",
                ]
            )

    def sample(self, *, step: int, state) -> VxProfileSample | None:
        if not self.cfg.enable:
            return None
        if self.cfg.every <= 0 or step % self.cfg.every != 0:
            return None

        fluid_ids = state.fluid_indices
        if fluid_ids.size == 0:
            return None

        pos = state.pos[fluid_ids]
        vel = state.vel[fluid_ids]
        active = pos[:, 0] < _ACTIVE_X_LIMIT
        pos = pos[active]
        vel = vel[active]
        if pos.size == 0:
            return None

        if self.cfg.use_x_slice and self.cfg.x_slice_width > 0.0 and self.cfg.x_mid is not None:
            half_w = 0.5 * float(self.cfg.x_slice_width)
            x = pos[:, self.cfg.component]
            m = np.abs(x - float(self.cfg.x_mid)) <= half_w
            pos = pos[m]
            vel = vel[m]
            if pos.size == 0:
                return None

        y_world = pos[:, self.cfg.axis]
        y_world_min = float(np.min(y_world))
        y_world_max = float(np.max(y_world))

        if self.cfg.y_extent_mode == "slice_auto":
            y0_eff = float(y_world_min)
            h_eff = float(y_world_max - y_world_min)
            if not np.isfinite(h_eff) or h_eff <= 0.0:
                return None
            y_mapped = y_world - y0_eff
            y_map_min = 0.0
            y_map_max = h_eff
            print(
                f"[VXPROF_AUTO] step={step} y0_eff={y0_eff:.6g} H_eff={h_eff:.6g} "
                f"y_world_range=[{y_world_min:.6g},{y_world_max:.6g}]"
            )
        else:
            # Mapped coordinate y' = y - y0 for Poiseuille profile in strict config mode.
            y0_eff = self._resolve_y0(y_world)
            y_mapped = y_world - y0_eff
            y_map_min, y_map_max = self._resolve_y_mapped_range(y_mapped)
            if not np.isfinite(y_map_min) or not np.isfinite(y_map_max) or y_map_max <= y_map_min:
                return None
            in_channel = (y_mapped >= y_map_min) & (y_mapped <= y_map_max)
            y_mapped = y_mapped[in_channel]
            vel = vel[in_channel]
            if y_mapped.size == 0:
                return None
            print(
                f"[VXPROF_INFO] step={step} y0={y0_eff:.6g} "
                f"y_world_range=[{y_world_min:.6g},{y_world_max:.6g}] "
                f"y_mapped_range=[{float(np.min(y_mapped)):.6g},{float(np.max(y_mapped)):.6g}]"
            )

        def _compute_profile(bin_edges: np.ndarray) -> tuple[np.ndarray, list[int], list[float], list[float], list[float], int]:
            ctr = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            cts: list[int] = []
            mns: list[float] = []
            vms: list[float] = []
            sds: list[float] = []
            empties = 0
            for b in range(self.cfg.bins):
                if b + 1 == self.cfg.bins:
                    m = (y_mapped >= bin_edges[b]) & (y_mapped <= bin_edges[b + 1])
                else:
                    m = (y_mapped >= bin_edges[b]) & (y_mapped < bin_edges[b + 1])
                n = int(np.count_nonzero(m))
                if n > 0:
                    vx = vel[m, self.cfg.component]
                    mean_vx = float(np.mean(vx))
                    vmax_vx = float(np.max(np.abs(vx)))
                    std_vx = float(np.std(vx))
                else:
                    mean_vx = 0.0
                    vmax_vx = 0.0
                    std_vx = 0.0
                    empties += 1
                cts.append(n)
                mns.append(mean_vx)
                vms.append(vmax_vx)
                sds.append(std_vx)
            return ctr, cts, mns, vms, sds, empties

        edges = np.linspace(y_map_min, y_map_max, self.cfg.bins + 1, dtype=np.float64)
        centers, counts, means, vmaxs, stds, empty_bins = _compute_profile(edges)

        # Robust mode fallback: avoid empty bins in diagnostics by using quantile edges
        # when possible (strict benchmark mode keeps fixed config extents/bins).
        if self.cfg.y_extent_mode == "slice_auto" and empty_bins > 0 and y_mapped.size >= self.cfg.bins:
            q = np.linspace(0.0, 1.0, self.cfg.bins + 1, dtype=np.float64)
            q_edges = np.quantile(y_mapped, q)
            # Ensure strictly increasing edges for stable bin membership.
            eps = 1e-12
            for i in range(1, q_edges.size):
                if q_edges[i] <= q_edges[i - 1]:
                    q_edges[i] = q_edges[i - 1] + eps
            centers_q, counts_q, means_q, vmaxs_q, stds_q, empty_bins_q = _compute_profile(q_edges)
            if empty_bins_q <= empty_bins:
                edges = q_edges
                centers = centers_q
                counts = counts_q
                means = means_q
                vmaxs = vmaxs_q
                stds = stds_q
                empty_bins = empty_bins_q
                print(f"[VXPROF_AUTO] step={step} quantile_fallback=true empty_bins={empty_bins}")

        analy: list[float] = []
        with self.cfg.out_file.open("a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            for b in range(self.cfg.bins):
                vx_analytic = self._analytic_vx(y_mapped=float(centers[b]), h=float(y_map_max - y_map_min))
                analy.append(vx_analytic)
                w.writerow(
                    [
                        int(step),
                        int(b),
                        float(centers[b]),
                        int(counts[b]),
                        float(means[b]),
                        float(vmaxs[b]),
                        float(stds[b]),
                        float(vx_analytic),
                        int(1 if counts[b] == 0 else 0),
                    ]
                )

        l2, linf = self._errors(means=means, analy=analy, counts=counts)
        used_bins = int(sum(1 for n in counts if n > 0))
        vmax_analytic = self._vmax_analytic(h=float(y_map_max - y_map_min))
        vmax_measured = float(np.max(np.abs(np.asarray(means, dtype=np.float64)))) if means else 0.0
        print(
            f"[VXANA] step={step} mode={self.cfg.y_extent_mode} "
            f"vmax_analytic={vmax_analytic:.3e} vmax_measured={vmax_measured:.3e}"
        )
        if vmax_measured > 1e-12 and vmax_analytic / vmax_measured > 10.0:
            ratio = vmax_analytic / vmax_measured
            print(
                f"[VXANA][WARN] step={step} analytic/measured vmax ratio={ratio:.2f}. "
                f"Consider lowering gx or increasing nu."
            )
        print(
            f"[VXERR] step={step} L2={l2:.3e} Linf={linf:.3e} "
            f"empty_bins={empty_bins} used_bins={used_bins}/{self.cfg.bins}"
        )

        return VxProfileSample(
            step=int(step),
            y_centers=[float(v) for v in centers.tolist()],
            counts=counts,
            mean_vx=means,
            vmax_vx=vmaxs,
            std_vx=stds,
            vx_analytic=analy,
            empty_bins=int(empty_bins),
            l2=float(l2),
            linf=float(linf),
        )

    def _resolve_y0(self, y_world_values: np.ndarray) -> float:
        if self.cfg.y0 is not None:
            return float(self.cfg.y0)
        if self.cfg.y_min is not None:
            return float(self.cfg.y_min)
        if self.cfg.channel_height is not None:
            return float(-0.5 * self.cfg.channel_height)
        return float(np.min(y_world_values))

    def _resolve_y_mapped_range(self, y_mapped_values: np.ndarray) -> tuple[float, float]:
        if self.cfg.channel_height is not None and self.cfg.channel_height > 0.0:
            return 0.0, float(self.cfg.channel_height)

        y_min = float(np.min(y_mapped_values))
        y_max = float(np.max(y_mapped_values))
        if y_max <= y_min:
            return y_min, y_max
        span = y_max - y_min
        margin = float(max(self.cfg.y_margin, 0.02 * span))
        return y_min - margin, y_max + margin

    def _analytic_vx(self, *, y_mapped: float, h: float) -> float:
        gx = float(self.cfg.gx if self.cfg.gx is not None else 0.0)
        nu = float(self.cfg.nu if self.cfg.nu is not None else 0.0)
        if nu <= 0.0:
            return 0.0
        if h <= 0.0:
            return 0.0
        yy = float(y_mapped)
        return float((gx / (2.0 * nu)) * yy * (h - yy))

    def _vmax_analytic(self, *, h: float) -> float:
        gx = float(self.cfg.gx if self.cfg.gx is not None else 0.0)
        nu = float(self.cfg.nu if self.cfg.nu is not None else 0.0)
        if nu <= 0.0 or h <= 0.0:
            return 0.0
        return float((gx * h * h) / (8.0 * nu))

    @staticmethod
    def _errors(*, means: list[float], analy: list[float], counts: list[int]) -> tuple[float, float]:
        valid_idx = [i for i, n in enumerate(counts) if n > 0]
        if not valid_idx:
            return float("nan"), float("nan")

        diff = np.asarray([means[i] - analy[i] for i in valid_idx], dtype=np.float64)
        l2 = float(math.sqrt(float(np.mean(diff * diff))))
        linf = float(np.max(np.abs(diff)))
        return l2, linf

