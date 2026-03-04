"""
Vx-profile debug: bin fluid particles by y and report mean vx per bin.

Supports y_extent_mode:
- "walls": use full domain height (y0 = domain min y, H = domain height).
- "walls_inner": use fluid interior implied by boundary layers:
  t = boundary_layers * fluid_spacing, y0_eff = y0_wall + t, H_eff = H_wall - 2*t.

Analytic vmax for plane Poiseuille (body-force gx, kinematic viscosity nu):
  vmax = gx * (H_eff**2) / (8*nu)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sph.core.state import ParticleState


def get_y_extent(scene: dict, mode: str) -> tuple[float, float]:
    """
    Compute effective y origin and height for vx profiling from scene.

    Args:
        scene: Scene config with domain.min/max and (for walls_inner)
               domain.boundary_layers and fluid.spacing.
        mode: "walls" | "walls_inner".

    Returns:
        (y0_eff, H_eff) in world coordinates.

    Raises:
        ValueError: if mode is unknown or (walls_inner) H_eff <= 0.
    """
    domain = scene.get("domain", {})
    domain_min = np.array(domain.get("min", [0.0, 0.0]), dtype=np.float64)
    domain_max = np.array(domain.get("max", [1.0, 1.0]), dtype=np.float64)
    y0_wall = float(domain_min[1])
    H_wall = float(domain_max[1] - domain_min[1])

    if mode == "walls":
        return (y0_wall, H_wall)

    if mode == "walls_inner":
        boundary_layers = int(domain.get("boundary_layers", 0))
        spacing = float(scene.get("fluid", {}).get("spacing", 0.02))
        t = boundary_layers * spacing
        y0_eff = y0_wall + t
        H_eff = H_wall - 2.0 * t
        if H_eff <= 0.0:
            raise ValueError(
                f"vx_profile walls_inner: H_eff must be > 0; got H_eff={H_eff:.6f} "
                f"(H_wall={H_wall:.6f}, boundary_layers={boundary_layers}, spacing={spacing:.6f}, t={t:.6f})"
            )
        return (y0_eff, H_eff)

    raise ValueError(f"vx_profile: unknown y_extent_mode={mode!r}; use 'walls' or 'walls_inner'")


@dataclass(frozen=True)
class VxProfileResult:
    """Result of vx profile computation with measured + analytic diagnostics."""

    step: int
    mode: str
    y0_eff: float
    H_eff: float
    y_world_range: tuple[float, float]
    n_bins: int
    used_bins: int
    empty_bins: int
    bin_centers: np.ndarray
    vx_mean_per_bin: np.ndarray
    count_per_bin: np.ndarray
    analytic_vx_per_bin: np.ndarray
    vmax_measured: float
    vmax_analytic: float | None
    l2_error: float | None
    linf_error: float | None
    gx: float | None
    nu: float | None
    use_x_slice: bool
    x_slice_width: float | None


def compute_vx_profile(
    step: int,
    state: ParticleState,
    scene: dict,
    *,
    y_extent_mode: str = "walls",
    n_bins: int = 8,
    gx: float | None = None,
    nu: float | None = None,
    use_x_slice: bool = False,
    x_slice_width: float | None = None,
) -> VxProfileResult:
    """
    Bin fluid particles by y in [y0_eff, y0_eff + H_eff] and compute mean vx per bin.

    Uses scene domain and (for walls_inner) boundary_layers and fluid.spacing.
    Optionally computes analytic vmax for plane Poiseuille if gx and nu are provided;
    if nu is None, uses scene material.viscosity.nu when enable is true.
    """
    if n_bins <= 0:
        raise ValueError(f"vx_profile: n_bins must be > 0, got {n_bins}")

    y0_eff, H_eff = get_y_extent(scene, y_extent_mode)
    y_world_range = (y0_eff, y0_eff + H_eff)

    fluid_mask = ~state.is_boundary
    fluid_ids = np.where(fluid_mask)[0]
    if use_x_slice:
        if x_slice_width is None:
            raise ValueError("vx_profile: use_x_slice=True requires x_slice_width")
        x_slice_width = float(x_slice_width)
        if x_slice_width <= 0.0:
            raise ValueError(f"vx_profile: x_slice_width must be > 0, got {x_slice_width}")
        fluid_cfg = scene.get("fluid", {})
        x_slice_center = 0.5 * (float(fluid_cfg.get("min", [0.0, 0.0])[0]) + float(fluid_cfg.get("max", [0.0, 0.0])[0]))
        x_fluid = state.pos[fluid_ids, 0]
        keep = np.abs(x_fluid - x_slice_center) <= (0.5 * x_slice_width)
        fluid_ids = fluid_ids[keep]

    if fluid_ids.size == 0:
        bin_centers = np.linspace(y0_eff + H_eff * 0.5 / max(1, n_bins), y0_eff + H_eff * (1.0 - 0.5 / max(1, n_bins)), n_bins)
        analytic_vx_per_bin = np.full(n_bins, np.nan, dtype=np.float64)
        if gx is not None and nu is not None and nu > 0.0:
            y_prime = bin_centers - y0_eff
            analytic_vx_per_bin = (float(gx) / (2.0 * float(nu))) * y_prime * (H_eff - y_prime)
        return VxProfileResult(
            step=step,
            mode=y_extent_mode,
            y0_eff=y0_eff,
            H_eff=H_eff,
            y_world_range=y_world_range,
            n_bins=n_bins,
            used_bins=0,
            empty_bins=n_bins,
            bin_centers=bin_centers,
            vx_mean_per_bin=np.full(n_bins, np.nan, dtype=np.float64),
            count_per_bin=np.zeros(n_bins, dtype=np.int64),
            analytic_vx_per_bin=analytic_vx_per_bin,
            vmax_measured=float("nan"),
            vmax_analytic=None,
            l2_error=None,
            linf_error=None,
            gx=None if gx is None else float(gx),
            nu=None if nu is None else float(nu),
            use_x_slice=bool(use_x_slice),
            x_slice_width=x_slice_width,
        )

    y_fluid = state.pos[fluid_ids, 1]
    vx_fluid = state.vel[fluid_ids, 0]

    # Bin edges: [y0_eff, ..., y0_eff + H_eff]; bins are [edges[i], edges[i+1]) for i=0..n_bins-2, last bin [edges[-2], edges[-1]]
    edges = np.linspace(y0_eff, y0_eff + H_eff, n_bins + 1, dtype=np.float64)
    # digitize(y, edges) returns i in 1..n_bins with edges[i-1] < y <= edges[i]; map to 0..n_bins-1
    bin_idx = np.digitize(y_fluid, edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    count_per_bin = np.zeros(n_bins, dtype=np.int64)
    vx_sum_per_bin = np.zeros(n_bins, dtype=np.float64)
    for b in range(n_bins):
        mask = bin_idx == b
        count_per_bin[b] = int(np.count_nonzero(mask))
        if count_per_bin[b] > 0:
            vx_sum_per_bin[b] = np.sum(vx_fluid[mask])

    vx_mean_per_bin = np.full(n_bins, np.nan, dtype=np.float64)
    np.true_divide(vx_sum_per_bin, count_per_bin, out=vx_mean_per_bin, where=count_per_bin > 0)
    bin_centers = (edges[:-1] + edges[1:]) * 0.5
    used_bins = int(np.count_nonzero(count_per_bin > 0))
    empty_bins = n_bins - used_bins

    # Analytic vmax: vmax = gx * H_eff^2 / (8*nu)
    vmax_analytic = None
    if nu is None:
        visc = scene.get("material", {}).get("viscosity", {})
        if visc.get("enable", False):
            nu = float(visc.get("nu", 0.0))
    if gx is not None:
        gx = float(gx)
    if nu is not None:
        nu = float(nu)
    if gx is not None and nu is not None and nu > 0.0:
        vmax_analytic = float(gx * (H_eff**2) / (8.0 * nu))

    analytic_vx_per_bin = np.full(n_bins, np.nan, dtype=np.float64)
    if gx is not None and nu is not None and nu > 0.0:
        y_prime = bin_centers - y0_eff
        analytic_vx_per_bin = (gx / (2.0 * nu)) * y_prime * (H_eff - y_prime)

    finite_measured = np.isfinite(vx_mean_per_bin)
    vmax_measured = float(np.max(vx_mean_per_bin[finite_measured])) if np.any(finite_measured) else float("nan")

    valid_for_error = finite_measured & np.isfinite(analytic_vx_per_bin)
    if np.any(valid_for_error):
        err = vx_mean_per_bin[valid_for_error] - analytic_vx_per_bin[valid_for_error]
        l2_error = float(np.sqrt(np.mean(err * err)))
        linf_error = float(np.max(np.abs(err)))
    else:
        l2_error = None
        linf_error = None

    return VxProfileResult(
        step=step,
        mode=y_extent_mode,
        y0_eff=y0_eff,
        H_eff=H_eff,
        y_world_range=y_world_range,
        n_bins=n_bins,
        used_bins=used_bins,
        empty_bins=empty_bins,
        bin_centers=bin_centers,
        vx_mean_per_bin=vx_mean_per_bin,
        count_per_bin=count_per_bin,
        analytic_vx_per_bin=analytic_vx_per_bin,
        vmax_measured=vmax_measured,
        vmax_analytic=vmax_analytic,
        l2_error=l2_error,
        linf_error=linf_error,
        gx=gx,
        nu=nu,
        use_x_slice=bool(use_x_slice),
        x_slice_width=x_slice_width,
    )


def format_vx_profile_log_line(result: VxProfileResult) -> str:
    """Single log line with extent, bins, and analytic-comparison metrics."""
    parts = [
        f"vx_profile mode={result.mode} y0_eff={result.y0_eff:.6f} H_eff={result.H_eff:.6f}",
        f"y_world_range=({result.y_world_range[0]:.6f},{result.y_world_range[1]:.6f})",
        f"used_bins={result.used_bins}/{result.n_bins} empty_bins={result.empty_bins} vmax_measured={result.vmax_measured:.6e}",
    ]
    if result.vmax_analytic is not None:
        parts.append(f"vmax_analytic={result.vmax_analytic:.6e}")
    if result.l2_error is not None and result.linf_error is not None:
        parts.append(f"L2={result.l2_error:.6e} Linf={result.linf_error:.6e}")
    return " ".join(parts)


def _format_meta_value(value: object) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "nan"
        return f"{float(value):.17g}"
    return str(value)


def export_vx_profile_csv(
    path: str | Path,
    result: VxProfileResult,
    *,
    scene_name: str | None = None,
    sim_time: float | None = None,
    timestamp: str | None = None,
) -> None:
    """
    Write vx profile with metadata header and per-bin table.
    Header lines are comments as '# key=value' and are followed by:
      y_center,vx_mean,vx_count,vx_analytic
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = timestamp or datetime.now(timezone.utc).isoformat()
    metadata = {
        "scene_name": scene_name or "unknown",
        "step": result.step,
        "timestamp": ts,
        "sim_time": sim_time,
        "bins": result.n_bins,
        "y_extent_mode": result.mode,
        "use_x_slice": result.use_x_slice,
        "x_slice_width": result.x_slice_width,
        "y0_eff": result.y0_eff,
        "H_eff": result.H_eff,
        "gx": result.gx,
        "nu": result.nu,
        "vmax_analytic": result.vmax_analytic,
        "vmax_measured": result.vmax_measured,
        "L2": result.l2_error,
        "Linf": result.linf_error,
        "used_bins": result.used_bins,
        "empty_bins": result.empty_bins,
    }
    required_keys = ("y0_eff", "H_eff", "gx", "nu", "bins")
    missing = [k for k in required_keys if metadata.get(k) is None]
    if missing:
        raise ValueError(f"vx_profile CSV metadata missing required keys: {missing}")

    header_lines = [f"# {k}={_format_meta_value(v)}" for k, v in metadata.items()]
    header = "y_center,vx_mean,vx_count,vx_analytic\n"
    rows = []
    for b in range(result.n_bins):
        vx_mean = result.vx_mean_per_bin[b]
        vx_analytic = result.analytic_vx_per_bin[b]
        vx_str = f"{vx_mean:.17g}" if np.isfinite(vx_mean) else ""
        vx_ana_str = f"{vx_analytic:.17g}" if np.isfinite(vx_analytic) else ""
        rows.append(
            f"{result.bin_centers[b]:.17g},{vx_str},{result.count_per_bin[b]},{vx_ana_str}"
        )
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(header_lines))
        f.write("\n")
        f.write(header)
        f.write("\n".join(rows))
        if rows:
            f.write("\n")
