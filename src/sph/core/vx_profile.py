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
    """Result of vx profile computation: bin stats and optional analytic vmax."""

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
    vmax_analytic: float | None


def compute_vx_profile(
    step: int,
    state: ParticleState,
    scene: dict,
    *,
    y_extent_mode: str = "walls",
    n_bins: int = 8,
    gx: float | None = None,
    nu: float | None = None,
) -> VxProfileResult:
    """
    Bin fluid particles by y in [y0_eff, y0_eff + H_eff] and compute mean vx per bin.

    Uses scene domain and (for walls_inner) boundary_layers and fluid.spacing.
    Optionally computes analytic vmax for plane Poiseuille if gx and nu are provided;
    if nu is None, uses scene material.viscosity.nu when enable is true.
    """
    y0_eff, H_eff = get_y_extent(scene, y_extent_mode)
    y_world_range = (y0_eff, y0_eff + H_eff)

    fluid_mask = ~state.is_boundary
    fluid_ids = np.where(fluid_mask)[0]
    if fluid_ids.size == 0:
        bin_centers = np.linspace(y0_eff + H_eff * 0.5 / max(1, n_bins), y0_eff + H_eff * (1.0 - 0.5 / max(1, n_bins)), n_bins)
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
            vmax_analytic=None,
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
    if gx is not None and nu is not None and nu > 0.0:
        vmax_analytic = float(gx * (H_eff**2) / (8.0 * nu))

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
        vmax_analytic=vmax_analytic,
    )


def format_vx_profile_log_line(result: VxProfileResult) -> str:
    """Single log line: mode, y0_eff, H_eff, y_world_range, used_bins, empty_bins, optional vmax_analytic."""
    parts = [
        f"vx_profile mode={result.mode} y0_eff={result.y0_eff:.6f} H_eff={result.H_eff:.6f}",
        f"y_world_range=({result.y_world_range[0]:.6f},{result.y_world_range[1]:.6f})",
        f"used_bins={result.used_bins}/{result.n_bins} empty_bins={result.empty_bins}",
    ]
    if result.vmax_analytic is not None:
        parts.append(f"vmax_analytic={result.vmax_analytic:.6e}")
    return " ".join(parts)


def export_vx_profile_csv(path: str | Path, result: VxProfileResult) -> None:
    """
    Write vx profile to CSV: step, mode, y0_eff, H_eff, then per-bin: bin_idx, y_center, vx_mean, count.

    Stable format for downstream analysis.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "step,mode,y0_eff,H_eff,bin_idx,y_center,vx_mean,count\n"
    rows = []
    for b in range(result.n_bins):
        vx_mean = result.vx_mean_per_bin[b]
        vx_str = f"{vx_mean:.17g}" if np.isfinite(vx_mean) else ""
        rows.append(
            f"{result.step},{result.mode},{result.y0_eff:.17g},{result.H_eff:.17g},"
            f"{b},{result.bin_centers[b]:.17g},{vx_str},{result.count_per_bin[b]}"
        )
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(header)
        f.write("\n".join(rows))
        if rows:
            f.write("\n")
