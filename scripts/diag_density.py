"""
Step 1 diagnostic: compare CPU vs CUDA density assembly term by term.

Run with CUDASIM:
    NUMBA_ENABLE_CUDASIM=1 python scripts/diag_density.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import os
os.environ.setdefault("NUMBA_ENABLE_CUDASIM", "1")

from numba import cuda

from sph.simulator_new import Simulator
from sph.cuda_pair_ops import (
    clip_upper_1d_kernel,
    combine_density_kernel,
    density_fb_kernel,
    density_ff_kernel,
    fill_1d_kernel,
    launch_config,
    pair_geometry_fb_kernel,
    pair_geometry_ff_kernel,
)


def cuda_density_from_snapshot(snap: dict, boundary_positions_d, periodic_length: float, use_periodic: int):
    """Replicate _run_density_kernels exactly on the snapshot data."""
    pairs = snap["pairs"]
    n = int(np.asarray(snap["densities"]).size)
    mass = float(snap["mass"])
    h = float(snap["h"])
    rho0 = float(snap["rho0"])

    positions = np.ascontiguousarray(np.asarray(snap["positions"]))
    d_positions = cuda.to_device(positions)

    ff_i = np.ascontiguousarray(pairs.ff_i)
    ff_j = np.ascontiguousarray(pairs.ff_j)
    fb_i = np.ascontiguousarray(pairs.fb_i)
    fb_j = np.ascontiguousarray(pairs.fb_j)

    d_ff_i = cuda.to_device(ff_i)
    d_ff_j = cuda.to_device(ff_j)
    d_fb_i = cuda.to_device(fb_i)
    d_fb_j = cuda.to_device(fb_j)

    ff_count = int(ff_i.size)
    fb_count = int(fb_i.size)

    d_ff_dx   = cuda.device_array(ff_count, dtype=np.float64)
    d_ff_dy   = cuda.device_array(ff_count, dtype=np.float64)
    d_ff_dist = cuda.device_array(ff_count, dtype=np.float64)
    d_fb_dx   = cuda.device_array(fb_count, dtype=np.float64)
    d_fb_dy   = cuda.device_array(fb_count, dtype=np.float64)
    d_fb_dist = cuda.device_array(fb_count, dtype=np.float64)

    d_rho_self = cuda.device_array(n, dtype=np.float64)
    d_rho_ff   = cuda.device_array(n, dtype=np.float64)
    d_rho_fb   = cuda.device_array(n, dtype=np.float64)
    d_density  = cuda.device_array(n, dtype=np.float64)

    cfg_p = launch_config(n)

    # Pair geometry
    if ff_count > 0:
        cfg_ff = launch_config(ff_count)
        pair_geometry_ff_kernel[cfg_ff](
            d_positions, d_ff_i, d_ff_j,
            periodic_length, use_periodic,
            d_ff_dx, d_ff_dy, d_ff_dist,
        )
    if fb_count > 0:
        cfg_fb = launch_config(fb_count)
        pair_geometry_fb_kernel[cfg_fb](
            d_positions, boundary_positions_d,
            d_fb_i, d_fb_j,
            periodic_length, use_periodic,
            d_fb_dx, d_fb_dy, d_fb_dist,
        )

    # Density
    w_self = mass * (10.0 / (7.0 * np.pi * h * h))
    fill_1d_kernel[cfg_p](d_rho_self, w_self)
    fill_1d_kernel[cfg_p](d_rho_ff, 0.0)
    fill_1d_kernel[cfg_p](d_rho_fb, 0.0)

    boundary_psi_host = np.asarray(snap["boundary_psi"])
    d_boundary_psi = cuda.to_device(np.ascontiguousarray(boundary_psi_host))

    if ff_count > 0:
        density_ff_kernel[cfg_ff](d_ff_i, d_ff_j, d_ff_dist, mass, h, d_rho_ff)
    if fb_count > 0:
        density_fb_kernel[cfg_fb](d_fb_i, d_fb_j, d_fb_dist, d_boundary_psi, h, d_rho_fb)

    combine_density_kernel[cfg_p](d_rho_self, d_rho_ff, d_rho_fb, d_density)
    clip_upper_1d_kernel[cfg_p](d_density, float(snap["rho0"]) * 1.02)

    cuda.synchronize()

    return (
        d_rho_self.copy_to_host(),
        d_rho_ff.copy_to_host(),
        d_rho_fb.copy_to_host(),
        d_density.copy_to_host(),
        d_ff_dist.copy_to_host() if ff_count > 0 else np.zeros(0),
    )


def compare_density_cpu_vs_cuda(snap: dict, boundary_positions, periodic_length: float, use_periodic: int):
    """
    Term-by-term comparison of CPU snapshot vs CUDA recomputed density.
    """
    d_bpos = cuda.to_device(np.ascontiguousarray(boundary_positions)) if boundary_positions.size else None

    rho_self_gpu, rho_ff_gpu, rho_fb_gpu, density_gpu, ff_dist_gpu = cuda_density_from_snapshot(
        snap, d_bpos, periodic_length, use_periodic
    )

    rho_self_ref = np.asarray(snap["rho_self"])
    rho_ff_ref   = np.asarray(snap["rho_ff"])
    rho_fb_ref   = np.asarray(snap["rho_fb"])
    density_ref  = np.asarray(snap["densities"])

    raw_cpu = rho_self_ref + rho_ff_ref + rho_fb_ref
    raw_gpu = rho_self_gpu + rho_ff_gpu + rho_fb_gpu

    self_diff    = np.abs(rho_self_gpu - rho_self_ref)
    ff_diff      = np.abs(rho_ff_gpu   - rho_ff_ref)
    fb_diff      = np.abs(rho_fb_gpu   - rho_fb_ref)
    density_diff = np.abs(density_gpu  - density_ref)
    raw_diff     = np.abs(raw_gpu - raw_cpu)

    worst = int(np.argmax(density_diff))

    print(f"\n{'='*70}")
    print(f"Stage: {snap.get('stage', '?')}")
    print(f"{'='*70}")
    print(f"\nTerm-by-term linf:")
    print(f"  rho_self : {self_diff.max():.6e}  (l1_mean={self_diff.mean():.6e})")
    print(f"  rho_ff   : {ff_diff.max():.6e}  (l1_mean={ff_diff.mean():.6e})")
    print(f"  rho_fb   : {fb_diff.max():.6e}  (l1_mean={fb_diff.mean():.6e})")
    print(f"  raw_sum  : {raw_diff.max():.6e}  (l1_mean={raw_diff.mean():.6e})")
    print(f"  density  : {density_diff.max():.6e}  (l1_mean={density_diff.mean():.6e})")
    print(f"  (density = snap[densities] which may include cap/diffusion/stabilization)")

    print(f"\nWorst particle (idx={worst}):")
    print(f"  rho_self cpu={rho_self_ref[worst]:.6f}  gpu={rho_self_gpu[worst]:.6f}  diff={self_diff[worst]:.6e}")
    print(f"  rho_ff   cpu={rho_ff_ref[worst]:.6f}  gpu={rho_ff_gpu[worst]:.6f}  diff={ff_diff[worst]:.6e}")
    print(f"  rho_fb   cpu={rho_fb_ref[worst]:.6f}  gpu={rho_fb_gpu[worst]:.6f}  diff={fb_diff[worst]:.6e}")
    print(f"  raw_cpu  = {raw_cpu[worst]:.6f}")
    print(f"  raw_gpu  = {raw_gpu[worst]:.6f}")
    print(f"  density_ref (with cap/diffusion) = {density_ref[worst]:.6f}")
    print(f"  density_gpu (raw CUDA + upper clip) = {density_gpu[worst]:.6f}")

    # Check if density_ref == clip(raw_cpu, 0, rho0*1.02)
    rho0 = float(snap["rho0"])
    clipped_raw = np.clip(raw_cpu, 0.0, rho0 * 1.02)
    cap_diff = np.abs(density_ref - clipped_raw)
    print(f"\nCheck: density_ref vs clip(raw_cpu, 0, rho0*1.02):")
    print(f"  linf={cap_diff.max():.6e}  l1_mean={cap_diff.mean():.6e}")
    print(f"  -> This nonzero part = diffusion + stabilization added by CPU")

    # Compare ff_dist CPU vs GPU
    ff_dist_cpu = np.asarray(snap["pairs"].ff_dist)
    if ff_dist_gpu.size > 0 and ff_dist_cpu.size > 0:
        dist_diff = np.abs(ff_dist_gpu - ff_dist_cpu)
        print(f"\nPair geometry (ff_dist):")
        print(f"  linf={dist_diff.max():.6e}  l1_mean={dist_diff.mean():.6e}")
        print(f"  cpu range: [{ff_dist_cpu.min():.4f}, {ff_dist_cpu.max():.4f}]")
        print(f"  gpu range: [{ff_dist_gpu.min():.4f}, {ff_dist_gpu.max():.4f}]")

    print(f"\nDensity ranges:")
    print(f"  rho_self: cpu [{rho_self_ref.min():.4f}, {rho_self_ref.max():.4f}]  gpu [{rho_self_gpu.min():.4f}, {rho_self_gpu.max():.4f}]")
    print(f"  rho_ff:   cpu [{rho_ff_ref.min():.4f}, {rho_ff_ref.max():.4f}]  gpu [{rho_ff_gpu.min():.4f}, {rho_ff_gpu.max():.4f}]")
    print(f"  rho_fb:   cpu [{rho_fb_ref.min():.4f}, {rho_fb_ref.max():.4f}]  gpu [{rho_fb_gpu.min():.4f}, {rho_fb_gpu.max():.4f}]")
    print(f"  density_ref: [{density_ref.min():.4f}, {density_ref.max():.4f}]")
    print(f"  density_gpu: [{density_gpu.min():.4f}, {density_gpu.max():.4f}]")


def main():
    scene_path = ROOT / "scenes/examples/pipe_flow_2d.json"
    print(f"Loading scene: {scene_path}")
    sim = Simulator(scene_path)

    # Run 1 step to get snapshots
    sim.step()

    time_step = sim.time_step
    snaps = getattr(time_step, "last_cuda_stage_snapshots", {})
    if not snaps:
        print("ERROR: No CUDA snapshots found on time_step")
        return

    boundary = sim.boundary
    boundary_positions = boundary.positions if boundary is not None and boundary.n else np.zeros((0, 2))

    periodic_length = 0.0
    use_periodic = 0
    if sim.x_min is not None and sim.x_max is not None:
        periodic_length = float(sim.x_max - sim.x_min)
        use_periodic = 1

    for stage_name in ("cd_input", "df_input"):
        if stage_name in snaps:
            compare_density_cpu_vs_cuda(snaps[stage_name], boundary_positions, periodic_length, use_periodic)


if __name__ == "__main__":
    main()
