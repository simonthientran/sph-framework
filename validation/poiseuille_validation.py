#!/usr/bin/env python3
"""
Poiseuille Flow Validation

Verify that the simulation reproduces the analytical parabolic velocity profile.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from sph.simulator_new import Simulator


def extract_velocity_profile(fluid, n_bins=20):
    """
    Extract velocity profile vx(y) by binning particles.

    Returns:
        bin_centers: y-coordinates of bin centers
        vx_mean: mean vx velocity per bin
        vx_std: standard deviation per bin
        populated: boolean mask of bins that contain at least one particle
    """
    y_min = fluid.positions[:, 1].min()
    y_max = fluid.positions[:, 1].max()
    # Extend upper bound by tiny epsilon so the top-most particle (at y_max)
    # is included in the last bin (avoids off-by-one empty bins)
    bins = np.linspace(y_min, y_max + 1e-10, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    vx_mean = np.zeros(n_bins)
    vx_std = np.zeros(n_bins)
    populated = np.zeros(n_bins, dtype=bool)

    for k in range(n_bins):
        mask = (fluid.positions[:, 1] >= bins[k]) & \
               (fluid.positions[:, 1] < bins[k+1])
        if mask.sum() > 0:
            vx_mean[k] = fluid.velocities[mask, 0].mean()
            vx_std[k] = fluid.velocities[mask, 0].std()
            populated[k] = True

    return bin_centers, vx_mean, vx_std, populated


def analytical_poiseuille(y, L, f, nu):
    """
    Analytical Poiseuille flow velocity profile.

    vx(y) = f / (2*nu) * (L^2 - y^2)

    Args:
        y: y-coordinate (distance from centerline)
        L: channel half-width
        f: body force magnitude
        nu: kinematic viscosity

    Returns:
        vx: velocity at y
    """
    return f / (2 * nu) * (L**2 - y**2)


def compute_L2_error(y_bins, vx_sim, L, f, nu, populated=None):
    """
    Compute relative L2 error between simulation and analytical solution.

    Only uses interior, populated bins (excludes boundary regions and empty bins).
    """
    vx_ana = analytical_poiseuille(y_bins, L, f, nu)

    # Only use interior bins (exclude boundary bins at edges)
    interior = (np.abs(y_bins) < 0.9 * L)

    # Also exclude bins with no particles (vx_sim=0 due to empty bin, not physics)
    if populated is not None:
        interior = interior & populated

    if interior.sum() == 0:
        return float('inf')

    error = np.sqrt(np.mean((vx_sim[interior] - vx_ana[interior])**2))
    norm = np.sqrt(np.mean(vx_ana[interior]**2))

    if norm < 1e-10:
        return float('inf')

    return error / norm  # relative L2 error


def main():
    scene_path = Path(__file__).parent.parent / "scenes" / "examples" / "pipe_flow_2d.json"
    sim = Simulator(scene_path)

    # Extract parameters from scene/simulator
    dx = sim.spacing
    h = sim.h
    rho0 = sim.rho0
    nu = sim.time_step.nu
    gravity = sim.time_step.gravity

    # Channel geometry
    fluid_cfg = sim.scene["fluid"]
    y_min = fluid_cfg["min"][1]
    y_max = fluid_cfg["max"][1]
    L = (y_max - y_min) / 2.0  # half-width
    y_center = (y_min + y_max) / 2.0  # centerline

    # Body force (gravity in x-direction)
    f = np.abs(gravity[0]) if len(gravity) > 0 else 0.0

    print("=" * 70)
    print("POISEUILLE FLOW VALIDATION (with periodic BC)")
    print("=" * 70)
    print(f"Scene: pipe_flow_2d.json")
    print(f"Steps to run: 5000")
    print()
    print("Parameters:")
    print(f"  dx = {dx:.6f}")
    print(f"  h  = {h:.6f}")
    print(f"  nu = {nu:.6e}")
    print(f"  f  = {f:.6e}")
    print(f"  L  = {L:.6f} (channel half-width)")
    print()

    # Analytical solution
    vmax_analytical = f * L**2 / (2 * nu)
    print("Analytical solution:")
    print(f"  vmax = f * L^2 / (2*nu) = {vmax_analytical:.6e}")
    print()

    # Snapshots to save
    snapshot_steps = [500, 1000, 2000, 3000, 5000]
    profiles = {}

    print("Running simulation...")
    print()

    for step in range(1, 5001):
        sim.step()

        if step in snapshot_steps:
            # Extract velocity profile
            y_bins, vx_mean, vx_std, populated = extract_velocity_profile(sim.fluid, n_bins=20)

            # Shift y to be relative to centerline
            y_bins_centered = y_bins - y_center

            # Compute L2 error (skip empty bins)
            l2_error = compute_L2_error(y_bins_centered, vx_mean, L, f, nu, populated)

            # Compute vmax (at centerline, over populated bins)
            centerline_idx = np.argmin(np.abs(y_bins_centered[populated]))
            vmax_sim = vx_mean[populated][centerline_idx]

            # Store profile
            profiles[step] = {
                'y': y_bins_centered,
                'vx': vx_mean,
                'vx_std': vx_std,
                'populated': populated,
                'l2_error': l2_error,
                'vmax_sim': vmax_sim,
            }

            print(f"Step {step:4d}:")
            print(f"  vmax_sim = {vmax_sim:.6e}")
            print(f"  vmax_ana = {vmax_analytical:.6e}")
            print(f"  vmax error = {abs(vmax_sim - vmax_analytical) / vmax_analytical * 100:.2f}%")
            print(f"  L2 error = {l2_error * 100:.2f}%")
            print()

    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    # Check success criteria and convergence
    final_profile = profiles[5000]
    final_l2_error = final_profile['l2_error']
    final_vmax_error = abs(final_profile['vmax_sim'] - vmax_analytical) / vmax_analytical

    print(f"Final L2 error (step 5000): {final_l2_error * 100:.2f}%")
    print(f"Final vmax error (step 5000): {final_vmax_error * 100:.2f}%")

    # Check if vmax converged (compare step 3000 vs 5000)
    vmax_3000 = profiles[3000]['vmax_sim']
    vmax_5000 = profiles[5000]['vmax_sim']
    vmax_change = abs(vmax_5000 - vmax_3000) / vmax_3000
    converged = vmax_change < 0.01  # Less than 1% change
    print(f"vmax convergence (3000→5000): {vmax_change * 100:.2f}% change")
    print(f"Converged: {'YES' if converged else 'NO'}")
    print()

    success_l2 = final_l2_error < 0.10  # < 10%
    success_vmax = final_vmax_error < 0.15  # < 15%

    print("Success criteria:")
    print(f"  L2 error < 10%: {'PASS' if success_l2 else 'FAIL'} ({final_l2_error * 100:.2f}%)")
    print(f"  vmax error < 15%: {'PASS' if success_vmax else 'FAIL'} ({final_vmax_error * 100:.2f}%)")
    print()

    # Create plot
    print("Creating plot...")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Analytical solution (fine grid)
    y_analytical = np.linspace(-L, L, 200)
    vx_analytical = analytical_poiseuille(y_analytical, L, f, nu)

    ax.plot(y_analytical, vx_analytical, 'k--', linewidth=2, label='Analytical', zorder=10)

    # Simulation profiles
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']

    for i, step in enumerate(snapshot_steps):
        profile = profiles[step]
        pop = profile['populated']
        ax.plot(profile['y'][pop], profile['vx'][pop],
                marker=markers[i], color=colors[i],
                markersize=6, linewidth=1.5, alpha=0.7,
                label=f'Step {step} (L2={profile["l2_error"]*100:.1f}%)')

    ax.set_xlabel('y - y_center (distance from centerline)', fontsize=12)
    ax.set_ylabel('v_x (velocity)', fontsize=12)
    ax.set_title('Poiseuille Flow: Velocity Profile Validation', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)

    # Save plot
    output_path = Path(__file__).parent / "poiseuille_profile.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")
    print()

    print("=" * 70)

    # Exit with appropriate code
    sys.exit(0 if (success_l2 and success_vmax) else 1)


if __name__ == "__main__":
    main()
