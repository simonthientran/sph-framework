#!/usr/bin/env python3
"""
Diagnose mass assignment and partition of unity.
"""
import sys
from pathlib import Path
import numpy as np

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from sph.simulator_new import Simulator
from sph.kernel import CubicSplineKernel

def main():
    scene_path = Path(__file__).parent.parent / "scenes" / "examples" / "pipe_flow_2d.json"
    sim = Simulator(scene_path)

    dx = sim.spacing
    h = sim.h
    rho0 = sim.rho0
    mass = sim.fluid.mass

    print("=" * 70)
    print("MASS AND KERNEL DIAGNOSTICS")
    print("=" * 70)
    print(f"dx = {dx:.6f}")
    print(f"h  = {h:.6f}  (h/dx = {h/dx:.3f})")
    print(f"rho0 = {rho0:.3f}")
    print(f"mass = {mass:.6f}")
    print()

    # Expected mass for uniform 2D grid (naive formula)
    mass_expected_naive = rho0 * dx**2
    print(f"mass_expected_naive = rho0 * dx^2 = {mass_expected_naive:.6f}")
    print(f"mass ratio = {mass / mass_expected_naive:.4f}  (should be ~1.0 for naive)")
    print()

    # Kernel partition of unity check
    # Build a test uniform grid patch
    kernel = CubicSplineKernel(h=h, dim=2)
    support = 2.0 * h

    print(f"Checking partition of unity...")
    print(f"support radius = {support:.6f}")
    print()

    # Create a uniform grid patch centered at origin
    n_cells = int(np.ceil(support / dx)) + 1
    print(f"Grid size for test: {2*n_cells+1} x {2*n_cells+1}")

    # Build grid
    grid_points = []
    for ix in range(-n_cells, n_cells + 1):
        for iy in range(-n_cells, n_cells + 1):
            x = ix * dx
            y = iy * dx
            r = np.sqrt(x**2 + y**2)
            if r <= support:
                grid_points.append(r)

    print(f"Neighbors within support: {len(grid_points)}")

    # Compute kernel sum
    W_values = []
    for r in grid_points:
        W = kernel.W(r)
        W_values.append(W)

    w_sum_raw = sum(W_values)
    print(f"Sum of W(r_ij) = {w_sum_raw:.6f}")

    # Partition of unity: sum of (volume_j * W_ij) should = 1
    volume = mass / rho0  # V = m / rho0
    w_sum_volume = w_sum_raw * volume
    print(f"Sum of (V_j * W_ij) = {w_sum_volume:.6f}  (should be ~1.0 for partition of unity)")
    print()

    # What mass SHOULD be for partition of unity?
    mass_correct = rho0 / w_sum_raw
    print(f"Correct mass for partition of unity:")
    print(f"mass_correct = rho0 / sum(W) = {mass_correct:.6f}")
    print(f"Ratio: mass_current / mass_correct = {mass / mass_correct:.6f}")
    print()

    # Density prediction
    density_predicted = mass * w_sum_raw
    density_error_pct = abs(density_predicted - rho0) / rho0 * 100
    print(f"Predicted density with current mass:")
    print(f"rho = mass * sum(W) = {density_predicted:.3f}")
    print(f"Density error: {density_error_pct:.2f}%")
    print()

    print("=" * 70)
    print("CONCLUSION:")
    print("=" * 70)
    if abs(mass / mass_correct - 1.0) > 0.01:
        print(f"❌ Mass is INCORRECT!")
        print(f"   Current:  {mass:.6f}")
        print(f"   Should be: {mass_correct:.6f}")
        print(f"   Fix: Update mass computation in simulator")
    else:
        print(f"✓ Mass is correct")

    if density_error_pct > 2:
        print(f"❌ Density error too high: {density_error_pct:.2f}%")
    else:
        print(f"✓ Density error acceptable: {density_error_pct:.2f}%")

if __name__ == "__main__":
    main()
