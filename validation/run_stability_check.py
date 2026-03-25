#!/usr/bin/env python3
"""
STEP 2 (revised): Quick Stability Check - 50 steps + incremental output

Run the pipe_flow_2d.json scene for 50 steps and collect diagnostic data.

This script:
1. Loads and runs the pipe flow scene for 50 steps
2. Records per step: avg_density, max_density_error_pct, avg_neighbors, min_neighbors, vmax, dt, has_nan
3. Writes one CSV row after EACH step (incremental output)
4. Prints a one-line diagnostic to console after every step
5. Prints a summary of stability criteria after 50 steps

Success criteria (must all be PASS):
- Max density error < 2% over all 50 steps
- Min neighbor count >= 15 at all times
- No NaN or Inf anywhere in positions, velocities, densities
- Simulation completed 50 steps
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from sph.core.simulation import SimulationRunner


def main():
    # Configuration
    scene_path = Path(__file__).parent.parent / "scenes" / "examples" / "pipe_flow_2d.json"
    output_csv = Path(__file__).parent / "stability_log.csv"
    num_steps = 50
    max_density_error_threshold = 0.02  # 2%
    min_neighbor_threshold = 15
    max_plausible_velocity = 1.0  # 10x expected max velocity for pipe flow

    print("=" * 70)
    print("STEP 2 (revised): Quick Stability Check - 50 steps")
    print("=" * 70)
    print(f"Scene: {scene_path.name}")
    print(f"Steps to run: {num_steps}")
    print(f"Output: {output_csv}")
    print()

    # Initialize simulation
    print("Loading scene and initializing simulation...")
    runner = SimulationRunner(scene_path)
    print(f"✓ Scene loaded: {runner.scene_name}")
    init_state = runner.state
    print(f"  - Fluid particles: {init_state.fluid_positions.shape[0]}")
    print(f"  - Boundary particles: {init_state.boundary_positions.shape[0]}")
    print(f"  - Reference density (rho0): {runner.rho0}")
    print(f"  - Support radius: {runner.support_radius}")
    print()

    # Open CSV file for incremental writing
    csv_file = open(output_csv, "w", newline="")
    fieldnames = ["step", "avg_density", "max_density_error_pct", "avg_neighbors", "min_neighbors", "vmax", "dt", "has_nan"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    csv_file.flush()

    # Tracking for statistics
    max_density_error_seen = 0.0
    min_neighbor_count_seen = float('inf')
    any_nan_detected = False
    completed_all_steps = False

    print("Running simulation...")
    print()

    try:
        for step in range(1, num_steps + 1):
            # Step the simulation
            result = runner.step()
            metrics = result.metrics

            avg_density = metrics.rho_mean
            max_density_error = metrics.rho_error_mean
            max_density_error_pct = max_density_error * 100.0
            avg_neighbor_count = metrics.neighbor_mean
            min_neighbor_count = metrics.neighbor_min
            vmax = metrics.velocity_max
            dt = metrics.dt

            state = result.state
            has_nan = (
                not np.isfinite(state.fluid_positions).all() or
                not np.isfinite(state.fluid_velocities).all() or
                not np.isfinite(state.fluid_densities).all()
            )

            # Update tracking statistics
            max_density_error_seen = max(max_density_error_seen, max_density_error)
            min_neighbor_count_seen = min(min_neighbor_count_seen, min_neighbor_count)
            if has_nan:
                any_nan_detected = True

            # Write CSV row immediately
            writer.writerow({
                "step": step,
                "avg_density": f"{avg_density:.6f}",
                "max_density_error_pct": f"{max_density_error_pct:.4f}",
                "avg_neighbors": f"{avg_neighbor_count:.2f}",
                "min_neighbors": min_neighbor_count,
                "vmax": f"{vmax:.6f}",
                "dt": f"{dt:.6e}",
                "has_nan": "YES" if has_nan else "NO",
            })
            csv_file.flush()

            # Print one-line diagnostic
            print(f"Step {step:03d} | rho_err={max_density_error_pct:5.2f}% | neighbors_min={min_neighbor_count:2d} | vmax={vmax:.6f} | dt={dt:.6e}")

            # Stop if NaN detected
            if has_nan:
                print()
                print(f"ERROR: NaN detected at step {step}. Stopping simulation.")
                break

        else:
            # Loop completed without break
            completed_all_steps = True

    finally:
        csv_file.close()

    print()
    print("=" * 70)
    print("STABILITY CHECK SUMMARY")
    print("=" * 70)

    # Summary statistics
    max_density_error_pct_seen = max_density_error_seen * 100.0

    print(f"Max density error seen: {max_density_error_pct_seen:.3f}% → {'PASS' if max_density_error_seen < max_density_error_threshold else 'FAIL'} (threshold {max_density_error_threshold*100}%)")
    print(f"Min neighbor count seen: {int(min_neighbor_count_seen)} → {'PASS' if min_neighbor_count_seen >= min_neighbor_threshold else 'FAIL'} (threshold {min_neighbor_threshold})")
    print(f"Any NaN detected: {'YES' if any_nan_detected else 'NO'} → {'FAIL' if any_nan_detected else 'PASS'}")
    print(f"Simulation completed {num_steps} steps: {'YES' if completed_all_steps else 'NO'} → {'PASS' if completed_all_steps else 'FAIL'}")

    print()

    # Overall result
    all_pass = (
        max_density_error_seen < max_density_error_threshold and
        min_neighbor_count_seen >= min_neighbor_threshold and
        not any_nan_detected and
        completed_all_steps
    )

    if all_pass:
        print("✓ ALL CRITERIA PASSED")
    else:
        print("✗ SOME CRITERIA FAILED")

    print()
    print(f"✓ Results saved to {output_csv}")
    print()

    # Display first and last 5 rows from CSV
    print("=" * 70)
    print("FIRST 5 ROWS OF stability_log.csv:")
    print("=" * 70)
    with open(output_csv, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        print(f"{'step':>4} {'avg_density':>12} {'err_pct':>8} {'avg_neigh':>10} {'min_neigh':>10} {'vmax':>10} {'dt':>12} {'nan':>4}")
        for row in rows[:5]:
            print(f"{row['step']:>4} {row['avg_density']:>12} {row['max_density_error_pct']:>8} {row['avg_neighbors']:>10} {row['min_neighbors']:>10} {row['vmax']:>10} {row['dt']:>12} {row['has_nan']:>4}")

    print()
    print("=" * 70)
    print("LAST 5 ROWS OF stability_log.csv:")
    print("=" * 70)
    with open(output_csv, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        print(f"{'step':>4} {'avg_density':>12} {'err_pct':>8} {'avg_neigh':>10} {'min_neigh':>10} {'vmax':>10} {'dt':>12} {'nan':>4}")
        for row in rows[-5:]:
            print(f"{row['step']:>4} {row['avg_density']:>12} {row['max_density_error_pct']:>8} {row['avg_neighbors']:>10} {row['min_neighbors']:>10} {row['vmax']:>10} {row['dt']:>12} {row['has_nan']:>4}")

    print()
    print("=" * 70)

    # Exit with appropriate code
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
