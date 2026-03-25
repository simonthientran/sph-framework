#!/usr/bin/env python3
"""
REFACTORED: Stability check using new vectorized SPH architecture.

Tests the new FluidModel + WCSPHTimeStep implementation.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from sph.simulator_new import Simulator


def main():
    # Configuration
    scene_path = Path(__file__).parent.parent / "scenes" / "examples" / "pipe_flow_2d.json"
    output_csv = Path(__file__).parent / "stability_log_new.csv"
    num_steps = 50
    max_density_error_threshold = 0.02  # 2%
    min_neighbor_threshold = 15

    print("=" * 70)
    print("REFACTORED STABILITY CHECK - New Vectorized Architecture")
    print("=" * 70)
    print(f"Scene: {scene_path.name}")
    print(f"Steps to run: {num_steps}")
    print(f"Output: {output_csv}")
    print()

    # Initialize simulator
    print("Loading scene and initializing simulator...")
    sim = Simulator(scene_path)
    print(f"✓ Simulator initialized")
    print(f"  - Fluid particles: {sim.fluid.n}")
    print(f"  - Boundary particles: {sim.boundary.n if sim.boundary else 0}")
    print(f"  - Reference density (rho0): {sim.rho0}")
    print(f"  - Particle spacing (dx): {sim.spacing}")
    print(f"  - Smoothing length (h): {sim.h}")
    print(f"  - h/dx ratio: {sim.h / sim.spacing:.2f}")
    print(f"  - Support radius: {sim.support_radius}")
    print()

    # Open CSV file for incremental writing
    csv_file = open(output_csv, "w", newline="")
    fieldnames = ["step", "avg_density", "max_density_error_pct", "avg_neighbors",
                  "min_neighbors", "vmax", "dt", "has_nan", "time_ms"]
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

    total_start_time = time.time()

    try:
        for step in range(1, num_steps + 1):
            step_start_time = time.time()

            # Step the simulation
            sim.step()

            step_end_time = time.time()
            step_time_ms = (step_end_time - step_start_time) * 1000

            # Compute diagnostics
            fluid = sim.fluid

            # Density metrics
            avg_density = np.mean(fluid.densities)
            density_errors = np.abs(fluid.densities - fluid.rho0) / fluid.rho0
            max_density_error = np.max(density_errors)
            max_density_error_pct = max_density_error * 100.0

            # Neighbor count (estimate from neighbor pairs)
            pairs = fluid.neighbor_pairs
            if pairs is not None:
                counts = np.bincount(pairs.ff_i, minlength=fluid.n)
                counts += np.bincount(pairs.ff_j, minlength=fluid.n)
                if pairs.fb_i.size:
                    counts += np.bincount(pairs.fb_i, minlength=fluid.n)
                avg_neighbor_count = float(np.mean(counts))
                min_neighbor_count = int(np.min(counts)) if counts.size else 0
            else:
                avg_neighbor_count = 0.0
                min_neighbor_count = 0

            # Velocity max
            vmax = np.max(np.linalg.norm(fluid.velocities, axis=1))

            # Check for NaN
            has_nan = (
                not np.isfinite(fluid.positions).all() or
                not np.isfinite(fluid.velocities).all() or
                not np.isfinite(fluid.densities).all()
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
                "min_neighbors": int(min_neighbor_count),
                "vmax": f"{vmax:.6f}",
                "dt": f"{sim.dt:.6e}",
                "has_nan": "YES" if has_nan else "NO",
                "time_ms": f"{step_time_ms:.2f}",
            })
            csv_file.flush()

            # Print one-line diagnostic
            print(f"Step {step:03d} | rho_err={max_density_error_pct:5.2f}% | neighbors_min={min_neighbor_count:2d} | vmax={vmax:.6f} | dt={sim.dt:.6e} | time={step_time_ms:6.1f}ms")

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

    total_end_time = time.time()
    total_time = total_end_time - total_start_time

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
    print(f"Total time for {num_steps} steps: {total_time:.2f} seconds ({total_time/num_steps*1000:.1f} ms/step)")
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
    print("FIRST 5 ROWS OF stability_log_new.csv:")
    print("=" * 70)
    with open(output_csv, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        print(f"{'step':>4} {'avg_density':>12} {'err_pct':>8} {'avg_neigh':>10} {'min_neigh':>10} {'vmax':>10} {'time_ms':>8}")
        for row in rows[:5]:
            print(f"{row['step']:>4} {row['avg_density']:>12} {row['max_density_error_pct']:>8} {row['avg_neighbors']:>10} {row['min_neighbors']:>10} {row['vmax']:>10} {row['time_ms']:>8}")

    print()
    print("=" * 70)
    print("LAST 5 ROWS OF stability_log_new.csv:")
    print("=" * 70)
    with open(output_csv, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        print(f"{'step':>4} {'avg_density':>12} {'err_pct':>8} {'avg_neigh':>10} {'min_neigh':>10} {'vmax':>10} {'time_ms':>8}")
        for row in rows[-5:]:
            print(f"{row['step']:>4} {row['avg_density']:>12} {row['max_density_error_pct']:>8} {row['avg_neighbors']:>10} {row['min_neighbors']:>10} {row['vmax']:>10} {row['time_ms']:>8}")

    print()
    print("=" * 70)

    # Exit with appropriate code
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
