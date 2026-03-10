"""
Simple particle visualization for SPH debugging and analysis.

Purpose:
    Visualize particle snapshots exported by sph.io.csv_export.export_particles_csv.
    This tool is read-only: it loads CSV files and plots them; it does not modify
    the simulation or any exported data.

How it works:
    - Loads a CSV snapshot with columns: id, is_boundary, x, y, vx, vy, rho, p, m
    - Splits particles into fluid (is_boundary=0) and boundary (is_boundary=1)
    - Fluid particles: scatter plot colored by velocity magnitude |v| = sqrt(vx² + vy²)
    - Boundary particles: black scatter with small markers

Relation to SPH debugging:
    Use this to inspect particle positions, velocity fields, and boundary layout
    after running a simulation. Helps verify geometry, detect overlap, and
    sanity-check flow patterns.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_particles_csv(path: str) -> None:
    """
    Load a CSV particle snapshot and plot fluid (colored by |v|) and boundary (black).

    CSV format: id, is_boundary, x, y, vx, vy, rho, p, m
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    data = np.loadtxt(path, delimiter=",", skiprows=1)
    # Columns: id, is_boundary, x, y, vx, vy, rho, p, m
    is_boundary = data[:, 1].astype(bool)
    x = data[:, 2]
    y = data[:, 3]
    vx = data[:, 4]
    vy = data[:, 5]

    # Velocity magnitude: |v| = sqrt(vx² + vy²)
    vmag = np.sqrt(vx * vx + vy * vy)

    fluid_mask = ~is_boundary
    boundary_mask = is_boundary

    fig, ax = plt.subplots()

    # Fluid particles: scatter colored by velocity magnitude
    if np.any(fluid_mask):
        sc_fluid = ax.scatter(
            x[fluid_mask],
            y[fluid_mask],
            c=vmag[fluid_mask],
            cmap="viridis",
            s=8,
        )

    # Boundary particles: black, small
    if np.any(boundary_mask):
        ax.scatter(
            x[boundary_mask],
            y[boundary_mask],
            color="black",
            s=2,
        )

    if np.any(fluid_mask):
        cbar = plt.colorbar(sc_fluid, ax=ax)
        cbar.set_label("velocity magnitude")

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("SPH particle visualization")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m sph.visualization.plot_particles <path_to_csv>")
        sys.exit(1)
    plot_particles_csv(sys.argv[1])
