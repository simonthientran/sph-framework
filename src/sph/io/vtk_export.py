from __future__ import annotations

"""
Observability export: VTK legacy ASCII PolyData for particle visualization.

What this module does:
- Writes a VTK legacy (ASCII) PolyData file containing:
  - POINTS (particle positions)
  - VERTICES (one vertex per particle)
  - POINT_DATA scalars/vectors for analysis in ParaView

How it works:
- This writer uses no external dependencies.
- For 2D, positions/velocities are padded with z=0 so ParaView can treat them
  as 3D vectors where required.

Physics / solver constraints:
- This module is pure I/O: it must not modify simulation state.

References:
- "SPH Techniques for the Physics Based Simulation of Fluids and Solids - SPH_Tutorial.pdf"
  - Algorithm 1: export step snapshots for debugging and analysis.
  - Eq. (83) / Eq. (84): rho and p are solver-produced quantities; this module
    exports them but does not compute them.
"""

from pathlib import Path

import numpy as np

from sph.core.state import ParticleState


def export_particles_vtk_legacy(path: str | Path, state: ParticleState) -> None:
    """
    Export particles as VTK legacy ASCII PolyData.

    Required fields:
    - POINTS and VERTICES
    - POINT_DATA:
        - is_boundary (int)
        - rho (float)
        - p (float)
        - m (float)
        - v (VECTORS, float) (pad z=0 for 2D)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    dim = int(state.dim)
    if dim not in (2, 3):
        raise ValueError("export_particles_vtk_legacy supports only dim=2 or dim=3")

    n = state.n

    # Pad to 3D for VTK vectors
    if dim == 2:
        pos3 = np.zeros((n, 3), dtype=np.float64)
        vel3 = np.zeros((n, 3), dtype=np.float64)
        pos3[:, 0:2] = state.pos
        vel3[:, 0:2] = state.vel
    else:
        pos3 = state.pos.astype(np.float64, copy=False)
        vel3 = state.vel.astype(np.float64, copy=False)

    is_b = state.is_boundary.astype(np.int32, copy=False)
    rho = state.rho.astype(np.float64, copy=False)
    p = state.p.astype(np.float64, copy=False)
    m = state.mass.astype(np.float64, copy=False)

    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("SPH particles (WCSPH) - legacy PolyData\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")

        # POINTS
        f.write(f"POINTS {n} float\n")
        for i in range(n):
            x, y, z = pos3[i]
            f.write(f"{x:.17g} {y:.17g} {z:.17g}\n")

        # VERTICES (n cells, 2*n indices)
        f.write(f"VERTICES {n} {2*n}\n")
        for i in range(n):
            f.write(f"1 {i}\n")

        # POINT_DATA
        f.write(f"POINT_DATA {n}\n")

        f.write("SCALARS is_boundary int 1\n")
        f.write("LOOKUP_TABLE default\n")
        for i in range(n):
            f.write(f"{int(is_b[i])}\n")

        f.write("SCALARS rho float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for i in range(n):
            f.write(f"{float(rho[i]):.17g}\n")

        f.write("SCALARS p float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for i in range(n):
            f.write(f"{float(p[i]):.17g}\n")

        f.write("SCALARS m float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for i in range(n):
            f.write(f"{float(m[i]):.17g}\n")

        f.write("VECTORS v float\n")
        for i in range(n):
            vx, vy, vz = vel3[i]
            f.write(f"{vx:.17g} {vy:.17g} {vz:.17g}\n")


