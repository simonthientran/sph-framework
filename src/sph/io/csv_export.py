from __future__ import annotations

"""
Observability export: CSV snapshots for particle data.

What this module does:
- Writes one CSV file containing per-particle attributes for offline analysis.

How it works:
- Reads arrays from `ParticleState` and writes them in a stable column order.
- Supports 2D and 3D (for 2D, z and vz are omitted).

Physics / solver constraints:
- This module is pure I/O: it must not modify simulation state.

References:
- "SPH Techniques for the Physics Based Simulation of Fluids and Solids - SPH_Tutorial.pdf"
  - Algorithm 1: rho and p are computed each step and used for forces/integration.
  - Eq. (83) / Eq. (84): rho and p are solver-produced quantities; this module
    exports them but does not compute them.
"""

from pathlib import Path

import numpy as np

from sph.core.state import ParticleState


def export_particles_csv(path: str | Path, state: ParticleState) -> None:
    """
    Export a snapshot of all particles to CSV.

    Columns:
      id, is_boundary,
      x, y, (z),
      vx, vy, (vz),
      rho, p, m
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    dim = int(state.dim)
    if dim not in (2, 3):
        raise ValueError("export_particles_csv supports only dim=2 or dim=3")

    n = state.n
    ids = np.arange(n, dtype=np.int64)
    is_b = state.is_boundary.astype(np.int64)

    if dim == 2:
        header = "id,is_boundary,x,y,vx,vy,rho,p,m\n"
        table = np.column_stack(
            [
                ids,
                is_b,
                state.pos[:, 0],
                state.pos[:, 1],
                state.vel[:, 0],
                state.vel[:, 1],
                state.rho,
                state.p,
                state.mass,
            ]
        )
        fmt = "%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g"
    else:
        header = "id,is_boundary,x,y,z,vx,vy,vz,rho,p,m\n"
        table = np.column_stack(
            [
                ids,
                is_b,
                state.pos[:, 0],
                state.pos[:, 1],
                state.pos[:, 2],
                state.vel[:, 0],
                state.vel[:, 1],
                state.vel[:, 2],
                state.rho,
                state.p,
                state.mass,
            ]
        )
        fmt = "%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g"

    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(header)
        np.savetxt(f, table, delimiter=",", fmt=fmt)


