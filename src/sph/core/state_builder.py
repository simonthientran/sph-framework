from __future__ import annotations

import numpy as np
from sph.core.state import ParticleState


def build_fluid_block(scene: dict) -> ParticleState:
    meta = scene["meta"]
    dim = int(meta["dimensions"])

    fluid = scene["fluid"]
    if fluid["type"] != "block":
        raise ValueError(f"unsupported fluid type: {fluid['type']}")

    pmin = np.array(fluid["min"], dtype=np.float64)
    pmax = np.array(fluid["max"], dtype=np.float64)
    spacing = float(fluid["spacing"])

    v0 = np.array(fluid.get("initial_velocity", [0.0] * dim), dtype=np.float64)

    if pmin.shape != (dim,) or pmax.shape != (dim,):
        raise ValueError("fluid.min/max must match dimensions")

    if spacing <= 0:
        raise ValueError("spacing must be > 0")

    # generate regular grid points in [pmin, pmax] (inclusive-ish)
    axes = []
    for d in range(dim):
        # +1e-12 to avoid floating issues at the boundary
        axes.append(np.arange(pmin[d], pmax[d] + 1e-12, spacing, dtype=np.float64))

    if dim == 2:
        X, Y = np.meshgrid(axes[0], axes[1], indexing="xy")
        pos = np.stack([X.ravel(), Y.ravel()], axis=1)
    elif dim == 3:
        X, Y, Z = np.meshgrid(axes[0], axes[1], axes[2], indexing="xy")
        pos = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    else:
        raise ValueError("dimensions must be 2 or 3")

    n = pos.shape[0]

    vel = np.repeat(v0[None, :], n, axis=0)
    acc = np.zeros((n, dim), dtype=np.float64)

    # Mass: for now use rho0 * spacing^dim (simple volumetric mass).
    # This is a good baseline and keeps density roughly in the correct scale.
    rho0 = float(scene["material"]["rho0"])
    mass_value = rho0 * (spacing ** dim)
    mass = np.full((n,), mass_value, dtype=np.float64)

    rho = np.full((n,), rho0, dtype=np.float64)
    p = np.zeros((n,), dtype=np.float64)

    state = ParticleState(dim=dim, pos=pos, vel=vel, acc=acc, mass=mass, rho=rho, p=p)
    state.validate()
    return state
