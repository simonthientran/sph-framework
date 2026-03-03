from __future__ import annotations

import numpy as np

from sph.core.state import ParticleState


def _grid_points_2d(pmin: np.ndarray, pmax: np.ndarray, spacing: float) -> np.ndarray:
    xs = np.arange(pmin[0], pmax[0] + 1e-12, spacing, dtype=np.float64)
    ys = np.arange(pmin[1], pmax[1] + 1e-12, spacing, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    return np.stack([X.ravel(), Y.ravel()], axis=1)


def _sample_box_boundary_2d(domain_min: np.ndarray, domain_max: np.ndarray, spacing: float, layers: int) -> np.ndarray:
    """
    Sample boundary as static particles in multiple layers (recommended in Section 5.1.1
    to avoid incomplete neighborhoods near boundaries).
    """
    pts = []

    # We generate layers inward from the domain edges.
    for k in range(layers):
        off = k * spacing

        x0 = domain_min[0] + off
        x1 = domain_max[0] - off
        y0 = domain_min[1] + off
        y1 = domain_max[1] - off

        xs = np.arange(x0, x1 + 1e-12, spacing, dtype=np.float64)
        ys = np.arange(y0, y1 + 1e-12, spacing, dtype=np.float64)

        # bottom and top edges
        pts.append(np.stack([xs, np.full_like(xs, y0)], axis=1))
        pts.append(np.stack([xs, np.full_like(xs, y1)], axis=1))

        # left and right edges (avoid double-count corners by skipping first/last)
        if len(ys) > 2:
            ys_inner = ys[1:-1]
            pts.append(np.stack([np.full_like(ys_inner, x0), ys_inner], axis=1))
            pts.append(np.stack([np.full_like(ys_inner, x1), ys_inner], axis=1))

    if not pts:
        return np.zeros((0, 2), dtype=np.float64)

    all_pts = np.concatenate(pts, axis=0)

    # Remove duplicates (important for corners / overlaps)
    all_pts = np.unique(np.round(all_pts / spacing).astype(np.int64), axis=0).astype(np.float64) * spacing
    return all_pts


def spawn_boundary_walls_channel(
    *,
    domain_min: np.ndarray,
    domain_max: np.ndarray,
    spacing: float,
    layers: int,
    walls: tuple[str, ...] = ("xmin", "xmax", "ymin", "ymax"),
) -> np.ndarray:
    """
    Spawn static boundary particles for a 2D channel/pipe segment.

    This is a *scene builder* utility: it defines geometry only, not physics.
    It reuses the same boundary sampling strategy as `_sample_box_boundary_2d`,
    but allows selecting which faces are walled.

    Args:
        domain_min/domain_max: AABB of the channel domain.
        spacing: particle spacing.
        layers: number of boundary layers.
        walls: subset of {"xmin","xmax","ymin","ymax"} to sample.

    Returns:
        (Nb,2) boundary particle positions.
    """
    walls_set = set(walls)
    allowed = {"xmin", "xmax", "ymin", "ymax"}
    unknown = walls_set - allowed
    if unknown:
        raise ValueError(f"unknown channel walls: {sorted(unknown)}")

    pts: list[np.ndarray] = []
    for k in range(int(layers)):
        off = k * float(spacing)
        x0 = float(domain_min[0] + off)
        x1 = float(domain_max[0] - off)
        y0 = float(domain_min[1] + off)
        y1 = float(domain_max[1] - off)

        xs = np.arange(x0, x1 + 1e-12, spacing, dtype=np.float64)
        ys = np.arange(y0, y1 + 1e-12, spacing, dtype=np.float64)

        if "ymin" in walls_set:
            pts.append(np.stack([xs, np.full_like(xs, y0)], axis=1))
        if "ymax" in walls_set:
            pts.append(np.stack([xs, np.full_like(xs, y1)], axis=1))

        if len(ys) > 2:
            ys_inner = ys[1:-1]
            if "xmin" in walls_set:
                pts.append(np.stack([np.full_like(ys_inner, x0), ys_inner], axis=1))
            if "xmax" in walls_set:
                pts.append(np.stack([np.full_like(ys_inner, x1), ys_inner], axis=1))

    if not pts:
        return np.zeros((0, 2), dtype=np.float64)
    all_pts = np.concatenate(pts, axis=0)
    all_pts = np.unique(np.round(all_pts / spacing).astype(np.int64), axis=0).astype(np.float64) * spacing
    return all_pts


def spawn_fluid_block_in_channel(
    *,
    domain_min: np.ndarray,
    domain_max: np.ndarray,
    spacing: float,
    wall_layers: int,
    pad_x: float = 0.0,
    pad_y: float = 0.0,
) -> np.ndarray:
    """
    Spawn a uniform fluid particle grid inside a 2D channel, leaving clearance
    from the walls.

    This is intended for pipe/channel examples where the geometry is "fill the
    channel interior". It is a geometry helper only.

    Args:
        domain_min/domain_max: AABB of the channel.
        spacing: particle spacing.
        wall_layers: boundary layers count; we leave at least wall_layers*spacing
            clearance from any wall to avoid initial overlap with boundary particles.
        pad_x/pad_y: additional clearance in x/y.

    Returns:
        (Nf,2) fluid positions.
    """
    clearance = float(max(0, int(wall_layers))) * float(spacing)
    pmin = np.array([domain_min[0] + clearance + pad_x, domain_min[1] + clearance + pad_y], dtype=np.float64)
    pmax = np.array([domain_max[0] - clearance - pad_x, domain_max[1] - clearance - pad_y], dtype=np.float64)
    if np.any(pmax <= pmin):
        raise ValueError("channel too small for requested wall clearance/padding")
    return _grid_points_2d(pmin, pmax, spacing)


def build_scene_state(scene: dict) -> ParticleState:
    meta = scene["meta"]
    dim = int(meta["dimensions"])
    if dim != 2:
        raise ValueError("This boundary builder currently supports only 2D (dim=2).")

    # --- scene parameters
    spacing = float(scene["fluid"]["spacing"])
    rho0 = float(scene["material"]["rho0"])

    # support radius used to decide how many boundary layers we need
    h = float(scene["neighbors"]["support_radius"])
    layers = int(scene.get("domain", {}).get("boundary_layers", int(np.ceil(h / spacing)) + 1))

    domain_min = np.array(scene["domain"]["min"], dtype=np.float64)
    domain_max = np.array(scene["domain"]["max"], dtype=np.float64)

    # --- fluid block
    fluid = scene["fluid"]
    ftype = str(fluid["type"]).lower()
    if ftype == "block":
        fmin = np.array(fluid["min"], dtype=np.float64)
        fmax = np.array(fluid["max"], dtype=np.float64)
        fluid_pos = _grid_points_2d(fmin, fmax, spacing)
    elif ftype == "channel_fill":
        # Pipe/channel example: fill channel interior with a uniform grid.
        pad = float(fluid.get("padding", 0.0))
        fluid_pos = spawn_fluid_block_in_channel(
            domain_min=domain_min,
            domain_max=domain_max,
            spacing=spacing,
            wall_layers=layers,
            pad_x=float(fluid.get("pad_x", pad)),
            pad_y=float(fluid.get("pad_y", pad)),
        )
    else:
        raise ValueError(f"unsupported fluid type: {fluid['type']}")

    v0 = np.array(fluid.get("initial_velocity", [0.0, 0.0]), dtype=np.float64)
    fluid_vel = np.repeat(v0[None, :], fluid_pos.shape[0], axis=0)

    # --- boundary sampling (static)
    domain_cfg = scene.get("domain", {})
    walls = tuple(str(w).lower() for w in domain_cfg.get("boundary_walls", ["xmin", "xmax", "ymin", "ymax"]))
    if str(domain_cfg.get("type", "box")).lower() in {"channel", "pipe", "box"}:
        # `box` remains the default; `boundary_walls` lets channel scenes omit end caps.
        boundary_pos = spawn_boundary_walls_channel(
            domain_min=domain_min,
            domain_max=domain_max,
            spacing=spacing,
            layers=layers,
            walls=walls,
        )
    else:
        boundary_pos = _sample_box_boundary_2d(domain_min, domain_max, spacing, layers=layers)
    boundary_vel = np.zeros((boundary_pos.shape[0], dim), dtype=np.float64)

    # --- combine
    pos = np.concatenate([fluid_pos, boundary_pos], axis=0)
    vel = np.concatenate([fluid_vel, boundary_vel], axis=0)
    acc = np.zeros_like(vel)

    n = pos.shape[0]
    is_boundary = np.zeros((n,), dtype=np.bool_)
    is_boundary[fluid_pos.shape[0]:] = True

    # masses: uniform boundary samples of same size use same mass as fluid samples
    # (Section 5.1.1 discusses uniform sampling where mi = mif = mib). Eq. (83) uses these masses.
    # We use baseline m = rho0 * spacing^dim
    mass_value = rho0 * (spacing ** dim)
    mass = np.full((n,), mass_value, dtype=np.float64)

    rho = np.full((n,), rho0, dtype=np.float64)
    p = np.zeros((n,), dtype=np.float64)

    state = ParticleState(dim=dim, pos=pos, vel=vel, acc=acc, mass=mass, rho=rho, p=p, is_boundary=is_boundary)
    state.validate()
    return state


def build_fluid_block(scene: dict) -> ParticleState:
    """
    Legacy fluid-only block builder used by existing tests.

    This function is a direct structural adaptation of the original
    implementation (see previous git history) to the new ParticleState
    layout that includes an is_boundary flag. We set all particles to
    non-boundary, so the physical configuration (positions, masses,
    densities, pressures) remains identical; only the container gains
    an explicit boundary flag.
    """
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

    # Mass: rho0 * spacing^dim, as in the original implementation.
    # This preserves the same volumetric mass distribution and therefore
    # keeps density-related physics identical.
    rho0 = float(scene["material"]["rho0"])
    mass_value = rho0 * (spacing ** dim)
    mass = np.full((n,), mass_value, dtype=np.float64)

    rho = np.full((n,), rho0, dtype=np.float64)
    p = np.zeros((n,), dtype=np.float64)

    # All particles are fluid; boundary flag is False everywhere.
    is_boundary = np.zeros((n,), dtype=np.bool_)

    state = ParticleState(
        dim=dim,
        pos=pos,
        vel=vel,
        acc=acc,
        mass=mass,
        rho=rho,
        p=p,
        is_boundary=is_boundary,
    )
    state.validate()
    return state
