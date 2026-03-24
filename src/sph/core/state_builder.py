from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from sph.boundary.mesh_sampling import (
    compute_fluid_boundary_distance_stats,
    generate_boundary_layers,
    sample_mesh_surface_uniform,
)
from sph.core.state import ParticleState
from sph.core.physics import compute_particle_mass
from sph.geometry.stl import load_stl_mesh


def _agent_log(hypothesis_id: str, message: str, data: dict) -> None:
    """
    Lightweight debug logger for the AI agent.

    Writes one NDJSON line per call to the shared debug log file. This is
    purely for instrumentation and does not affect any physics or algorithms.
    """
    try:
        entry = {
            "id": f"log_{int(time.time() * 1000)}",
            "timestamp": int(time.time() * 1000),
            "location": "sph/core/state_builder.py",
            "message": message,
            "data": data,
            "runId": "pre-fix",
            "hypothesisId": hypothesis_id,
        }
        log_path = Path(".cursor/debug.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


_agent_log("H1", "module_imported", {})


def _grid_points_2d(pmin: np.ndarray, pmax: np.ndarray, spacing: float) -> np.ndarray:
    xs = np.arange(pmin[0], pmax[0] + 1e-12, spacing, dtype=np.float64)
    ys = np.arange(pmin[1], pmax[1] + 1e-12, spacing, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    return np.stack([X.ravel(), Y.ravel()], axis=1)


def _sample_box_boundary_2d(
    domain_min: np.ndarray,
    domain_max: np.ndarray,
    spacing: float,
    layers: int,
    periodic_axes: tuple[bool, bool] | None = None,
) -> np.ndarray:
    """
    Sample box boundary particles for a 2D domain.

    Important:
    - periodic_axes[d] == True means this axis is periodic
    - no solid wall particles are created on periodic faces

    Example:
    periodic_axes = (True, False)
      -> no left/right walls
      -> only bottom/top walls

    This is the correct setup for a fully filled, streamwise-periodic pipe/channel.
    """
    if periodic_axes is None:
        periodic_axes = (False, False)

    periodic_x = bool(periodic_axes[0])
    periodic_y = bool(periodic_axes[1])

    pts: list[np.ndarray] = []

    for k in range(layers):
        off = k * spacing

        x0 = domain_min[0] + off
        x1 = domain_max[0] - off
        y0 = domain_min[1] + off
        y1 = domain_max[1] - off

        if x1 < x0 or y1 < y0:
            continue

        xs = np.arange(x0, x1 + 1e-12, spacing, dtype=np.float64)
        ys = np.arange(y0, y1 + 1e-12, spacing, dtype=np.float64)

        # bottom / top walls only if y is NOT periodic
        if not periodic_y:
            pts.append(np.stack([xs, np.full_like(xs, y0)], axis=1))
            pts.append(np.stack([xs, np.full_like(xs, y1)], axis=1))

        # left / right walls only if x is NOT periodic
        if not periodic_x and len(ys) > 2:
            ys_inner = ys[1:-1]
            pts.append(np.stack([np.full_like(ys_inner, x0), ys_inner], axis=1))
            pts.append(np.stack([np.full_like(ys_inner, x1), ys_inner], axis=1))

    if not pts:
        return np.zeros((0, 2), dtype=np.float64)

    all_pts = np.concatenate(pts, axis=0)

    # remove duplicates robustly on spacing grid
    all_pts = np.unique(
        np.round(all_pts / spacing).astype(np.int64),
        axis=0,
    ).astype(np.float64) * spacing

    return all_pts


def _print_mesh_quality_warnings(
    mesh_name: str,
    triangle_count: int,
    dropped: int,
    diag: float,
    domain_diag: float | None,
) -> None:
    if triangle_count < 100:
        print(
            f"[GEOM][WARN] mesh={mesh_name} has only {triangle_count} triangles. "
            "Low triangle count can cause poor boundary resolution."
        )
    if dropped > 0:
        print(f"[GEOM][WARN] mesh={mesh_name} dropped degenerate triangles={dropped}")
    if domain_diag is not None and domain_diag > 0.0:
        ratio = float(diag / domain_diag)
        if ratio < 1e-3 or ratio > 1e3:
            print(
                f"[GEOM][WARN] mesh={mesh_name} bbox diag={diag:.6e} vs domain diag={domain_diag:.6e} (ratio={ratio:.3e}). "
                "Likely unit/scale mismatch; check units_hint/scale."
            )


def build_scene_state(scene: dict) -> ParticleState:
    _agent_log(
        "H1",
        "build_scene_state_called",
        {"keys": sorted(list(scene.keys()))},
    )

    meta = scene["meta"]
    dim = int(meta["dimensions"])
    if dim != 2:
        raise ValueError("This boundary builder currently supports only 2D (dim=2).")

    dx = float(scene["fluid"]["spacing"])
    rho0 = float(scene["material"]["rho0"])

    support_radius = float(scene["neighbors"]["support_radius"])
    layers = int(scene.get("domain", {}).get("boundary_layers", int(np.ceil(support_radius / dx)) + 1))

    domain_min = np.array(scene["domain"]["min"], dtype=np.float64)
    domain_max = np.array(scene["domain"]["max"], dtype=np.float64)

    periodic_axes_cfg = scene.get("domain", {}).get("periodic_axes", [False, False])
    periodic_axes = (
        bool(periodic_axes_cfg[0]) if len(periodic_axes_cfg) > 0 else False,
        bool(periodic_axes_cfg[1]) if len(periodic_axes_cfg) > 1 else False,
    )

    fluid = scene["fluid"]
    if fluid["type"] != "block":
        raise ValueError(f"unsupported fluid type: {fluid['type']}")

    fmin = np.array(fluid["min"], dtype=np.float64)
    fmax = np.array(fluid["max"], dtype=np.float64)

    fluid_pos = _grid_points_2d(fmin, fmax, dx)

    v0 = np.array(fluid.get("initial_velocity", [0.0, 0.0]), dtype=np.float64)
    fluid_vel = np.repeat(v0[None, :], fluid_pos.shape[0], axis=0)

    # Procedural boundary sampling:
    # For periodic axes we must NOT create wall particles on those faces.
    boundary_pos = _sample_box_boundary_2d(
        domain_min=domain_min,
        domain_max=domain_max,
        spacing=dx,
        layers=layers,
        periodic_axes=periodic_axes,
    )
    boundary_vel = np.zeros((boundary_pos.shape[0], dim), dtype=np.float64)
    boundary_mass_val = compute_particle_mass(dx, rho0, dim)
    boundary_mass = np.full((boundary_pos.shape[0],), boundary_mass_val, dtype=np.float64)
    boundary_rho = np.full((boundary_pos.shape[0],), rho0, dtype=np.float64)
    boundary_p = np.zeros((boundary_pos.shape[0],), dtype=np.float64)
    boundary_is = np.ones((boundary_pos.shape[0],), dtype=np.bool_)

    mesh_states: list[ParticleState] = []
    mesh_reports: list[dict] = []
    geometry_cfg = scene.get("geometry", {})
    meshes = geometry_cfg.get("meshes", [])
    debug_geom = bool(scene.get("__debug_geom__", False))
    scene_dir = Path(str(scene.get("__scene_dir__", ".")))
    domain_diag = float(np.linalg.norm(domain_max - domain_min))

    for mesh_cfg in meshes:
        if str(mesh_cfg.get("type", "boundary")).lower() != "boundary":
            continue

        mesh_file = mesh_cfg.get("path", mesh_cfg.get("file"))
        if not mesh_file:
            raise ValueError("geometry.meshes[].path (or file) is required for mesh boundary import")

        mesh_path = Path(str(mesh_file))
        if not mesh_path.is_absolute():
            mesh_path = scene_dir / mesh_path
        mesh_path = mesh_path.resolve()
        mesh_name = mesh_path.name

        print(f"[GEOM] loaded mesh {mesh_name}")
        mesh = load_stl_mesh(mesh_path)
        mesh = mesh.transformed(
            scale=mesh_cfg.get("scale", 1.0),
            translate=mesh_cfg.get("translate", [0.0, 0.0, 0.0]),
            rotate_euler_deg=mesh_cfg.get("rotate_euler_deg", [0.0, 0.0, 0.0]),
            units_hint=mesh_cfg.get("units_hint", None),
        )
        print(f"[GEOM] triangles={mesh.triangle_count}")

        _print_mesh_quality_warnings(
            mesh_name=mesh_name,
            triangle_count=mesh.triangle_count,
            dropped=mesh.degenerate_triangles_dropped,
            diag=mesh.characteristic_length,
            domain_diag=domain_diag,
        )

        if mesh.normal_consistency_ratio is not None and mesh.normal_consistency_ratio < 0.7:
            print(
                f"[GEOM][WARN] mesh={mesh_name} normal consistency is low: "
                f"{mesh.normal_consistency_ratio * 100.0:.1f}%"
            )

        boundary_spacing = float(mesh_cfg.get("boundary_spacing", mesh_cfg.get("spacing", dx)))
        layer_count = int(mesh_cfg.get("layers", 1))
        layer_mode = str(mesh_cfg.get("layer_mode", "outward")).lower()

        pts3, nrm3 = sample_mesh_surface_uniform(mesh, spacing=boundary_spacing)
        pts3, nrm3 = generate_boundary_layers(
            pts3,
            nrm3,
            layers=max(1, layer_count),
            layer_spacing=boundary_spacing,
            direction=layer_mode,
        )

        overlap_mode = str(mesh_cfg.get("overlap_resolution", "warn")).lower()
        overlap_threshold = float(mesh_cfg.get("overlap_threshold", 0.5 * dx))

        overlap_before = compute_fluid_boundary_distance_stats(
            fluid_positions=fluid_pos,
            boundary_positions=pts3[:, :dim],
            threshold=overlap_threshold,
        )

        if overlap_mode == "push_outward" and overlap_before.close_fraction > 0.05:
            push_dist = float(mesh_cfg.get("overlap_push_distance", overlap_threshold))
            pts3 = pts3 + push_dist * nrm3
            overlap_after = compute_fluid_boundary_distance_stats(
                fluid_positions=fluid_pos,
                boundary_positions=pts3[:, :dim],
                threshold=overlap_threshold,
            )
            print(
                f"[GEOM] overlap push_outward applied mesh={mesh_name} push={push_dist:.6e} "
                f"close_fraction {overlap_before.close_fraction:.3f}->{overlap_after.close_fraction:.3f}"
            )

        bpos = pts3[:, :dim].astype(np.float64)
        bvel = np.zeros((bpos.shape[0], dim), dtype=np.float64)
        bmass_val = compute_particle_mass(boundary_spacing, rho0, dim)
        bmass = np.full((bpos.shape[0],), bmass_val, dtype=np.float64)
        brho = np.full((bpos.shape[0],), rho0, dtype=np.float64)
        bp = np.zeros((bpos.shape[0],), dtype=np.float64)
        bis = np.ones((bpos.shape[0],), dtype=np.bool_)

        mesh_state = ParticleState(
            dim=dim,
            pos=bpos,
            vel=bvel,
            acc=np.zeros_like(bvel),
            mass=bmass,
            rho=brho,
            p=bp,
            is_boundary=bis,
        )
        mesh_state.validate()
        print(f"[GEOM] boundary particles created={mesh_state.n}")

        mesh_reports.append(
            {
                "name": mesh_name,
                "triangle_count": mesh.triangle_count,
                "degenerate_dropped": mesh.degenerate_triangles_dropped,
                "surface_area": mesh.surface_area,
                "bbox_min": mesh.bbox_min.tolist(),
                "bbox_max": mesh.bbox_max.tolist(),
                "diag": mesh.characteristic_length,
                "normal_consistency_ratio": mesh.normal_consistency_ratio,
                "open_edge_ratio": mesh.open_edge_ratio,
                "non_manifold_edge_ratio": mesh.non_manifold_edge_ratio,
                "boundary_spacing": boundary_spacing,
                "layers": layer_count,
                "layer_mode": layer_mode,
            }
        )
        mesh_states.append(mesh_state)

    pos_parts = [fluid_pos, boundary_pos]
    vel_parts = [fluid_vel, boundary_vel]

    fmass_val = compute_particle_mass(dx, rho0, dim)
    mass_parts = [
        np.full((fluid_pos.shape[0],), fmass_val, dtype=np.float64),
        boundary_mass,
    ]
    rho_parts = [
        np.full((fluid_pos.shape[0],), rho0, dtype=np.float64),
        boundary_rho,
    ]
    p_parts = [
        np.zeros((fluid_pos.shape[0],), dtype=np.float64),
        boundary_p,
    ]
    is_parts = [
        np.zeros((fluid_pos.shape[0],), dtype=np.bool_),
        boundary_is,
    ]

    for mesh_state in mesh_states:
        pos_parts.append(mesh_state.pos)
        vel_parts.append(mesh_state.vel)
        mass_parts.append(mesh_state.mass)
        rho_parts.append(mesh_state.rho)
        p_parts.append(mesh_state.p)
        is_parts.append(mesh_state.is_boundary)

    pos = np.concatenate(pos_parts, axis=0)
    vel = np.concatenate(vel_parts, axis=0)
    acc = np.zeros_like(vel)
    mass = np.concatenate(mass_parts, axis=0)
    rho = np.concatenate(rho_parts, axis=0)
    p = np.concatenate(p_parts, axis=0)
    is_boundary = np.concatenate(is_parts, axis=0)

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

    overlap_threshold = 0.5 * dx
    if state.boundary_indices.size > 0:
        overlap_stats = compute_fluid_boundary_distance_stats(
            fluid_positions=state.pos[state.fluid_indices],
            boundary_positions=state.pos[state.boundary_indices],
            threshold=overlap_threshold,
        )
    else:
        overlap_stats = compute_fluid_boundary_distance_stats(
            fluid_positions=state.pos[state.fluid_indices],
            boundary_positions=np.zeros((0, dim), dtype=np.float64),
            threshold=overlap_threshold,
        )

    if overlap_stats.close_fraction > 0.05:
        print(
            f"[GEOM][WARN] fluid-boundary overlap risk: min_dist={overlap_stats.min_distance:.6e} "
            f"mean_min_dist={overlap_stats.mean_min_distance:.6e} "
            f"close<{overlap_threshold:.6e} count={overlap_stats.close_count}/{overlap_stats.fluid_count}"
        )
        if debug_geom:
            print(
                "[GEOM][WARN] likely causes: scale mismatch, mesh intersects fluid, "
                "boundary spacing too coarse/fine, or missing transform."
            )

    scene["__geom_report__"] = {
        "meshes": mesh_reports,
        "overlap": {
            "threshold": overlap_threshold,
            "min_distance": overlap_stats.min_distance,
            "mean_min_distance": overlap_stats.mean_min_distance,
            "close_fraction": overlap_stats.close_fraction,
            "close_count": overlap_stats.close_count,
            "fluid_count": overlap_stats.fluid_count,
        },
    }

    return state


def build_fluid_block(scene: dict) -> ParticleState:
    """
    Legacy fluid-only block builder used by existing tests.
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

    axes = []
    for d in range(dim):
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

    rho0 = float(scene["material"]["rho0"])
    mass_value = compute_particle_mass(spacing, rho0, dim)
    mass = np.full((n,), mass_value, dtype=np.float64)

    rho = np.full((n,), rho0, dtype=np.float64)
    p = np.zeros((n,), dtype=np.float64)
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