"""Build a CLOSED, fully-filled torus-bend elbow (experiment: recover the
centrifugal bend pressure pattern).

Same centerline geometry as build_elbow_analytical.py, but:
  - fluid disks filled up to (r - 0.5*dx)  → fuller, fewer near-wall gaps
  - boundary = 3 lateral rings PLUS 3-layer flat end caps closing BOTH ends
    (solid disks normal to the local tangent), so the system is sealed and can
    pressurise (rho ~ rho0) instead of draining.

Writes assets/elbow_closed/{fluid,boundary}.npy  (does NOT touch elbow_analytical).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from sph.core.state_builder import deduplicate_positions  # noqa: E402

# ── Geometry (matches build_elbow_analytical.py) ────────────────────────────
R_PIPE = 0.045
D = 2 * R_PIPE
R_BEND = 1.5 * D          # 0.135
L_H = 0.30
L_V = 0.30
DX = 0.0105
N_LAYERS = 3


def _frame(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n2 = np.array([0.0, 0.0, 1.0])
    n1 = np.cross(t, n2)
    nrm = np.linalg.norm(n1)
    if nrm < 1e-9:
        n1 = np.array([1.0, 0.0, 0.0]); nrm = 1.0
    return n1 / nrm, n2


def _disk_offsets(fill_radius: float, dx: float) -> np.ndarray:
    n = int(np.ceil(fill_radius / dx))
    coords = np.arange(-n, n + 1) * dx
    a, b = np.meshgrid(coords, coords, indexing="ij")
    a = a.ravel(); b = b.ravel()
    mask = a * a + b * b < fill_radius * fill_radius
    return np.column_stack([a[mask], b[mask]])


def _ring_offsets(dx: float, n_layers: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    for layer in range(n_layers):
        R = R_PIPE + layer * dx
        n_theta = max(8, int(round(2 * np.pi * R / dx)))
        th = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
        rows.append(np.column_stack([R * np.cos(th), R * np.sin(th)]))
    return np.vstack(rows)


def _centerline() -> list[tuple[np.ndarray, np.ndarray]]:
    pts: list[tuple[np.ndarray, np.ndarray]] = []
    for x in np.arange(0.0, L_H + DX / 2, DX):
        pts.append((np.array([x, R_BEND, 0.0]), np.array([1.0, 0.0, 0.0])))
    center = np.array([L_H, 2 * R_BEND, 0.0])
    n_arc = max(1, int(round(R_BEND * (np.pi / 2) / DX)))
    for k in range(1, n_arc + 1):
        theta = (np.pi / 2) * k / n_arc
        pt = center + R_BEND * np.array([np.sin(theta), -np.cos(theta), 0.0])
        t = np.array([np.cos(theta), np.sin(theta), 0.0])
        pts.append((pt, t))
    x_v = L_H + R_BEND
    y0 = 2 * R_BEND
    for y in np.arange(y0 + DX, y0 + L_V + DX / 2, DX):
        pts.append((np.array([x_v, y, 0.0]), np.array([0.0, 1.0, 0.0])))
    return pts


def _sweep(offsets2d: np.ndarray, cline) -> np.ndarray:
    out: list[np.ndarray] = []
    for pt, t in cline:
        n1, n2 = _frame(t)
        out.append(pt[None, :] + offsets2d[:, 0:1] * n1[None, :] + offsets2d[:, 1:2] * n2[None, :])
    return np.vstack(out)


def _end_cap(pt: np.ndarray, t: np.ndarray, sign: float, n_layers: int) -> np.ndarray:
    """Solid-disk end cap of n_layers normal to t, placed `sign` side of pt.
    Radius up to R_PIPE so it seals the opening and meets the wall rings."""
    n1, n2 = _frame(t)
    disk = _disk_offsets(R_PIPE + 0.5 * DX, DX)   # solid disk covering the bore
    out: list[np.ndarray] = []
    for layer in range(n_layers):
        axial = sign * (layer + 0.5) * DX
        base = pt + axial * t
        out.append(base[None, :] + disk[:, 0:1] * n1[None, :] + disk[:, 1:2] * n2[None, :])
    return np.vstack(out)


def main() -> None:
    cline = _centerline()

    # Fluid: fill up to (r - 0.5 dx) along the whole centerline.
    fill_r = R_PIPE - 0.5 * DX
    fluid = deduplicate_positions(_sweep(_disk_offsets(fill_r, DX), cline), DX)

    # Boundary: lateral rings + end caps at both centerline ends.
    walls = _sweep(_ring_offsets(DX, N_LAYERS), cline)
    p_in, t_in = cline[0]           # inlet end (tangent +x) → cap on -t side
    p_out, t_out = cline[-1]        # outlet end (tangent +y) → cap on +t side
    cap_in = _end_cap(p_in, t_in, -1.0, N_LAYERS)
    cap_out = _end_cap(p_out, t_out, +1.0, N_LAYERS)
    boundary = deduplicate_positions(np.vstack([walls, cap_in, cap_out]), DX)

    out_dir = ROOT / "assets" / "elbow_closed"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "fluid.npy", fluid)
    np.save(out_dir / "boundary.npy", boundary)

    # Verify caps: boundary particles near each end covering the full disk.
    def cap_count(pt, t):
        rel = boundary - pt[None, :]
        axial = rel @ t
        return int(np.sum((axial < 0.0) if (t[0] > 0.5) else (axial > 0.0)
                          ) if False else 0)
    # Simpler verification: boundary particles within 3.5dx axially behind each end.
    rel_in = boundary - p_in[None, :]
    ax_in = rel_in @ t_in
    n_cap_in = int(np.sum((ax_in < 0.0) & (ax_in > -3.5 * DX)))
    rel_out = boundary - p_out[None, :]
    ax_out = rel_out @ t_out
    n_cap_out = int(np.sum((ax_out > 0.0) & (ax_out < 3.5 * DX)))

    print(f"Geometry: r={R_PIPE} R_bend={R_BEND} dx={DX} (CLOSED, filled)")
    print(f"n_fluid    = {len(fluid)}")
    print(f"n_boundary = {len(boundary)}")
    print(f"caps: inlet cap particles={n_cap_in}  outlet cap particles={n_cap_out}")
    print(f"caps present: {n_cap_in > 50 and n_cap_out > 50}")
    print(f"fluid fill radius = {fill_r:.4f} (r-0.5dx)")
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
