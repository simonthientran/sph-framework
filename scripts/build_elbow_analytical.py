"""Build a smooth curved (torus-bend) pipe elbow: boundary + fluid layouts.

The two straight arms are joined by a 90 degree TORUS BEND of centerline
radius R_bend = 1.5 * D, so the outer wall of the bend is a smooth arc and
the cross-section stays circular all the way through the curve.

Method (centerline sweep)
-------------------------
1. Build the centerline as straight-arm + quarter-circle arc + straight-arm.
2. At each centerline point take an orthonormal frame (t, n1, n2):
     t  = unit tangent
     n2 = (0,0,1)              (bend is planar in xy, so z is always normal)
     n1 = normalize(t x n2)    (in-plane normal)
   Fill the circular cross-section in the (n1, n2) plane.
3. Deduplicate at the segment joins.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from sph.core.state_builder import deduplicate_positions  # noqa: E402


# ── Geometry ────────────────────────────────────────────────────────────────
R_PIPE = 0.045            # pipe inner radius (D = 0.09)
D = 2 * R_PIPE
R_BEND = 1.5 * D          # = 0.135 centerline curvature radius
L_H = 0.30               # horizontal straight arm length
L_V = 0.30               # vertical straight arm length
DX = 0.012               # particle spacing
N_LAYERS = 3


def _frame(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal cross-section normals (n1, n2) for tangent t."""
    n2 = np.array([0.0, 0.0, 1.0])
    n1 = np.cross(t, n2)
    nrm = np.linalg.norm(n1)
    if nrm < 1e-9:                      # tangent parallel to z (not used here)
        n1 = np.array([1.0, 0.0, 0.0])
        nrm = 1.0
    n1 = n1 / nrm
    return n1, n2


def _disk_offsets(fill_radius: float, dx: float) -> np.ndarray:
    """(k,2) array of (a,b) offsets filling a disk of radius fill_radius.

    The grid is symmetric about 0 (coords = {-n*dx ... +n*dx}) so the swept
    cross-section is circular even when fill_radius is not a multiple of dx.
    """
    n = int(np.ceil(fill_radius / dx))
    coords = np.arange(-n, n + 1) * dx
    a, b = np.meshgrid(coords, coords, indexing="ij")
    a = a.ravel(); b = b.ravel()
    mask = a * a + b * b < fill_radius * fill_radius
    return np.column_stack([a[mask], b[mask]])


def _ring_offsets(dx: float, n_layers: int) -> np.ndarray:
    """(k,2) offsets for n_layers boundary rings at radii r, r+dx, r+2dx."""
    rows: list[np.ndarray] = []
    for layer in range(n_layers):
        R = R_PIPE + layer * dx
        n_theta = max(8, int(round(2 * np.pi * R / dx)))
        th = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
        rows.append(np.column_stack([R * np.cos(th), R * np.sin(th)]))
    return np.vstack(rows)


def _centerline() -> list[tuple[np.ndarray, np.ndarray]]:
    """List of (point, unit_tangent) along the curved centerline."""
    pts: list[tuple[np.ndarray, np.ndarray]] = []

    # Horizontal arm: centerline at y = R_BEND, flowing +x.
    for x in np.arange(0.0, L_H + DX / 2, DX):
        pts.append((np.array([x, R_BEND, 0.0]), np.array([1.0, 0.0, 0.0])))

    # 90 deg torus arc, centered above the horizontal-arm end.
    center = np.array([L_H, 2 * R_BEND, 0.0])
    arc_len = R_BEND * (np.pi / 2)
    n_arc = max(1, int(round(arc_len / DX)))
    for k in range(1, n_arc + 1):                 # skip theta=0 (arm-end dup)
        theta = (np.pi / 2) * k / n_arc
        pt = center + R_BEND * np.array([np.sin(theta), -np.cos(theta), 0.0])
        t = np.array([np.cos(theta), np.sin(theta), 0.0])
        pts.append((pt, t))

    # Vertical arm: centerline at x = L_H + R_BEND, flowing +y.
    x_v = L_H + R_BEND
    y0 = 2 * R_BEND
    for y in np.arange(y0 + DX, y0 + L_V + DX / 2, DX):
        pts.append((np.array([x_v, y, 0.0]), np.array([0.0, 1.0, 0.0])))

    return pts


def _sweep(offsets2d: np.ndarray) -> np.ndarray:
    """Sweep a 2D cross-section pattern along the centerline frame."""
    out: list[np.ndarray] = []
    for pt, t in _centerline():
        n1, n2 = _frame(t)
        ring = pt[None, :] + offsets2d[:, 0:1] * n1[None, :] + offsets2d[:, 1:2] * n2[None, :]
        out.append(ring)
    return np.vstack(out)


def _round_ratio(particles: np.ndarray, center: np.ndarray, t: np.ndarray) -> tuple[float, int]:
    """Cross-section roundness near a centerline point: min/max extent in the
    (n1, n2) frame for particles within dx of the plane through center."""
    n1, n2 = _frame(t)
    rel = particles - center[None, :]
    along = rel @ t
    slab = np.abs(along) < 0.6 * DX
    if slab.sum() < 6:
        return float("nan"), int(slab.sum())
    s = rel[slab]
    a = s @ n1
    b = s @ n2
    ext_a = a.max() - a.min()
    ext_b = b.max() - b.min()
    return float(min(ext_a, ext_b) / max(ext_a, ext_b)), int(slab.sum())


def main() -> None:
    fill_r = R_PIPE - DX
    fluid = _sweep(_disk_offsets(fill_r, DX))
    boundary = _sweep(_ring_offsets(DX, N_LAYERS))

    fluid = deduplicate_positions(fluid, DX)
    boundary = deduplicate_positions(boundary, DX)

    out_dir = ROOT / "assets" / "elbow_analytical"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "fluid.npy", fluid)
    np.save(out_dir / "boundary.npy", boundary)

    print(f"Geometry: r={R_PIPE} D={D} R_bend={R_BEND} (1.5*D) dx={DX}")
    print(f"n_fluid    = {len(fluid)}")
    print(f"n_boundary = {len(boundary)}")
    print(f"fluid bbox x=[{fluid[:,0].min():.3f},{fluid[:,0].max():.3f}] "
          f"y=[{fluid[:,1].min():.3f},{fluid[:,1].max():.3f}] "
          f"z=[{fluid[:,2].min():.3f},{fluid[:,2].max():.3f}]")

    # Round-ratio checks at the three key cross-sections. Snap each target to
    # the nearest actual centerline ring so the measurement slab is centred.
    cline = _centerline()
    cl_pts = np.array([p for p, _ in cline])
    targets = [
        ("mid horizontal arm", np.array([0.15, R_BEND, 0.0])),
        ("mid bend (45 deg)",
         np.array([L_H, 2 * R_BEND, 0.0]) +
         R_BEND * np.array([np.sin(np.pi / 4), -np.cos(np.pi / 4), 0.0])),
        ("mid vertical arm", np.array([L_H + R_BEND, 2 * R_BEND + 0.15, 0.0])),
    ]
    print("Round-ratio (1.0 = perfectly circular):")
    for name, target in targets:
        idx = int(np.argmin(np.linalg.norm(cl_pts - target[None, :], axis=1)))
        c, t = cline[idx]
        ratio, npart = _round_ratio(fluid, c, t)
        print(f"  {name:22s} ratio={ratio:.3f}  (n_slab={npart})")

    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
