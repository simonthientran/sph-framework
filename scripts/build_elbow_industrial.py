"""
Build industrial L-elbow boundary + fluid particle layouts.

dx=0.015, R_in=0.060 (4 particle layers), L=0.30m arm length.
Fluid uses boolean subtraction to eliminate overlap at bend corner.
Zero near-duplicate pairs guaranteed.

Physics:
  gx=0.75 m/s², nu=0.001 → Re_ss=40.5 (above Dean threshold De_crit=11.6)
  tau_visc = R²/nu = 3.6 s
  Peak Re≈77 observed at t≈1.2 s (step 2500 at dt≈0.00085 s)
"""
from __future__ import annotations

import numpy as np
from pathlib import Path


def cylinder_wall(
    center: np.ndarray,
    axis: int,
    R_inner: float,
    R_outer: float,
    length: float,
    dx: float,
    n_layers: int = 2,
) -> np.ndarray:
    r1, r2 = [i for i in range(3) if i != axis]
    n_theta = max(12, int(2 * np.pi * R_outer / dx))
    thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    along = np.arange(-length / 2, length / 2, dx)
    pts: list[np.ndarray] = []
    for layer in range(n_layers):
        R = R_inner + (layer + 0.5) * dx
        if R > R_outer:
            break
        for theta in thetas:
            for s in along:
                p = center.astype(np.float64).copy()
                p[axis] += s
                p[r1] += R * np.cos(theta)
                p[r2] += R * np.sin(theta)
                pts.append(p)
    return np.asarray(pts, dtype=np.float64) if pts else np.zeros((0, 3), dtype=np.float64)


def main() -> None:
    dx = 0.015
    R_in = 0.060    # 4 particle layers across radius
    R_out = 0.090   # 2 boundary layers
    L = 0.30

    # --- Boundary: cylindrical walls ---
    arm_h = cylinder_wall(np.array([L / 2, 0.0, 0.0]), axis=0,
                          R_inner=R_in, R_outer=R_out, length=L, dx=dx)
    arm_v = cylinder_wall(np.array([L, L / 2, 0.0]), axis=1,
                          R_inner=R_in, R_outer=R_out, length=L, dx=dx)
    boundary = np.vstack([arm_h, arm_v])

    # --- Fluid: boolean subtraction avoids overlap at bend corner ---
    fluid_h: list[list[float]] = []
    for x in np.arange(dx, L, dx):
        for y in np.arange(-(R_in - dx), R_in, dx):
            for z in np.arange(-(R_in - dx), R_in, dx):
                if y * y + z * z >= (R_in - dx) ** 2:
                    continue
                if (x - L) ** 2 + z * z < (R_in - dx) ** 2:
                    continue  # inside vertical arm → skip
                fluid_h.append([x, y, z])

    fluid_v: list[list[float]] = []
    for y in np.arange(dx, L, dx):
        for x in np.arange(L - (R_in - dx), L + (R_in - dx) + dx / 2, dx):
            for z in np.arange(-(R_in - dx), R_in, dx):
                if (x - L) ** 2 + z * z >= (R_in - dx) ** 2:
                    continue
                if y * y + z * z < (R_in - dx) ** 2:
                    continue  # inside horizontal arm → skip
                fluid_v.append([x, y, z])

    fluid = np.vstack([np.asarray(fluid_h, dtype=np.float64),
                       np.asarray(fluid_v, dtype=np.float64)])

    # Sanity check
    from scipy.spatial import cKDTree
    tree = cKDTree(fluid)
    n_dups = len(tree.query_pairs(0.002))
    assert n_dups == 0, f"Found {n_dups} near-duplicate pairs!"

    out_dir = Path(__file__).parent.parent / "assets" / "elbow_industrial"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "fluid.npy", fluid)
    np.save(out_dir / "boundary.npy", boundary)

    print(f"dx={dx} R_in={R_in} R_out={R_out} L={L}")
    print(f"Fluid:    {len(fluid)} particles  (h={len(fluid_h)}, v={len(fluid_v)})")
    print(f"Boundary: {len(boundary)} particles")
    print(f"Near-duplicate pairs: {n_dups}")
    h = fluid[:, 0] < L / 2
    v = fluid[:, 1] > L / 2
    print(f"Horizontal arm (x<{L/2:.2f}): {h.sum()}")
    print(f"Vertical arm   (y>{L/2:.2f}): {v.sum()}")
    print(f"Saved to {out_dir}/")

    gx, nu = 0.75, 0.001
    vmean = gx * R_in ** 2 / (8 * nu)
    Re = vmean * 2 * R_in / nu
    tau = R_in ** 2 / nu
    print(f"\nPhysics: gx={gx} nu={nu}")
    print(f"  v_mean_ss = {vmean:.4f} m/s   Re_ss = {Re:.1f}   tau_visc = {tau:.2f} s")


if __name__ == "__main__":
    main()
