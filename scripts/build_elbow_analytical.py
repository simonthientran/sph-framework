"""Build analytical L-elbow boundary + fluid particle layouts (no STL)."""
from __future__ import annotations

import numpy as np
from pathlib import Path


def sample_cylinder_wall(
    center: np.ndarray,
    axis: int,
    R_inner: float,
    R_outer: float,
    length: float,
    dx: float,
    n_layers: int = 3,
) -> np.ndarray:
    """Sample boundary particles on a cylindrical wall."""
    particles: list[np.ndarray] = []
    r1, r2 = [i for i in range(3) if i != axis]

    n_theta = max(8, int(2 * np.pi * R_outer / dx))
    thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    along = np.arange(-length / 2, length / 2, dx)

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
                particles.append(p)

    if not particles:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(particles, dtype=np.float64)


def main() -> None:
    dx = 0.01
    R_in = 0.05
    R_out = 0.08
    L = 0.40

    arm_h = sample_cylinder_wall(
        center=np.array([L / 2, 0.0, 0.0]),
        axis=0,
        R_inner=R_in,
        R_outer=R_out,
        length=L,
        dx=dx,
    )

    arm_v = sample_cylinder_wall(
        center=np.array([L, L / 2, 0.0]),
        axis=1,
        R_inner=R_in,
        R_outer=R_out,
        length=L,
        dx=dx,
    )

    boundary = np.vstack([arm_h, arm_v])

    fluid_h: list[list[float]] = []
    for x in np.arange(0.01, L - 0.01, dx):
        for y in np.arange(-R_in + dx, R_in - dx, dx):
            for z in np.arange(-R_in + dx, R_in - dx, dx):
                if y * y + z * z < (R_in - dx) ** 2:
                    fluid_h.append([x, y, z])

    fluid_v: list[list[float]] = []
    for y in np.arange(0.01, L - 0.01, dx):
        for x in np.arange(L - R_in + dx, L + R_in - dx, dx):
            for z in np.arange(-R_in + dx, R_in - dx, dx):
                if (x - L) ** 2 + z * z < (R_in - dx) ** 2:
                    fluid_v.append([x, y, z])

    fluid = np.vstack([np.asarray(fluid_h), np.asarray(fluid_v)])

    print(f"Boundary particles: {len(boundary)}")
    print(f"Fluid particles:    {len(fluid)}")
    print(f"Boundary x: {boundary[:, 0].min():.2f} .. {boundary[:, 0].max():.2f}")
    print(f"Boundary y: {boundary[:, 1].min():.2f} .. {boundary[:, 1].max():.2f}")

    out_dir = Path(__file__).parent.parent / "assets" / "elbow_analytical"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "boundary.npy", boundary)
    np.save(out_dir / "fluid.npy", fluid)
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
