from __future__ import annotations

"""
Animated particle visualization for SPH CSV snapshots.

Purpose
-------
Visualize a time sequence of exported particle CSV files as an animation.
This is intended as a lightweight debugging / analysis viewer for SPH runs.

Expected CSV columns
--------------------
id,is_boundary,x,y,vx,vy,rho,p,m

What this tool does
-------------------
- Loads multiple CSV snapshot files from a directory
- Splits fluid and boundary particles
- Animates the particle motion over time
- Colors fluid particles by one selected field:
    - velocity magnitude ("vmag")
    - density ("rho")
    - pressure ("p")
- Draws boundary particles in dark gray / black

What this tool does NOT do
--------------------------
- It does not modify solver data
- It does not compute SPH quantities
- It is purely post-processing / visualization

Typical use
-----------
python -m sph.visualization.animate_particles out/csv --color-by vmag
python -m sph.visualization.animate_particles out/csv --color-by rho --interval 80
python -m sph.visualization.animate_particles out/csv --color-by p --save out/anim.gif
python -m sph.visualization.animate_particles out/csv --vectors
"""

import argparse
from pathlib import Path
from typing import Literal

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


ColorMode = Literal["vmag", "rho", "p"]
REQUIRED_COLUMNS = ("id", "is_boundary", "x", "y", "vx", "vy", "rho", "p", "m")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animate SPH particle CSV snapshots.")
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing particle_step_XXXX.csv files",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="particles_step_*.csv",
        help="Glob pattern for snapshot files",
    )
    parser.add_argument(
        "--color-by",
        type=str,
        default="vmag",
        choices=["vmag", "rho", "p"],
        help="Field used to color fluid particles",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=80,
        help="Animation interval in milliseconds",
    )
    parser.add_argument(
        "--fluid-size",
        type=float,
        default=10.0,
        help="Marker size for fluid particles",
    )
    parser.add_argument(
        "--boundary-size",
        type=float,
        default=6.0,
        help="Marker size for boundary particles",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every n-th frame only",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional output path (.gif or .mp4) to save animation",
    )
    parser.add_argument(
        "--vectors",
        action="store_true",
        help="Draw velocity vectors for fluid particles (stride every 5th)",
    )
    parser.add_argument(
        "--vector-stride",
        type=int,
        default=5,
        help="Use every n-th fluid particle for velocity vectors (default: 5)",
    )
    return parser.parse_args()


def _load_csv_snapshot(path: Path) -> np.ndarray:
    data = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )

    if data.size == 0:
        raise ValueError(f"CSV snapshot is empty: {path}")

    if data.shape == ():
        data = np.array([data], dtype=data.dtype)

    if data.dtype.names is None:
        raise ValueError(f"Could not read CSV header from: {path}")

    missing = [c for c in REQUIRED_COLUMNS if c not in data.dtype.names]
    if missing:
        raise ValueError(f"CSV snapshot {path} is missing columns: {missing}")

    return data


def _get_color_values(data: np.ndarray, color_by: ColorMode) -> np.ndarray:
    if color_by == "rho":
        return data["rho"].astype(float)
    if color_by == "p":
        return data["p"].astype(float)

    vx = data["vx"].astype(float)
    vy = data["vy"].astype(float)
    return np.sqrt(vx * vx + vy * vy)


def _discover_files(input_dir: Path, pattern: str, stride: int) -> list[Path]:
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files found in {input_dir} matching pattern {pattern!r}"
        )
    stride = max(1, int(stride))
    return files[::stride]


def _compute_global_ranges(files: list[Path], color_by: ColorMode) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    x_min = np.inf
    x_max = -np.inf
    y_min = np.inf
    y_max = -np.inf
    c_min = np.inf
    c_max = -np.inf

    for path in files:
        data = _load_csv_snapshot(path)

        x = data["x"].astype(float)
        y = data["y"].astype(float)
        c = _get_color_values(data, color_by)

        x_min = min(x_min, float(np.min(x)))
        x_max = max(x_max, float(np.max(x)))
        y_min = min(y_min, float(np.min(y)))
        y_max = max(y_max, float(np.max(y)))
        c_min = min(c_min, float(np.min(c)))
        c_max = max(c_max, float(np.max(c)))

    if not np.isfinite(c_min) or not np.isfinite(c_max):
        c_min, c_max = 0.0, 1.0

    if abs(c_max - c_min) < 1e-12:
        c_max = c_min + 1e-12

    return (x_min, x_max), (y_min, y_max), (c_min, c_max)


def animate_particles(
    input_dir: str,
    pattern: str = "particles_step_*.csv",
    color_by: ColorMode = "vmag",
    interval: int = 80,
    fluid_size: float = 10.0,
    boundary_size: float = 6.0,
    stride: int = 1,
    save: str | None = None,
    vectors: bool = False,
    vector_stride: int = 5,
) -> None:
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    files = _discover_files(input_path, pattern, stride)
    (x_min, x_max), (y_min, y_max), (c_min, c_max) = _compute_global_ranges(files, color_by)

    first = _load_csv_snapshot(files[0])
    fluid_mask = ~first["is_boundary"].astype(bool)
    boundary_mask = first["is_boundary"].astype(bool)

    fig, ax = plt.subplots(figsize=(11, 4.5))

    fluid_x = first["x"].astype(float)[fluid_mask]
    fluid_y = first["y"].astype(float)[fluid_mask]
    fluid_c = _get_color_values(first, color_by)[fluid_mask]

    boundary_x = first["x"].astype(float)[boundary_mask]
    boundary_y = first["y"].astype(float)[boundary_mask]

    boundary_scatter = ax.scatter(
        boundary_x,
        boundary_y,
        c="black",
        s=boundary_size,
        linewidths=0.0,
        alpha=0.9,
    )

    fluid_scatter = ax.scatter(
        fluid_x,
        fluid_y,
        c=fluid_c,
        s=fluid_size,
        cmap="turbo",
        vmin=c_min,
        vmax=c_max,
        linewidths=0.0,
    )

    cbar = fig.colorbar(fluid_scatter, ax=ax)
    if color_by == "vmag":
        cbar.set_label("velocity magnitude")
    elif color_by == "rho":
        cbar.set_label("density")
    else:
        cbar.set_label("pressure")

    margin_x = 0.02 * max(1e-12, x_max - x_min)
    margin_y = 0.05 * max(1e-12, y_max - y_min)

    ax.set_xlim(x_min - margin_x, x_max + margin_x)
    ax.set_ylim(y_min - margin_y, y_max + margin_y)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    title = ax.set_title(f"SPH animation | {files[0].name} | color={color_by}")

    # Velocity vectors (fluid particles only, strided)
    quiver_artist = None
    if vectors:
        v_stride = max(1, int(vector_stride))
        idx = np.where(fluid_mask)[0][::v_stride]
        qx = first["x"].astype(float)[idx]
        qy = first["y"].astype(float)[idx]
        qvx = first["vx"].astype(float)[idx]
        qvy = first["vy"].astype(float)[idx]
        vmag_q = np.sqrt(qvx * qvx + qvy * qvy)
        scale_len = 0.05 * min(x_max - x_min, y_max - y_min)
        scale = scale_len / (np.max(vmag_q) + 1e-12)
        quiver_artist = ax.quiver(qx, qy, qvx * scale, qvy * scale, scale_units="xy", scale=1.0)

    def update(frame_idx: int):
        path = files[frame_idx]
        data = _load_csv_snapshot(path)

        fluid_mask_local = ~data["is_boundary"].astype(bool)
        boundary_mask_local = data["is_boundary"].astype(bool)

        x = data["x"].astype(float)
        y = data["y"].astype(float)
        c = _get_color_values(data, color_by)

        fluid_offsets = np.column_stack([x[fluid_mask_local], y[fluid_mask_local]])
        boundary_offsets = np.column_stack([x[boundary_mask_local], y[boundary_mask_local]])

        fluid_scatter.set_offsets(fluid_offsets)
        fluid_scatter.set_array(c[fluid_mask_local])

        boundary_scatter.set_offsets(boundary_offsets)

        if quiver_artist is not None:
            v_stride = max(1, int(vector_stride))
            idx = np.where(fluid_mask_local)[0][::v_stride]
            qx = x[idx]
            qy = y[idx]
            qvx = data["vx"].astype(float)[idx]
            qvy = data["vy"].astype(float)[idx]
            vmag_q = np.sqrt(qvx * qvx + qvy * qvy)
            scale_len = 0.05 * min(x_max - x_min, y_max - y_min)
            scale = scale_len / (np.max(vmag_q) + 1e-12)
            quiver_artist.set_offsets(np.column_stack([qx, qy]))
            quiver_artist.set_UVC(qvx * scale, qvy * scale)

        title.set_text(f"SPH animation | {path.name} | color={color_by}")
        return fluid_scatter, boundary_scatter, title

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(files),
        interval=interval,
        blit=False,
        repeat=True,
    )

    fig.tight_layout()

    if save is not None:
        save_path = Path(save)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        suffix = save_path.suffix.lower()
        if suffix == ".gif":
            anim.save(save_path, writer="pillow")
        elif suffix == ".mp4":
            anim.save(save_path, writer="ffmpeg")
        else:
            raise ValueError("Save path must end with .gif or .mp4")

        print(f"[ANIM] saved animation to {save_path}")

    plt.show()


def main() -> int:
    args = _parse_args()
    animate_particles(
        input_dir=args.input_dir,
        pattern=args.pattern,
        color_by=args.color_by,
        interval=args.interval,
        fluid_size=args.fluid_size,
        boundary_size=args.boundary_size,
        stride=args.stride,
        save=args.save,
        vectors=args.vectors,
        vector_stride=args.vector_stride,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())