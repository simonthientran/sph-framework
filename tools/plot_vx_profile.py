from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _parse_scalar(value: str) -> object:
    v = value.strip()
    lower = v.lower()
    if lower == "none":
        return None
    if lower == "nan":
        return float("nan")
    if lower in ("true", "false"):
        return lower == "true"
    try:
        if any(ch in v for ch in (".", "e", "E")):
            return float(v)
        return int(v)
    except ValueError:
        return v


def read_profile_csv(path: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    metadata: dict[str, object] = {}
    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    table_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("# "):
            payload = stripped[2:]
            if "=" in payload:
                k, v = payload.split("=", 1)
                metadata[k.strip()] = _parse_scalar(v)
            continue
        if stripped:
            table_start = i
            break

    required_meta = ("y0_eff", "H_eff", "gx", "nu", "step")
    missing = [k for k in required_meta if k not in metadata]
    if missing:
        raise ValueError(f"Missing required metadata keys in header: {missing}")

    reader = csv.DictReader(lines[table_start:])
    rows = list(reader)
    if not rows:
        raise ValueError("No profile rows found in CSV table.")

    required_cols = ("y_center", "vx_mean", "vx_count")
    missing_cols = [c for c in required_cols if c not in reader.fieldnames]
    if missing_cols:
        raise ValueError(f"Missing required table columns: {missing_cols}")

    y_center = np.array([float(r["y_center"]) for r in rows], dtype=np.float64)
    vx_count = np.array([int(r["vx_count"]) for r in rows], dtype=np.int64)
    vx_mean = np.array(
        [float(r["vx_mean"]) if str(r["vx_mean"]).strip() != "" else np.nan for r in rows],
        dtype=np.float64,
    )
    data = {"y_center": y_center, "vx_mean": vx_mean, "vx_count": vx_count}
    return metadata, data


def analytic_profile(y: np.ndarray, y0_eff: float, H_eff: float, gx: float, nu: float) -> np.ndarray:
    y_prime = y - y0_eff
    return (gx / (2.0 * nu)) * y_prime * (H_eff - y_prime)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot measured vx profile against analytic Poiseuille profile.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("out/pipe_flow_2d/vx_profile_bins.csv"),
        help="Input vx-profile CSV with metadata header.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/pipe_flow_2d/vx_profile.png"),
        help="Output PNG path.",
    )
    args = parser.parse_args()

    in_path = args.input
    out_path = args.output
    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    metadata, data = read_profile_csv(in_path)
    y0_eff = float(metadata["y0_eff"])
    H_eff = float(metadata["H_eff"])
    gx = float(metadata["gx"])
    nu = float(metadata["nu"])
    step = int(metadata["step"])
    vmax_measured = metadata.get("vmax_measured", float("nan"))
    vmax_analytic = metadata.get("vmax_analytic", float("nan"))
    l2 = metadata.get("L2", float("nan"))
    linf = metadata.get("Linf", float("nan"))

    y_dense = np.linspace(y0_eff, y0_eff + H_eff, 400, dtype=np.float64)
    u_dense = analytic_profile(y_dense, y0_eff, H_eff, gx, nu)
    u_bins = analytic_profile(data["y_center"], y0_eff, H_eff, gx, nu)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=120)
    ax.plot(y_dense, u_dense, "-", linewidth=2.0, label="analytic")
    ax.plot(data["y_center"], data["vx_mean"], "o-", linewidth=1.5, markersize=4.0, label="measured")
    ax.set_xlabel("y")
    ax.set_ylabel("vx")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(
        "Pipe-flow vx-profile\n"
        f"step={step} vmax_measured={vmax_measured:.6g} vmax_analytic={vmax_analytic:.6g} "
        f"L2={l2:.3e} Linf={linf:.3e}"
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    finite_meas = np.isfinite(data["vx_mean"])
    used_bins = int(np.count_nonzero((data["vx_count"] > 0) & finite_meas))
    print(
        f"[VX_PLOT] input={in_path} output={out_path} "
        f"step={step} bins={len(data['y_center'])} used_bins={used_bins} "
        f"vmax_measured={vmax_measured} vmax_analytic={vmax_analytic} L2={l2} Linf={linf}"
    )
    # Keep u_bins computed to validate reconstruction path in script runtime.
    if not np.all(np.isfinite(u_bins)):
        raise ValueError("Analytic profile evaluation produced non-finite values.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
