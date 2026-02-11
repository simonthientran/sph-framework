from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from sph.core.state_builder import build_fluid_block
from sph.neighbors.spatial_hash import SpatialHash
from sph.sph.density import compute_density_summation


def main() -> int:
    print("[BOOT] bootstrap started")

    if len(sys.argv) < 2:
        print("Usage: python -m sph.core.bootstrap <scene.json>")
        return 2

    scene_path = Path(sys.argv[1]).resolve()
    print(f"[BOOT] loading scene: {scene_path}")

    if not scene_path.exists():
        print("[ERROR] scene file not found")
        return 1

    # -----------------------------
    # Load scene
    # -----------------------------
    with scene_path.open("r", encoding="utf-8") as f:
        scene = json.load(f)

    print("[BOOT] scene loaded successfully")

    # -----------------------------
    # Build particle state
    # -----------------------------
    state = build_fluid_block(scene)

    print(f"[BOOT] particles: N={state.n}, dim={state.dim}")
    print(f"[BOOT] pos min: {state.pos.min(axis=0)}")
    print(f"[BOOT] pos max: {state.pos.max(axis=0)}")

    # -----------------------------
    # Neighbor search
    # -----------------------------
    h = float(scene["neighbors"]["support_radius"])
    search = SpatialHash(support_radius=h, dim=state.dim)
    search.build(state.pos)

    neighbor_counts = []

    for i in range(state.n):
        nbs = search.query(i, state.pos)
        neighbor_counts.append(len(nbs))

    neighbor_counts = np.array(neighbor_counts)

    print("[BOOT] neighbor stats:")
    print(f"        min: {neighbor_counts.min()}")
    print(f"        avg: {neighbor_counts.mean():.2f}")
    print(f"        max: {neighbor_counts.max()}")

    # -----------------------------
    # Density reconstruction
    # -----------------------------
    rho0 = float(scene["material"]["rho0"])

    rho = compute_density_summation(
        state=state,
        neighbor_search=search,
        h=h
    )

    state.rho[:] = rho

    rel_err = (rho - rho0) / rho0

    print("[BOOT] density stats:")
    print(f"        rho min/avg/max: {rho.min():.3f} / {rho.mean():.3f} / {rho.max():.3f}")
    print(f"        rel err min/avg/max: {rel_err.min():.3%} / {rel_err.mean():.3%} / {rel_err.max():.3%}")
    print("        note: density underestimation near free surfaces is expected.")

    print("[BOOT] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
