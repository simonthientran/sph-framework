#!/usr/bin/env python3
"""
CPU-vs-CUDA validation CLI.

Runs the same scene on both backends and compares final state.
Exits 0 on PASS, 1 on FAIL, 2 on error.

Usage
-----
    # Single-phase benchmark (default, 50 steps)
    NUMBA_ENABLE_CUDASIM=0 PYTHONPATH=src python scripts/validate_cpu_cuda.py

    # Explicit scene and step count
    NUMBA_ENABLE_CUDASIM=0 PYTHONPATH=src python scripts/validate_cpu_cuda.py \\
        --scene scenes/examples/pipe_flow_2d.json --steps 100

    # CPU-only validation (no CUDA device required)
    PYTHONPATH=src python scripts/validate_cpu_cuda.py --cpu-only --steps 50

    # JSON output for CI / log capture
    NUMBA_ENABLE_CUDASIM=0 PYTHONPATH=src python scripts/validate_cpu_cuda.py \\
        --json --steps 50 > validation_result.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sph.validation.compare import cpu_only_validate, run_comparison


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_SCENE = ROOT / "scenes/examples/pipe_flow_2d.json"
DEFAULT_STEPS = 50
DEFAULT_WARMUP = 0

# Tolerances (can be tightened; these match the well-validated single-phase path)
DEFAULT_RHO_TOL = 0.005       # 0.5% relative for scalar density metrics
DEFAULT_VEL_TOL = 0.01        # 1.0% relative for velocity_max
DEFAULT_DENSITY_ARR_TOL = 0.01  # 1.0% L2-relative for density array
DEFAULT_VEL_ARR_TOL = 0.02    # 2.0% L2-relative for velocity-magnitude array


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_cuda() -> bool:
    """Check whether a real CUDA device is available."""
    try:
        from numba import cuda
        return bool(cuda.is_available())
    except Exception:
        return False


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CPU-vs-CUDA SPH validation tool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--scene",
        type=Path,
        default=DEFAULT_SCENE,
        help="Path to scene JSON file.",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help="Number of simulation steps to compare.",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help="Warmup steps excluded from timing (JIT etc.).",
    )
    p.add_argument(
        "--rho-tol",
        type=float,
        default=DEFAULT_RHO_TOL,
        metavar="REL",
        help="Relative tolerance for scalar density checks.",
    )
    p.add_argument(
        "--vel-tol",
        type=float,
        default=DEFAULT_VEL_TOL,
        metavar="REL",
        help="Relative tolerance for velocity_max check.",
    )
    p.add_argument(
        "--density-arr-tol",
        type=float,
        default=DEFAULT_DENSITY_ARR_TOL,
        metavar="L2",
        help="L2-relative tolerance for density array comparison.",
    )
    p.add_argument(
        "--vel-arr-tol",
        type=float,
        default=DEFAULT_VEL_ARR_TOL,
        metavar="L2",
        help="L2-relative tolerance for velocity-magnitude array comparison.",
    )
    p.add_argument(
        "--cpu-only",
        action="store_true",
        help="Skip CUDA run; validate CPU backend output only.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Print result as JSON (stdout) instead of human text.",
    )
    p.add_argument(
        "--no-cudasim-guard",
        action="store_true",
        help="Allow running under NUMBA_ENABLE_CUDASIM=1 (timings will be meaningless).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()

    scene = args.scene.resolve()
    if not scene.exists():
        print(f"ERROR: scene file not found: {scene}", file=sys.stderr)
        return 2

    # CUDASIM guard: meaningful CPU-vs-CUDA comparison requires real hardware.
    cudasim = os.environ.get("NUMBA_ENABLE_CUDASIM", "").strip() == "1"
    if cudasim and not args.cpu_only and not args.no_cudasim_guard:
        print(
            "WARNING: NUMBA_ENABLE_CUDASIM=1 detected.  Timing will not reflect real GPU.\n"
            "Use --no-cudasim-guard to run anyway, or --cpu-only to skip CUDA.",
            file=sys.stderr,
        )
        # Don't abort — CUDASIM still exercises the code path.

    # --- CPU-only path -------------------------------------------------------
    if args.cpu_only:
        result = cpu_only_validate(
            scene_path=scene,
            steps=args.steps,
            rho_tol_rel=args.rho_tol,
        )
        if args.json_output:
            payload = {
                "mode": "cpu_only",
                "scene": str(scene),
                "steps": args.steps,
                "passed": result["passed"],
                "summary": result["summary"],
            }
            rt = result.get("runtime")
            if rt is not None:
                payload["final_step"] = {
                    "rho_mean": float(rt.rho_mean),
                    "rho_error_mean": float(rt.rho_error_mean),
                    "velocity_max": float(rt.velocity_max),
                    "neighbor_min": int(rt.neighbor_min),
                    "stability": str(rt.stability),
                    "health": str(rt.solver_health_summary),
                }
            print(json.dumps(payload, indent=2))
        else:
            print(result["summary"])
        return 0 if result["passed"] else 1

    # --- CPU-vs-CUDA comparison ----------------------------------------------
    if not _detect_cuda() and not cudasim:
        print(
            "ERROR: No CUDA device detected.  Use --cpu-only for non-GPU environments.",
            file=sys.stderr,
        )
        return 2

    if not args.json_output:
        print(
            f"Running CPU-vs-CUDA validation: {scene.name}  "
            f"steps={args.steps}  warmup={args.warmup}"
        )

    try:
        comparison = run_comparison(
            scene_path=scene,
            steps=args.steps,
            warmup_steps=args.warmup,
            rho_tol_rel=args.rho_tol,
            velocity_tol_rel=args.vel_tol,
            density_array_l2_tol=args.density_arr_tol,
            velocity_array_l2_tol=args.vel_arr_tol,
        )
    except Exception as exc:
        print(f"ERROR: Comparison run failed: {exc}", file=sys.stderr)
        return 2

    if args.json_output:
        print(json.dumps(comparison.to_dict(), indent=2))
    else:
        for line in comparison.summary_lines():
            print(line)

    return 0 if comparison.overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
