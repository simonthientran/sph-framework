#!/usr/bin/env python3
"""Boundary-representation validation CLI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sph.validation.boundary import validate_boundary_representations


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate boundary representations and preprocessing outputs for a scene.",
    )
    parser.add_argument(
        "--scene",
        type=Path,
        default=ROOT / "scenes/examples/box_fill_3d.json",
        help="Path to scene JSON file.",
    )
    parser.add_argument(
        "--with-sdf",
        action="store_true",
        help="Force SDF representation building in addition to current runtime particle representation.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Print structured JSON output instead of human-readable lines.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    scene = args.scene.resolve()
    if not scene.exists():
        print(f"ERROR: scene file not found: {scene}", file=sys.stderr)
        return 2

    requested = ("particles", "sdf") if args.with_sdf else ()
    try:
        result = validate_boundary_representations(
            scene,
            requested_representations=requested,
        )
    except Exception as exc:
        print(f"ERROR: boundary validation failed: {exc}", file=sys.stderr)
        return 2

    if args.json_output:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        for line in result.summary_lines:
            print(line)

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
