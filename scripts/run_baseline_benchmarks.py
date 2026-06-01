#!/usr/bin/env python3
"""Run frozen single-phase benchmark presets."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from sph.validation.baseline_registry import BASELINE_RUN_PRESETS, build_preset_command


def _with_pythonpath(env: dict[str, str]) -> dict[str, str]:
    updated = dict(env)
    current = updated.get("PYTHONPATH", "").strip()
    updated["PYTHONPATH"] = str(SRC) if not current else f"{SRC}:{current}"
    return updated


def _build_command(
    preset_name: str,
    python_bin: str,
    steps: int | None,
    warmup: int | None,
    log_every: int | None,
) -> tuple[list[str], dict[str, str], str]:
    env = _with_pythonpath(os.environ)
    cmd, env, preset = build_preset_command(
        repo_root=ROOT,
        python_bin=python_bin,
        preset_name=preset_name,
        steps=steps,
        warmup=warmup,
        log_every=log_every,
        base_env=env,
    )
    return cmd, env, preset.description


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen single-phase benchmark presets.")
    parser.add_argument("preset", choices=sorted(BASELINE_RUN_PRESETS.keys()), help="Benchmark preset to run")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to use")
    parser.add_argument("--steps", type=int, default=None, help="Override steady-state steps")
    parser.add_argument("--warmup", type=int, default=None, help="Override warmup steps for profile presets")
    parser.add_argument("--log-every", type=int, default=None, help="Override log frequency for bench presets")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without running it")
    args = parser.parse_args()

    cmd, env, description = _build_command(
        preset_name=args.preset,
        python_bin=args.python,
        steps=args.steps,
        warmup=args.warmup,
        log_every=args.log_every,
    )

    print(f"Preset      : {args.preset}")
    print(f"Description : {description}")
    print(f"Command     : {' '.join(cmd)}")
    if args.dry_run:
        return

    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


if __name__ == "__main__":
    main()
