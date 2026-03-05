from __future__ import annotations

import json
from pathlib import Path

from sph.verification.harness import run_verification


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_scene(scene_rel: str, *, fast: bool = False) -> tuple[bool, dict]:
    root = Path(__file__).resolve().parent.parent
    scene_path = root / scene_rel
    scene = _load_json(scene_path)
    return run_verification(scene, scene_path=str(scene_path), report_path=None, fast=fast)


def _assert_in_ranges(report: dict, baseline: dict) -> None:
    summary = report["summary"]
    ranges = baseline["summary_ranges"]
    for key, bounds in ranges.items():
        lo, hi = float(bounds[0]), float(bounds[1])
        v = float(summary[key])
        assert lo <= v <= hi, f"{key}={v} out of range [{lo}, {hi}]"


def test_verify_hydrostatic_scene_passes_and_matches_baseline():
    passed, report = _run_scene("scenes/verification/hydrostatic_2d.json", fast=True)
    assert passed, f"verification failed checks={report.get('checks')}"
    assert report["overall_pass"] is True
    assert report["gates"]["vmax_max"]["pass"] is True
    assert report["gates"]["density_rel_err_mean_max"]["pass"] is True
    assert report["gates"]["density_rel_err_max_max"]["pass"] is True
    assert report["gates"]["pressure_abs_max"]["pass"] is True
    baseline = _load_json(Path(__file__).resolve().parent / "baselines" / "hydrostatic_2d.json")
    _assert_in_ranges(report, baseline)


def test_verify_poiseuille_scene_passes_and_matches_baseline():
    passed, report = _run_scene("scenes/verification/poiseuille_2d.json", fast=True)
    assert passed, f"verification failed checks={report.get('checks')}"
    assert report["overall_pass"] is True
    assert report["gates"]["poiseuille_empty_bins_max"]["pass"] is True
    assert report["gates"]["poiseuille_l2_max"]["pass"] is True
    assert report["gates"]["poiseuille_linf_max"]["pass"] is True
    assert report["gates"]["poiseuille_vmax_ratio"]["pass"] is True
    baseline = _load_json(Path(__file__).resolve().parent / "baselines" / "poiseuille_2d.json")
    _assert_in_ranges(report, baseline)

