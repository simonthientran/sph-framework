from __future__ import annotations

from pathlib import Path

import pytest

from sph.scene.schema import load_scene_dict


def test_splishsplash_3d_scene_emits_explicit_neighbor_support_radius():
    normalized = load_scene_dict(Path("scenes/dam_break_splishsplash.json"))

    assert normalized["meta"]["dimensions"] == 3
    assert normalized["fluids"][0]["spacing"] == 0.025
    assert normalized["neighbors"]["type"] == "kdtree"
    assert normalized["neighbors"]["support_radius"] == pytest.approx(0.09)
    assert normalized["fluids"][0]["min"][2] == pytest.approx(0.05)
    assert normalized["fluids"][0]["max"][2] == pytest.approx(0.35)
