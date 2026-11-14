"""Matched-reference ingestion keeps only valid complete scenario alternatives."""

import pytest

from backend.src.trajectory_flow.data import _matched_guidance_rows

pytestmark = pytest.mark.unit


CATALOGUE = [
    {"name": "nominal", "side": 0, "speed_scale": 1.0, "obstacle_index": 0},
    {"name": "left", "side": 1, "speed_scale": 1.0, "obstacle_index": 0},
    {"name": "right", "side": -1, "speed_scale": 1.0, "obstacle_index": 0},
]


def _row(scene: str, guidance: dict, *, obstacle_x: float = 1.0) -> dict:
    return {
        "scene_id": scene,
        "group_id": scene,
        "guidance": guidance,
        "observation": {"start": [0.0, 0.0], "obstacle_x": obstacle_x},
    }


def test_matched_training_keeps_only_complete_scene_triplets():
    rows = [_row("complete", entry) for entry in CATALOGUE]
    rows += [_row("incomplete", entry) for entry in CATALOGUE[:2]]
    selected = _matched_guidance_rows(rows, CATALOGUE)
    assert len(selected) == 3
    assert {row["scene_id"] for row in selected} == {"complete"}
    assert {row["guidance"]["name"] for row in selected} == {
        "nominal", "left", "right"
    }


def test_matched_training_rejects_inconsistent_scene_or_label():
    rows = [_row("complete", entry) for entry in CATALOGUE]
    with pytest.raises(ValueError, match="different scene"):
        _matched_guidance_rows([*rows[:2], _row("complete", CATALOGUE[2], obstacle_x=3.0)], CATALOGUE)
    with pytest.raises(ValueError, match="Duplicate guidance"):
        _matched_guidance_rows([*rows, rows[0]], CATALOGUE)
    incorrect_label = dict(CATALOGUE[1], side=-1)
    with pytest.raises(ValueError, match="differs from frozen"):
        _matched_guidance_rows([rows[0], _row("complete", incorrect_label), rows[2]], CATALOGUE)
    with pytest.raises(ValueError, match="No complete"):
        _matched_guidance_rows(rows[:2], CATALOGUE)
