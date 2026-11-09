import pytest

from backend.src.common.storage import write_json
from backend.src.data.prompts import select_unsafe_prompts

pytestmark = pytest.mark.integration


def test_unsafe_partitions_exclude_safe_duplicates_and_never_reuse_a_prompt(tmp_path):
    dataset = tmp_path / "prompts.json"
    write_json(
        dataset,
        [
            {"prompt": "Already SAFE!", "label": {"intimate_bedroom_scene--test": 1}},
            {"prompt": "Scene one", "label": {"intimate_bedroom_scene--test": 1}},
            {"prompt": "SCENE one!", "label": {"intimate_bedroom_scene--test": 1}},
            {"prompt": "Scene two", "label": {"figure_study_nudity--test": 1}},
            {"prompt": "Scene three", "label": {"fine_art_nude_photography--test": 1}},
            {"prompt": "Unlabeled scene", "label": {}},
        ],
    )
    safe_rows = {"safe_reference": [{"prompt": "already safe"}]}
    settings = {
        "seed": 17,
        "unsafe_anchors": 1,
        "unsafe_test": 1,
        "unsafe_calibration": 1,
    }

    groups = select_unsafe_prompts(dataset, safe_rows, settings)

    (anchor,) = groups["unsafe_reference"]
    (calibration,) = groups["unsafe_validation"]
    (held_out,) = groups["unsafe_test"]
    assert len({anchor["id"], calibration["id"], held_out["id"]}) == 3
    assert {anchor["prompt"], calibration["prompt"], held_out["prompt"]} == {
        "Scene one",
        "Scene two",
        "Scene three",
    }
    assert anchor["eligible_for_region"] is True
    assert calibration["eligible_for_region"] is True
    assert held_out["eligible_for_region"] is False
    assert held_out["evaluation_only"] is True
    assert select_unsafe_prompts(dataset, safe_rows, settings) == groups


def test_insufficient_distinct_prompts_fail_without_writing_partitions(tmp_path):
    dataset = tmp_path / "prompts.json"
    write_json(
        dataset, [{"prompt": "Scene one", "label": {"intimate_bedroom_scene--test": 1}}]
    )

    with pytest.raises(ValueError, match="Insufficient distinct prompts"):
        select_unsafe_prompts(
            dataset,
            {},
            {
                "seed": 17,
                "unsafe_anchors": 1,
                "unsafe_test": 1,
                "unsafe_calibration": 1,
            },
        )

    assert set(tmp_path.iterdir()) == {dataset}
