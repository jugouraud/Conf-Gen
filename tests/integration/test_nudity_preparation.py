"""The retained detection stage freezes only the declared five-category design."""

import csv

import pytest

from backend.experiments.five_category_nudity_detection_20261101.definition import (
    CATEGORIES,
)
from backend.experiments.five_category_nudity_detection_20261101.package import (
    run_detection_packaging,
)
from backend.experiments.five_category_nudity_detection_20261101.prepare import (
    run_nudity_detection_preparation,
)
from backend.src.common.storage import read_json
from backend.src.detection.contracts import read_resident_detection_inputs

pytestmark = pytest.mark.integration


def nudity_input(tmp_path, *, wrong_category=False):
    dataset = tmp_path / "pairs.csv"
    with dataset.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "pair_id",
                "scenario_id",
                "category",
                "split",
                "variant_id",
                "control_prompt",
                "target_prompt",
            ],
        )
        writer.writeheader()
        for category in CATEGORIES:
            for index in range(200):
                identifier = f"{category}-{index}"
                writer.writerow(
                    {
                        "pair_id": identifier,
                        "scenario_id": identifier,
                        "category": "unexpected"
                        if wrong_category and category == CATEGORIES[0]
                        else category,
                        "split": "train"
                        if index < 140
                        else "validation"
                        if index < 160
                        else "test",
                        "variant_id": 1,
                        "control_prompt": f"clothed {identifier}",
                        "target_prompt": f"nude {identifier}",
                    }
                )
    config = tmp_path / "config.toml"
    config.write_text(
        f'''[paths]
dataset = "{dataset.as_posix()}"
[dataset.prompt_columns]
safe = "control_prompt"
unsafe = "target_prompt"
[model]
height = 512
width = 512
[probes]
time_points = 8
trajectories = 8
''',
        encoding="utf-8",
    )
    return config


def test_nudity_stage_freezes_strata_and_packages_matching_implementation(tmp_path):
    config = nudity_input(tmp_path)
    output = tmp_path / "prepared"
    assert run_nudity_detection_preparation(config, output) == {
        "safe_reference": 700,
        "unsafe_reference": 700,
        "safe_validation": 100,
        "unsafe_validation": 100,
        "safe_test": 200,
        "unsafe_test": 200,
    }
    manifest, groups = read_resident_detection_inputs(output)
    assert len(groups["safe_reference"]) == 700
    assert len(manifest["variants"]["trajectory"]) == 3
    assert read_json(output / "resource_plan.json")["gpu_started"] is False
    package = run_detection_packaging(output)
    import zipfile

    with zipfile.ZipFile(package) as archive:
        assert "backend/src/fields/resident.py" in archive.namelist()
        assert "results/trajectory/splits/unsafe_test.jsonl" in archive.namelist()
    before = (output / "experiment.json").read_bytes()
    with pytest.raises(ValueError, match="new output directory"):
        run_nudity_detection_preparation(config, output)
    assert (output / "experiment.json").read_bytes() == before


def test_nudity_stage_rejects_wrong_category_before_writing_output(tmp_path):
    config = nudity_input(tmp_path, wrong_category=True)
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="five categories"):
        run_nudity_detection_preparation(config, output)
    assert not output.exists()
