"""Full real optimization, subprocess parallelism, replay, resume and dataset audit."""

import json
from dataclasses import replace

import numpy as np
import pytest

from backend.experiments.expert_trajectories.pipeline import main
from backend.src.expert.audit import read_jsonl, run_dataset_validation
from backend.src.expert.pipeline import run_expert_pipeline
from backend.src.expert.storage import publish_record, read_record

pytestmark = pytest.mark.e2e


def test_expert_dataset_reproduces_across_workers_resumes_and_rejects_corruption(
    tmp_path, expert_config, expert_scenes, capsys
):
    config = replace(
        expert_config,
        planner=replace(expert_config.planner, initial_guess_offsets=(0.0,)),
    )
    first, second = tmp_path / "first", tmp_path / "second"
    report = run_expert_pipeline(config, expert_scenes, first, workers=1)
    assert report["status"] == "complete"
    assert report["metrics"]["validated_targets"] == 4
    again = run_expert_pipeline(config, expert_scenes, second, workers=2)
    assert again["dataset_id"] == report["dataset_id"]
    left, right = (
        read_jsonl(first / "dataset.jsonl"),
        read_jsonl(second / "dataset.jsonl"),
    )
    for a, b in zip(left, right, strict=True):
        assert (a["scene_id"], a["group_id"], a["split"], a["seed"]) == (
            b["scene_id"],
            b["group_id"],
            b["split"],
            b["seed"],
        )
        np.testing.assert_allclose(a["states"], b["states"], atol=1e-10)
        np.testing.assert_allclose(a["controls"], b["controls"], atol=1e-10)
    receipts = {
        path.name: path.read_bytes() for path in (first / "attempts").glob("*.json")
    }
    # Model an interruption after only three receipts reached durable publication.
    missing = sorted(receipts)[-1]
    publish_record(
        first / "status.json", {"state": "running", "dataset_id": report["dataset_id"]}
    )
    (first / "attempts" / missing).unlink()
    resumed = run_expert_pipeline(config, expert_scenes, first, resume=True, workers=2)
    session = resumed["generation_sessions"][-1]
    assert session["executed_attempts"] == 1 and session["reused_attempts"] == 3
    for name, data in receipts.items():
        if name != missing:
            assert (first / "attempts" / name).read_bytes() == data
    assert run_dataset_validation(first)["valid"]
    assert main(["validate", "--output", str(first)]) == 0
    assert main(["summary", "--output", str(first)]) == 0
    capsys.readouterr()
    normalization = read_record(first / "normalization.json")
    training_ids = [row["target_id"] for row in left if row["split"] == "train"]
    assert normalization["target_ids"] == training_ids
    assert set(normalization["groups"]).isdisjoint(
        {row["group_id"] for row in left if row["split"] != "train"}
    )
    data = json.loads(
        (first / "targets" / (left[0]["target_id"] + ".json")).read_text()
    )
    data["y_ref_star"][1][0] += 1
    (first / "targets" / (left[0]["target_id"] + ".json")).write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checksum"):
        run_dataset_validation(first)
    with pytest.raises(ValueError, match="checksum"):
        run_expert_pipeline(config, expert_scenes, first, resume=True)


def test_failed_solver_retries_are_bounded_recorded_and_never_exported(
    tmp_path, expert_config, expert_scenes
):
    config = replace(
        expert_config,
        planner=replace(
            expert_config.planner,
            initial_guess_offsets=(0.0,),
            max_iterations=1,
            retries=1,
        ),
    )
    report = run_expert_pipeline(config, expert_scenes, tmp_path, workers=2)
    assert report["status"] == "no_training_targets"
    assert report["metrics"]["attempts"] == 8
    assert report["metrics"]["validated_targets"] == 0
    assert report["metrics"]["eligible_attempts"] == 0
    assert not run_dataset_validation(tmp_path)["training_ready"]
    attempts = [
        read_record(path)["payload"] for path in (tmp_path / "attempts").glob("*.json")
    ]
    assert {row["job"]["retry"] for row in attempts} == {0, 1}
    assert len({row["job"]["seed"] for row in attempts}) == 8
    assert all(not row["result"]["solver_success"] for row in attempts)
    replay = run_expert_pipeline(config, expert_scenes, tmp_path, resume=True)
    assert replay["generation_sessions"][-1]["executed_attempts"] == 0
    assert replay["generation_sessions"][-1]["reused_attempts"] == 8
