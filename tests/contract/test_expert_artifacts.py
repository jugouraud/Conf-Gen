"""Real file, process, schema and checkpoint boundaries for expert data."""

from dataclasses import replace

import pytest

from backend.experiments.expert_trajectories.scenes import run_scene_sampling
from backend.src.expert.generate import attempt_job, execute_attempt, load_attempt
from backend.src.expert.prepare import run_expert_preparation
from backend.src.expert.storage import (
    dataset_lock,
    digest,
    parse_json,
    publish_jsonl,
    publish_record,
    read_record,
)

pytestmark = pytest.mark.contract


@pytest.mark.parametrize(
    "payload",
    [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}', b"[" * 33 + b"0" + b"]" * 33],
)
def test_strict_scene_json_rejects_ambiguous_or_unbounded_input(payload):
    with pytest.raises(ValueError):
        parse_json(payload)


def test_user_scenes_are_preserved_and_related_variants_share_group(
    tmp_path, expert_config, expert_scenes
):
    supplied = tmp_path / "scenes.json"
    variant = dict(expert_scenes[0], scene_id="augmented", goal=[2.2, 0])
    publish_record(
        supplied, {"schema": "expert-scenes-v1", "scenes": expert_scenes + [variant]}
    )
    rows, source = run_scene_sampling({}, expert_config, supplied)
    assert rows == expert_scenes + [variant] and source["kind"] == "user_provided"
    output = tmp_path / "dataset"
    with dataset_lock(output):
        manifest = run_expert_preparation(
            expert_config, rows, output, resume=False, scene_source=source
        )
    assert manifest["scenes"][0]["split"] == manifest["scenes"][-1]["split"]
    assert manifest["scenes"][0]["group_id"] == manifest["scenes"][-1]["group_id"]


def test_real_worker_hard_timeout_cannot_publish_a_target(
    tmp_path, expert_config, expert_scenes
):
    with dataset_lock(tmp_path):
        manifest = run_expert_preparation(
            expert_config, expert_scenes, tmp_path, resume=False
        )
    job = attempt_job(manifest, manifest["scenes"][0], 0, 0, 0)
    result = execute_attempt(job, timeout=0.001, threads=1)
    assert result["status"] == "worker_timeout" and not result["eligible"]
    assert result["states"] is None and result["y_ref_star"] is None


def test_invalid_initial_scene_is_retained_before_augmentation(
    tmp_path, expert_config, expert_scenes
):
    invalid = dict(expert_scenes[0], initial_state=[-2.0, 0.0])
    rows = [invalid, *expert_scenes[1:]]
    with dataset_lock(tmp_path):
        manifest = run_expert_preparation(expert_config, rows, tmp_path, resume=False)
    assert manifest["scenes"][0]["raw_observation"] == invalid
    assert manifest["scenes"][0]["status"] == "rejected"
    assert manifest["scenes"][0]["split"] in {
        "train",
        "reference",
        "future_calibration",
        "test",
    }


def test_committed_attempt_rejects_changed_bytes_and_new_config(
    tmp_path, expert_config, expert_scenes
):
    with dataset_lock(tmp_path):
        manifest = run_expert_preparation(
            expert_config, expert_scenes, tmp_path, resume=False
        )
    job = attempt_job(manifest, manifest["scenes"][0], 0, 0, 0)
    result = execute_attempt(job, timeout=20, threads=1)
    assert result["eligible"], result
    payload = {"input_id": digest(job), "job": job, "result": result}
    path = tmp_path / "attempt.json"
    publish_record(path, {"payload_sha256": digest(payload), "payload": payload})
    assert load_attempt(path, job)["result"]["eligible"]
    record = read_record(path)
    record["payload"]["result"]["seed"] = -1
    publish_record(path, record)
    with pytest.raises(ValueError, match="changed"):
        load_attempt(path, job)
    changed = replace(expert_config, planner=replace(expert_config.planner, dt=0.6))
    with pytest.raises(ValueError, match="changed"):
        run_expert_preparation(changed, expert_scenes, tmp_path, resume=True)


def test_single_coordinator_lease_is_released_without_stale_pid_files(tmp_path):
    with dataset_lock(tmp_path):
        with pytest.raises(RuntimeError, match="coordinator"):
            with dataset_lock(tmp_path):
                pytest.fail("Concurrent writer entered")
    with dataset_lock(tmp_path):
        assert (tmp_path / ".writer.lock").exists()


def test_dataset_export_accepts_the_configured_maximum_size(tmp_path, monkeypatch):
    import backend.src.expert.storage as storage

    monkeypatch.setattr(storage, "MAX_DATASET_JSONL_BYTES", 29)
    publish_jsonl(tmp_path / "dataset.jsonl", [{"value": "x" * 16}])
    assert (tmp_path / "dataset.jsonl").stat().st_size == 29
    with pytest.raises(ValueError, match=r"total=30 limit=29 line=30"):
        publish_jsonl(tmp_path / "dataset.jsonl", [{"value": "x" * 17}])
