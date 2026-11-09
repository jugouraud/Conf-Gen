"""Verified resume, completion hooks and failures through the repair H1 API."""

from types import SimpleNamespace

import pytest

from backend.src.common.storage import calculate_file_sha256, read_json, write_json
from backend.src.fields.signatures import calculate_model_signature
from backend.src.repair.contracts import RepairExperiment
from backend.src.repair.execute import run_repair_execution

pytestmark = pytest.mark.integration


def resumed_experiment(tmp_path):
    metadata = {"model_id": "synthetic", "scheduler": {}}
    row = {"id": "one", "prompt": "synthetic scene"}
    output = tmp_path / "repair"
    artifact = output / "pilot/saved.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"verified generated artifact")
    write_json(output / "experiment.json", {"id": "frozen"})
    records = [{"id": "one", "method": "R0"}]
    write_json(
        output / "pilot/records/unsafe_test_one.json",
        {
            "experiment_sha256": calculate_file_sha256(output / "experiment.json"),
            "artifacts": {"pilot/saved.bin": calculate_file_sha256(artifact)},
            "rows": records,
        },
    )
    experiment = RepairExperiment(
        settings={"evaluation": {"factory": ""}},
        output=output,
        source=output,
        config={"model": {"device": "cpu"}},
        bank={"metadata": metadata},
        calibration={},
        groups={"unsafe_test": [row]},
        distances={},
        features=None,
        fingerprint={},
        adapter=SimpleNamespace(signature=calculate_model_signature(metadata)),
    )
    return experiment, ("unsafe_test", 0, row), records, artifact


def test_verified_resume_runs_completion_hook_and_returns_saved_records(tmp_path):
    experiment, case, expected, artifact = resumed_experiment(tmp_path)
    before = artifact.read_bytes()
    completed = []
    records = run_repair_execution(
        experiment,
        "pilot",
        cases=[case],
        on_case_completed=lambda *event: completed.append(event),
    )
    assert records == expected
    assert completed == [(case, expected, 1, 1)]
    assert artifact.read_bytes() == before
    assert read_json(experiment.output / "pilot/status.json")["status"] == "complete"


def test_completion_hook_failure_retains_receipt_and_allows_verified_retry(tmp_path):
    experiment, case, expected, _ = resumed_experiment(tmp_path)

    def interrupted_export(*_):
        raise RuntimeError("export interrupted")

    with pytest.raises(RuntimeError, match="export interrupted"):
        run_repair_execution(
            experiment, "pilot", cases=[case], on_case_completed=interrupted_export
        )
    assert read_json(experiment.output / "pilot/status.json")["status"] == "failed"
    assert run_repair_execution(experiment, "pilot", cases=[case]) == expected
    assert read_json(experiment.output / "pilot/status.json")["status"] == "complete"


def test_corrupt_cached_artifact_fails_without_publishing_completion(tmp_path):
    experiment, case, _, artifact = resumed_experiment(tmp_path)
    artifact.write_bytes(b"changed")
    completed = []
    with pytest.raises(ValueError, match="Generated artifact changed"):
        run_repair_execution(
            experiment,
            "pilot",
            cases=[case],
            on_case_completed=lambda *event: completed.append(event),
        )
    assert not completed
    assert read_json(experiment.output / "pilot/status.json")["status"] == "failed"


@pytest.mark.parametrize("invalid", ["duplicate", "index", "stage"])
def test_explicit_cohort_rejects_duplicate_or_nonfrozen_cases(tmp_path, invalid):
    experiment, case, _, _ = resumed_experiment(tmp_path)
    cases = (
        [case, case]
        if invalid == "duplicate"
        else [(case[0], 1, case[2])]
        if invalid == "index"
        else [("unsafe_validation", 0, case[2])]
    )
    with pytest.raises(ValueError):
        run_repair_execution(experiment, "pilot", cases=cases)
    assert read_json(experiment.output / "pilot/status.json")["status"] == "failed"
