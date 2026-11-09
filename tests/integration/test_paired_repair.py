"""Matched selection, detector reconstruction gates, and durable repair collection."""

import ast
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from backend.runtime.repair_session import BOOTSTRAP, snapshot, verify_completed_cases
from backend.runtime.repair_worker import CachedRepairAdapter, measure_rebuilt_distances
from backend.src.common.archives import extract_verified_archive
from backend.src.common.storage import read_json, write_json
from backend.src.repair.artifacts import compare_reconstruction, export_files
from backend.src.repair.prepare import select_matched_pairs


@pytest.mark.integration
def test_progress_continues_polling_while_export_download_is_pending(
    tmp_path, monkeypatch
):
    from backend.runtime import repair_session as session

    write_json(
        tmp_path / "experiment.json",
        {
            "expected_prompts": 20,
            "expected_pairs": 20,
            "expected_generation_records": 120,
        },
    )
    connection = SimpleNamespace(session="existing-worker")
    monkeypatch.setattr(session, "connection_for", lambda output: connection)
    polls = iter((10, 11))

    def snapshot(_):
        return {
            "records": {
                "status.json": {"phase": "extract_fields_gpu", "completed": next(polls)}
            },
            "exports": [{"name": "field_pending.zip", "sha256": "digest"}],
            "worker_pid": 123,
            "worker_alive": True,
        }

    class SlowDownload:
        def done(self):
            return False

        def result(self):
            pytest.fail("Progress polling must not wait for an unfinished download")

    class Collectors:
        def __init__(self, **kwargs):
            pass

        def submit(self, *args):
            return SlowDownload()

    observations = []

    def sleep(_):
        state = read_json(tmp_path / "progress.json")
        observations.append(state["stage_report"]["completed"])
        assert state["collection"]["pending_exports"] == 1
        if len(observations) == 2:
            raise StopIteration

    monkeypatch.setattr(session, "snapshot", snapshot)
    monkeypatch.setattr(session, "ThreadPoolExecutor", Collectors)
    monkeypatch.setattr(session.time, "sleep", sleep)
    with pytest.raises(StopIteration):
        session.monitor(tmp_path)
    assert observations == [10, 11]


def paired_groups():
    groups = {"safe_test": [], "unsafe_test": []}
    for category in range(5):
        for scenario in range(8):
            for variant in range(3):
                pair = f"{category}-{scenario}-{variant}"
                for label in ("safe", "unsafe"):
                    groups[f"{label}_test"].append(
                        {
                            "id": f"{pair}_{label}",
                            "pair_id": pair,
                            "scenario_id": f"{category}-{scenario}",
                            "category": str(category),
                            "variant_id": str(variant),
                            "prompt": f"{pair} {label}",
                            "label": label,
                        }
                    )
    return groups


@pytest.mark.unit
def test_selection_preserves_pairs_and_spreads_across_twenty_test_scenarios():
    groups = paired_groups()
    before = copy.deepcopy(groups)
    selected = select_matched_pairs(groups, 4, 1729)
    assert selected == select_matched_pairs(groups, 4, 1729)
    assert groups == before
    assert len(selected) == 40
    assert len({row["pair_id"] for row in selected}) == 20
    assert len({row["scenario_id"] for row in selected}) == 20
    for row in selected:
        original = groups[row["split"]][row["source_index"]]
        assert original["id"] == row["id"]
    for category in map(str, range(5)):
        assert sum(row["category"] == category for row in selected) == 8


@pytest.mark.unit
def test_mismatched_pair_cannot_enter_repair():
    groups = paired_groups()
    groups["unsafe_test"][0]["scenario_id"] = "changed"
    with pytest.raises(ValueError, match="metadata"):
        select_matched_pairs(groups, 4, 1729)


@pytest.mark.unit
def test_reconstruction_gate_checks_every_distance_and_rejects_nonfinite_values():
    expected = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert compare_reconstruction(expected + 1e-5, expected, 1e-3, 1e-3)["passed"]
    changed = expected.copy()
    changed[1, 1] += 0.1
    assert not compare_reconstruction(changed, expected, 1e-3, 1e-3)["passed"]
    changed[1, 1] = np.nan
    with pytest.raises(ValueError, match="invalid"):
        compare_reconstruction(changed, expected, 1e-3, 1e-3)


@pytest.mark.integration
def test_case_export_survives_independent_checksumming_and_extraction(tmp_path):
    output = tmp_path / "remote"
    write_json(output / "pilot/records/test.json", {"rows": [1]})
    image = output / "pilot/generations/example.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image fixture")
    export_files(output, "case", [image, output / "pilot/records/test.json"])
    metadata = read_json(output / "exports/case.json")
    assert metadata["name"] == "case.zip"
    destination = tmp_path / "collected"
    extract_verified_archive(output / "exports/case.zip", destination)
    assert (
        destination / "pilot/generations/example.png"
    ).read_bytes() == b"image fixture"
    assert read_json(destination / "pilot/records/test.json") == {"rows": [1]}


@pytest.mark.unit
def test_cached_adapter_uses_native_cached_embeddings_without_reencoding():
    values = (torch.tensor([1.0]), torch.tensor([2.0]))
    encoder = SimpleNamespace(read_embeddings=lambda row: values)
    adapter = CachedRepairAdapter(
        SimpleNamespace(signature="fixed", device="cpu"),
        encoder,
        {"safe_reference": [{"prompt": "reference", "id": "one"}]},
    )
    actual = adapter.encode_prompt("reference")
    for observed, expected in zip(actual, values, strict=True):
        torch.testing.assert_close(observed, expected)
    assert adapter.signature == "fixed"
    with pytest.raises(KeyError):
        adapter.encode_prompt("not preregistered")


@pytest.mark.integration
def test_bootstrap_and_status_scripts_compile_without_live_gpu():
    compile(ast.parse(BOOTSTRAP), "bootstrap.py", "exec")

    def execute(source):
        compile(ast.parse(source), "snapshot.py", "exec")
        return "REPAIR_SNAPSHOT=" + json.dumps({"worker_alive": True})

    connection = SimpleNamespace(root="/content/repair-test", execute=execute)
    assert snapshot(connection)["worker_alive"] is True


@pytest.mark.integration
def test_incomplete_results_cannot_trigger_success_and_shutdown(tmp_path):
    write_json(tmp_path / "experiment.json", {"expected_prompts": 40})
    with pytest.raises(ValueError, match="Missing completed"):
        verify_completed_cases(tmp_path)


@pytest.mark.integration
def test_rebuilt_reference_bridge_preserves_threshold_distances_and_cross_orientation(
    tmp_path, monkeypatch
):
    from backend.src.fields.distances import ResidentFieldDistances

    def cpu_cache(*args, **kwargs):
        kwargs["device"] = "cpu"
        return ResidentFieldDistances(*args, **kwargs)

    monkeypatch.setattr(
        "backend.runtime.repair_worker.ResidentFieldDistances", cpu_cache
    )
    values = {
        "s0": 0.0,
        "s1": 2.0,
        "u0": 1.0,
        "u1": 5.0,
        "v": 0.2,
        "st": 0.4,
        "ut": 4.0,
    }
    groups = {
        "safe_reference": [{"id": "s0"}, {"id": "s1"}],
        "unsafe_reference": [{"id": "u0"}, {"id": "u1"}],
        "safe_validation": [{"id": "v"}],
        "safe_test": [{"id": "st"}],
        "unsafe_test": [{"id": "ut"}],
    }
    store = SimpleNamespace(
        load_signature=lambda split, row: torch.full((2, 1, 1, 1), values[row["id"]])
    )
    bank = {"states": torch.zeros(2, 1, 1, 1), "weights": torch.full((2,), 0.5)}
    tables = {
        f"{label}__{split}": np.array(
            [
                [
                    abs(values[row["id"]] - values[ref["id"]])
                    for ref in groups[f"{label}_reference"]
                ]
                for row in groups[split]
            ]
        )
        for label in ("safe", "unsafe")
        for split in ("safe_validation", "safe_test", "unsafe_test")
    }
    selected = [
        {**groups[split][0], "split": split, "source_index": 0}
        for split in ("safe_test", "unsafe_test")
    ]
    settings = {
        "experiment": {"normalization_limit": 1},
        "execution": {"reconstruction_rtol": 0.001, "reconstruction_atol": 0.001},
    }
    measure_rebuilt_distances(tmp_path, store, groups, bank, tables, selected, settings)
    np.testing.assert_allclose(
        tables["unsafe__safe_reference"], [[1, 5], [1, 3]], atol=1e-5
    )
    assert read_json(tmp_path / "reconstruction.json")["passed"] is True
    assert read_json(tmp_path / "reconstruction.json")["thresholds_refitted"] is False
