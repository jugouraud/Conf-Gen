"""Compare scorers through real frozen artifacts without invoking a model backend."""

import csv
from copy import deepcopy

import numpy as np
import pytest
import torch

from backend.src.common.storage import calculate_file_sha256, read_json
from backend.src.detection.calibration import calibrate_paired_detection_thresholds
from backend.src.detection.extract import run_detection_extraction
from backend.src.detection.measure import run_detection_measurement
from backend.src.detection.prepare import run_detection_preparation
from backend.src.detection.score import run_detection_scoring
from backend.src.fields.distances import calculate_streamed_reference_distances
from backend.src.fields.signatures import calculate_model_signature


class ScalarFlowAdapter:
    device = "cpu"
    shape = (1, 1, 1)
    metadata = {"scheduler": {}, "model": "scalar-scoring-fixture"}
    signature = calculate_model_signature(metadata)

    def encode_prompt(self, prompt):
        label, split, index = prompt.split()
        return (
            (100 if label == "illegal" else 0)
            + int(index) / 10
            + {"train": 0, "validation": 0.01, "test": 0.005}[split]
        )

    def evaluate_guided_velocity(self, states, time, embeddings):
        return torch.zeros_like(states) + embeddings

    def evaluate_prompt_effect(self, states, time, embeddings):
        return states * 0.01 + embeddings


@pytest.fixture
def completed_scalar_detection(tmp_path):
    dataset, config, source = (
        tmp_path / "pairs.csv",
        tmp_path / "config.toml",
        tmp_path / "source",
    )
    with dataset.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "pair_id",
                "scenario_id",
                "category",
                "split",
                "variant_id",
                "illegal_prompt",
                "legal_prompt",
            )
        )
        for split in ("train", "validation", "test"):
            for index in range(24):
                writer.writerow(
                    (
                        f"{split}-{index}",
                        f"{split}-{index}",
                        "fixture",
                        split,
                        1,
                        f"illegal {split} {index}",
                        f"legal {split} {index}",
                    )
                )
    config.write_text(
        f'[paths]\ndataset = "{dataset.as_posix()}"\n'
        "[experiment]\nseed = 1729\nalpha = 0.05\nunsafe_alpha = 0.05\ncap_alpha = 0.05\nguidance_scale = 7.5\n"
        '[model]\nid = "injected"\n'
        "[probes]\ntime_points = 2\ntrajectories = 2\nintegration_steps = 4\n"
    )
    run_detection_preparation(config, source)
    run_detection_extraction(source, mode="native", adapter=ScalarFlowAdapter())
    run_detection_measurement(source, device="cpu", mode="paired")
    run_detection_scoring(source, mode="paired")
    return source


@pytest.mark.integration
def test_all_scorers_reuse_frozen_data_and_reproduce_baseline(
    completed_scalar_detection, tmp_path
):
    source, output = completed_scalar_detection, tmp_path / "comparison"
    artifacts = {
        name: calculate_file_sha256(source / name)
        for name in (
            "experiment.json",
            "calibration.json",
            "report.json",
            "trajectory/distances.npz",
        )
    }
    rows = run_detection_scoring(
        output,
        source=source,
        mode="comparison",
        options={"working_directory": tmp_path},
    )
    assert len(rows) == 6
    assert {row["method"] for row in rows} == {
        "euclidean",
        "anchor_static",
        "anchor_adaptive",
    }
    assert read_json(output / "comparison.json")["euclidean_baseline_reproduced"]
    assert read_json(output / "status.json")["status"] == "complete"
    assert all(
        calculate_file_sha256(source / name) == digest
        for name, digest in artifacts.items()
    )
    for method in ("euclidean", "anchor_static", "anchor_adaptive"):
        calibration = read_json(output / method / "calibration.json")
        assert calibration["fit_policy"] == "validation_only"
        assert all(
            not row_id.startswith("test-")
            for ids in calibration["calibration_ids"].values()
            for row_id in ids
        )
    with np.load(output / "training_geometry/safe.npz") as archive:
        matrix = archive["distances"]
        expected_bandwidth = np.median(matrix[np.triu_indices(24, 1)])
    assert (
        read_json(output / "anchor_adaptive/calibration.json")["safe_rule"]["bandwidth"]
        == expected_bandwidth
    )
    with np.load(source / "trajectory/distances.npz") as archive:
        tables = {key: archive[key].copy() for key in archive.files}
    changed = deepcopy(tables)
    for key in changed:
        if key.endswith("test"):
            changed[key] *= 100
    settings = read_json(source / "experiment.json")["config"]["experiment"]
    for method in ("anchor_static", "anchor_adaptive"):
        calibration = read_json(output / method / "calibration.json")
        rules = {key: calibration[key] for key in ("safe_rule", "unsafe_rule")}
        assert calibrate_paired_detection_thresholds(
            tables, settings, **rules
        ) == calibrate_paired_detection_thresholds(changed, settings, **rules)


@pytest.mark.integration
def test_comparison_rejects_modified_training_signature(
    completed_scalar_detection, tmp_path
):
    source = completed_scalar_detection
    feature = next((source / "trajectory/features/safe_reference").glob("*.pt"))
    with feature.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Training signature bytes changed"):
        run_detection_scoring(
            tmp_path / "rejected",
            source=source,
            mode="comparison",
            options={"working_directory": tmp_path},
        )


@pytest.mark.unit
def test_streamed_training_geometry_matches_direct_subtraction(tmp_path):
    vectors = torch.randn(9, 37, generator=torch.Generator().manual_seed(1729))
    actual = calculate_streamed_reference_distances(
        list(range(9)),
        lambda row: vectors[row],
        working_directory=tmp_path,
        feature_block_size=7,
    )
    expected = torch.linalg.vector_norm(
        vectors.double()[:, None] - vectors.double()[None], dim=-1
    ).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
