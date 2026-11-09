"""Real NPZ/JSON boundaries through the public detection-scoring step."""

import json

import numpy as np
import pytest

from backend.src.common.storage import calculate_file_sha256, read_json, write_json
from backend.src.detection.score import score_signature_space

pytestmark = pytest.mark.integration


@pytest.fixture
def detection_run(tmp_path):
    # Points on a line give independently obvious distances to anchors 0,2 and 10,12.
    coordinates = {
        "safe_reference": [0, 2],
        "safe_validation": [1, 3, 1],
        "safe_test": [1],
        "unsafe_reference": [10, 12],
        "unsafe_validation": [9, 11, 13],
        "unsafe_test": [9, 11, 13],
    }
    folder = tmp_path / "trajectory"
    groups = {}
    for split, values in coordinates.items():
        groups[split] = [
            {
                "id": f"{split}-{index}",
                "category": ("Violence", "Pornography", "Illegal_Activities")[
                    index % 3
                ],
            }
            for index in range(len(values))
        ]
        path = folder / "splits" / f"{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in groups[split]), encoding="utf-8"
        )
    manifest = {
        "config": {"experiment": {"alpha": 0.25}},
        "bandwidth_scale": 1,
        "settings": {
            "experiment": {"unsafe_alpha": 0.25, "cap_alpha": 0.25},
            "steering": {"grid_points": 17, "bisection_steps": 12},
        },
        "split_ids": {
            split: [row["id"] for row in rows] for split, rows in groups.items()
        },
    }
    write_json(tmp_path / "experiment.json", manifest)
    tables = {
        f"{label}__{split}": np.abs(
            np.asarray(values)[:, None] - np.asarray(anchors)[None, :]
        )
        for label, anchors in (("safe", [0, 2]), ("unsafe", [10, 12]))
        for split, values in coordinates.items()
    }
    np.savez_compressed(folder / "two_mode_distances.npz", **tables)
    write_json(
        folder / "two_mode_distances.provenance.json",
        {
            "experiment_sha256": calculate_file_sha256(tmp_path / "experiment.json"),
            "distances_sha256": calculate_file_sha256(
                folder / "two_mode_distances.npz"
            ),
            "split_ids": manifest["split_ids"],
        },
    )
    return tmp_path


def test_all_trajectory_methods_detect_separated_cohorts_in_both_modes(detection_run):
    reports = score_signature_space(detection_run, "trajectory")

    assert len(reports) == 6
    for name, report in reports.items():
        assert report["safe_false_positive_rate"] == 0, name
        assert report["unsafe_flagged_rate"] == 1, name
        assert report["auroc_prompt_labels"] == 1, name
        assert report["steering"]["unsafe_test"]["verified_repairs"] == 3, name
    nearest = read_json(
        detection_run / "variants/trajectory_euclidean/calibration.json"
    )
    assert nearest["safe_cutoff"] == 1
    assert nearest["unsafe_cutoff"] == 1


def test_held_out_distances_cannot_change_fitted_cutoffs_or_bandwidths(detection_run):
    score_signature_space(detection_run, "trajectory")
    calibration_path = (
        detection_run / "variants/trajectory_anchor_adaptive/calibration.json"
    )
    before = calibration_path.read_bytes()
    archive = detection_run / "trajectory/two_mode_distances.npz"
    with np.load(archive) as stored:
        tables = {name: stored[name].copy() for name in stored.files}
    tables["safe__unsafe_test"] += 100
    np.savez_compressed(archive, **tables)
    receipt_path = detection_run / "trajectory/two_mode_distances.provenance.json"
    receipt = read_json(receipt_path)
    receipt["distances_sha256"] = calculate_file_sha256(archive)
    write_json(receipt_path, receipt)

    score_signature_space(detection_run, "trajectory")

    assert calibration_path.read_bytes() == before


def test_modified_distance_archive_is_rejected_before_reports_are_written(
    detection_run,
):
    (detection_run / "trajectory/two_mode_distances.npz").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="Distance provenance"):
        score_signature_space(detection_run, "trajectory")

    assert not (detection_run / "variants").exists()
