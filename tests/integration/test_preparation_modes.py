"""Preparation modes preserve frozen artifacts and accept configurable cohorts."""

import json
import zipfile

import numpy as np
import pytest
import torch

from backend.src.common.storage import (
    calculate_file_sha256,
    read_json,
    write_json,
)
from backend.src.detection.contracts import IMPLEMENTATION_FILES
from backend.src.detection.extract import run_detection_extraction
from backend.src.detection.measure import run_detection_measurement
from backend.src.detection.prepare import run_detection_preparation
from backend.src.detection.score import run_detection_scoring, score_signature_space
from backend.src.fields.signatures import save_field_signature
from backend.src.repair.prepare import run_repair_preparation

pytestmark = pytest.mark.integration


def repair_settings(source, output):
    return {
        "paths": {"source": str(source), "output": str(output)},
        "execution": {},
        "colab": {},
        "experiment": {
            "seed": 17,
            "generation_seeds": [7, 11],
            "methods": ["R0", "R1"],
            "modes": ["safe_attraction", "safe_attraction_unsafe_avoidance"],
            "inference_steps": 2,
            "pairs_per_category": 2,
            "cohort_selection": "matched_pairs",
            "pilot_per_category": 1,
            "tuning_per_category": 1,
            "normalization_limit": 2,
            "threads": 1,
        },
        "steering": {
            "grid_points": 3,
            "refinement_steps": 0,
            "optimization_knots": 2,
            "optimization_sweeps": 1,
            "top_k": 1,
            "window_widths": [0.5],
            "ramp_fraction": 0.2,
            "matched_budgets": [],
            "coordinate_steps": [0.25],
            "smoothness_weight": 0.01,
            "constraint_penalty": 1,
            "unsafe_margin": 0,
        },
    }


@pytest.fixture
def paired_source(tmp_path):
    from tests.integration.test_paired_detection import (
        TinyFlowAdapter,
        create_paired_inputs,
    )

    _, config = create_paired_inputs(tmp_path)
    source = tmp_path / "source"
    run_detection_preparation(config, source)
    run_detection_extraction(source, mode="native", adapter=TinyFlowAdapter())
    run_detection_measurement(source, device="cpu")
    run_detection_scoring(source, mode="paired")
    write_json(
        source / "scores/trajectory/euclidean/calibration.json",
        read_json(source / "calibration.json"),
    )
    manifest = read_json(source / "experiment.json")
    # Native and resident producers share this calibration/distance contract.
    from backend.src.common.storage import calculate_source_fingerprints

    manifest["implementation"] = calculate_source_fingerprints(IMPLEMENTATION_FILES)
    write_json(source / "experiment.json", manifest)
    receipt_path = source / "trajectory/distances.provenance.json"
    receipt = read_json(receipt_path)
    receipt["experiment_sha256"] = calculate_file_sha256(source / "experiment.json")
    write_json(receipt_path, receipt)
    write_json(
        source / "banks.provenance.json",
        {"banks": {"trajectory": receipt["bank_sha256"]}},
    )
    return source


@pytest.mark.parametrize(
    "policy,prompts", [("matched_pairs", 4), ("unsafe_test_only", 2)]
)
def test_paired_preparation_derives_cohort_and_seed_counts(
    tmp_path, paired_source, policy, prompts
):
    output = tmp_path / "prepared"
    settings = repair_settings(paired_source, output)
    settings["experiment"]["cohort_selection"] = policy
    prepared = run_repair_preparation(settings, mode="paired")
    assert prepared.mode == "paired" and prepared.experiment is None
    assert prepared.summary == {"pairs": 2, "prompts": prompts, "records": prompts * 8}
    manifest = read_json(output / "experiment.json")
    assert {row["category"] for row in manifest["selected_cohort"]} == {"synthetic"}
    assert manifest["expected_generation_records"] == prompts * 8
    assert (output / "source/experiment.json").read_bytes() == (
        paired_source / "experiment.json"
    ).read_bytes()
    with zipfile.ZipFile(tmp_path / "gpu-inputs.zip") as archive:
        assert "backend/src/repair/prepare.py" in archive.namelist()
        assert (
            "backend/experiments/five_category_nudity_detection_20261101/definition.py"
            in archive.namelist()
        )
    with pytest.raises(ValueError, match="fresh output"):
        run_repair_preparation(settings, mode="paired")


def test_paired_preparation_rejects_corrupt_parent_before_writes(
    tmp_path, paired_source
):
    (paired_source / "trajectory/distances.npz").write_bytes(b"corrupt")
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="Parent distances changed"):
        run_repair_preparation(repair_settings(paired_source, output), mode="paired")
    assert not output.exists()


@pytest.mark.parametrize("mode", ["paired", "controls", "unknown"])
def test_preparation_rejects_invalid_mode_or_cohort_before_writes(tmp_path, mode):
    output = tmp_path / "prepared"
    settings = repair_settings(tmp_path / "missing", output)
    settings["experiment"]["cohort_selection"] = (
        "stratified" if mode == "paired" else "matched_pairs"
    )
    with pytest.raises(ValueError, match="mode|selection"):
        run_repair_preparation(settings, mode=mode)
    assert not output.exists()


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_controls_preparation_binds_real_frozen_fields_and_reuses_normalization(
    tmp_path,
):
    source, controls, output = (
        tmp_path / name for name in ("source", "controls", "prepared")
    )
    values = {
        "safe_reference": [0, 2],
        "unsafe_reference": [10, 12],
        "safe_validation": [1, 3, 1],
        "unsafe_validation": [9, 11, 13],
        "safe_test": [1],
        "unsafe_test": [9],
        "additional_safe_test": [2],
    }
    groups = {
        split: [
            {
                "id": f"{split}-{i}",
                "prompt": f"prompt {split} {i}",
                "label": "benign_intended"
                if split == "additional_safe_test"
                else "safe"
                if split.startswith("safe")
                else "risky",
                "category": "synthetic",
            }
            for i in range(len(coords))
        ]
        for split, coords in values.items()
    }
    for split, rows in groups.items():
        root = controls if split == "additional_safe_test" else source
        write_rows(root / "trajectory/splits" / f"{split}.jsonl", rows)
    ids = {
        split: [row["id"] for row in rows]
        for split, rows in groups.items()
        if split != "additional_safe_test"
    }
    manifest = {
        "config": {
            "paths": {},
            "model": {},
            "experiment": {"alpha": 0.25},
            "probes": {"time_points": 1, "trajectories": 1, "integration_steps": 1},
        },
        "split_ids": ids,
        "bandwidth_scale": 1,
        "settings": {
            "experiment": {"unsafe_alpha": 0.25, "cap_alpha": 0.25},
            "steering": {"grid_points": 3, "bisection_steps": 1},
        },
    }
    write_json(source / "experiment.json", manifest)
    fit = source / "trajectory/splits/safe_reference.jsonl"
    bank = {
        "states": torch.zeros(1, 1, 1, 1),
        "weights": torch.ones(1),
        "times": torch.tensor([0.5]),
        "midpoints": torch.tensor([0.5]),
        "time_points": 1,
        "trajectories": 1,
        "integration_steps": 1,
        "fit_sha256": calculate_file_sha256(fit),
        "fit_ids": ids["safe_reference"],
        "provenance": [{"trajectory_id": 0}],
        "adapter_signature": "synthetic",
    }
    bank_path = source / "trajectory/probes/probe_bank.pt"
    bank_path.parent.mkdir(parents=True)
    torch.save(bank, bank_path)
    bank_hash = calculate_file_sha256(bank_path)
    hashes = {}
    for split, rows in groups.items():
        root = controls if split == "additional_safe_test" else source
        hashes[split] = {}
        for row, coordinate in zip(rows, values[split], strict=True):
            path = root / "trajectory/features" / split / f"{row['id']}.pt"
            save_field_signature(
                path,
                row["id"],
                torch.full((1, 1, 1, 1), float(coordinate)),
                bank_hash,
                "synthetic",
            )
            hashes[split][f"{split}/{row['id']}.pt"] = calculate_file_sha256(path)
    for name, splits in (
        ("safe_distances", [s for s in ids if s.startswith("safe")]),
        ("unsafe_features", [s for s in ids if s.startswith("unsafe")]),
    ):
        write_json(
            source / "trajectory" / f"{name}.provenance.json",
            {
                "bank_sha256": bank_hash,
                "features": {k: v for s in splits for k, v in hashes[s].items()},
            },
        )
    write_json(
        controls / "experiment.json",
        {
            "parent_experiment_sha256": calculate_file_sha256(
                source / "experiment.json"
            ),
            "split_ids": {
                "additional_safe_test": [
                    row["id"] for row in groups["additional_safe_test"]
                ]
            },
        },
    )
    for name in ("additional_features", "reference_features"):
        write_json(
            controls / "trajectory" / f"{name}.provenance.json",
            {"bank_sha256": bank_hash, "features": hashes["additional_safe_test"]},
        )
    tables = {
        f"{label}__{split}": np.abs(
            np.array(coords)[:, None] - np.array(values[f"{label}_reference"])[None, :]
        )
        for label in ("safe", "unsafe")
        for split, coords in values.items()
        if split in ids
    }
    distance_path = source / "trajectory/two_mode_distances.npz"
    np.savez_compressed(distance_path, **tables)
    write_json(
        distance_path.with_suffix(".provenance.json"),
        {
            "experiment_sha256": calculate_file_sha256(source / "experiment.json"),
            "distances_sha256": calculate_file_sha256(distance_path),
            "split_ids": ids,
            "bank_sha256": bank_hash,
            "safe_receipt_sha256": calculate_file_sha256(
                source / "trajectory/safe_distances.provenance.json"
            ),
            "unsafe_receipt_sha256": calculate_file_sha256(
                source / "trajectory/unsafe_features.provenance.json"
            ),
        },
    )
    score_signature_space(source, "trajectory")
    distance_path = controls / "trajectory/additional_distances.npz"
    np.savez_compressed(
        distance_path, safe=np.array([[2, 0]]), unsafe=np.array([[8, 10]])
    )
    write_json(
        distance_path.with_suffix(".provenance.json"),
        {
            "distances_sha256": calculate_file_sha256(distance_path),
            "bank_sha256": bank_hash,
            "experiment_sha256": calculate_file_sha256(controls / "experiment.json"),
            "query_receipt_sha256": calculate_file_sha256(
                controls / "trajectory/additional_features.provenance.json"
            ),
            "reference_receipt_sha256": calculate_file_sha256(
                controls / "trajectory/reference_features.provenance.json"
            ),
        },
    )
    settings = repair_settings(source, output)
    settings["paths"]["controls"] = str(controls)
    settings["experiment"]["cohort_selection"] = "stratified"
    result = run_repair_preparation(settings, mode="controls")
    assert result.mode == "controls" and result.experiment is not None
    assert result.experiment.normalization["fit_ids"] == ids["safe_validation"][:2]
    assert result.experiment.normalization["location"] == [1.0]
    assert result.experiment.calibration["safe_cutoff"] == 1
    before = (output / "experiment.json").read_bytes()
    repeated = run_repair_preparation(settings, mode="controls")
    assert repeated.experiment.normalization == result.experiment.normalization
    assert (output / "experiment.json").read_bytes() == before
