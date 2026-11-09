"""Frozen paired detection identity and shared artifact contract."""

from pathlib import Path

from backend.src.common.storage import (
    calculate_file_sha256,
    calculate_source_fingerprints,
    read_json,
    read_jsonl,
)

QUERY_SPLITS = ("safe_validation", "unsafe_validation", "safe_test", "unsafe_test")
SPACE_METHODS = {
    "trajectory": ("euclidean", "anchor_static", "anchor_adaptive"),
    "source_slice": ("anchor_static", "anchor_adaptive"),
}
IMPLEMENTATION_FILES = (
    "backend/runtime/colab.py",
    "backend/src/common/archives.py",
    "backend/runtime/detection.py",
    "backend/src/detection/extract.py",
    "backend/src/detection/score.py",
    "backend/src/data/definitions.py",
    "backend/src/fields/definitions.py",
    "backend/src/fields/checkpoints.py",
    "backend/src/fields/constraints.py",
    "backend/src/detection/definitions.py",
    "backend/src/detection/measure.py",
    "backend/src/detection/rendering.py",
    "backend/src/detection/contracts.py",
    "backend/src/fields/resident.py",
    "backend/src/detection/pipeline.py",
    "backend/src/detection/prepare.py",
    "backend/src/fields/distances.py",
    "backend/src/inference/encoding.py",
    "backend/src/data/paired.py",
    "backend/src/inference/flow.py",
    "backend/src/fields/signatures.py",
    "backend/src/detection/calibration.py",
    "backend/src/common/storage.py",
    "backend/src/data/prompts.py",
    "backend/src/common/paths.py",
)


def read_resident_detection_inputs(output: Path) -> tuple[dict, dict]:
    """Validate implementation, split checksums and frozen row ordering."""
    manifest = read_json(output / "experiment.json")
    if manifest["implementation"] != calculate_source_fingerprints(
        IMPLEMENTATION_FILES
    ):
        raise ValueError("Prepared implementation changed; prepare a fresh output")
    if manifest.get("experiment_implementation", {}) != calculate_source_fingerprints(
        manifest.get("experiment_implementation", {})
    ):
        raise ValueError("Prepared experiment implementation changed")
    groups = {}
    for split, digest in manifest["split_sha256"].items():
        path = output / "trajectory/splits" / f"{split}.jsonl"
        if calculate_file_sha256(path) != digest:
            raise ValueError("Frozen split changed")
        groups[split] = read_jsonl(path)
        if [r["id"] for r in groups[split]] != manifest["split_ids"][split]:
            raise ValueError("Frozen split order changed")
    return manifest, groups


def read_frozen_control_inputs(output):
    manifest = read_json(output / "experiment.json")
    if (
        calculate_file_sha256(output / "parent_comparison.json")
        != manifest["parent_comparison_sha256"]
    ):
        raise ValueError("Frozen parent comparison changed")
    parent = read_json(output / "parent_comparison.json")
    frozen = read_json(output / "frozen_calibration.json")
    for key, calibration in frozen.items():
        if calibration != parent["methods"][key + "__safe_attraction"]["calibration"]:
            raise ValueError(
                "Additional test attempted to change a frozen calibration rule"
            )
    return manifest, parent, frozen
