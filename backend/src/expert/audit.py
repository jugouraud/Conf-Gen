"""Revalidate saved targets, receipts, splits and training-only normalization."""

from pathlib import Path, PurePosixPath

from backend.src.common.storage import calculate_file_sha256
from backend.src.expert.config import SPLITS, ExpertConfig, Guidance
from backend.src.expert.generate import attempt_job, load_attempt
from backend.src.expert.normalize import training_statistics
from backend.src.expert.scenes import Scene
from backend.src.expert.storage import (
    MAX_DATASET_JSONL_BYTES,
    MAX_JSON_BYTES,
    OPERATIONAL_SOURCE_FILES,
    digest,
    parse_json,
    read_record,
)
from backend.src.expert.validate import run_expert_validation


def verified_path(root: Path, name: str) -> Path:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError("Invalid artifact path")
    target = root.joinpath(*path.parts)
    if not target.resolve().is_relative_to(root.resolve()) or target.is_symlink():
        raise ValueError("Artifact escapes dataset root")
    return target


def read_jsonl(path: Path) -> list[dict]:
    result, size = [], 0
    with path.open("rb") as stream:
        while line := stream.readline(MAX_JSON_BYTES + 1):
            size += len(line)
            if size > MAX_DATASET_JSONL_BYTES or len(result) >= 10000:
                raise ValueError("Dataset size limit exceeded")
            result.append(parse_json(line))
    return result


def run_dataset_validation(output: Path, *, mode: str = "complete") -> dict:
    if mode not in {"complete", "building"}:
        raise ValueError("Unknown dataset validation mode")
    status = read_record(output / "status.json")
    if mode == "complete" and status["state"] not in {
        "complete",
        "no_training_targets",
    }:
        raise ValueError("Dataset is not complete")
    manifest, index = (
        read_record(output / "frozen.json"),
        read_record(output / "index.json"),
    )
    identity = {
        key: value
        for key, value in manifest.items()
        if key not in {"config", "dataset_id"}
    }
    if (
        digest(identity) != manifest["dataset_id"]
        or index["dataset_id"] != manifest["dataset_id"]
        or status["dataset_id"] != manifest["dataset_id"]
    ):
        raise ValueError("Dataset identity mismatch")
    config = ExpertConfig.from_record(manifest["config"])
    science = config.to_record()
    science["execution"].pop("workers")
    if digest(science) != digest(manifest["scientific_config"]):
        raise ValueError("Frozen configuration changed")
    source_root = Path(__file__).resolve().parents[3]
    for name, checksum in manifest["sources"].items():
        if (
            calculate_file_sha256(verified_path(source_root, name)) != checksum
            and name not in OPERATIONAL_SOURCE_FILES
        ):
            raise ValueError(
                "Validation source changed; use the recorded source revision"
            )
    for name, checksum in index["files"].items():
        if calculate_file_sha256(verified_path(output, name)) != checksum:
            raise ValueError("Dataset artifact checksum mismatch")
    attempts = {}
    for outcome in index["outcomes"]:
        path = verified_path(output, "attempts/" + outcome["attempt_id"] + ".json")
        record = read_record(path)["payload"]
        job = record["job"]
        row = manifest["scenes"][job["scene_index"]]
        expected = attempt_job(
            manifest, row, job["guidance_index"], job["guess"], job["retry"]
        )
        attempts[job["attempt_id"]] = load_attempt(path, expected)
    targets = read_jsonl(output / "dataset.jsonl")
    if (
        len({row["target_id"] for row in targets}) != len(targets)
        or [row["target_id"] for row in targets] != index["target_ids"]
    ):
        raise ValueError("Duplicate or mismatched target IDs")
    for target in targets:
        attempt = attempts[target["target_id"]]
        job, result = attempt["job"], attempt["result"]
        if (
            target["group_id"] != job["group_id"]
            or target["observation"] != job["observation"]
            or target["guidance"] != job["guidance"]
            or target["split"] != manifest["group_splits"][target["group_id"]]
            or target["dataset_id"] != manifest["dataset_id"]
        ):
            raise ValueError(
                "Scene, guidance or group split changed in an exported target"
            )
        if (
            not result["eligible"]
            or (config.planner.require_convergence and not result["solver_success"])
            or any(
                target[name] != result[name]
                for name in ("states", "controls", "timestamps", "y_ref_star")
            )
            or target["seed"] != job["seed"]
            or target["input_id"] != attempt["input_id"]
            or target["scene_id"] != job["observation"]["scene_id"]
            or digest(target["vehicle"]) != digest(manifest["config"]["vehicle"])
            or digest(target["planner"]) != digest(manifest["config"]["planner"])
        ):
            raise ValueError(
                "Exported target differs from its validated solver receipt"
            )
        if (
            read_record(
                verified_path(output, "targets/" + target["target_id"] + ".json")
            )
            != target
        ):
            raise ValueError("Target JSON and JSONL disagree")
        validation = run_expert_validation(
            Scene.from_record(target["observation"]),
            Guidance(**target["guidance"]),
            config,
            target["states"],
            target["controls"],
            target["timestamps"],
            target["y_ref_star"],
        )
        if not validation["valid"]:
            raise ValueError("Saved target failed independent physical validation")
        if validation != target["validation"]:
            raise ValueError("Stored physical diagnostics differ from revalidation")
    for split in SPLITS:
        if read_jsonl(output / "splits" / (split + ".jsonl")) != [
            row for row in targets if row["split"] == split
        ]:
            raise ValueError("Split export mismatch")
    if read_record(output / "normalization.json") != training_statistics(
        targets, config
    ):
        raise ValueError("Normalization was not fitted exclusively on training targets")
    if mode == "complete":
        summary = read_record(output / "summary.json")
        if calculate_file_sha256(output / "summary.json") != status["summary_sha256"]:
            raise ValueError("Final summary checksum mismatch")
        for name, checksum in summary["artifacts"].items():
            if calculate_file_sha256(verified_path(output, name)) != checksum:
                raise ValueError("Final dataset receipt mismatch")
    return {
        "valid": True,
        "dataset_id": manifest["dataset_id"],
        "targets": len(targets),
        "attempts": len(attempts),
        "scenes": len(manifest["scenes"]),
        "training_ready": any(row["split"] == "train" for row in targets),
        "reserved_calibration_use": "evaluate deployed generator outputs; do not calibrate on experts",
    }
