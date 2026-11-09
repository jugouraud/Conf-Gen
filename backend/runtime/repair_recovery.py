"""Recover verified nudity repair checkpoints without modifying the stopped run."""

import shutil
import tempfile
from pathlib import Path

import torch

from backend.src.common.archives import resolve_contained_artifact_path
from backend.src.common.storage import calculate_file_sha256, read_json, write_json
from backend.src.fields.signatures import (
    save_field_signature,
    validate_cached_signature,
)


def clone_and_correct_source(previous: Path, output: Path) -> dict:
    """Clone a recovered source and undo a known null-field shift in new fields.

    Original frozen fields retain their bytes. Only additional checkpoints from
    the stopped attempt receive CFG * (wrong_null - original_null). Identities,
    finite values, recovery evidence and checksums are checked before publishing.
    """
    previous, output = Path(previous).resolve(), Path(output).resolve()
    target = output / "source"
    if (
        target.exists()
        or output.is_relative_to(previous)
        or previous.is_relative_to(output)
    ):
        raise ValueError("Cache recovery requires a fresh, separate source directory")
    if any(path.is_symlink() for path in previous.rglob("*")):
        raise ValueError("Recovered source must not contain symlinks")
    recovery = read_json(previous.parent / "native_null_recovery.json")
    if (
        recovery.get("status") != "complete"
        or recovery.get("comparison_to_original_frozen_null_velocity", {}).get(
            "bitwise_equal"
        )
        is not True
    ):
        raise ValueError("Native null recovery is unverified")
    reuse = read_json(output / "experiment.json")["reuse"]
    bank_path = previous / "trajectory/probes/probe_bank.pt"
    bank_hash = calculate_file_sha256(bank_path)
    if bank_hash != reuse["bank_sha256"]:
        raise ValueError("Recovered bank changed")
    bank = torch.load(bank_path, map_location="cpu", weights_only=True)
    signature = bank["adapter_signature"]
    nulls = []
    for name in ("trajectory.null.pt", "optimized-null-16.pt"):
        record = torch.load(
            previous / "optimization" / name, map_location="cpu", weights_only=True
        )
        if (
            record["adapter_signature"] != signature
            or record["bank_sha256"] != bank_hash
        ):
            raise ValueError("Recovered null identity changed")
        nulls.append(record["values"])
    original, wrong = nulls
    if original.shape != wrong.shape or not all(
        torch.isfinite(value).all() for value in nulls
    ):
        raise ValueError("Invalid recovered null values")
    shift = bank["metadata"]["guidance_scale"] * (wrong - original)
    for name, digest in reuse["files"].items():
        if not name.startswith("source/"):
            raise ValueError("Recovered artifact is outside source")
        path = resolve_contained_artifact_path(previous, name.removeprefix("source/"))
        if calculate_file_sha256(path) != digest:
            raise ValueError("Frozen recovered artifact changed")
    fields = []
    for path in sorted((previous / "trajectory/features").glob("*/*.pt")):
        if (
            calculate_file_sha256(path)
            != read_json(path.with_suffix(".sha256.json"))["sha256"]
        ):
            raise ValueError("Recovered field checksum changed")
        validate_cached_signature(path, bank_hash, signature)
        record = torch.load(path, map_location="cpu", weights_only=True)
        if (
            record["values"].shape != shift.shape
            or not torch.isfinite(record["values"]).all()
        ):
            raise ValueError("Invalid recovered field values")
        if "source/" + path.relative_to(previous).as_posix() not in reuse["files"]:
            fields.append((path, record))
    corrected = []
    with tempfile.TemporaryDirectory(dir=output, prefix=".recover-") as scratch:
        staged = Path(scratch) / "source"
        # copytree copies bytes; writing corrected files cannot mutate previous.
        shutil.copytree(previous, staged, symlinks=False)
        for path, record in fields:
            destination = staged / path.relative_to(previous)
            save_field_signature(
                destination,
                record["id"],
                record["values"] + shift,
                bank_hash,
                signature,
            )
            write_json(
                destination.with_suffix(".sha256.json"),
                {"sha256": calculate_file_sha256(destination)},
            )
            corrected.append(record["id"])
        staged.rename(target)
    result = {
        "status": "complete",
        "corrected_ids": corrected,
        "source": str(previous),
        "bank_sha256": bank_hash,
    }
    write_json(output / "cache_recovery.json", result)
    return result
