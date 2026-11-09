import hashlib
import json
import zipfile

import pytest

from backend.src.common.archives import (
    create_manifested_archive,
    extract_verified_archive,
)

pytestmark = pytest.mark.integration


def benchmark_receipt(seconds=1.0, **changes):
    return dict(
        {
            "passed": True,
            "bank_sha256": "bank",
            "adapter_signature": "model",
            "acceptance": "bitwise equal",
            "batch_size": 16,
            "encoding_seconds": {"t5": seconds},
            "records": [
                {
                    "id": "train_safe",
                    "bitwise_equal": True,
                    "embedding_bitwise_equal": True,
                    "max_absolute_error": 0.0,
                    "relative_l2_error": 0.0,
                    "baseline_seconds": seconds,
                    "optimized_field_seconds": seconds,
                }
            ],
        },
        **changes,
    )


def test_benchmark_timing_refresh_preserves_old_evidence(tmp_path):
    destination = tmp_path / "results"
    relative = "optimization/compatibility.json"
    previous = destination / relative
    previous.parent.mkdir(parents=True)
    previous.write_text(json.dumps(benchmark_receipt(1.0)))
    old_bytes = previous.read_bytes()
    incoming = tmp_path / "incoming.json"
    report = benchmark_receipt(2.0)
    report["records"][0]["saved_checkpoint_bitwise_equal"] = True
    incoming.write_text(json.dumps(report))
    feature_index = tmp_path / "features.provenance.json"
    feature_index.write_text(
        json.dumps(
            {"features": {"safe_reference/train_safe.pt": "verified-feature-hash"}}
        )
    )
    archive = tmp_path / "bundle.zip"
    create_manifested_archive(
        archive,
        {relative: incoming, "source_slice/features.provenance.json": feature_index},
    )
    extract_verified_archive(archive, destination, refresh_benchmarks={relative})
    assert previous.read_bytes() == incoming.read_bytes()
    assert (
        destination / "source_slice/features.provenance.json"
    ).read_bytes() == feature_index.read_bytes()
    history = list((destination / "gpu/benchmark_history").rglob("*.json"))
    assert len(history) == 1
    assert history[0].read_bytes() == old_bytes
    extract_verified_archive(archive, destination, refresh_benchmarks={relative})
    assert list((destination / "gpu/benchmark_history").rglob("*.json")) == history


@pytest.mark.parametrize(
    "change",
    ["bank_sha256", "adapter_signature", "passed", "field_equality", "frozen_feature"],
)
def test_benchmark_refresh_keeps_scientific_artifacts_strict(tmp_path, change):
    destination = tmp_path / "results"
    relative = "optimization/compatibility.json"
    previous = destination / relative
    previous.parent.mkdir(parents=True)
    previous.write_text(json.dumps(benchmark_receipt()))
    original = previous.read_bytes()
    report = benchmark_receipt(3.0)
    if change in ("bank_sha256", "adapter_signature"):
        report[change] = "changed"
    elif change == "passed":
        report[change] = False
    elif change == "field_equality":
        report["records"][0]["bitwise_equal"] = False
    incoming = tmp_path / "incoming.json"
    incoming.write_text(json.dumps(report))
    files = {relative: incoming}
    frozen = destination / "source_slice/features/train.pt"
    if change == "frozen_feature":
        frozen.parent.mkdir(parents=True)
        frozen.write_bytes(b"old feature")
        replacement = tmp_path / "new.pt"
        replacement.write_bytes(b"changed feature")
        files["source_slice/features/train.pt"] = replacement
    archive = tmp_path / "bundle.zip"
    create_manifested_archive(archive, files)
    with pytest.raises(ValueError):
        extract_verified_archive(archive, destination, refresh_benchmarks={relative})
    assert previous.read_bytes() == original
    assert not (destination / "gpu/benchmark_history").exists()
    if change == "frozen_feature":
        assert frozen.read_bytes() == b"old feature"


def test_verified_transfer_can_be_repeated_without_changing_frozen_bytes(tmp_path):
    source = tmp_path / "source.txt"
    source.write_bytes(b"frozen\x00bytes")
    archive = tmp_path / "bundle.zip"
    create_manifested_archive(archive, {"nested/result.txt": source})
    destination = tmp_path / "destination"

    extract_verified_archive(archive, destination)
    extract_verified_archive(archive, destination)

    assert (destination / "nested/result.txt").read_bytes() == b"frozen\x00bytes"
    assert not list(destination.rglob("*.transfer"))


@pytest.mark.parametrize(
    "bad_name,bad_bytes,digest",
    [
        ("bad.txt", b"tampered", "0" * 64),
        ("../escape.txt", b"x", hashlib.sha256(b"x").hexdigest()),
        ("C:/escape.txt", b"x", hashlib.sha256(b"x").hexdigest()),
    ],
)
def test_invalid_archive_commits_no_valid_members(
    tmp_path, bad_name, bad_bytes, digest
):
    archive = tmp_path / "invalid.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("valid.txt", b"valid")
        bundle.writestr(bad_name, bad_bytes)
        bundle.writestr(
            "artifact_manifest.json",
            json.dumps(
                {
                    "valid.txt": hashlib.sha256(b"valid").hexdigest(),
                    bad_name: digest,
                }
            ),
        )
    destination = tmp_path / "destination"

    with pytest.raises(ValueError):
        extract_verified_archive(archive, destination)

    assert not destination.exists()
    assert not (tmp_path / "escape.txt").exists()


def test_frozen_conflict_does_not_overwrite_any_other_member(tmp_path):
    first, second = tmp_path / "first.txt", tmp_path / "second.txt"
    first.write_text("new status", encoding="utf-8")
    second.write_text("new frozen", encoding="utf-8")
    archive = tmp_path / "bundle.zip"
    create_manifested_archive(archive, {"status.json": first, "frozen.txt": second})
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "status.json").write_text("old status", encoding="utf-8")
    (destination / "frozen.txt").write_text("old frozen", encoding="utf-8")

    with pytest.raises(ValueError, match="Conflicting"):
        extract_verified_archive(archive, destination, replace_names={"status.json"})

    assert (destination / "status.json").read_text() == "old status"
    assert (destination / "frozen.txt").read_text() == "old frozen"


def test_only_explicitly_refreshable_status_is_replaced(tmp_path):
    source = tmp_path / "status.json"
    source.write_text('"complete"', encoding="utf-8")
    archive = tmp_path / "bundle.zip"
    create_manifested_archive(archive, {"status.json": source})
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "status.json").write_text('"running"', encoding="utf-8")

    extract_verified_archive(archive, destination, replace_names={"status.json"})

    assert json.loads((destination / "status.json").read_text()) == "complete"
