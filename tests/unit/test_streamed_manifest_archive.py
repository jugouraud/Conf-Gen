"""Single-pass staging must validate hashes and preserve verified archive semantics."""

import hashlib
import zipfile

import pytest

from backend.src.common.archives import (
    create_manifested_archive,
    extract_verified_archive,
)

pytestmark = pytest.mark.unit


def test_streamed_archive_verifies_frozen_query_hash_and_round_trips(tmp_path):
    source = tmp_path / "query.pt"
    source.write_bytes(bytes(range(256)) * 4096)
    name = "results/queries/B1/query.pt"
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    archive = tmp_path / "inputs.zip"
    manifest = create_manifested_archive(archive, {name: source}, {name: expected})
    assert manifest == {name: expected}
    destination = tmp_path / "received"
    extract_verified_archive(archive, destination)
    assert (destination / name).read_bytes() == source.read_bytes()
    with zipfile.ZipFile(archive) as packed:
        assert packed.getinfo(name).compress_type == zipfile.ZIP_DEFLATED


def test_stale_query_is_rejected_during_staging(tmp_path):
    source = tmp_path / "query.pt"
    source.write_bytes(b"changed query")
    with pytest.raises(ValueError, match="Stale or corrupt staged"):
        create_manifested_archive(
            tmp_path / "inputs.zip", {"query.pt": source}, {"query.pt": "0" * 64}
        )


def test_unsafe_archive_paths_and_unstaged_expected_hashes_are_rejected(tmp_path):
    source = tmp_path / "query.pt"
    source.write_bytes(b"query")
    with pytest.raises(ValueError, match="Invalid artifact path"):
        create_manifested_archive(tmp_path / "bad.zip", {"../outside.pt": source}, {})
    with pytest.raises(ValueError, match="unstaged"):
        create_manifested_archive(
            tmp_path / "bad.zip", {"query.pt": source}, {"absent": "0" * 64}
        )
