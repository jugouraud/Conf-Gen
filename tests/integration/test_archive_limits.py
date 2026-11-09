"""Expansion budgets and frozen recovery selection at the real ZIP boundary."""

import stat
import zipfile

import pytest

from backend.src.common.archives import (
    ArchiveLimits,
    create_manifested_archive,
    extract_verified_archive,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "limit", ["members", "manifest", "member_bytes", "total_bytes"]
)
def test_archive_limits_reject_before_committing_any_files(tmp_path, limit):
    source = tmp_path / "source.txt"
    source.write_bytes(b"a" * 64)
    archive = tmp_path / "archive.zip"
    create_manifested_archive(archive, {"first.txt": source, "second.txt": source})
    options = (
        {"max_members": 2}
        if limit == "members"
        else {"max_manifest_bytes": 1}
        if limit == "manifest"
        else {"max_member_bytes": 32}
        if limit == "member_bytes"
        else {"max_total_bytes": 64}
    )
    destination = tmp_path / "received"
    with pytest.raises(ValueError, match="limit"):
        extract_verified_archive(archive, destination, limits=ArchiveLimits(**options))
    assert not destination.exists()


def test_recovery_selection_binds_file_bytes_to_trusted_manifest(tmp_path):
    source = tmp_path / "source.txt"
    source.write_bytes(b"frozen source")
    archive = tmp_path / "cache.zip"
    manifest = create_manifested_archive(
        archive, {"source/field.pt": source, "unselected.txt": source}
    )
    destination = tmp_path / "received"
    with pytest.raises(ValueError, match="frozen inputs"):
        extract_verified_archive(
            archive, destination, expected_hashes={"source/field.pt": "0" * 64}
        )
    assert not destination.exists()
    selected = {"source/field.pt": manifest["source/field.pt"]}
    assert (
        extract_verified_archive(archive, destination, expected_hashes=selected)
        == selected
    )
    assert (destination / "source/field.pt").read_bytes() == source.read_bytes()
    assert not (destination / "unselected.txt").exists()


def test_archive_symlinks_are_rejected_before_extraction(tmp_path):
    archive = tmp_path / "symlink.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        info = zipfile.ZipInfo("link")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        bundle.writestr(info, "../outside")
    with pytest.raises(ValueError, match="symlink"):
        extract_verified_archive(archive, tmp_path / "received")
    assert not (tmp_path / "received").exists()
