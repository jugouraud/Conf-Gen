"""Failed serialization and packaging keep the last complete artifact usable."""

import pytest

from backend.src.common.archives import (
    create_manifested_archive,
    extract_verified_archive,
)
from backend.src.common.storage import read_json, write_json

pytestmark = pytest.mark.integration


def test_json_serialization_failure_preserves_previous_complete_document(tmp_path):
    path = tmp_path / "status.json"
    write_json(path, {"status": "complete"})
    before = path.read_bytes()
    cycle = {}
    cycle["self"] = cycle
    with pytest.raises(ValueError, match="Circular reference"):
        write_json(path, cycle)
    assert path.read_bytes() == before
    assert read_json(path) == {"status": "complete"}
    assert set(tmp_path.iterdir()) == {path}


def test_failed_new_package_preserves_previous_verified_zip(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"frozen source")
    target = tmp_path / "package.zip"
    create_manifested_archive(target, {"source.bin": source}, {})
    before = target.read_bytes()
    source.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="Stale or corrupt"):
        create_manifested_archive(
            target, {"source.bin": source}, {"source.bin": "0" * 64}
        )
    assert target.read_bytes() == before
    destination = tmp_path / "unpacked"
    extract_verified_archive(target, destination)
    assert (destination / "source.bin").read_bytes() == b"frozen source"
    assert not list(tmp_path.glob("*.tmp"))
