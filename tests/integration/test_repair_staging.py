"""Execute staged repair programs against real isolated files without a Colab account."""

import shutil
import subprocess
import sys

import pytest

from backend.runtime.repair_colab import RepairColabConnection
from backend.src.common.archives import create_manifested_archive
from backend.src.common.paths import PROJECT_ROOT
from backend.src.common.storage import calculate_file_sha256, read_json, write_json

pytestmark = pytest.mark.integration


def staging_inputs(tmp_path, monkeypatch):
    output = tmp_path / "local"
    remote = tmp_path / "remote"
    field = tmp_path / "field.bin"
    field.write_bytes(b"verified recovered field")
    cache = tmp_path / "recovered.zip"
    hashes = create_manifested_archive(
        cache, {"source/field.bin": field, "unselected.bin": field}
    )
    config = {
        "command": ["colab"],
        "session": "synthetic",
        "gpu": "A100",
        # Rooted POSIX spelling also resolves on the current Windows drive for
        # this local transport fixture; production still requires a POSIX root.
        "remote_root": remote.as_posix().removeprefix(remote.drive),
    }
    write_json(
        output / "experiment.json",
        {
            "config": {"colab": config},
            "reuse": {
                "archive_sha256": calculate_file_sha256(cache),
                "files": {"source/field.bin": hashes["source/field.bin"]},
            },
        },
    )
    package = tmp_path / "prepared.zip"
    files = {
        name: PROJECT_ROOT / name
        for name in (
            "backend/__init__.py",
            "backend/runtime/__init__.py",
            "backend/src/common/archives.py",
            "backend/src/__init__.py",
            "backend/src/common/storage.py",
            "backend/src/common/paths.py",
        )
    }
    files["results/experiment.json"] = output / "experiment.json"
    create_manifested_archive(package, files)
    write_json(
        output / "validation.json",
        {
            "package_path": str(package),
            "package_sha256": calculate_file_sha256(package),
        },
    )
    metadata = tmp_path / "parts.json"
    write_json(
        metadata,
        {
            "sha256": calculate_file_sha256(cache),
            "parts": [{"name": cache.name, "sha256": calculate_file_sha256(cache)}],
        },
    )
    write_json(
        output / "gpu/reuse_upload.json",
        {"metadata": str(metadata), "parts_directory": str(tmp_path)},
    )
    connection = RepairColabConnection(output)
    monkeypatch.setattr(connection, "verify_hardware", lambda: None)
    monkeypatch.setattr(connection, "stage_model_auth", lambda: None)

    def upload(operation, _session_flag, _session, local, destination):
        assert operation == "upload"
        shutil.copyfile(local, destination)

    monkeypatch.setattr(connection, "invoke", upload)

    def execute(source):
        return subprocess.run(
            [sys.executable, "-c", source],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout

    monkeypatch.setattr(connection, "execute", execute)
    return connection, output, remote, cache


def test_prepared_package_and_selected_cache_round_trip_through_staging(
    tmp_path, monkeypatch
):
    connection, output, remote, _ = staging_inputs(tmp_path, monkeypatch)
    assert connection.stage()["status"] == "complete"
    assert (
        remote / "results/source/field.bin"
    ).read_bytes() == b"verified recovered field"
    assert not (remote / "results/unselected.bin").exists()
    assert read_json(output / "progress.json")["status"] == "starting"


def test_changed_cache_part_is_rejected_before_any_remote_file_write(
    tmp_path, monkeypatch
):
    connection, _, remote, cache = staging_inputs(tmp_path, monkeypatch)
    cache.write_bytes(b"changed cache")
    with pytest.raises(ValueError, match="archive part changed"):
        connection.stage()
    assert not remote.exists()
    assert not remote.with_name(remote.name + "-inputs.zip").exists()
