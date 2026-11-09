"""Artifact I/O, integrity checks, and repository-relative path resolution."""

import csv
import hashlib
import json
import os
import shutil
import tempfile
import tomllib
from pathlib import Path, PurePosixPath

from backend.src.common.paths import PROJECT_ROOT

HASH_READ_SIZE = 1024 * 1024


def relocate_path(value: str | Path) -> Path:
    """Resolve absolute paths or paths relative to the repository root."""
    normalized = str(value).replace("\\", "/")
    path = Path(normalized)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def read_toml_config(path: Path) -> dict:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def resolve_config_path(cfg: dict, name: str) -> Path:
    path = Path(cfg["paths"][name])
    return relocate_path(path)


def calculate_file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(HASH_READ_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def calculate_source_fingerprints(paths):
    """Hash the explicitly selected implementation files relative to the checkout."""
    return {name: calculate_file_sha256(PROJECT_ROOT / name) for name in paths}


def write_json(path: Path, data: object) -> None:
    """Publish complete UTF-8 JSON with a unique temporary file on the same disk."""
    encoded = json.dumps(data, sort_keys=True, indent=2, default=str) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def link_or_copy_frozen_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if source.samefile(target):
            return
        if calculate_file_sha256(source) != calculate_file_sha256(target):
            raise ValueError(f"Different frozen artifact: {target}")
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def resolve_image_path(root, relative):
    """Input: receipt image path. Output: an existing image contained in the run."""
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or any(c in relative for c in "\\:"):
        raise ValueError("Image receipt path must stay within the repair run")
    target = (root / relative).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError(f"Missing or invalid scoring image: {relative}")
    return target
