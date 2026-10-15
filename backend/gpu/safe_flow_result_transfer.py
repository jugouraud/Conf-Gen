"""Split and verify large Colab result archives using bounded transfers."""

import hashlib
import json
import os
from pathlib import Path


PART_BYTES = 64 * 1024 * 1024
REMOTE_RESULT = Path("/content/conf-gen-safe-flow-result.zip")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def split_result(archive: Path, part_bytes: int = PART_BYTES) -> Path:
    if part_bytes < 1:
        raise ValueError("Result part size must be positive")
    parts = []
    with archive.open("rb") as source:
        while data := source.read(part_bytes):
            part = Path(f"{archive}.part{len(parts):04d}")
            part.write_bytes(data)
            parts.append({"size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    if not parts:
        raise ValueError("Empty result archive")
    manifest = Path(f"{archive}.parts.json")
    manifest.write_text(json.dumps({
        "archive_size": archive.stat().st_size,
        "archive_sha256": _sha(archive), "parts": parts,
    }), encoding="utf-8")
    return manifest


def assemble_result(manifest: Path, destination: Path) -> None:
    details = json.loads(manifest.read_text(encoding="utf-8"))
    parts = details["parts"]
    if not isinstance(parts, list) or not 1 <= len(parts) <= 10000:
        raise ValueError("Invalid result part count")
    temporary = Path(f"{destination}.assembling")
    digest = hashlib.sha256()
    size = 0
    try:
        with temporary.open("wb") as output:
            for index, expected in enumerate(parts):
                part = Path(f"{destination}.part{index:04d}")
                if part.stat().st_size != expected["size"] or _sha(part) != expected["sha256"]:
                    raise ValueError(f"Invalid result part {index}")
                with part.open("rb") as source:
                    for block in iter(lambda: source.read(1048576), b""):
                        output.write(block)
                        digest.update(block)
                        size += len(block)
        if size != details["archive_size"] or digest.hexdigest() != details["archive_sha256"]:
            raise ValueError("Result archive failed integrity check")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    manifest_path = split_result(REMOTE_RESULT)
    print(f"Result archive split into {len(json.loads(manifest_path.read_text())['parts'])} parts")
