"""Configuration, paths, and artifact serialization for flow runs."""

import hashlib
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("safe_probe_source", "safe_reference", "safe_validation", "safe_test", "unsafe_test")
SAFE_INPUT = {"safe_reference", "safe_calibration", "safe_test"}
HASH_READ_SIZE = 1024 * 1024


# Hierarchy 3: atomic actions.

def load_config(path: Path) -> dict:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def resolved_path(cfg: dict, name: str) -> Path:
    path = Path(cfg["paths"][name])
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(HASH_READ_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True, indent=2, default=str) + "\n", encoding="utf-8")
