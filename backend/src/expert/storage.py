"""Bounded strict JSON, scientific identities, atomic receipts and one-owner lock."""

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np

from backend.src.common.storage import write_json

MAX_JSON_BYTES = 16 * 1024 * 1024


def json_data(value: object) -> object:
    if isinstance(value, np.ndarray):
        return json_data(value.tolist())
    if isinstance(value, dict):
        return {key: json_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_data(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            json_data(value), sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _unique_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def parse_json(data: bytes) -> object:
    if len(data) > MAX_JSON_BYTES:
        raise ValueError("JSON byte limit exceeded")
    # Bound nesting before the standard parser, ignoring brackets within strings.
    depth, quoted, escaped = 0, False, False
    for char in data:
        if quoted:
            if escaped:
                escaped = False
            elif char == 92:
                escaped = True
            elif char == 34:
                quoted = False
        elif char == 34:
            quoted = True
        elif char in (91, 123):
            depth += 1
            if depth > 32:
                raise ValueError("JSON nesting limit exceeded")
        elif char in (93, 125):
            depth -= 1

    def reject_constant(value: str):
        raise ValueError("Nonfinite JSON number")

    def finite_float(value: str) -> float:
        number = float(value)
        if not np.isfinite(number):
            raise ValueError("Nonfinite JSON number")
        return number

    return json.loads(
        data.decode("utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=reject_constant,
        parse_float=finite_float,
    )


def read_record(path: Path) -> object:
    with path.open("rb") as stream:
        return parse_json(stream.read(MAX_JSON_BYTES + 1))


def publish_record(path: Path, value: object) -> None:
    payload = json_data(value)
    # Validate strict serialization before the existing atomic writer is invoked.
    encoded = json.dumps(payload, allow_nan=False)
    if len(encoded.encode()) > MAX_JSON_BYTES:
        raise ValueError("Artifact byte limit exceeded")
    write_json(path, payload)


def publish_jsonl(path: Path, rows: list[dict]) -> None:
    """Stream bounded dataset exports and replace only a complete file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    total = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=".dataset-", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            for row in rows:
                line = (
                    json.dumps(json_data(row), sort_keys=True, allow_nan=False) + "\n"
                ).encode()
                total += len(line)
                if len(line) > MAX_JSON_BYTES or total > 64 * 1024 * 1024:
                    raise ValueError("Dataset export byte budget exceeded")
                stream.write(line)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def dataset_lock(output: Path) -> Iterator[None]:
    """OS lock releases on coordinator death; no stale PID lock to override."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".writer.lock").open("a+b") as handle:
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("Another coordinator owns this dataset") from error
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError("Another coordinator owns this dataset") from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
