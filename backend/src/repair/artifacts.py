"""Repair progress, reconstruction evidence, and per-case export artifacts."""

import json
import zipfile

import numpy as np

from backend.src.common.storage import calculate_file_sha256, write_json


def update(output, phase, **details):
    temporary = output / "status.tmp.json"
    write_json(temporary, {"status": "running", "phase": phase, **details})
    temporary.replace(output / "status.json")


def export_files(output, name, paths):
    """Publish a closed checksummed archive atomically; include only explicit artifacts."""
    target = output / "exports" / f"{name}.zip"
    target.parent.mkdir(exist_ok=True)
    files = {path.relative_to(output).as_posix(): path for path in paths}
    temporary = target.with_suffix(".tmp")
    with zipfile.ZipFile(
        temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=1
    ) as bundle:
        for relative, path in files.items():
            bundle.write(path, relative)
        bundle.writestr(
            "artifact_manifest.json",
            json.dumps(
                {
                    relative: calculate_file_sha256(path)
                    for relative, path in files.items()
                }
            ),
        )
    temporary.replace(target)
    write_json(
        target.with_suffix(".json"),
        {
            "name": target.name,
            "sha256": calculate_file_sha256(target),
            "bytes": target.stat().st_size,
        },
    )


def compare_reconstruction(observed, expected, rtol, atol):
    if observed.shape != expected.shape or not np.isfinite(observed).all():
        raise ValueError("Reconstructed distance dimensions or values are invalid")
    difference = np.abs(observed - expected)
    return {
        "passed": bool(np.allclose(observed, expected, rtol=rtol, atol=atol)),
        "max_absolute_error": float(difference.max()),
        "max_relative_error": float(
            (difference / np.maximum(np.abs(expected), atol)).max()
        ),
        "rtol": rtol,
        "atol": atol,
        "comparisons": int(observed.size),
    }
