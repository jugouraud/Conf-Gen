"""Run the five-category nudity detection experiment."""

from pathlib import Path

from backend.src.common.archives import (
    collect_python_sources,
    create_manifested_archive,
)
from backend.src.common.paths import PROJECT_ROOT


def run_detection_packaging(output: Path) -> str:
    """Package prepared inputs and source with a checksum manifest; never launch compute."""
    files = {
        f"results/{path.relative_to(output).as_posix()}": path
        for pattern in (
            "experiment.json",
            "resource_plan.json",
            "trajectory/splits/*.jsonl",
        )
        for path in output.glob(pattern)
    }
    if "results/experiment.json" not in files:
        raise ValueError("Prepare the experiment before packaging")
    files.update(collect_python_sources())
    for name in (
        "backend/runtime/detection.py",
        "backend/runtime/colab.py",
        "backend/src/common/archives.py",
        "pyproject.toml",
        "uv.lock",
    ):
        files[name] = PROJECT_ROOT / name
    target = output.parent / "gpu-inputs.zip"
    create_manifested_archive(target, files)
    return str(target)
