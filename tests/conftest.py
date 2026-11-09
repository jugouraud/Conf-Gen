"""Deterministic generated examples and a saved run isolated to each test."""

import pytest
from hypothesis import settings
from PIL import Image

from backend.src.common.storage import calculate_file_sha256, write_json

settings.register_profile(
    "default", max_examples=60, derandomize=True, deadline=None, database=None
)
settings.load_profile("default")


@pytest.fixture
def saved_repair_run(tmp_path):
    """A single same-prompt/seed pair with real PNG files and checksum receipts."""
    root = tmp_path / "repair run"
    write_json(root / "experiment.json", {"name": "isolated repair"})
    images = root / "pilot/images"
    images.mkdir(parents=True)
    Image.new("RGB", (3, 3), "red").save(images / "original.png")
    Image.new("RGB", (3, 3), "blue").save(images / "repaired.png")
    common = {
        "cohort": "pilot",
        "split": "unsafe_test",
        "id": "prompt-1",
        "prompt": "A synthetic scene <script>alert(1)</script>",
        "category": "Violence",
        "mode": "safe_attraction",
        "seed": 17,
        "variant": "constant",
        "budget": None,
    }
    rows = [
        dict(common, method="R0", image="pilot/images/original.png"),
        dict(common, method="R1", image="pilot/images/repaired.png"),
    ]
    write_json(
        root / "pilot/records/prompt-1.json",
        {
            "experiment_sha256": calculate_file_sha256(root / "experiment.json"),
            "artifacts": {
                row["image"]: calculate_file_sha256(root / row["image"]) for row in rows
            },
            "rows": rows,
        },
    )
    return root
