"""Repository roots shared by source, experiment entry points and test tooling."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_ROOT = PROJECT_ROOT / "data"

VALIDATION_ROOT = DATA_ROOT / "validation"

EXPERIMENT_ROOT = PROJECT_ROOT / "backend" / "experiments"
