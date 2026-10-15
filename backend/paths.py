"""Default project data paths used by local commands."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
VALIDATION_ROOT = DATA_ROOT / "validation"
