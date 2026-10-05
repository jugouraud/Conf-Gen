"""Default project data paths used by local commands."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
FAIRNESS_ROOT = DATA_ROOT / "fairness"
COCO_ROOT = DATA_ROOT / "coco"
VALIDATION_ROOT = DATA_ROOT / "validation"
RUNS_ROOT = DATA_ROOT / "runs"
MODELS_ROOT = DATA_ROOT / "models"
