"""Compose offline expert stages under one dataset-owner lease."""

from pathlib import Path

from backend.src.expert.audit import run_dataset_validation
from backend.src.expert.config import ExpertConfig
from backend.src.expert.export import run_expert_export
from backend.src.expert.generate import run_expert_generation
from backend.src.expert.normalize import run_training_normalization
from backend.src.expert.prepare import run_expert_preparation
from backend.src.expert.storage import dataset_lock
from backend.src.expert.summary import run_dataset_summary


def run_expert_pipeline(
    config: ExpertConfig,
    scenes: list[dict],
    output: Path,
    *,
    resume: bool = False,
    workers: int | None = None,
    scene_source: dict | None = None,
) -> dict:
    """A missing complete marker means unfinished work, even if receipts exist."""
    with dataset_lock(output):
        manifest = run_expert_preparation(
            config, scenes, output, resume=resume, scene_source=scene_source
        )
        attempts = run_expert_generation(manifest, output, workers=workers)
        targets = run_expert_export(manifest, attempts, output)
        normalization = run_training_normalization(targets, config, output)
        validation = run_dataset_validation(output, mode="building")
        return run_dataset_summary(
            manifest, attempts, targets, normalization, validation, output
        )
