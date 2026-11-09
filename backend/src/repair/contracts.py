"""Explicit state passed between repair workflow steps."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.src.repair.steering import (
        RepairConstraintScores,
        RepairScheduleChoice,
        TemporalSteeringGeometry,
    )


@dataclass
class RepairExperiment:
    """Frozen detection inputs, repair settings, and initialized runtime adapters."""

    settings: dict
    output: Path
    source: Path
    config: dict
    bank: dict
    calibration: dict
    groups: dict
    distances: dict
    features: Any
    fingerprint: dict
    normalization: dict = field(default_factory=dict)
    adapter: Any = None
    evaluator: Any = None


@dataclass
class PreparedRepair:
    """A preparation result with stable artifact metadata and optional bound runtime."""

    mode: str
    output: Path
    summary: dict
    experiment: RepairExperiment | None = None


@dataclass
class RepairSearchResult:
    """Search decisions and fresh field evidence for one prompt and admission mode."""

    original: RepairConstraintScores
    anchor: dict | None
    profiles: dict
    variants: list[tuple[RepairScheduleChoice, str]]
    stats: dict
    anchor_embeddings: Any = None
    geometry_context: tuple[Any, Any, TemporalSteeringGeometry] | None = None
