"""Episode-level ownership, shared by probe fitting and both calibrations."""

from collections.abc import Mapping, Sequence

from backend.src.control.contracts import SPLITS, Example


def validate_partitions(
    groups: Mapping[str, Sequence[Example]],
) -> dict[str, tuple[str, ...]]:
    if set(groups) != set(SPLITS):
        raise ValueError("Exactly four episode partitions are required")
    owners: dict[str, str] = {}
    memberships = {}
    for split in SPLITS:
        if not groups[split]:
            raise ValueError("Empty episode partition")
        episodes = tuple(sorted({row.context.episode for row in groups[split]}))
        for episode in episodes:
            if episode in owners:
                raise ValueError("Shared-episode leakage across partitions")
            owners[episode] = split
        memberships[split] = episodes
    return memberships


def require_independent_examples(examples: Sequence[Example]) -> None:
    """One score per independent episode for each claimed marginal target."""
    if len({e.context.episode for e in examples}) != len(examples):
        raise ValueError(
            "Calibration needs one prespecified query per independent episode"
        )
