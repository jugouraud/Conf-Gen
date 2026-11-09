"""Safe-only normalization has an independently calculated temporal baseline."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from backend.src.repair.normalization import fit_safe_deviation_normalization

pytestmark = pytest.mark.unit


def normalization_experiment():
    fields = {
        "anchor": torch.zeros(4, 1, 1, 1),
        "first": torch.tensor([1.0, 1.0, 3.0, 3.0]).reshape(4, 1, 1, 1),
        "second": torch.tensor([3.0, 3.0, 5.0, 5.0]).reshape(4, 1, 1, 1),
    }
    return SimpleNamespace(
        settings={"experiment": {"normalization_limit": 2}},
        groups={
            "safe_reference": [{"id": "anchor"}],
            "safe_validation": [{"id": "first"}, {"id": "second"}],
            "unsafe_test": [{"id": "never-load"}],
        },
        distances={"safe__safe_validation": np.array([[1.0], [3.0]])},
        bank={"time_points": 2, "trajectories": 2},
        features=SimpleNamespace(load_signature=lambda split, row: fields[row["id"]]),
    )


def test_normalization_uses_only_ordered_safe_validation_and_safe_references():
    actual = fit_safe_deviation_normalization(normalization_experiment())
    assert actual["location"] == [2.0, 4.0]
    assert actual["scale"] == [1.0, 1.0]
    assert actual["fit_ids"] == ["first", "second"]
    assert actual["fit_split"] == "safe_validation"


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_normalization_rejects_nonpositive_or_noninteger_limit(limit):
    experiment = normalization_experiment()
    experiment.settings["experiment"]["normalization_limit"] = limit
    with pytest.raises(ValueError, match="positive integer"):
        fit_safe_deviation_normalization(experiment)
