import json

import pytest
import torch

from backend.src.fields.signatures import (
    load_weighted_signature_vector,
    save_field_signature,
    validate_cached_signature,
)
from backend.src.repair.steering import SteeringSchedule

pytestmark = pytest.mark.contract


def test_saved_signature_is_loaded_as_a_quadrature_weighted_vector(tmp_path):
    cfg = {"paths": {"output": str(tmp_path)}}
    values = torch.tensor([2.0, 4.0]).reshape(2, 1, 1, 1)
    path = tmp_path / "features/safe_test/prompt-1.pt"
    save_field_signature(path, "prompt-1", values, "bank-1", "model-1")
    bank = {"states": torch.zeros_like(values), "weights": torch.tensor([0.25, 0.75])}

    loaded = load_weighted_signature_vector(
        cfg, "safe_test", {"id": "prompt-1"}, bank, "bank-1"
    )

    torch.testing.assert_close(loaded, torch.tensor([1.0, 3.4641016]))
    assert not path.with_suffix(".tmp").exists()


@pytest.mark.parametrize(
    "bank,model", [("new-bank", "model-1"), ("bank-1", "new-model")]
)
def test_cached_signature_rejects_a_changed_bank_or_model(tmp_path, bank, model):
    path = tmp_path / "signature.pt"
    save_field_signature(path, "prompt-1", torch.ones(1, 1, 1, 1), "bank-1", "model-1")

    with pytest.raises(ValueError, match="Stale cached feature"):
        validate_cached_signature(path, bank, model)


def test_schedule_json_record_preserves_replayable_strengths():
    schedule = SteeringSchedule("window", (0, 0.5, 1), (0, 1, 0))
    record = json.loads(json.dumps(schedule.to_record()))
    assert record == {
        "name": "window",
        "times": [0, 0.5, 1],
        "strengths": [0, 1, 0],
        "interpolation": "smoothstep",
        "integrated_strength": 0.5,
    }
    replayed = SteeringSchedule(
        record["name"], tuple(record["times"]), tuple(record["strengths"])
    )
    assert replayed(0.25) == 0.5
