import pytest

from backend.src.common.storage import read_json, write_json
from backend.src.inference.scorers import validate_image_policy_scores
from backend.src.repair.score import collect_image_pairs, configure_image_scoring

pytestmark = pytest.mark.contract


def test_image_policies_accept_closed_probability_boundaries():
    assert validate_image_policy_scores(
        {"sexual": 0, "dangerous": 1, "violence": 0.5}
    ) == {
        "sexual": 0.0,
        "dangerous": 1.0,
        "violence": 0.5,
    }


@pytest.mark.parametrize("probability", [-0.001, 1.001, float("nan"), float("inf")])
def test_image_policies_reject_non_probabilities(probability):
    with pytest.raises(ValueError, match="finite probabilities"):
        validate_image_policy_scores(
            {"sexual": probability, "dangerous": 0, "violence": 0}
        )


@pytest.mark.parametrize(
    "scores", [{"sexual": 0}, {"sexual": 0, "dangerous": 0, "violence": 0, "extra": 0}]
)
def test_image_policy_schema_requires_exactly_the_declared_policies(scores):
    with pytest.raises(ValueError, match="Missing or unexpected"):
        validate_image_policy_scores(scores)


@pytest.mark.parametrize(
    "defect,message",
    [
        ("seed", "Missing same-prompt"),
        ("duplicate_baseline", "Ambiguous original"),
        ("duplicate_repair", "Duplicate repair"),
    ],
)
def test_saved_pair_contract_rejects_unmatched_or_ambiguous_records(
    saved_repair_run, defect, message
):
    receipt_path = saved_repair_run / "pilot/records/prompt-1.json"
    receipt = read_json(receipt_path)
    if defect == "seed":
        receipt["rows"][1]["seed"] = 18
    elif defect == "duplicate_baseline":
        receipt["rows"].append(dict(receipt["rows"][0]))
    else:
        receipt["rows"].append(dict(receipt["rows"][1]))
    write_json(receipt_path, receipt)
    context = configure_image_scoring(["--output", str(saved_repair_run)], scorer=None)

    with pytest.raises(ValueError, match=message):
        collect_image_pairs(context)

    assert not (saved_repair_run / "image_scoring").exists()
