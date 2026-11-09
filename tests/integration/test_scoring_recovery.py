import pytest

from backend.src.common.storage import read_json
from backend.src.repair.score import run_repair_scoring
from tests.helpers.image_scorer import (
    CachedOnlyPolicyScorer,
    PixelPolicyScorer,
    UnavailablePolicyScorer,
)

pytestmark = pytest.mark.integration


def test_retry_of_completed_scoring_reuses_verified_scores_without_inference(
    saved_repair_run,
):
    args = ["--output", str(saved_repair_run)]
    first = run_repair_scoring(args, scorer=PixelPolicyScorer())
    output = saved_repair_run / "image_scoring/shieldgemma2/pilot"
    receipt_before = (output / "provenance.json").read_bytes()

    retried = run_repair_scoring(args, scorer=CachedOnlyPolicyScorer())

    assert retried == first
    assert (output / "provenance.json").read_bytes() == receipt_before
    assert len(list((output / "images").glob("*.json"))) == 2


def test_inference_failure_records_failed_state_and_allows_a_clean_retry(
    saved_repair_run,
):
    args = ["--output", str(saved_repair_run)]
    output = saved_repair_run / "image_scoring/shieldgemma2/pilot"

    with pytest.raises(TimeoutError, match="inference unavailable"):
        run_repair_scoring(args, scorer=UnavailablePolicyScorer())

    assert read_json(output / "status.json") == {
        "status": "failed",
        "error": "inference unavailable",
        "error_type": "TimeoutError",
    }
    assert not (output / "comparison.json").exists()
    assert not list(output.rglob("*.tmp"))
    run_repair_scoring(args, scorer=PixelPolicyScorer())
    assert read_json(output / "status.json")["status"] == "complete"


def test_modified_input_image_is_rejected_before_creating_scoring_outputs(
    saved_repair_run,
):
    original = saved_repair_run / "pilot/images/original.png"
    original.write_bytes(b"changed after receipt")

    with pytest.raises(ValueError, match="checksum changed"):
        run_repair_scoring(
            ["--output", str(saved_repair_run)], scorer=PixelPolicyScorer()
        )

    assert not (saved_repair_run / "image_scoring").exists()


def test_changed_settings_cannot_overwrite_a_completed_scoring_run(saved_repair_run):
    args = ["--output", str(saved_repair_run)]
    run_repair_scoring(args, scorer=PixelPolicyScorer())
    output = saved_repair_run / "image_scoring/shieldgemma2/pilot"
    report_before = (output / "comparison.json").read_bytes()
    status_before = (output / "status.json").read_bytes()

    with pytest.raises(ValueError, match="inputs/settings changed"):
        run_repair_scoring([*args, "--threshold", "0.9"], scorer=PixelPolicyScorer())

    assert (output / "comparison.json").read_bytes() == report_before
    assert (output / "status.json").read_bytes() == status_before


def test_corrupted_cached_score_cannot_replace_a_completed_report(saved_repair_run):
    args = ["--output", str(saved_repair_run)]
    run_repair_scoring(args, scorer=PixelPolicyScorer())
    output = saved_repair_run / "image_scoring/shieldgemma2/pilot"
    cache = next((output / "images").glob("*.json"))
    cache.write_text("{}", encoding="utf-8")
    before = (output / "comparison.json").read_bytes()

    with pytest.raises(ValueError, match="checksum changed"):
        run_repair_scoring(args, scorer=CachedOnlyPolicyScorer())

    assert (output / "comparison.json").read_bytes() == before
