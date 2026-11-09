"""Exercise the saved-image workflow with only expensive inference replaced."""

import csv

import pytest

from backend.src.common.storage import calculate_file_sha256, read_json
from backend.src.repair.score import run_repair_scoring
from tests.helpers.image_scorer import PixelPolicyScorer

pytestmark = pytest.mark.e2e


def test_saved_images_produce_paired_reports_and_verified_provenance(saved_repair_run):
    result = run_repair_scoring(
        ["--output", str(saved_repair_run)], scorer=PixelPolicyScorer()
    )

    assert result["status"] == "complete"
    assert result["unique_images"] == 2
    (pair,) = result["pairs"]
    assert pair["original_toxicity"] == 0.8
    assert pair["repaired_toxicity"] == 0.2
    assert pair["toxicity_reduction"] == pytest.approx(0.6)
    assert pair["unsafe_to_safe"] is True
    output = saved_repair_run / "image_scoring/shieldgemma2/pilot"
    with (output / "pairs.csv").open(encoding="utf-8", newline="") as stream:
        (csv_pair,) = csv.DictReader(stream)
    assert csv_pair["id"] == "prompt-1"
    assert float(csv_pair["toxicity_delta"]) == pytest.approx(-0.6)
    assert read_json(output / "comparison.json") == result
    gallery = (output / "report.html").read_text(encoding="utf-8")
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in gallery
    assert "<script>alert(1)</script>" not in gallery
    assert "pilot/images/original.png" in gallery
    assert "pilot/images/repaired.png" in gallery
    provenance = read_json(output / "provenance.json")
    for relative, digest in provenance["artifacts"].items():
        assert calculate_file_sha256(output / relative) == digest, relative
