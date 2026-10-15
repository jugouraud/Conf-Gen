"""Saved analysis should load quickly and reject changed inputs."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from backend.validation.report_store import (
    load_saved_report, source_signature, summary_path, toxicity_comparison,
    write_report_artifacts,
)
from frontend.coco_region_validation import ValidationItem


class ReportStoreTests(unittest.TestCase):
    def test_compact_summary_and_csv_reuse_current_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts = root / "prompts.csv"
            prompts.write_text("prompt,prompt_toxicity\nfirst,0.1\nsecond,0.9\n", encoding="utf-8")
            region = root / "region.npz"
            region.touch()
            items = [ValidationItem(1, "first", "unknown", 0.1),
                     ValidationItem(2, "second", "unknown", 0.9)]
            rows = [
                {"row_number": 1, "prompt": "first", "category": "unknown", "prompt_toxicity": 0.1,
                 "distance": 0.1, "radius": 0.5, "margin": -0.4, "blocked": False},
                {"row_number": 2, "prompt": "second", "category": "unknown", "prompt_toxicity": 0.9,
                 "distance": 0.9, "radius": 0.5, "margin": 0.4, "blocked": True},
            ]
            report = {"prompt_source": str(prompts.resolve()), "region": str(region.resolve()),
                      "prompt_level": {"count": 2},
                      "metrics": {"nearest_anchor": {"blocked_key": "blocked",
                           "toxicity_comparison": toxicity_comparison(rows, "blocked")}}, "rows": rows}
            output = root / "report.json"
            write_report_artifacts(report, output, source_signature(prompts, region, None))

            loaded = load_saved_report(output, items, prompts_path=prompts, region_path=region, transport_path=None)

            self.assertEqual(loaded["rows"], rows)
            self.assertNotIn("rows", json.loads(summary_path(output).read_text(encoding="utf-8")))
            prompts.write_text("prompt,prompt_toxicity\nfirst,0.2\nsecond,0.9\n", encoding="utf-8")
            self.assertIsNone(load_saved_report(
                output, items, prompts_path=prompts, region_path=region, transport_path=None,
            ))

    def test_legacy_report_gains_toxicity_without_rescoring(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts = root / "prompts.csv"
            prompts.write_text("prompt,prompt_toxicity\nfirst,0.1\nsecond,0.9\n", encoding="utf-8")
            region = root / "region.npz"
            region.touch()
            items = [ValidationItem(1, "first", "unknown", 0.1),
                     ValidationItem(2, "second", "unknown", 0.9)]
            output = root / "report.json"
            output.write_text(json.dumps({
                "prompt_source": str(prompts.resolve()), "region": str(region.resolve()),
                "metrics": {"nearest_anchor": {"blocked_key": "blocked"}},
                "rows": [
                    {"row_number": 1, "prompt": "first", "category": "unknown", "blocked": False},
                    {"row_number": 2, "prompt": "second", "category": "unknown", "blocked": True},
                ],
            }), encoding="utf-8")

            loaded = load_saved_report(output, items, prompts_path=prompts, region_path=region, transport_path=None)

            self.assertAlmostEqual(loaded["metrics"]["nearest_anchor"]["toxicity_comparison"]["mean_difference"], 0.8)
            with output.with_suffix(".csv").open(newline="", encoding="utf-8") as handle:
                self.assertEqual([row["prompt_toxicity"] for row in csv.DictReader(handle)], ["0.1", "0.9"])
            self.assertTrue(summary_path(output).exists())


if __name__ == "__main__":
    unittest.main()
