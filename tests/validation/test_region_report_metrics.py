"""Check per-method validation report aggregation."""

import csv
import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from backend.validation.region_report import _metric_report, analyze_validation_region
from frontend.coco_region_validation import ValidationItem


class RegionReportMetricTests(unittest.TestCase):
    def test_each_method_uses_its_own_costs_and_decisions(self) -> None:
        rows = [
            {"row_number": 1, "category": "a", "prompt_toxicity": 0.1,
             "order1_cost": 0.1, "order1_budget": 0.2, "order1_margin": -0.1, "order1_blocked": False},
            {"row_number": 2, "category": "b", "prompt_toxicity": 0.9,
             "order1_cost": 0.3, "order1_budget": 0.2, "order1_margin": 0.1, "order1_blocked": True},
        ]
        result = _metric_report(
            rows, rows, score_key="order1_cost", budget_key="order1_budget",
            blocked_key="order1_blocked",
        )
        self.assertEqual(result["prompt_level"]["blocked"], 1)
        self.assertEqual(result["prompt_level"]["block_rate"], 0.5)
        self.assertAlmostEqual(result["prompt_level"]["distance_median"], 0.2)
        self.assertEqual(len(result["by_category"]), 2)
        self.assertAlmostEqual(result["toxicity_comparison"]["mean_difference"], 0.8)

    def test_csv_report_includes_all_three_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mapping = root / "prompts.csv"
            mapping.write_text("prompt,categories,prompt_toxicity\nfirst,safe,0.1\nsecond,other,0.9\nfirst,safe,0.1\n", encoding="utf-8")
            region = root / "region.npz"
            region.touch()
            transport = root / "transport.sqlite3"
            transport.touch()
            items = [ValidationItem(number, prompt, category, toxicity) for number, prompt, category, toxicity in (
                (1, "first", "safe", 0.1), (2, "second", "other", 0.9), (3, "first", "safe", 0.1),
            )]
            validator = Mock()
            validator.encode_prompts_with_clouds.return_value = (
                np.asarray([[0.2, 0.0], [0.9, 0.0]], dtype=np.float32),
                [np.zeros((1, 2), dtype=np.float32)] * 2,
            )
            validator.region.score.return_value = np.asarray([0.2, 0.9])
            validator.region.radius = 0.5
            validator.region.anchors = np.zeros((10, 2), dtype=np.float32)
            validator.region.whiten.side_effect = lambda vectors: vectors
            validator.transport.summary = {"order1_radius": 0.02, "orderinf_radius": 1.0,
                                           "n_task_reference": 5}
            validator.transport.score.side_effect = [(0.01, 1.2), (0.03, 0.8)]
            report = analyze_validation_region(
                prompts_path=mapping, region_path=region, transport_path=transport,
                output_path=root / "report.json",
                validator=validator, items=items,
            )
            self.assertEqual(set(report["metrics"]), {"nearest_anchor", "order1", "orderinf"})
            self.assertEqual(report["metrics"]["nearest_anchor"]["prompt_level"]["blocked"], 1)
            self.assertEqual(report["metrics"]["order1"]["prompt_level"]["blocked"], 1)
            self.assertEqual(report["metrics"]["orderinf"]["prompt_level"]["blocked"], 2)
            self.assertEqual(report["metrics"]["orderinf"]["unique_prompt_level"]["count"], 2)
            nearest_toxicity = report["metrics"]["nearest_anchor"]["toxicity_comparison"]
            self.assertEqual(nearest_toxicity["blocked"]["count"], 1)
            self.assertAlmostEqual(nearest_toxicity["mean_difference"], 0.8)
            self.assertAlmostEqual(report["metrics"]["orderinf"]["toxicity_comparison"]["mean_difference"], -0.8)
            self.assertEqual([row["prompt_toxicity"] for row in report["rows"]], [0.1, 0.9, 0.1])
            with (root / "report.csv").open(newline="", encoding="utf-8") as handle:
                saved_rows = list(csv.DictReader(handle))
            self.assertEqual([row["prompt_toxicity"] for row in saved_rows], ["0.1", "0.9", "0.1"])


if __name__ == "__main__":
    unittest.main()
