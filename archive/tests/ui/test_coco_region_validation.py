"""Checks CSV prompt loading for live COCO validation."""

import csv
import tempfile
import unittest
from pathlib import Path

from frontend.coco_region_validation import load_validation_items


class ValidationItemTests(unittest.TestCase):
    def test_csv_rows_keep_order_and_duplicate_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts = root / "prompts.csv"
            with prompts.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["prompt", "categories", "prompt_toxicity"])
                writer.writerows([("The first prompt", "violence", "0.1"),
                                  ("The second prompt", "sexual", "0.8"),
                                  ("The first prompt", "violence", "0.2")])

            items = load_validation_items(prompts)

            self.assertEqual([item.prompt for item in items],
                             ["The first prompt", "The second prompt", "The first prompt"])
            self.assertEqual([item.row_number for item in items], [1, 2, 3])
            self.assertEqual([item.categories for item in items], ["violence", "sexual", "violence"])
            self.assertEqual([item.prompt_toxicity for item in items], [0.1, 0.8, 0.2])

    def test_rejects_non_numeric_toxicity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prompts = Path(directory) / "prompts.csv"
            prompts.write_text("prompt,prompt_toxicity\nexample,NaN\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid prompt_toxicity"):
                load_validation_items(prompts)


if __name__ == "__main__":
    unittest.main()
